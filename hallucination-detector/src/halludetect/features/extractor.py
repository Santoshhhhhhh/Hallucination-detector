"""Per-layer hidden-state feature extraction from a frozen causal LM.

Two backends:

  * HFBackend   : loads a real HuggingFace causal LM (e.g. Qwen2.5-3B), runs a
                  forward pass with output_hidden_states=True, and mean-pools the
                  *answer-span* tokens at every layer.

  * MockBackend : produces deterministic pseudo hidden-states that ENCODE the
                  label-relevant signal in specific layers, so the sweep/probe
                  pipeline is fully runnable and testable without torch/HF or a GPU.
                  It mimics the empirical finding that groundedness signal peaks in
                  intermediate layers.

Both return a tensor of shape [num_layers, hidden_dim] per example. The extractor
stacks these into X of shape [N, num_layers, hidden_dim].
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..data.schema import Example
from .. import LABEL2ID


class Backend(Protocol):
    num_layers: int
    hidden_dim: int

    def encode(self, ex: Example) -> np.ndarray:
        """Return per-layer features, shape [num_layers, hidden_dim]."""
        ...


# --------------------------------------------------------------------------- #
# Mock backend                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class MockBackend:
    """Deterministic synthetic hidden states.

    The signal for the 3-class label is injected with a per-layer *gain* that
    peaks in the middle of the network (a raised-cosine bump), reproducing the
    "hallucination signal concentrates in intermediate layers" phenomenon so a
    layer sweep will meaningfully prefer a middle layer.
    """
    num_layers: int = 28
    hidden_dim: int = 256
    noise: float = 1.0
    signal: float = 3.0
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # Fixed class prototype directions in hidden space.
        self._proto = rng.standard_normal((len(LABEL2ID), self.hidden_dim))
        self._proto /= np.linalg.norm(self._proto, axis=1, keepdims=True)
        # Per-layer gain: raised-cosine bump peaking mid-network.
        L = self.num_layers
        peak = 0.55 * L
        width = 0.28 * L
        idx = np.arange(L)
        self._gain = np.exp(-0.5 * ((idx - peak) / width) ** 2)

    def _example_noise(self, ex: Example) -> np.ndarray:
        h = hashlib.sha256(ex.id.encode()).digest()
        seed = int.from_bytes(h[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self.num_layers, self.hidden_dim))

    def encode(self, ex: Example) -> np.ndarray:
        base = self.noise * self._example_noise(ex)
        proto = self._proto[LABEL2ID[ex.label]]           # [hidden_dim]
        signal = self.signal * np.outer(self._gain, proto)  # [L, hidden_dim]
        return (base + signal).astype(np.float32)


# --------------------------------------------------------------------------- #
# HuggingFace backend                                                          #
# --------------------------------------------------------------------------- #
class HFBackend:
    """Real frozen causal LM backend.

    Requires `torch` and `transformers`. The model is loaded once, kept in eval
    mode with gradients disabled. For each example we build a RAG-style prompt,
    find the token indices belonging to the answer span, and mean-pool those
    tokens' hidden states at every layer.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B",
                 device: str | None = None, dtype: str = "bfloat16",
                 max_length: int = 2048):
        import torch  # local import so the mock path needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = getattr(torch, dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, output_hidden_states=True,
        ).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # hidden_states includes the embedding layer -> num_layers+1 tensors.
        self.num_layers = self.model.config.num_hidden_layers + 1
        self.hidden_dim = self.model.config.hidden_size

    def _prompt(self, ex: Example) -> tuple[str, str]:
        prefix = (
            f"Context:\n{ex.context}\n\n"
            f"Question: {ex.question}\n\n"
            f"Answer: "
        )
        return prefix, ex.answer

    @staticmethod
    def _mean_pool(hidden, start: int, end: int):
        # hidden: [seq, dim]; pool tokens [start, end)
        span = hidden[start:end]
        if span.shape[0] == 0:
            span = hidden[-1:]  # fallback: last token
        return span.mean(dim=0)

    def encode(self, ex: Example) -> np.ndarray:
        torch = self.torch
        prefix, answer = self._prompt(ex)
        prefix_ids = self.tokenizer(prefix, add_special_tokens=True).input_ids
        full_ids = self.tokenizer(prefix + answer, add_special_tokens=True).input_ids
        full_ids = full_ids[: self.max_length]
        start = min(len(prefix_ids), len(full_ids))
        end = len(full_ids)
        input_ids = torch.tensor([full_ids], device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        # out.hidden_states: tuple(len=num_layers) of [1, seq, dim]
        feats = []
        for layer_h in out.hidden_states:
            h = layer_h[0]  # [seq, dim]
            pooled = self._mean_pool(h, start, end)
            feats.append(pooled.float().cpu().numpy())
        return np.stack(feats, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def extract_features(examples: list[Example], backend: Backend,
                     verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for a list of examples.

    Returns:
        X: float32 array [N, num_layers, hidden_dim]
        y: int64 array   [N]
    """
    N = len(examples)
    X = np.empty((N, backend.num_layers, backend.hidden_dim), dtype=np.float32)
    y = np.empty((N,), dtype=np.int64)
    for i, ex in enumerate(examples):
        X[i] = backend.encode(ex)
        y[i] = LABEL2ID[ex.label]
        if verbose and (i + 1) % max(1, N // 10) == 0:
            print(f"  extracted {i + 1}/{N}")
    return X, y


def build_backend(kind: str = "auto", **kwargs) -> Backend:
    """Factory. kind in {auto, mock, hf}. `auto` uses HF if torch+transformers
    import, otherwise mock."""
    if kind == "mock":
        return MockBackend(**{k: v for k, v in kwargs.items()
                              if k in {"num_layers", "hidden_dim", "noise",
                                       "signal", "seed"}})
    if kind == "hf":
        return HFBackend(**{k: v for k, v in kwargs.items()
                            if k in {"model_name", "device", "dtype", "max_length"}})
    # auto
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return HFBackend(**{k: v for k, v in kwargs.items()
                            if k in {"model_name", "device", "dtype", "max_length"}})
    except Exception:
        print("[backend] torch/transformers unavailable -> MockBackend.")
        return MockBackend(**{k: v for k, v in kwargs.items()
                              if k in {"num_layers", "hidden_dim", "noise",
                                       "signal", "seed"}})
