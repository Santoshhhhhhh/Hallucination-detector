"""Serving: load a trained probe + backend and detect on new RAG triples."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.schema import Example
from ..features.extractor import build_backend, Backend
from ..probe.probe import Probe
from .. import ID2LABEL


class HallucinationDetector:
    """End-to-end detector: (context, question, answer) -> label + probabilities."""

    def __init__(self, probe: Probe, backend: Backend):
        if probe.layer is None:
            raise ValueError("probe has no recorded layer; retrain via the sweep.")
        self.probe = probe
        self.backend = backend
        self.layer = probe.layer

    @classmethod
    def from_paths(cls, probe_path: str | Path, backend_kind: str = "auto",
                   **backend_kwargs) -> "HallucinationDetector":
        probe = Probe.load(probe_path)
        backend = build_backend(backend_kind, **backend_kwargs)
        return cls(probe, backend)

    def detect(self, context: str, question: str, answer: str) -> dict:
        ex = Example(id="query", context=context, question=question,
                     answer=answer, label="hallucinated")  # label unused
        feats = self.backend.encode(ex)              # [num_layers, hidden_dim]
        x = feats[self.layer][None, :]               # [1, hidden_dim]
        proba = self.probe.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return {
            "label": ID2LABEL[idx],
            "confidence": float(proba[idx]),
            "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(proba)},
            "layer": self.layer,
        }
