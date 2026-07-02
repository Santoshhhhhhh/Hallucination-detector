"""Per-layer probe sweep: fit a probe at each layer, pick the best by dev macro-F1.

This is where we operationalize the core hypothesis — hallucination signal
concentrates in intermediate layers — by scanning every layer and reporting the
full macro-F1 curve, then selecting L* = argmax dev macro-F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score

from .probe import Probe


@dataclass
class SweepResult:
    per_layer_f1: list[float] = field(default_factory=list)  # dev macro-F1 by layer
    best_layer: int = -1
    best_dev_f1: float = -1.0

    def as_dict(self) -> dict:
        return {
            "per_layer_f1": self.per_layer_f1,
            "best_layer": self.best_layer,
            "best_dev_f1": self.best_dev_f1,
        }


def run_sweep(
    X_train: np.ndarray, y_train: np.ndarray,
    X_dev: np.ndarray, y_dev: np.ndarray,
    C: float = 1.0, seed: int = 0, verbose: bool = True,
) -> tuple[SweepResult, Probe]:
    """Sweep all layers; return the sweep result and the best refit probe.

    X_*: arrays [N, num_layers, hidden_dim].
    """
    assert X_train.ndim == 3, "expected [N, num_layers, hidden_dim]"
    num_layers = X_train.shape[1]
    result = SweepResult()

    for L in range(num_layers):
        probe = Probe(C=C, seed=seed).fit(X_train[:, L, :], y_train, layer=L)
        pred = probe.predict(X_dev[:, L, :])
        f1 = f1_score(y_dev, pred, average="macro")
        result.per_layer_f1.append(float(f1))
        if verbose:
            marker = ""
            print(f"  layer {L:>2}: dev macro-F1 = {f1:.4f}{marker}")
        if f1 > result.best_dev_f1:
            result.best_dev_f1 = float(f1)
            result.best_layer = L

    if verbose:
        print(f"[sweep] best layer = {result.best_layer} "
              f"(dev macro-F1 = {result.best_dev_f1:.4f})")

    # Refit best probe on train (optionally could refit on train+dev).
    best_probe = Probe(C=C, seed=seed).fit(
        X_train[:, result.best_layer, :], y_train, layer=result.best_layer
    )
    return result, best_probe
