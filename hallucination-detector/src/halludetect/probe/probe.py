"""Linear probe: a standardizer + multinomial logistic regression head.

Operates on a single layer's pooled hidden states, X_layer of shape [N, hidden_dim].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from .. import ID2LABEL


class Probe:
    """A thin, serializable linear probe over one layer's features."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000, seed: int = 0):
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            # NOTE: sklearn >=1.7 dropped the `multi_class` kwarg; multinomial
            # is the default for LogisticRegression, so we no longer pass it.
            ("clf", LogisticRegression(
                C=C, max_iter=max_iter,
                class_weight="balanced", random_state=seed,
            )),
        ])
        self.layer: int | None = None

    def fit(self, X_layer: np.ndarray, y: np.ndarray, layer: int | None = None) -> "Probe":
        self.pipe.fit(X_layer, y)
        self.layer = layer
        return self

    def predict(self, X_layer: np.ndarray) -> np.ndarray:
        return self.pipe.predict(X_layer)

    def predict_proba(self, X_layer: np.ndarray) -> np.ndarray:
        return self.pipe.predict_proba(X_layer)

    def predict_labels(self, X_layer: np.ndarray) -> list[str]:
        return [ID2LABEL[int(i)] for i in self.predict(X_layer)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipe": self.pipe, "layer": self.layer}, path)

    @classmethod
    def load(cls, path: str | Path) -> "Probe":
        obj = joblib.load(path)
        p = cls()
        p.pipe = obj["pipe"]
        p.layer = obj["layer"]
        return p
