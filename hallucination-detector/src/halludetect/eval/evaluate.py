"""Evaluation: macro-F1, per-class report, confusion matrix, cost comparison."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
)

from .. import LABELS


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the evaluation bundle."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))),
        target_names=LABELS, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS)))).tolist()
    return {"macro_f1": macro_f1, "report": report, "confusion_matrix": cm}


def format_report(metrics: dict) -> str:
    lines = []
    lines.append(f"Macro-F1: {metrics['macro_f1']:.4f}\n")
    rep = metrics["report"]
    lines.append(f"{'class':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for cls in LABELS:
        r = rep[cls]
        lines.append(f"{cls:<20}{r['precision']:>10.3f}{r['recall']:>10.3f}"
                     f"{r['f1-score']:>10.3f}{int(r['support']):>10d}")
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):")
    lines.append(f"{'':<20}" + "".join(f"{c[:14]:>16}" for c in LABELS))
    for i, row in enumerate(metrics["confusion_matrix"]):
        lines.append(f"{LABELS[i]:<20}" + "".join(f"{v:>16d}" for v in row))
    return "\n".join(lines)


def cost_comparison(n_examples: int,
                    probe_ms_per_ex: float = 8.0,
                    judge_ms_per_ex: float = 1400.0,
                    judge_usd_per_1k: float = 3.0,
                    probe_usd_per_1k: float = 0.02) -> dict:
    """Illustrative latency/cost comparison: probe vs. LLM-as-judge.

    Numbers are indicative defaults; override with measured values.
    """
    return {
        "n_examples": n_examples,
        "probe": {
            "total_latency_s": round(n_examples * probe_ms_per_ex / 1000, 2),
            "est_usd": round(n_examples / 1000 * probe_usd_per_1k, 4),
        },
        "llm_judge": {
            "total_latency_s": round(n_examples * judge_ms_per_ex / 1000, 2),
            "est_usd": round(n_examples / 1000 * judge_usd_per_1k, 4),
        },
        "speedup_x": round(judge_ms_per_ex / probe_ms_per_ex, 1),
        "cost_reduction_x": round(judge_usd_per_1k / probe_usd_per_1k, 1),
    }
