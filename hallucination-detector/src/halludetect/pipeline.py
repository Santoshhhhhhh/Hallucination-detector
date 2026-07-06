"""End-to-end training pipeline: curate -> extract -> sweep -> evaluate -> save."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data.curate import curate
from .data.schema import load_split
from .features.extractor import build_backend, extract_features
from .probe.sweep import run_sweep
from .eval.evaluate import evaluate, format_report, cost_comparison


def run(
    data_dir: str = "data",
    out_dir: str = "outputs",
    backend_kind: str = "auto",
    n_per_class: int = 200,
    C: float = 1.0,
    seed: int = 7,
    force_synthetic: bool = False,
    backend_kwargs: dict | None = None,
) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_kwargs = backend_kwargs or {}

    # 1. Curate
    print("== [1/5] curating dataset ==")
    counts = curate(data_dir, n_per_class=n_per_class, seed=seed,
                    force_synthetic=force_synthetic)
    print(f"  splits: {counts}")

    train = load_split(data_dir / "train.jsonl")
    dev = load_split(data_dir / "dev.jsonl")
    test = load_split(data_dir / "test.jsonl")

    # 2. Backend + feature extraction
    print("== [2/5] building backend ==")
    backend = build_backend(backend_kind, **backend_kwargs)
    print(f"  backend={type(backend).__name__} "
          f"layers={backend.num_layers} dim={backend.hidden_dim}")

    print("== [3/5] extracting features ==")
    Xtr, ytr = extract_features(train, backend)
    Xdev, ydev = extract_features(dev, backend)
    Xte, yte = extract_features(test, backend)

    # 3. Layer sweep + best probe
    print("== [4/5] layer sweep ==")
    sweep, probe = run_sweep(Xtr, ytr, Xdev, ydev, C=C, seed=seed)

    # 4. Evaluate on test
    print("== [5/5] evaluating on test ==")
    pred = probe.predict(Xte[:, probe.layer, :])
    metrics = evaluate(yte, pred)
    print("\n" + format_report(metrics))

    cost = cost_comparison(len(test))

    # 5. Persist artifacts
    probe_path = out_dir / "probe.joblib"
    probe.save(probe_path)
    report = {
        "backend": type(backend).__name__,
        "num_layers": backend.num_layers,
        "hidden_dim": backend.hidden_dim,
        "split_counts": counts,
        "sweep": sweep.as_dict(),
        "test_metrics": {
            "macro_f1": metrics["macro_f1"],
            "report": metrics["report"],
            "confusion_matrix": metrics["confusion_matrix"],
        },
        "cost_comparison": cost,
        "probe_path": str(probe_path),
    }
    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved probe -> {probe_path}")
    print(f"Saved report -> {out_dir / 'report.json'}")
    print(f"\nBest layer L* = {probe.layer}  |  test macro-F1 = {metrics['macro_f1']:.4f}")
    return report
