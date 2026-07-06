#!/usr/bin/env python3
"""Train the hallucination-detection probe end-to-end.

Examples:
    # Offline / CI-friendly (mock backend, no torch needed):
    python scripts/train.py --backend mock

    # Real 3B model (requires torch + transformers + GPU recommended):
    python scripts/train.py --backend hf --model Qwen/Qwen2.5-3B
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halludetect.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Train the hallucination detector probe.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B",
                    help="HF model name (hf backend only).")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-per-class", type=int, default=200,
                    help="synthetic examples per class when RAGTruth is unavailable.")
    ap.add_argument("--C", type=float, default=1.0, help="probe L2 inverse strength.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force-synthetic", action="store_true")
    # mock backend knobs
    ap.add_argument("--mock-layers", type=int, default=28)
    ap.add_argument("--mock-dim", type=int, default=256)
    args = ap.parse_args()

    backend_kwargs = {}
    if args.backend in ("hf", "auto"):
        backend_kwargs.update(model_name=args.model, dtype=args.dtype)
    if args.backend in ("mock", "auto"):
        backend_kwargs.update(num_layers=args.mock_layers, hidden_dim=args.mock_dim)

    run(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        backend_kind=args.backend,
        n_per_class=args.n_per_class,
        C=args.C,
        seed=args.seed,
        force_synthetic=args.force_synthetic,
        backend_kwargs=backend_kwargs,
    )


if __name__ == "__main__":
    main()
