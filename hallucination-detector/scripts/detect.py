#!/usr/bin/env python3
"""Run the trained detector on a single RAG triple, or a demo batch.

Examples:
    python scripts/detect.py --demo
    python scripts/detect.py \
        --context "The report states the plant has a capacity of 480 MW." \
        --question "What is the plant capacity?" \
        --answer "The plant has a capacity of 9900 MW."
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halludetect.serve.detector import HallucinationDetector  # noqa: E402


DEMO = [
    ("The report states the new solar farm has a capacity of 480 megawatts.",
     "What is the capacity of the new solar farm?",
     "The new solar farm has a capacity of 480 megawatts."),
    ("The report states the new solar farm has a capacity of 480 megawatts.",
     "What is the capacity of the new solar farm?",
     "Solar panels convert sunlight into electricity via the photovoltaic effect."),
    ("The report states the new solar farm has a capacity of 480 megawatts.",
     "What is the capacity of the new solar farm?",
     "The new solar farm has a capacity of 9,900 megawatts."),
]


def main():
    ap = argparse.ArgumentParser(description="Detect hallucination on a RAG triple.")
    ap.add_argument("--probe", default="outputs/probe.joblib")
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "hf"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--mock-layers", type=int, default=28)
    ap.add_argument("--mock-dim", type=int, default=256)
    ap.add_argument("--context")
    ap.add_argument("--question")
    ap.add_argument("--answer")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    backend_kwargs = dict(model_name=args.model,
                          num_layers=args.mock_layers, hidden_dim=args.mock_dim)
    det = HallucinationDetector.from_paths(
        args.probe, backend_kind=args.backend, **backend_kwargs)

    if type(det.backend).__name__ == "MockBackend":
        print("NOTE: MockBackend derives features from the example id/label, not "
              "the text,\n      so per-query predictions here are not meaningful. "
              "Use --backend hf\n      with a real model for content-sensitive "
              "detection.\n")

    if args.demo or not (args.context and args.question and args.answer):
        for ctx, q, a in DEMO:
            res = det.detect(ctx, q, a)
            print("-" * 70)
            print(f"Q: {q}\nA: {a}")
            print(json.dumps(res, indent=2))
    else:
        res = det.detect(args.context, args.question, args.answer)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
