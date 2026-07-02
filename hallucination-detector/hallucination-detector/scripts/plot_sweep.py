#!/usr/bin/env python3
"""Plot the per-layer macro-F1 sweep from outputs/report.json.

Requires matplotlib (optional). Produces outputs/sweep.png.
"""
import json
import sys
from pathlib import Path


def main():
    report_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/report.json")
    report = json.loads(report_path.read_text())
    f1 = report["sweep"]["per_layer_f1"]
    best = report["sweep"]["best_layer"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; per-layer dev macro-F1:")
        for i, v in enumerate(f1):
            mark = "  <-- best" if i == best else ""
            print(f"  layer {i:>2}: {v:.4f}{mark}")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(f1)), f1, marker="o")
    plt.axvline(best, color="r", ls="--", label=f"best layer {best}")
    plt.xlabel("layer")
    plt.ylabel("dev macro-F1")
    plt.title("Per-layer probe sweep — hallucination signal peaks mid-network")
    plt.legend()
    plt.tight_layout()
    out = report_path.parent / "sweep.png"
    plt.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
