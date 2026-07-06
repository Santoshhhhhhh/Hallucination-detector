"""Curate a 3-class dataset and write train/dev/test splits.

Tries to load the real RAGTruth dataset from HuggingFace; if that is not
available (no network / datasets not installed), falls back to the bundled
synthetic generator so the pipeline still runs.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .schema import Example, write_jsonl
from . import synthetic


def _try_load_ragtruth() -> list[Example] | None:
    """Attempt to load RAGTruth from HuggingFace and map to the 3-class schema.

    RAGTruth annotates spans as hallucinated or not, per task. We map:
      - answers with no hallucination span that restate context -> context_supported
      - answers with no hallucination span that add outside facts  -> common_knowledge
      - answers with a hallucination span                         -> hallucinated

    Returns None if the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = load_dataset("wandb/RAGTruth-processed")  # may vary by mirror
    except Exception:
        try:
            ds = load_dataset("ParticleMedia/RAGTruth")
        except Exception:
            return None

    examples: list[Example] = []
    split = ds["train"] if "train" in ds else next(iter(ds.values()))
    for i, row in enumerate(split):
        ctx = row.get("context") or row.get("prompt") or ""
        q = row.get("question") or row.get("query") or ""
        ans = row.get("response") or row.get("answer") or ""
        halluc = row.get("hallucination") or row.get("labels") or row.get("label")
        if not ans:
            continue
        if halluc and (halluc is True or (isinstance(halluc, (list, str)) and len(halluc) > 0)):
            label = "hallucinated"
        else:
            # Heuristic split of "faithful" answers into supported vs common-knowledge.
            label = "context_supported" if _overlaps(ctx, ans) else "common_knowledge"
        examples.append(Example(
            id=f"rt-{i:06d}", context=str(ctx), question=str(q),
            answer=str(ans), label=label, source="ragtruth",
        ))
    return examples or None


def _overlaps(context: str, answer: str, thresh: float = 0.25) -> bool:
    c = set(context.lower().split())
    a = set(answer.lower().split())
    if not a:
        return False
    return len(c & a) / len(a) >= thresh


def stratified_split(
    examples: list[Example],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 7,
) -> dict[str, list[Example]]:
    """Split examples into train/dev/test, stratified by label."""
    rng = random.Random(seed)
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)

    splits = {"train": [], "dev": [], "test": []}
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(n * ratios[0])
        n_dev = int(n * ratios[1])
        splits["train"] += items[:n_tr]
        splits["dev"] += items[n_tr:n_tr + n_dev]
        splits["test"] += items[n_tr + n_dev:]
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def curate(out_dir: str | Path, n_per_class: int = 200, seed: int = 7,
           force_synthetic: bool = False) -> dict[str, int]:
    """Build the dataset and write JSONL splits. Returns per-split counts."""
    out_dir = Path(out_dir)
    examples = None if force_synthetic else _try_load_ragtruth()
    if examples is None:
        print("[curate] RAGTruth unavailable -> using synthetic generator.")
        examples = synthetic.generate(n_per_class=n_per_class, seed=seed)
    else:
        print(f"[curate] loaded {len(examples)} RAGTruth examples.")

    splits = stratified_split(examples, seed=seed)
    counts = {}
    for name, items in splits.items():
        counts[name] = write_jsonl(items, out_dir / f"{name}.jsonl")
    return counts
