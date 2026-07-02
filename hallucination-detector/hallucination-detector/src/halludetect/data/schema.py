"""Data schema + JSONL IO for RAG hallucination examples."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator

from .. import LABEL2ID


@dataclass
class Example:
    """A single RAG triple with a 3-class label.

    Attributes:
        id: stable unique identifier.
        context: the retrieved passage(s) shown to the generator.
        question: the user query.
        answer: the model-generated answer span to be judged.
        label: one of context_supported / common_knowledge / hallucinated.
        source: provenance tag (e.g. "ragtruth", "synthetic").
    """
    id: str
    context: str
    question: str
    answer: str
    label: str
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABEL2ID:
            raise ValueError(
                f"Unknown label {self.label!r}; expected one of {list(LABEL2ID)}"
            )

    @property
    def label_id(self) -> int:
        return LABEL2ID[self.label]


def write_jsonl(examples: Iterable[Example], path: str | Path) -> int:
    """Write examples to a JSONL file. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Example]:
    """Stream examples from a JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield Example(**obj)


def load_split(path: str | Path) -> list[Example]:
    return list(read_jsonl(path))
