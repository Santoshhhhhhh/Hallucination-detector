"""Hallucination Detection Engine.

A three-class RAG hallucination detector that trains a linear probe on the
frozen middle-layer hidden states of a 3B causal LM.
"""

__version__ = "0.1.0"

LABELS = ["context_supported", "common_knowledge", "hallucinated"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for i, name in enumerate(LABELS)}
