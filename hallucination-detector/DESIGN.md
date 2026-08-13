# Hallucination Detection Engine — System Design

A lightweight, three-class RAG hallucination detector. Instead of calling a large
LLM-as-judge on every generation, we train a small **linear probe** on the frozen
hidden states of a 3B LLM and read the answer straight out of its residual stream.

## Problem framing

Given a RAG triple `(context, question, answer)`, classify the answer into one of
three classes:

| Class | Meaning |
|-------|---------|
| `context_supported` | The claim is entailed by the retrieved context. |
| `common_knowledge`  | The claim is true world knowledge but **not** in the context. |
| `hallucinated`      | The claim is unsupported by context and not reliable world knowledge. |

The middle `common_knowledge` class stops the detector from flagging
correct-but-out-of-context answers as hallucinations.

## Core idea

Groundedness signal concentrates in the **intermediate** transformer layers (final
layers specialize toward next-token prediction), so a probe reads a middle layer of a
frozen model. A per-layer sweep selects the best layer `L*`.

## Architecture

```
   RAGTruth-style triple (context, question, answer, label)
                          │
                          ▼   frozen 3B LLM, hidden states at every layer
                mean-pool answer-span tokens per layer
                          │
                          ▼   per-layer sweep
          fit a logistic-regression probe on each layer, pick L*
                          │
                          ▼
        probe at L*  →  detect(context, question, answer)
                     →  {label, confidence, layer}
```

## Why a linear probe

- **Cheap**: one forward pass plus a linear probe per query, versus a full judge call.
- **Interpretable**: a direction in activation space ≈ a concept.
- **Frozen base**: the 3B model is never updated; the probe is a thin, swappable head.
