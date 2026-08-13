# Hallucination Detection Engine

A lightweight **three-class RAG hallucination detector**. Instead of paying an
LLM-as-judge to grade every generation, it trains a small **linear probe** on the
**frozen middle-layer hidden states** of a 3B causal LM.

## Classes

| Label | Meaning |
|-------|---------|
| `context_supported` | Entailed by the retrieved context. |
| `common_knowledge`  | True world knowledge but **not** in the context. |
| `hallucinated`      | Unsupported by context and not reliable world knowledge. |

The `common_knowledge` class stops the detector from flagging
correct-but-out-of-context answers as hallucinations.

## Design

Groundedness signal concentrates in the **intermediate** transformer layers (final
layers specialize toward next-token prediction), so the probe reads a middle layer of a
frozen model. A per-layer sweep selects the best layer.

```
   (context, question, answer)
             │
             ▼   frozen 3B LLM → hidden states per layer
     mean-pool answer-span tokens
             │
             ▼   per-layer sweep → pick best layer L*
     probe at L* → {label, confidence}
```
