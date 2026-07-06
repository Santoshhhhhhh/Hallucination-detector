# Hallucination Detection Engine — System Design

A lightweight, three-class RAG hallucination detector. Instead of calling a large
LLM-as-judge on every generation, we train a small **linear probe** on the frozen
hidden states of a 3B LLM and read the answer straight out of its residual stream.

## 1. Problem framing

Given a RAG triple `(context, question, answer)`, classify each answer span into one
of three classes:

| Class | Meaning |
|-------|---------|
| `context_supported` | The claim is entailed by the retrieved context. |
| `common_knowledge`  | The claim is true/plausible world knowledge but **not** in the context. |
| `hallucinated`      | The claim is unsupported by context **and** not reliable world knowledge. |

The two-way (supported vs. hallucinated) split is the usual framing; the middle
`common_knowledge` class is what makes this useful in practice — it stops the detector
from flagging correct-but-out-of-context answers as hallucinations, which is the main
false-positive mode of naive RAG faithfulness checks.

## 2. Core hypothesis

Truthfulness / groundedness signals concentrate in the **intermediate** transformer
layers, not the final ones (final layers specialize toward next-token prediction).
So a probe reading a **middle layer** of a frozen model should recover most of an
LLM-judge's discriminative power at a tiny fraction of the cost and latency.

We validate this with a **per-layer sweep** and pick the best layer by dev macro-F1.

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                RAGTruth-style                │
                    │        (context, question, answer, label)    │
                    └───────────────────────┬─────────────────────┘
                                            │  data/curate.py
                                            ▼
                    ┌─────────────────────────────────────────────┐
                    │            Curated 3-class dataset           │
                    │        train / dev / test JSONL splits       │
                    └───────────────────────┬─────────────────────┘
                                            │  features/extractor.py
                       ┌────────────────────┼────────────────────┐
                       ▼                    ▼                    ▼
              ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
              │  Frozen 3B   │     │  hook every  │     │  mean-pool   │
              │     LLM      │────▶│    layer     │────▶│  answer-span │
              │ (no grad)    │     │  hidden[L]   │     │   tokens     │
              └──────────────┘     └──────────────┘     └──────┬───────┘
                                                               │
                                      per-layer feature matrix │  X[L] ∈ R^{N×d}
                                                               ▼
                    ┌─────────────────────────────────────────────┐
                    │        probe/sweep.py  (layer selection)     │
                    │   fit LogisticRegression probe on each L,    │
                    │   score dev macro-F1, pick argmax → L*       │
                    └───────────────────────┬─────────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────┐
                    │       Probe at L*  →  serialized .joblib      │
                    │       + calibrated 3-way softmax head        │
                    └───────────────────────┬─────────────────────┘
                                            │  serve/detector.py
                                            ▼
                    ┌─────────────────────────────────────────────┐
                    │   Inference: one forward pass to layer L*,   │
                    │   probe → {label, confidence, per-class p}   │
                    └─────────────────────────────────────────────┘
```

## 4. Components

### 4.1 Data curation (`data/curate.py`)
- Loads a RAGTruth-style source (real dataset via HF, or the bundled synthetic
  generator when offline).
- Maps the source annotations to the 3-class taxonomy.
- Produces stratified `train/dev/test` JSONL splits with a fixed seed.

### 4.2 Feature extractor (`features/extractor.py`)
- Wraps a frozen HF causal LM (default `Qwen/Qwen2.5-3B`, configurable).
- Runs `output_hidden_states=True`, keeps **every** layer.
- Locates the answer-span token indices and **mean-pools** those tokens per layer.
- A `MockBackend` produces deterministic pseudo-hidden-states so the whole pipeline
  runs (and tests pass) with **zero** heavyweight dependencies.

### 4.3 Layer sweep + probe (`probe/`)
- `sweep.py`: for each layer, fit a logistic-regression probe, evaluate dev macro-F1,
  emit a sweep report + plot data. Selects `L*`.
- `probe.py`: the probe object (standardizer + multinomial LR), fit/predict/save/load.

### 4.4 Serving (`serve/detector.py`)
- Loads probe + records `L*`, exposes `detect(context, question, answer)` returning
  `{label, confidence, probabilities, layer}`.

### 4.5 Evaluation (`eval/evaluate.py`)
- Macro-F1, per-class P/R/F1, confusion matrix.
- Cost/latency comparison stub vs. an LLM-judge baseline.

## 5. Why a linear probe (not a fine-tune)
- **Cheap**: closed-form-ish convex fit, seconds on CPU.
- **Interpretable**: a direction in activation space ≈ a concept.
- **Frozen base**: the 3B model is never updated, so the probe is a thin,
  swappable head; re-training on new data is trivial.

## 6. Metrics & target
- Primary: **macro-F1** on the 3-class test split. Target ≈ **0.80**.
- Report the best layer `L*` and the full per-layer curve to show the
  intermediate-layer concentration.

## 7. Extensibility
- Swap `Qwen2.5-3B` for any HF causal LM via config.
- Add token-level probing (per-claim) instead of answer-mean-pool.
- Replace LR with a small MLP probe if capacity-bound.
