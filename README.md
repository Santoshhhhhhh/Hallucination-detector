# Hallucination Detection Engine

A lightweight **three-class RAG hallucination detector**. Instead of paying an
LLM-as-judge to grade every generation, it trains a small **linear probe** on the
**frozen middle-layer hidden states** of a 3B causal LM and reads the answer straight
out of the residual stream — matching a judge's discriminative power at a fraction of
the cost and latency.

Classes:

| Label | Meaning |
|-------|---------|
| `context_supported` | The answer is entailed by the retrieved context. |
| `common_knowledge`  | The answer is true world knowledge but **not** in the context. |
| `hallucinated`      | The answer is unsupported by context and not reliable world knowledge. |

The middle `common_knowledge` class is the point: it stops the detector from flagging
correct-but-out-of-context answers as hallucinations, the main false-positive mode of
naive RAG faithfulness checks.

See **[DESIGN.md](DESIGN.md)** for the full system design and diagram.

## Core idea

Groundedness/truthfulness signal concentrates in **intermediate** transformer layers
(final layers specialize toward next-token prediction). So we:

1. Curate a 3-class dataset from RAGTruth (with an offline synthetic fallback).
2. Run one frozen forward pass per example, mean-pooling the **answer-span** tokens at
   **every** layer.
3. **Sweep all layers**, fitting a logistic-regression probe on each and scoring dev
   macro-F1, then select `L*` = argmax.
4. Serve: one forward pass to `L*` → probe → `{label, confidence, per-class p}`.

## Install

```bash
pip install -e .            # core only (numpy, scikit-learn, joblib, pyyaml)
pip install -e ".[hf]"      # + torch/transformers/datasets for the real 3B backend
pip install -e ".[dev]"     # + pytest
```

## Quickstart

The project runs **with zero heavyweight dependencies** using a deterministic
**mock backend** that injects a middle-layer-peaked signal — so the whole
curate → extract → sweep → probe → serve chain (and the test suite) is runnable and
CI-friendly out of the box.

```bash
# Train (offline mock backend):
python scripts/train.py --backend mock

# Inspect the per-layer sweep + test metrics:
cat outputs/report.json

# Demo inference:
python scripts/detect.py --backend mock --demo

# Tests:
pytest -q
```

### Real 3B model

On a machine with a GPU and network access to HuggingFace:

```bash
pip install -e ".[hf]"
python scripts/train.py --backend hf --model Qwen/Qwen2.5-3B
python scripts/detect.py --backend hf --model Qwen/Qwen2.5-3B \
  --context "The report states the plant capacity is 480 MW." \
  --question "What is the plant capacity?" \
  --answer  "The plant capacity is 9,900 MW."
```

Swap in any HF causal LM via `--model`. The backend keeps the model frozen
(`requires_grad=False`, eval mode) and extracts `output_hidden_states` at every layer.

## What the mock backend is (and isn't)

The mock backend derives features from each example's **id and label**, not its text.
It exists to validate the *pipeline mechanics* deterministically without a GPU. It will
happily reach ~0.85 macro-F1 on synthetic data and show the intermediate-layer peak in
the sweep — but its **per-query inference is not content-sensitive**. For real,
text-driven detection you must use `--backend hf`.


## Results (mock backend, illustrative)

The sweep reproduces the intended shape — macro-F1 rises into the middle of the network,
peaks (here around layer 17 of 28), and falls off toward the final layers — and the
selected probe reaches **test macro-F1 ≈ 0.85**. On real RAGTruth data with a 3B model
the target is **≈ 0.80 macro-F1**, with the probe running ~100–1000× cheaper and faster
than an LLM-as-judge (see `eval.cost_comparison`)
