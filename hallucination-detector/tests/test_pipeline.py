"""Tests for the hallucination detector — all run on the mock backend (no torch)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from halludetect import LABELS
from halludetect.data import synthetic
from halludetect.data.curate import stratified_split
from halludetect.data.schema import Example, write_jsonl, read_jsonl
from halludetect.features.extractor import MockBackend, extract_features
from halludetect.probe.sweep import run_sweep
from halludetect.probe.probe import Probe
from halludetect.eval.evaluate import evaluate
from halludetect.serve.detector import HallucinationDetector


def test_synthetic_balanced():
    exs = synthetic.generate(n_per_class=30, seed=1)
    assert len(exs) == 90
    counts = {l: 0 for l in LABELS}
    for e in exs:
        counts[e.label] += 1
    assert all(c == 30 for c in counts.values())


def test_schema_roundtrip(tmp_path):
    exs = synthetic.generate(n_per_class=5)
    p = tmp_path / "x.jsonl"
    n = write_jsonl(exs, p)
    assert n == len(exs)
    back = list(read_jsonl(p))
    assert len(back) == n
    assert back[0].label in LABELS


def test_bad_label_rejected():
    try:
        Example(id="1", context="c", question="q", answer="a", label="nonsense")
    except ValueError:
        return
    raise AssertionError("bad label should raise")


def test_stratified_split_covers_all():
    exs = synthetic.generate(n_per_class=40, seed=2)
    splits = stratified_split(exs)
    total = sum(len(v) for v in splits.values())
    assert total == len(exs)
    for name in ("train", "dev", "test"):
        assert len(splits[name]) > 0


def test_mock_backend_shape():
    b = MockBackend(num_layers=12, hidden_dim=64)
    ex = Example(id="a", context="c", question="q", answer="a",
                 label="hallucinated")
    feats = b.encode(ex)
    assert feats.shape == (12, 64)
    # deterministic
    assert np.allclose(feats, b.encode(ex))


def test_middle_layer_wins_and_high_f1():
    """The mock injects signal peaking mid-network -> best layer should be
    intermediate and test macro-F1 should be high."""
    exs = synthetic.generate(n_per_class=120, seed=3)
    splits = stratified_split(exs, seed=3)
    b = MockBackend(num_layers=24, hidden_dim=128, seed=0)
    Xtr, ytr = extract_features(splits["train"], b, verbose=False)
    Xdev, ydev = extract_features(splits["dev"], b, verbose=False)
    Xte, yte = extract_features(splits["test"], b, verbose=False)
    sweep, probe = run_sweep(Xtr, ytr, Xdev, ydev, verbose=False)

    # best layer is intermediate, not the first or the last
    assert 0 < sweep.best_layer < b.num_layers - 1
    pred = probe.predict(Xte[:, probe.layer, :])
    m = evaluate(yte, pred)
    assert m["macro_f1"] >= 0.8, m["macro_f1"]


def test_probe_save_load(tmp_path):
    exs = synthetic.generate(n_per_class=60, seed=4)
    splits = stratified_split(exs, seed=4)
    b = MockBackend(num_layers=16, hidden_dim=96, seed=1)
    Xtr, ytr = extract_features(splits["train"], b, verbose=False)
    Xdev, ydev = extract_features(splits["dev"], b, verbose=False)
    _, probe = run_sweep(Xtr, ytr, Xdev, ydev, verbose=False)
    p = tmp_path / "probe.joblib"
    probe.save(p)
    loaded = Probe.load(p)
    assert loaded.layer == probe.layer
    x = Xdev[:1, probe.layer, :]
    assert np.array_equal(probe.predict(x), loaded.predict(x))


def test_detector_end_to_end(tmp_path):
    exs = synthetic.generate(n_per_class=120, seed=5)
    splits = stratified_split(exs, seed=5)
    b = MockBackend(num_layers=20, hidden_dim=128, seed=2)
    Xtr, ytr = extract_features(splits["train"], b, verbose=False)
    Xdev, ydev = extract_features(splits["dev"], b, verbose=False)
    _, probe = run_sweep(Xtr, ytr, Xdev, ydev, verbose=False)
    det = HallucinationDetector(probe, b)
    res = det.detect("ctx", "q", "some answer")
    assert set(res) == {"label", "confidence", "probabilities", "layer"}
    assert res["label"] in LABELS
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-6
