import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import smooth_report as sr  # noqa: E402
from bpm.model import N_BINS  # noqa: E402

Q = {"multisite-ppg": 1.0}


def seq(idx, bpm, score, label=None):
    n = len(idx)
    return {"idx": np.asarray(idx), "bpm": np.asarray(bpm, float), "score": np.asarray(score, float),
            "label": np.asarray(label if label is not None else bpm, float), "dataset_source": "multisite-ppg",
            "subject": "s", "split": "test", "logits": np.log(np.full((n, N_BINS), 1.0 / N_BINS))}


def test_all_gated_short_segment_is_not_covered():
    s = seq(np.arange(10), np.full(10, 70.0), np.full(10, 9.0))  # 20 s, every window above the limit
    for name, params in [("kalman", {"m": 1, "c": 1}), ("hmm", {"m": 1})]:
        _, _, cov = sr.smooth(s, name, params, 1.0, Q, None, 30)
        assert not cov.any(), name


def test_short_gated_run_between_passed_windows_is_bridged():
    score = np.zeros(30)
    score[10:15] = 9.0  # 10 s gated run inside a passed segment
    _, _, cov = sr.smooth(seq(np.arange(30), np.full(30, 70.0), score), "kalman", {"m": 1, "c": 1}, 1.0, Q, None, 30)
    assert cov.all()


def test_lag_jump_never_spans_a_gap():
    # two segments 10 min apart; label steps 60 -> 120 across the gap, flat inside each
    idx = np.r_[np.arange(40), np.arange(340, 380)]
    label = np.r_[np.full(40, 60.0), np.full(40, 120.0)]
    label[60:] = 125.0  # real 5 BPM jump inside segment 2 at position 60
    out = label.copy()
    out[60:75] += 4.0  # smoother lags 4 BPM for 30 s after the real jump
    s = seq(idx, out, np.zeros(80), label)
    res = [(out, None, np.ones(80, bool))]
    assert np.isclose(sr.lag_error([s], res, "test", np.inf), 4.0)
