import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import evaluate  # noqa: E402
from bpm.model import BIN_MIN, N_BINS  # noqa: E402


class Fixed(nn.Module):
    def __init__(self, head, out):
        super().__init__()
        self.head_type, self.out = head, torch.tensor(out, dtype=torch.float32)

    def forward(self, x):
        return self.out


def data(n):
    return {"windows": np.random.default_rng(0).normal(size=(n, 1, 1000)).astype(np.float32)}


def test_predict_reg_mean_and_spread(monkeypatch):
    monkeypatch.setattr(evaluate, "DEVICE", "cpu")
    p = evaluate.predict([Fixed("reg", [60, 70, 80]), Fixed("reg", [80, 70, 60])], data(3))
    np.testing.assert_allclose(p["pred"], [70, 70, 70])
    np.testing.assert_allclose(p["scores"], [10, 0, 10])


def test_predict_gauss_mixture(monkeypatch):
    monkeypatch.setattr(evaluate, "DEVICE", "cpu")
    p = evaluate.predict([Fixed("gauss", [[60, 0]]), Fixed("gauss", [[80, 0]])], data(1))
    np.testing.assert_allclose(p["pred"], [70])
    np.testing.assert_allclose(p["scores"], [np.sqrt(1 + 100)], rtol=1e-5)  # var exp(0) + spread 10^2


def test_predict_dist_averages_probs(monkeypatch):
    monkeypatch.setattr(evaluate, "DEVICE", "cpu")
    peak = lambda bpm: [[20.0 if b == bpm - BIN_MIN else -20.0 for b in range(N_BINS)]]
    p = evaluate.predict([Fixed("dist", peak(60)), Fixed("dist", peak(100))], data(1))
    np.testing.assert_allclose(p["score_std"], [20], rtol=1e-4)  # half mass at 60, half at 100
    assert p["pred"][0] in (60, 100)  # probs averaged (bimodal), not logits
