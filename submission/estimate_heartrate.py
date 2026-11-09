from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, resample_poly, sosfiltfilt

FS = 100                 # model input rate (Hz)
WIN, STEP = 10, 2        # window length and stride (s)
BIN_MIN, N_BINS = 30, 191
GATE_LIMIT = 16.205      # curve-std limit, median of the 5 CV fold limits
KALMAN_M, KALMAN_C = 1.0, 0.5
STEP_STD = 1.0           # HR random-walk std per 2 s step (BPM), from train labels
WEIGHTS = Path(__file__).resolve().parent / "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding="same"), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding="same"), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding="same"), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, N_BINS))

    def forward(self, x):
        out, _ = self.lstm(self.cnn(x).transpose(1, 2))
        return self.head(out.mean(dim=1))


_models = []


def _load_models():
    if not _models:
        for p in sorted(WEIGHTS.glob("*.pt")):
            m = CNNBiLSTM().to(DEVICE)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            _models.append(m.eval())
    return _models


def _predict(windows):
    """windows (n, 1000) -> (bpm, curve std) per window, softmax curves averaged over all models."""
    mean = windows.mean(1, keepdims=True)
    std = windows.std(1, keepdims=True)
    x = np.where(std > 0, (windows - mean) / np.where(std > 0, std, 1), windows)
    c = np.arange(BIN_MIN, BIN_MIN + N_BINS, dtype=np.float64)
    bpm, sd = [], []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            xb = torch.from_numpy(x[i:i + 2048]).float().unsqueeze(1).to(DEVICE)
            p = torch.stack([torch.softmax(m(xb), 1) for m in _load_models()]).mean(0).double().cpu().numpy()
            peak = c[p.argmax(1)][:, None]
            q = p * (np.abs(c[None] - peak) <= 10)  # expected value within +-10 bins of the peak
            bpm.append((q * c).sum(1) / q.sum(1))
            mu = (p * c).sum(1, keepdims=True)
            sd.append(np.sqrt((p * (c - mu) ** 2).sum(1)))
    return np.concatenate(bpm), np.concatenate(sd)


def _kalman_rts(z, r, q):
    n = len(z)
    m_f, p_f, p_p = np.empty(n), np.empty(n), np.empty(n)
    m, p = z[np.isfinite(r)][0], 1e6
    for t in range(n):
        p_p[t] = p = p + (q if t else 0.0)
        if np.isfinite(r[t]):
            g = p / (p + r[t])
            m, p = m + g * (z[t] - m), (1 - g) * p
        m_f[t], p_f[t] = m, p
    out = m_f.copy()
    for t in range(n - 2, -1, -1):
        out[t] = m_f[t] + p_f[t] / p_p[t + 1] * (out[t + 1] - m_f[t])
    return out


def estimate_heartrate(ppg, fs):
    ppg = np.asarray(ppg, dtype=np.float64).ravel()
    n_out = int(len(ppg) / fs * 2)
    bad = ~np.isfinite(ppg)
    if bad.all():
        return np.full(n_out, np.nan)
    if bad.any():
        idx = np.arange(len(ppg))
        ppg[bad] = np.interp(idx[bad], idx[~bad], ppg[~bad])

    ratio = Fraction(FS / fs).limit_denominator(1000)
    x = resample_poly(ppg, ratio.numerator, ratio.denominator) if ratio != 1 else ppg
    if len(x) > 27:  # sosfiltfilt needs more samples than its padding
        x = sosfiltfilt(butter(4, [0.5, 8.0], btype="bandpass", fs=FS, output="sos"), x)
    if len(x) < WIN * FS:
        x = np.pad(x, (0, WIN * FS - len(x)))

    starts = np.arange(0, len(x) - WIN * FS + 1, STEP * FS)
    windows = np.stack([x[s:s + WIN * FS] for s in starts])
    bpm, score = _predict(windows)

    ok = score <= GATE_LIMIT
    hr = bpm
    if ok.any():  # gated windows get no weight and are bridged by the smoother
        r = np.where(ok, (KALMAN_C * score) ** 2, np.inf)
        hr = _kalman_rts(bpm, r, (KALMAN_M * STEP_STD) ** 2)

    centres = starts / FS + WIN / 2
    heartrate = np.interp(np.arange(n_out) * 0.5, centres, hr)
    return heartrate
