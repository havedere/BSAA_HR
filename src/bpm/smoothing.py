"""Sequence smoothers over consecutive 2 s-stride window predictions of one recording segment."""
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from bpm.model import BIN_MIN, N_BINS
from bpm.windowing import STRIDE_S


def segments(idx, max_gap_s=30, stride_s=STRIDE_S):
    """Window indices (start_s = stride_s * idx, sorted) -> list of slices, split where the start gap > max_gap_s."""
    cuts = np.flatnonzero(np.diff(idx) * stride_s > max_gap_s) + 1
    bounds = [0, *cuts, len(idx)]
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def median_smooth(bpm, k, ok=None):
    """Centred median over k (odd) windows; gated windows (ok False) left out; NaN if all k are gated."""
    x = np.asarray(bpm, float).copy()
    if ok is not None:
        x[~np.asarray(ok)] = np.nan
    padded = np.pad(x, k // 2, constant_values=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN window -> NaN, expected
        return np.nanmedian(sliding_window_view(padded, k), axis=1)


def kalman_rts(bpm, r, q):
    """Random-walk Kalman filter + RTS smoother. r: measurement variance (scalar or per window,
    np.inf = missing), q: step variance. Returns (mean, posterior variance)."""
    z = np.asarray(bpm, float)
    r = np.broadcast_to(np.asarray(r, float), z.shape)
    n = len(z)
    m_f, p_f, p_p = np.empty(n), np.empty(n), np.empty(n)
    finite = np.isfinite(r) & np.isfinite(z)
    m, p = (z[finite][0] if finite.any() else 0.0), 1e6
    for t in range(n):
        p_p[t] = p = p + (q if t else 0.0)
        if finite[t]:
            g = p / (p + r[t])
            m, p = m + g * (z[t] - m), (1 - g) * p
        m_f[t], p_f[t] = m, p
    mean, var = m_f.copy(), p_f.copy()
    for t in range(n - 2, -1, -1):
        c = p_f[t] / p_p[t + 1]
        mean[t] = m_f[t] + c * (mean[t + 1] - m_f[t])
        var[t] = p_f[t] + c**2 * (var[t + 1] - p_p[t + 1])
    return mean, var


def _transition(sigma_t, floor):
    d = np.arange(N_BINS)[:, None] - np.arange(N_BINS)[None, :]
    a = np.exp(-0.5 * (d / sigma_t) ** 2) * (np.abs(d) <= 5 * sigma_t)  # banded at +-5 sigma_T
    return (1 - floor) * a / a.sum(1, keepdims=True) + floor / N_BINS


def hmm_forward_backward(probs, sigma_t, floor=1e-3, log_prior=None, ok=None):
    """HMM over 1-BPM bins. Emission = probs / prior (scaled likelihood; prior = exp(log_prior), none = uniform),
    flat for gated windows (ok False); Gaussian transition sigma_t bins with a uniform floor.
    Returns (posterior (T, N_BINS), posterior mean BPM, posterior std)."""
    e = np.asarray(probs, float)
    if log_prior is not None:
        e = e / np.exp(log_prior)[None, :]
    e = e / e.sum(1, keepdims=True)
    if ok is not None:
        e[~np.asarray(ok)] = 1.0 / N_BINS
    a = _transition(sigma_t, floor)
    n = len(e)
    alpha, beta = np.empty_like(e), np.ones_like(e)
    alpha[0] = e[0] / e[0].sum()
    for t in range(1, n):
        f = (alpha[t - 1] @ a) * e[t]
        alpha[t] = f / f.sum()
    for t in range(n - 2, -1, -1):
        b = a @ (e[t + 1] * beta[t + 1])
        beta[t] = b / b.sum()
    post = alpha * beta
    post /= post.sum(1, keepdims=True)
    c = np.arange(BIN_MIN, BIN_MIN + N_BINS)
    mean = post @ c
    std = np.sqrt(np.clip(post @ c**2 - mean**2, 0, None))
    return post, mean, std


def estimate_q(labels_2s):
    """Robust step std (BPM per 2 s window): MAD * 1.4826 of first differences, NaN steps dropped.
    Kalman q = (m * estimate_q)^2, HMM sigma_T = m * estimate_q."""
    d = np.diff(np.asarray(labels_2s, float))
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))))
