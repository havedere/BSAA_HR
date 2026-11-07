import numpy as np

from bpm.model import BIN_MIN, N_BINS
from bpm.smoothing import estimate_q, hmm_forward_backward, kalman_rts, median_smooth, segments

CENTRES = np.arange(BIN_MIN, BIN_MIN + N_BINS)


def peaked(bpms, sigma=2.0):
    p = np.exp(-0.5 * ((CENTRES[None, :] - np.asarray(bpms)[:, None]) / sigma) ** 2)
    return p / p.sum(1, keepdims=True)


def test_median_pulls_outlier_back_and_skips_gated():
    bpm = np.full(21, 70.0)
    bpm[10] = 140.0
    assert median_smooth(bpm, 5)[10] == 70.0
    ok = np.ones(21, bool)
    ok[8:13] = False  # whole k-window around 10 gated
    out = median_smooth(bpm, 5, ok)
    assert np.isnan(out[10]) and out[7] == 70.0


def test_kalman_pulls_outlier_back_and_bridges_missing():
    bpm = np.full(60, 70.0)
    bpm[30] = 140.0
    mean, _ = kalman_rts(bpm, r=25.0, q=0.25)
    assert abs(mean[30] - 70) < 10

    ramp = np.linspace(60, 90, 60)
    r = np.full(60, 1.0)
    r[20:40] = np.inf
    mean, var = kalman_rts(ramp, r=r, q=1.0)
    assert np.all(np.isfinite(mean))
    assert abs(mean[30] - ramp[30]) < 2  # bridged between the two known ends
    assert var[30] > var[10]


def test_hmm_corrects_harmonic_outlier():
    probs = peaked(np.full(40, 70.0))
    probs[20] = 0.9 * peaked([140.0])[0] + 0.1 * probs[20]  # 2x harmonic wins the argmax
    _, mean, std = hmm_forward_backward(probs, sigma_t=2.0)
    assert abs(mean[20] - 70) < 3
    assert np.all(std > 0)


def naive_fb(probs, sigma_t, floor):
    d = CENTRES[:, None] - CENTRES[None, :]
    a = np.exp(-0.5 * (d / sigma_t) ** 2) * (np.abs(d) <= 5 * sigma_t)
    a = (1 - floor) * a / a.sum(1, keepdims=True) + floor / N_BINS
    alpha = [probs[0] / probs[0].sum()]
    for t in range(1, len(probs)):
        f = (alpha[-1] @ a) * probs[t]
        alpha.append(f / f.sum())
    beta = [np.ones(N_BINS)]
    for t in range(len(probs) - 1, 0, -1):
        b = a @ (probs[t] * beta[0])
        beta.insert(0, b / b.sum())
    post = np.array(alpha) * np.array(beta)
    return post / post.sum(1, keepdims=True)


def test_hmm_uniform_prior_equals_plain_forward_backward():
    rng = np.random.default_rng(0)
    probs = peaked(70 + np.cumsum(rng.normal(0, 1, 30))) + 1e-4
    probs /= probs.sum(1, keepdims=True)
    post, _, _ = hmm_forward_backward(probs, sigma_t=1.5, log_prior=np.full(N_BINS, -np.log(N_BINS)))
    np.testing.assert_allclose(post, naive_fb(probs, 1.5, 1e-3), atol=1e-8)
    np.testing.assert_allclose(post, hmm_forward_backward(probs, sigma_t=1.5)[0], atol=1e-12)


def test_hmm_gated_window_gets_flat_emission():
    obs = np.full(20, 70.0)
    obs[10] = 150.0
    ok = np.ones(20, bool)
    ok[10] = False
    _, mean, _ = hmm_forward_backward(peaked(obs), sigma_t=2.0, ok=ok)
    assert abs(mean[10] - 70) < 1


def test_estimate_q_on_known_random_walk():
    rng = np.random.default_rng(1)
    walk = 70 + np.cumsum(rng.normal(0, 0.8, 20000))
    walk[::50] = np.nan  # unlabelled windows dropped
    assert abs(estimate_q(walk) - 0.8) < 0.05


def test_segments_split_at_gaps_over_30s():
    idx = np.array([0, 1, 2, 10, 11, 30, 31])  # start_s = 2*idx; gaps 16 s and 38 s
    assert [list(idx[s]) for s in segments(idx)] == [[0, 1, 2, 10, 11], [30, 31]]
