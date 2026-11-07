import numpy as np

from bpm.quality import aurc, balanced_mae, fit_limit, gate, risk_coverage, skewness, spearman_ci, spectral_prominence


def table(n_subj=10, per=100, seed=0):
    rng = np.random.default_rng(seed)
    subject = np.repeat([f"s{i}" for i in range(n_subj)], per)
    dataset = np.where(np.arange(n_subj * per) < n_subj * per // 2, "a", "b")
    score = rng.uniform(size=n_subj * per)
    return score, subject, dataset, rng


def test_coverage_limit_keeps_requested_share():
    score, subject, dataset, _ = table()
    err = score * 10
    limit, info = fit_limit(score, err, subject, dataset, coverage=0.9)
    assert abs(gate(score, limit).mean() - 0.9) < 0.002
    assert not info["infeasible"] and info["n_subjects_out"] == 0


def test_mae_mode_finds_limit_and_falls_back_when_infeasible():
    score, subject, dataset, _ = table()
    err = score * 10  # passed MAE = limit * 5 for uniform score
    limit, info = fit_limit(score, err, subject, dataset, mode="mae", max_mae=3.0)
    assert info["mae"] <= 3.0 and 0.5 < limit <= 0.62 and not info["infeasible"]

    limit, info = fit_limit(score, err + 10, subject, dataset, mode="mae", max_mae=3.0)
    cov_limit, _ = fit_limit(score, err + 10, subject, dataset, coverage=0.9)
    assert info["infeasible"] and limit == cov_limit


def test_mae_mode_all_pass_gives_max_score():
    score, subject, dataset, _ = table()
    limit, info = fit_limit(score, score, subject, dataset, mode="mae", max_mae=5.0)
    assert limit == score.max() and info["coverage"] == 1.0


def test_subject_with_no_passed_window_is_left_out_and_counted():
    score, subject, dataset, _ = table()
    score[subject == "s3"] += 10  # s3 always gated
    err = np.where(subject == "s3", 100.0, 1.0)
    passed = gate(score, 1.0)
    mae, n_out = balanced_mae(err, subject, dataset, passed)
    assert mae == 1.0 and n_out == 1


def test_aurc_perfect_score_below_random():
    score, _, _, rng = table()
    err = rng.exponential(5, size=len(score))
    cov, risk = risk_coverage(err, err)
    assert cov[-1] == 1.0 and np.isclose(risk[-1], err.mean())
    assert aurc(err, err) < aurc(rng.uniform(size=len(err)), err)


def test_spearman_ci_contains_strong_positive():
    score, subject, _, rng = table()
    rho, lo, hi = spearman_ci(score, score + rng.normal(0, 0.1, len(score)), subject, n_boot=200)
    assert lo <= rho <= hi and lo > 0.8


def test_prominence_and_skewness_scores():
    t = np.arange(1000) / 100
    sine = np.sin(2 * np.pi * 1.2 * t)
    noise = np.random.default_rng(0).normal(size=1000)
    w = np.stack([sine, noise])[:, None, :]
    score = spectral_prominence(w)  # score = 1 - prominence
    assert score[0] < 0.1 and score[1] > 0.5
    pulse = np.stack([np.exp(-((t % 0.8) - 0.1) ** 2 / 0.002)])[:, None, :]  # sharp positive peaks
    assert skewness(pulse)[0] < 0  # positive skew -> negative score (good)
