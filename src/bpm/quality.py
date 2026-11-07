"""Window quality scores and the quality gate. Every score is "higher = worse": a window passes when
score <= limit. Limits are fitted on val rows and applied unchanged to test."""
import numpy as np
from scipy.stats import skew, spearmanr

BAND_HZ = (0.6, 3.5)
PEAK_HALF_WIDTH_HZ = 0.1
# shortcut: mae-mode limit searched on a coverage-quantile grid, not every unique score; refine if the
# passed MAE jumps between neighbouring grid points
MAE_GRID = np.linspace(0.01, 1.0, 199)


def skewness(windows):
    """(N, C, S) windows -> score = -skewness of the first channel (clean PPG is positively skewed)."""
    return -skew(windows[:, 0], axis=1)


def spectral_prominence(windows, fs=100):
    """Score = 1 - (power within +-0.1 Hz of the dominant 0.6-3.5 Hz peak / total 0.6-3.5 Hz power)."""
    x = windows[:, 0] - windows[:, 0].mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    freqs = np.fft.rfftfreq(x.shape[1], 1 / fs)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    p, f = power[:, band], freqs[band]
    peak = f[p.argmax(axis=1)]
    near = np.abs(f[None, :] - peak[:, None]) <= PEAK_HALF_WIDTH_HZ + 1e-9
    return 1 - (p * near).sum(axis=1) / p.sum(axis=1)


def gate(score, limit):
    return np.asarray(score) <= limit


def balanced_mae(err, subject, dataset, passed=None):
    """Subject-macro balanced MAE (mean over sub-datasets of mean per-subject MAE) on passed windows.
    Returns (mae, number of subjects left out because none of their windows passed)."""
    passed = np.ones(len(err), bool) if passed is None else np.asarray(passed)
    per_dataset, n_out = [], 0
    for d in np.unique(dataset):
        maes = []
        for s in np.unique(subject[dataset == d]):
            rows = (subject == s) & (dataset == d)
            if (rows & passed).any():
                maes.append(err[rows & passed].mean())
            else:
                n_out += 1
        if maes:
            per_dataset.append(np.mean(maes))
    return (float(np.mean(per_dataset)) if per_dataset else np.nan), n_out


def fit_limit(score, err, subject, dataset, mode="coverage", coverage=0.9, max_mae=5.0):
    """coverage: limit = coverage quantile of score. mae: largest grid limit whose passed balanced MAE
    <= max_mae (all windows pass -> max score); none -> coverage limit, infeasible=True.
    Returns (limit, info {coverage, mae, n_subjects_out, infeasible})."""
    score, err = np.asarray(score, float), np.asarray(err, float)
    limit, infeasible = float(np.quantile(score, coverage)), False
    if mode == "mae":
        if balanced_mae(err, subject, dataset)[0] <= max_mae:
            limit = float(score.max())
        else:
            ok = [q for q in np.quantile(score, MAE_GRID)
                  if balanced_mae(err, subject, dataset, score <= q)[0] <= max_mae]
            if ok:
                limit = float(max(ok))
            else:
                infeasible = True
    elif mode != "coverage":
        raise ValueError(mode)
    passed = gate(score, limit)
    mae, n_out = balanced_mae(err, subject, dataset, passed)
    return limit, {"coverage": float(passed.mean()), "mae": mae, "n_subjects_out": n_out, "infeasible": infeasible}


def risk_coverage(score, err):
    """Windows sorted by score (best first) -> (coverage, window-pooled MAE of the kept windows)."""
    e = np.asarray(err, float)[np.argsort(score, kind="stable")]
    n = np.arange(1, len(e) + 1)
    return n / len(e), np.cumsum(e) / n


def aurc(score, err):
    """Area under the risk-coverage curve (mean risk over coverage steps); lower = better ranking."""
    return float(risk_coverage(score, err)[1].mean())


def spearman_ci(score, err, subject, n_boot=1000, seed=0):
    """Spearman rho(score, err) with a subject-bootstrap 95% CI -> (rho, lo, hi)."""
    score, err, subject = map(np.asarray, (score, err, subject))
    rho = spearmanr(score, err).statistic
    subjects = np.unique(subject)
    rows = {s: np.flatnonzero(subject == s) for s in subjects}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[s] for s in rng.choice(subjects, len(subjects))])
        boot.append(spearmanr(score[idx], err[idx]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(rho), float(lo), float(hi)
