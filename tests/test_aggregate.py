import numpy as np
import pandas as pd
import pytest
from scipy.stats import wilcoxon

from bpm.aggregate import (
    bin_balanced, decide, holm, ltgo_folds, parse_tag, rank, seed_median, summarize, vs_baseline,
)

K = 5


def make_windows(offsets, noise=0.0, seeds=(42, 1, 2), n_win=25, seed=0):
    """Synthetic per-window table: 10 subjects per sub-dataset, subject i in group i % 5.
    err = offsets[config] + per-subject noise (same for all seeds unless noise > 0)."""
    rng = np.random.default_rng(seed)
    rows = []
    subjects = [(ds, f"{ds}{i}", i % K) for ds in ("ms", "wf") for i in range(10)]
    subj_noise = {s: rng.uniform(0, 1) for _, s, _ in subjects}
    truth = np.linspace(45, 135, n_win)
    for config, off in offsets.items():
        for seed_ in seeds:
            for fold in range(K):
                for ds, s, g in subjects:
                    split = "test" if g == fold else "val" if g == (fold + 1) % K else None
                    if split is None:
                        continue
                    err = off + subj_noise[s] + noise * rng.standard_normal(n_win) ** 2
                    rows.append(pd.DataFrame({
                        "config": config, "fold": fold, "seed": seed_, "split": split,
                        "dataset": ds, "subject": s, "truth": truth, "err": err,
                    }))
    return pd.concat(rows, ignore_index=True)


def test_parse_tag():
    assert parse_tag("mse_lds0.5_fold0_seed1") == ("mse_lds0.5", 0, 1)
    assert parse_tag("ens_huber5_bootHL_fold3") == ("huber5_bootHL", 3, None)
    assert parse_tag("mse_seed2") is None


def test_ltgo_folds_drop_fold_whose_val_is_group_k():
    for k in range(K):
        folds = ltgo_folds(k)
        assert len(folds) == K - 1
        assert all((j + 1) % K != k for j in folds)


def test_bin_balanced_skips_bins_under_20_windows():
    w = pd.DataFrame({"truth": [65.0] * 20 + [75.0] * 19 + [30.0] * 50,
                      "err": [2.0] * 20 + [100.0] * 19 + [100.0] * 50})
    assert bin_balanced(w) == 2.0


def test_seed_median_ignores_one_collapsed_seed():
    w = make_windows({"A": 1.0})
    collapsed = (w["seed"] == 2) & (w["split"] == "val")
    w2 = w.copy()
    w2.loc[collapsed, "err"] = 50.0
    a, b = seed_median(w[w["split"] == "val"]), seed_median(w2[w2["split"] == "val"])
    # 3 seeds: median of (x, x, 50) = x
    pd.testing.assert_series_equal(a["mae"], b["mae"])
    assert (a["n_seeds"] == 3).all()


def test_rank_key_is_paired_diff_vs_argmin():
    w = make_windows({"A": 1.0, "B": 2.0, "C": 1.5})
    r = rank(seed_median(w[w["split"] == "val"]))
    assert r.loc["A", "key"] == pytest.approx(0.0)
    assert r.loc["B", "key"] == pytest.approx(1.0)
    assert r.loc["C", "key"] == pytest.approx(0.5)
    assert r["se"].to_numpy() == pytest.approx(0.0)  # constant offsets: no subject spread
    assert r["argmin"].tolist() == [True, False, False]


def test_tie_set_and_tie_break():
    w = make_windows({"A": 3.0, "B": 3.0, "C": 3.0, "D": 3.3})
    idx = w["subject"].str[2:].astype(int)
    is_c = w["config"] == "C"
    # C: paired diff +1 / -0.8 alternating -> key 0.1, SE ~0.2 -> tie; D: +0.3 on every subject, SE 0 -> no tie
    w.loc[is_c, "err"] += np.where(idx[is_c] % 2 == 0, 1.0, -0.8)
    # B: same per-subject MAE as A (-2 on 2 windows >= 130, +1 on 4 windows < 60), lower bin-balanced MAE
    is_b = w["config"] == "B"
    w.loc[is_b & (w["truth"] >= 130), "err"] -= 2.0
    w.loc[is_b & (w["truth"] < 60), "err"] += 1.0
    d = decide(w, range(K))
    assert d.loc["C", "key"] == pytest.approx(0.1) and d.loc["C", "se"] > 0.1
    assert d.loc[["A", "B", "C"], "tie"].all() and not d.loc["D", "tie"]
    assert d.loc["B", "win"] and d["win"].sum() == 1
    assert d.loc[["A", "B", "C"], "top3"].all() and not d.loc["D", "top3"]


def test_top3_is_three_lowest_keys():
    w = make_windows({"A": 1.0, "B": 2.0, "C": 1.5, "D": 3.0})
    d = decide(w, range(K))
    assert d.loc[["A", "B", "C"], "top3"].all() and not d.loc["D", "top3"]


def test_ltgo_decision_k_ignores_group_k_val_rows():
    base = make_windows({"A": 1.0, "B": 1.5, "C": 2.0}, noise=0.5)
    for k in range(K):
        w = base.copy()
        group_k_val = (w["split"] == "val") & (w["fold"] == (k - 1) % K)
        rng = np.random.default_rng(k)
        w.loc[group_k_val, "err"] = rng.uniform(0, 50, group_k_val.sum())
        w.loc[group_k_val & (w["config"] == "C"), "err"] = 0.0  # C would win if these rows counted
        pd.testing.assert_frame_equal(decide(base, ltgo_folds(k)), decide(w, ltgo_folds(k)))
        assert decide(w, range(K)).loc["C", "win"]  # global sees them


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([0.02, np.nan, 0.5])
    assert out[0] == pytest.approx(0.04) and np.isnan(out[1]) and out[2] == pytest.approx(0.5)


def test_vs_baseline_wilcoxon_per_dataset():
    w = make_windows({"mse": 1.0, "huber10": 1.0}, noise=1.0)
    w.loc[w["config"] == "huber10", "err"] += np.where(w.loc[w["config"] == "huber10", "dataset"] == "ms", 1.0, 0.0)
    v = vs_baseline(w)
    t = w[w["split"] == "test"]
    per_subj = t.groupby(["config", "dataset", "subject", "seed"])["err"].mean().groupby(
        ["config", "dataset", "subject"]).mean().unstack("config")
    ms = per_subj.loc["ms"]
    assert v.loc["huber10", "p_ms"] == pytest.approx(wilcoxon(ms["huber10"] - ms["mse"]).pvalue)
    assert np.isnan(v.loc["mse", "p_ms"])
    assert v.loc["huber10", "diff_ci_lo"] > 0  # +0.5 balanced difference, all subjects worse on ms


def test_summarize_survives_one_fold_two_seeds():
    w = make_windows({"mse_lds0.5": 1.0}, seeds=(1, 2))
    w = w[w["fold"] == 0]
    runs = pd.DataFrame({"config": ["mse_lds0.5"] * 2, "collapsed": [False, True]})
    s = summarize(w, w.iloc[:0], runs)
    assert s.loc["mse_lds0.5", "collapse_rate"] == 0.5
    assert s.loc["mse_lds0.5", "few_seeds"]
    assert s.loc["mse_lds0.5", "global_win"]
