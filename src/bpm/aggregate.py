"""CV run aggregation: metrics, LTGO decisions, tests vs baseline.

Implements "Metrics" and "Selection, statistics, reporting" of `.scratch/model-run-plan/spec.md`.
Input is one per-window table `w` with columns config, fold, seed, split ("val"/"test"),
dataset, subject, truth, err (= |pred - truth|).
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

K = 5
BASELINE = "mse"  # MSE, basic, LDS off (run plan story design 2026-11-07)
BIN_EDGES = [40, 60, 70, 80, 90, 100, 110, 120, 130, np.inf]
N_BOOT = 1000
RNG_SEED = 0


def parse_tag(tag):
    """(config id, fold, seed) of a CV run tag; ensemble files `ens_<config>_fold<k>` give seed None."""
    if m := re.fullmatch(r"ens_(.+)_fold(\d+)", tag):
        return m[1], int(m[2]), None
    if m := re.fullmatch(r"(.+)_fold(\d+)_seed(\d+)", tag):
        return m[1], int(m[2]), int(m[3])
    return None


def ltgo_folds(k):
    """Folds whose val may enter decision k: fold j's val is group (j+1) % K, so drop j = (k-1) % K."""
    return [j for j in range(K) if j != (k - 1) % K]


def subject_mae(w, by=()):
    return w.groupby([*by, "dataset", "subject"])["err"].mean().rename("mae").reset_index()


def macro(subj):
    """Subject-macro balanced MAE: mean over sub-datasets of the mean per-subject MAE."""
    return subj.groupby("dataset")["mae"].mean().mean()


def window_balanced(w):
    return w.groupby("dataset")["err"].mean().mean()


def bin_balanced(w, min_n=20):
    """Window-pooled MAE per BPM bin (sub-datasets pooled), unweighted mean over bins with >= min_n windows."""
    g = w.groupby(pd.cut(w["truth"], BIN_EDGES, right=False), observed=True)["err"].agg(["mean", "size"])
    return g.loc[g["size"] >= min_n, "mean"].mean()


def run_metrics(w):
    """Per-split metrics of one run (or ensemble) window table."""
    out = {}
    for split, ws in w.groupby("split"):
        out[f"{split}_macro"] = macro(subject_mae(ws))
        out[f"{split}_window_bal"] = window_balanced(ws)
        out[f"{split}_bin_bal"] = bin_balanced(ws)
    return out


def seed_median(v):
    """Per (config, dataset, subject): median over seeds of the per-subject MAE, and the seed count."""
    s = subject_mae(v, by=("config", "seed"))
    return s.groupby(["config", "dataset", "subject"])["mae"].agg(mae="median", n_seeds="size").reset_index()


def _nanmean(a, axis):
    ok = ~np.isnan(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, a, 0).sum(axis) / ok.sum(axis)


def rank(cells, n_boot=N_BOOT, rng_seed=RNG_SEED):
    """Ranking key per config: per-subject seed-median MAE paired vs the argmin config, mean per
    sub-dataset, then mean over sub-datasets. SE: bootstrap over subjects stratified by sub-dataset,
    same resample for every config (paired)."""
    m = cells.pivot_table(index=["dataset", "subject"], columns="config", values="mae")
    best = m.groupby(level="dataset").mean().mean().idxmin()
    d = m.sub(m[best], axis=0)
    rng = np.random.default_rng(rng_seed)
    point, boot = [], []
    for _, dd in d.groupby(level="dataset"):
        a = dd.to_numpy()
        point.append(_nanmean(a, 0))
        boot.append(_nanmean(a[rng.integers(0, len(a), (n_boot, len(a)))], 1))
    out = pd.DataFrame({
        "key": _nanmean(np.stack(point), 0),
        "se": pd.DataFrame(_nanmean(np.stack(boot), 0)).std().to_numpy(),
    }, index=m.columns)
    out["argmin"] = out.index == best
    g = cells.groupby("config")
    out["n_subj"], out["min_seeds"] = g.size(), g["n_seeds"].min()
    return out


def decide(w, folds, candidates=None, top=3):
    """One decision over the val rows of `folds`: key, SE, tie set (within 1 SE of the argmin),
    tie-break bin-balanced val MAE (seed median over the pooled val of `folds`), top-n keys, winner."""
    v = w[(w["split"] == "val") & w["fold"].isin(list(folds))]
    if candidates is not None:
        v = v[v["config"].isin(candidates)]
    if v.empty:
        return pd.DataFrame(columns=["key", "se", "argmin", "n_subj", "min_seeds", "tie", "bin_bal",
                                     "bin_bal_sd", "top3", "win"])
    r = rank(seed_median(v))
    r["tie"] = r["key"] <= r["se"].fillna(0) + 1e-9  # eps: float noise on exactly equal configs
    bb = v.groupby(["config", "seed"])[["truth", "err"]].apply(bin_balanced).groupby("config")
    r["bin_bal"], r["bin_bal_sd"] = bb.median(), bb.std()
    r["top3"] = r["key"].rank(method="first") <= top
    winner = r[r["tie"]].sort_values(["bin_bal", "key"]).index[0]
    r["win"] = r.index == winner
    return r


def holm(p):
    """Holm step-down adjusted p-values; NaN entries stay NaN and do not count."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.empty_like(q)
    adj[order] = np.minimum(1, np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order]))
    out[ok] = adj
    return out


def _boot_macro(s, rng, n_boot=N_BOOT):
    """Bootstrap distribution of the subject-macro balanced mean of a (dataset, subject)-indexed Series."""
    s = s.dropna()
    if s.empty:
        return np.full(n_boot, np.nan)
    parts = [rng.choice(g.to_numpy(), (n_boot, len(g))).mean(1) for _, g in s.groupby(level="dataset")]
    return np.mean(parts, 0)


def _ci(boot):
    boot = boot[~np.isnan(boot)]
    return np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)


def vs_baseline(w, baseline=BASELINE, rng_seed=RNG_SEED):
    """Per config: Wilcoxon signed-rank p (raw + Holm across configs) per sub-dataset on the paired
    per-subject OOF test MAE (mean over seeds) vs baseline, and subject-bootstrap 95% CIs of the
    OOF subject-macro balanced MAE and of its difference to baseline."""
    t = w[w["split"] == "test"]
    m = subject_mae(t, by=("config", "seed")).groupby(["config", "dataset", "subject"])["mae"].mean()
    m = m.unstack("config")
    base = m[baseline] if baseline in m else pd.Series(np.nan, index=m.index)
    rows = {}
    for c in m.columns:
        row = {}
        for ds, mm in m.groupby(level="dataset"):
            diff = (mm[c] - base.loc[mm.index]).dropna()
            row[f"p_{ds}"] = wilcoxon(diff).pvalue if c != baseline and (diff != 0).any() else np.nan
        lo, hi = _ci(_boot_macro(m[c], np.random.default_rng(rng_seed)))
        dlo, dhi = _ci(_boot_macro(m[c] - base, np.random.default_rng(rng_seed)))
        rows[c] = {**row, "oof_ci_lo": lo, "oof_ci_hi": hi, "diff_ci_lo": dlo, "diff_ci_hi": dhi}
    out = pd.DataFrame.from_dict(rows, orient="index")
    for col in [c for c in out if c.startswith("p_")]:
        out[f"{col}_holm"] = holm(out[col])
    return out


def oof(w):
    """Per config: OOF test subject-macro balanced MAE over seeds (each seed pools its test folds)."""
    s = subject_mae(w[w["split"] == "test"], by=("config", "seed"))
    per_seed = s.groupby(["config", "seed"])[["dataset", "mae"]].apply(macro).groupby("config")
    return pd.DataFrame({"oof_mean": per_seed.mean(), "oof_std": per_seed.std(),
                         "oof_median": per_seed.median(), "oof_n_seeds": per_seed.size()})


def summarize(w, ens, runs):
    """One row per config. `ens`: ensemble window table (same columns, seed unused);
    `runs`: sidecar table with config and collapsed."""
    parts = [runs.groupby("config").agg(n_runs=("config", "size"),
                                        collapse_rate=("collapsed", "mean"))]
    for name, folds in [*((f"k{k}", ltgo_folds(k)) for k in range(K)), ("global", range(K))]:
        d = decide(w, folds)
        cols = ["key", "se", "tie", "bin_bal", "top3", "win"]
        if name == "global":
            cols += ["bin_bal_sd", "n_subj", "min_seeds"]
        parts.append(d[cols].add_prefix(f"{name}_"))
    parts += [oof(w), vs_baseline(w)]
    if not ens.empty:
        parts.append(ens[ens["split"] == "test"].groupby("config")[["dataset", "subject", "err"]]
                     .apply(lambda e: macro(subject_mae(e))).rename("oof_ens"))
    s = pd.concat(parts, axis=1)
    s["few_seeds"] = s["global_min_seeds"] < 3
    return s
