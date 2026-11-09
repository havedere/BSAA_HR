"""Generate notebooks/final_results.ipynb (final results of the winner dist_bootHL, shipped pipeline).
Usage: .venv/bin/python .scratch/cv-story/gen_nb_final.py notebooks/final_results.ipynb"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from common import SETUP, STYLE, STYLE_MD

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Final results: `dist_bootHL` (dist head + bootstrap HL, 3-seed ensemble)

Out-of-fold (OOF) test results of the winner over the 5 CV folds: every subject is tested exactly once, by the
3 models of the fold whose test group it is in (they never saw that subject). 47 subjects: 20 multisite-ppg, 27 wrist-finger-ppg.

**Shipped pipeline** (what `estimate_heartrate` runs): 10 s windows every 2 s -> 3-seed ensemble -> input gate
(dist curve std above the 90 % val quantile = no weight) -> Kalman + RTS (R = (c · score)², q = (m · step std)²).
The interface must return a value at every time point, so there is **no output gate**: gated windows are bridged by
the smoother. Per fold the smoother parameters and the gate limit are the ones `smooth_report.py` tuned on that
fold's val group.

Two scopes:
- **covered**: windows `smooth_report.py` scores (gated runs <= 30 s bridged; longer runs left out). Reproduces its CSV.
- **all**: every labelled window, as the shipped function returns them (also long gated runs; a segment with no
  passing window falls back to the raw ensemble value).

Labels: mean ECG-derived HR over each 10 s window. All labelled windows of the test recordings (not capped).
""")
code(SETUP)
code(r'''
from bpm.quality import balanced_mae, fit_limit
from bpm.smoothing import segments
from bpm.split import fold_split, load_folds
from smooth_report import FOLDS_PATH, smooth, train_q

CFG = "dist_bootHL"
OUT = ROOT / "cache" / "plots" / "final"   # figures + values.json of this notebook (save()/put() write here)
OUT.mkdir(parents=True, exist_ok=True)
VALUES_PATH, VALUES = OUT / "values.json", {}
''')
md(STYLE_MD)
code(STYLE)

md("""
## Re-run the shipped pipeline per fold

Same functions and tuned parameters as `smooth_report.py` (row `kalman`, input gate `coverage`, output gate `none`).
Check: the covered subject-macro MAE per fold equals the CSV value.
""")
code(r'''
F = pd.concat([pd.read_csv(RESULTS / f"smooth_ens_{CFG}_fold{k}.csv") for k in range(5)], ignore_index=True)
groups, _ = load_folds(FOLDS_PATH)
rows, PARAMS = [], {}
for k in range(5):
    row = F[(F["fold"] == k) & (F["smoother"] == "kalman") & (F["in_gate"] == "coverage") & (F["out_gate"] == "none")]
    assert len(row) == 1, (k, len(row))
    row = row.iloc[0]
    params = eval(row["params"])  # dict literal written by smooth_report
    z = np.load(PREDS / f"ens_{CFG}_fold{k}.npz")
    v = z["split"] == "val"
    in_limit = fit_limit(z["scores"][v], np.abs(z["pred"] - z["truth"])[v], z["subject"][v], z["dataset_source"][v])[0]
    q = train_q(set(fold_split(groups, k)["train"]))
    for p in sorted((ROOT / "cache" / "sequences" / f"ens_{CFG}_fold{k}").glob("*.npz")):
        zz = np.load(p)
        s = {key: (zz[key].item() if zz[key].ndim == 0 else zz[key]) for key in zz.files}
        if s["split"] != "test":
            continue
        out, _, cov = smooth(s, "kalman", params, in_limit, q, None, 30)
        raw = s["bpm"].astype(float)
        ok = ~(s["score"].astype(float) > in_limit)
        shipped = out.copy()
        for sl in segments(s["idx"]):
            if not ok[sl].any():  # nothing to bridge from: Kalman has no measurement, fall back to raw
                shipped[sl] = raw[sl]
        lab = np.isfinite(s["label"])
        rows.append(pd.DataFrame({
            "fold": k, "recording": p.stem, "subject": s["subject"], "dataset": s["dataset_source"],
            "t_s": s["idx"][lab] * 2, "truth": s["label"][lab].astype(float), "raw": raw[lab],
            "kalman": shipped[lab], "covered": cov[lab], "gated": ~ok[lab]}))
    PARAMS[k] = {**params, "in_limit": in_limit, **{f"q_{d}": v for d, v in q.items()}, "csv_mae": row["mae_macro"]}
W = pd.concat(rows, ignore_index=True)
for k, g in W.groupby("fold"):
    c = g["covered"].to_numpy()
    mae = balanced_mae(np.abs(g["kalman"] - g["truth"]).to_numpy(), g["subject"].to_numpy(), g["dataset"].to_numpy(), c)[0]
    assert abs(mae - PARAMS[k]["csv_mae"]) < 1e-6, (k, mae, PARAMS[k]["csv_mae"])
print(len(W), "labelled test windows,", W["subject"].nunique(), "subjects; covered", f"{W['covered'].mean():.1%}",
      "| gated by input gate", f"{W['gated'].mean():.1%}")
pd.DataFrame(PARAMS).T.round(3)
''')

md("""
## Headline table

Per sub-dataset and both (mean over the 2 sub-datasets, so the 37k multisite windows do not dominate).
`mae_subject` = mean of per-subject MAE (the CV ranking metric); `mae_window`, `rmse`, `bias` (pred - truth),
Pearson `r`, `within_5` / `within_10` = share of windows with |error| <= 5 / 10 BPM.
""")
code(r'''
def summarize(d, col):
    e = d[col] - d["truth"]
    return {"subjects": d["subject"].nunique(), "windows": len(d),
            "mae_subject": d.assign(err=e.abs()).groupby("subject")["err"].mean().mean(),
            "mae_window": e.abs().mean(), "rmse": np.sqrt((e ** 2).mean()), "bias": e.mean(),
            "r": np.corrcoef(d[col], d["truth"])[0, 1],
            "within_5": (e.abs() <= 5).mean(), "within_10": (e.abs() <= 10).mean()}


PIPES = {"raw ensemble": ("raw", None), "shipped, all windows": ("kalman", None), "shipped, covered": ("kalman", "covered")}
T = []
for name, (col, mask) in PIPES.items():
    d = W[W[mask]] if mask else W
    per = {ds: summarize(d[d["dataset"] == ds], col) for ds in DATASETS}
    both = {m: np.mean([per[ds][m] for ds in DATASETS]) for m in per[DATASETS[0]]}
    both.update(subjects=d["subject"].nunique(), windows=len(d))
    for ds, m in [*per.items(), ("both", both)]:
        T.append({"pipeline": name, "dataset": ds, "coverage": len(d[d["dataset"] == ds]) / len(W[W["dataset"] == ds])
                  if ds != "both" else len(d) / len(W), **m})
T = pd.DataFrame(T).set_index(["pipeline", "dataset"])
T.to_csv(RESULTS / f"final_{CFG}.csv")
put("table", **{f"{p}|{d}|{m}": T.loc[(p, d), m] for p, d in T.index for m in T.columns})
T.round(3)
''')

md("""
## Predicted vs ECG

Density of all test windows (shipped pipeline, all windows). The diagonal is perfect agreement; the dashed line is
the least-squares fit (a slope below 1 = pull towards the middle).
""")
code(r'''
fig, axes = plt.subplots(1, 2, figsize=figsize(1.35), sharex=True, sharey=True)
lim = (35, 175)
for ax, ds in zip(axes, DATASETS):
    d = W[W["dataset"] == ds]
    hb = ax.hexbin(d["truth"], d["kalman"], gridsize=60, extent=(*lim, *lim), bins="log", cmap=DENSITY_CMAP, mincnt=1)
    ax.plot(lim, lim, color=COLORS["ideal"], lw=STYLE["lw_thin"])
    slope, icpt = np.polyfit(d["truth"], d["kalman"], 1)
    ax.plot(lim, np.polyval([slope, icpt], lim), color=COLORS["bad"], ls="--", lw=STYLE["lw_thin"],
            label=f"fit: slope {slope:.2f}")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(f"{ds} (r = {T.loc[('shipped, all windows', ds), 'r']:.2f})")
    ax.set_xlabel("ECG truth (BPM)")
    ax.legend(loc="upper left")
    put("scatter", **{f"slope_{ds}": slope, f"intercept_{ds}": icpt})
axes[0].set_ylabel("predicted (BPM)")
fig.colorbar(hb, ax=axes, label="windows (log)", shrink=0.8)
save(fig, "scatter")
''')

md("""
## Bland-Altman

Difference (pred - truth) against the mean of both, as window density (log color; 414k multisite windows would
hide each other as points). Solid = bias, dashed = 95 % limits of agreement (bias ± 1.96 SD), window-pooled per sub-dataset.
""")
code(r'''
fig, axes = plt.subplots(1, 2, figsize=figsize(1.1), sharey=True)
BA = {}
for ax, ds in zip(axes, DATASETS):
    d = W[W["dataset"] == ds]
    diff, mean = d["kalman"] - d["truth"], (d["kalman"] + d["truth"]) / 2
    bias, sd = diff.mean(), diff.std()
    BA[ds] = {"bias": bias, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}
    hb = ax.hexbin(mean, diff, gridsize=60, extent=(35, 175, -60, 60), bins="log", cmap=DENSITY_CMAP, mincnt=1)
    ax.axhline(bias, color=COLORS["text"], lw=STYLE["lw"], label=f"bias {bias:+.2f}")
    for y in (BA[ds]["loa_low"], BA[ds]["loa_high"]):
        ax.axhline(y, color=COLORS["bad"], ls="--", lw=STYLE["lw_thin"])
    ax.plot([], [], color=COLORS["bad"], ls="--", label=f"LoA {BA[ds]['loa_low']:+.1f} / {BA[ds]['loa_high']:+.1f}")
    ax.set_ylim(-60, 60); ax.set_xlim(35, 175)
    ax.set_xlabel("mean of pred and truth (BPM)")
    ax.set_title(ds)
    ax.legend(loc="lower left")
axes[0].set_ylabel("pred - truth (BPM)")
fig.colorbar(hb, ax=axes, label="windows (log)", shrink=0.8)
fig.suptitle("Bland-Altman, shipped pipeline (all test windows)")
save(fig, "bland_altman")
put("bland_altman", **{f"{ds}_{k}": v for ds, b in BA.items() for k, v in b.items()})
pd.DataFrame(BA).T.round(2)
''')

md("""
## Error per BPM bin

Window-pooled MAE and bias per ECG truth bin, raw ensemble vs shipped pipeline. Numbers on top: windows per bin.
The high-BPM tail is rare in the data (2.8 % of windows >= 110 BPM) and stays the weak spot.
""")
code(r'''
W["bin"] = bin_of(W["truth"])
g = W.assign(e_raw=(W["raw"] - W["truth"]), e_k=(W["kalman"] - W["truth"])).groupby("bin", observed=False)
B = pd.DataFrame({"n": g.size(), "mae_raw": g["e_raw"].apply(lambda e: e.abs().mean()),
                  "mae_shipped": g["e_k"].apply(lambda e: e.abs().mean()), "bias_shipped": g["e_k"].mean()})
fig, (a1, a2) = plt.subplots(2, 1, figsize=figsize(1.6), sharex=True)
x = np.arange(len(B))
a1.bar(x - 0.2, B["mae_raw"], 0.4, color=COLORS["neutral"], edgecolor=COLORS["muted"], label="raw ensemble")
a1.bar(x + 0.2, B["mae_shipped"], 0.4, color=COLORS[CFG], label="shipped (Kalman + input gate)")
for xi, (n, h) in enumerate(zip(B["n"], B[["mae_raw", "mae_shipped"]].max(1))):
    a1.annotate(f"{n}", (xi, h), xytext=(0, 3), textcoords="offset points", ha="center", color=COLORS["muted"])
a1.set_ylabel("MAE (BPM)"); a1.legend(loc="upper center")
a1.set_ylim(0, B[["mae_raw", "mae_shipped"]].max().max() * 1.2)
a2.bar(x, B["bias_shipped"], color=COLORS[CFG])
a2.axhline(0, color=COLORS["muted"], lw=STYLE["lw_thin"])
a2.set_ylabel("bias pred - truth (BPM)")
a2.set_xticks(x, BIN_LABELS, rotation=45); a2.set_xlabel("ECG truth (BPM bin)")
fig.suptitle("Error per heart-rate bin (test, all windows)")
save(fig, "bins")
put("bins", **{f"{m}_{b}": B.loc[b, m] for b in B.index for m in B.columns})
B.round(2)
''')

md("""
## Per subject

MAE per test subject (shipped pipeline, all windows), sorted, colored by sub-dataset. Shows how much of the mean comes
from a few hard subjects.
""")
code(r'''
S = (W.assign(err=(W["kalman"] - W["truth"]).abs()).groupby(["dataset", "subject"])
     .agg(mae=("err", "mean"), windows=("err", "size"), mean_hr=("truth", "mean")).reset_index().sort_values("mae"))
fig, ax = plt.subplots(figsize=figsize(0.9))
ax.bar(np.arange(len(S)), S["mae"], color=[COLORS[d] for d in S["dataset"]])
from matplotlib.patches import Patch
ax.set_xticks([]); ax.set_xlabel("test subjects, sorted")
ax.set_ylabel("MAE (BPM)")
ax.set_title("Per-subject MAE, out-of-fold test")
ax.legend(handles=[Patch(color=COLORS[ds], label=f"{ds} (median {S[S['dataset'] == ds]['mae'].median():.2f})")
                   for ds in DATASETS], loc="upper left")
save(fig, "subjects")
put("subjects", **{f"median_{ds}": S[S["dataset"] == ds]["mae"].median() for ds in DATASETS},
    **{f"max_{ds}": S[S["dataset"] == ds]["mae"].max() for ds in DATASETS})
S.tail(8).round(2)
''')

md("""
## Error distribution

Cumulative share of windows with |error| <= x. Read off: share within 5 / 10 BPM.
""")
code(r'''
fig, ax = plt.subplots(figsize=figsize(0.85))
for ds in DATASETS:
    d = W[W["dataset"] == ds]
    for col, ls, lab in [("raw", "--", "raw ensemble"), ("kalman", "-", "shipped")]:
        e = np.sort((d[col] - d["truth"]).abs().to_numpy())
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLORS[ds], ls=ls, label=f"{ds}, {lab}")
for x in (5, 10):
    ax.axvline(x, color=COLORS["muted"], ls=":", lw=STYLE["lw_thin"])
ax.set_xlim(0, 30); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("|pred - truth| (BPM)"); ax.set_ylabel("share of windows")
ax.set_title("Cumulative absolute error (test)")
ax.legend(loc="lower right")
save(fig, "error_cdf")
''')
md("""
## Train vs test: the 5 fold ensembles

Each fold's 3-seed ensemble (raw, no gate, no smoothing) scored on its own **train** subjects, its **val** group and
its **test** group. Windows: the real labelled windows of `cache/windows/` (the same capped set training sampled
from; no bootstrap windows, no noise augmentation). So train MAE = error on data the model was fitted on; the
train-to-test gap measures over-fitting. Metric: subject-macro MAE per sub-dataset.
""")
code(r'''
from bpm.data_loading import load_subject_npz
from evaluate import load_model, predict

SPLITS = ["train", "val", "test"]
rows = []
for k in range(5):
    sp = fold_split(groups, k)
    models = [load_model(ROOT / "cache" / f"best_{CFG}_fold{k}_seed{s}.pt") for s in (42, 1, 2)]
    for split_name in SPLITS:
        for subj in sorted(sp[split_name]):
            d = load_subject_npz(subj, ROOT / "cache" / "windows")
            if d is None or not len(d["labels"]):
                continue
            pred = predict(models, d)["pred"]
            rows.append({"fold": k, "split": split_name, "subject": subj, "dataset": str(d["dataset_source"][0]),
                         "mae": np.abs(pred - d["labels"]).mean(), "windows": len(pred)})
TT = pd.DataFrame(rows)
# check: test/val MAE per subject equals the Stage D preds file of the same ensemble
for k in range(5):
    z = np.load(PREDS / f"ens_{CFG}_fold{k}.npz")
    ref = pd.DataFrame({"subject": z["subject"], "split": z["split"], "err": np.abs(z["pred"] - z["truth"])}) \
        .groupby(["split", "subject"])["err"].mean()
    mine = TT[(TT["fold"] == k) & (TT["split"] != "train")].set_index(["split", "subject"])["mae"]
    assert np.allclose(mine.sort_index(), ref.sort_index(), atol=1e-3), k
print(TT.groupby("split")["subject"].count().to_dict(), "subject-fold rows;", TT.groupby("split")["windows"].sum().to_dict(), "windows")
''')
code(r'''
FOLDS = TT.groupby(["dataset", "fold", "split"])["mae"].mean().unstack("split")[SPLITS]
SUMMARY = FOLDS.groupby("dataset").agg(["mean", "std"])
both = FOLDS.groupby("fold").mean()  # mean of the 2 sub-datasets per fold
SUMMARY.loc["both"] = both.agg(["mean", "std"]).T.stack()
FOLDS.to_csv(RESULTS / f"final_{CFG}_train_test.csv")
put("train_test", **{f"{ds}_{s}_{m}": SUMMARY.loc[ds, (s, m)] for ds in SUMMARY.index for s in SPLITS for m in ("mean", "std")})
display(FOLDS.round(2))
SUMMARY.round(2)
''')
code(r'''
MK_SPLIT = {"train": "o", "val": "o", "test": "D"}
FILL = {"train": True, "val": False, "test": True}
fig, axes = plt.subplots(1, 2, figsize=figsize(0.95))
for ax, ds in zip(axes, DATASETS):
    f = FOLDS.loc[ds]
    ylab = [f"fold {k}" for k in f.index] + ["mean ± sd"]
    y = np.arange(len(ylab))[::-1]
    for yi, (_, r) in zip(y, f.iterrows()):
        ax.plot([r.min(), r.max()], [yi, yi], color=COLORS["neutral"], lw=3, zorder=1)
    m, sd = SUMMARY.loc[ds].xs("mean", level=1), SUMMARY.loc[ds].xs("std", level=1)
    ax.plot([m.min(), m.max()], [y[-1], y[-1]], color=COLORS["neutral"], lw=3, zorder=1)
    for s in SPLITS:
        c = COLORS[s]
        ax.scatter(f[s], y[:-1], s=STYLE["marker"], marker=MK_SPLIT[s], color=c if FILL[s] else COLORS["surface"],
                   edgecolor=c, linewidth=1.5, zorder=3, label=s)
        ax.errorbar(m[s], y[-1], xerr=sd[s], fmt=MK_SPLIT[s], ms=8, color=c, mfc=c if FILL[s] else COLORS["surface"],
                    mec=c, mew=1.5, capsize=3, zorder=3)
    ax.axhline(y[-1] + 0.5, color=COLORS["grid"], lw=STYLE["lw_thin"])
    ax.set_yticks(y, ylab)
    ax.set_xlim(0, None)
    ax.set_xlabel("MAE (BPM), subject-macro")
    ax.set_title(f"{ds}: train {m['train']:.2f} vs test {m['test']:.2f}")
axes[0].legend(loc="upper left")
fig.suptitle("dist + boot, 3-seed ensemble per fold: train vs val vs test")
save(fig, "train_test_folds")
''')
md("""
One dot per subject. Train: mean over the 3 folds in which the subject was in train; val and test: the one fold each.
Thin lines join the same subject across splits.
""")
code(r'''
PS = TT.groupby(["dataset", "subject", "split"])["mae"].mean().unstack("split")[SPLITS]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=figsize(1))
for ax, ds in zip(axes, DATASETS):
    p = PS.loc[ds]
    jit = rng.uniform(-0.12, 0.12, len(p))
    ax.plot(np.arange(3)[:, None] + jit[None, :], p[SPLITS].T.to_numpy(), color=COLORS["neutral"], lw=0.6, zorder=1)
    bp = ax.boxplot([p[s] for s in SPLITS], positions=range(3), widths=0.5, showfliers=False, patch_artist=True,
                    medianprops={"color": COLORS["text"], "lw": STYLE["lw"]},
                    boxprops={"facecolor": COLORS["surface"], "edgecolor": COLORS["muted"]},
                    whiskerprops={"color": COLORS["muted"]}, capprops={"color": COLORS["muted"]}, zorder=2)
    for i, s in enumerate(SPLITS):
        ax.scatter(i + jit, p[s], s=STYLE["marker"] * 0.5, color=COLORS[s], alpha=0.85, zorder=3, lw=0)
    ax.set_xticks(range(3), [f"{s}\nmedian {p[s].median():.2f}" for s in SPLITS])
    ax.set_ylim(0, None)
    ax.set_ylabel("MAE per subject (BPM)")
    ax.set_title(f"{ds} ({len(p)} subjects)")
fig.suptitle("Per-subject MAE: train vs val vs test (raw 3-seed ensembles)")
save(fig, "train_test_subjects")
put("train_test", **{f"{ds}_median_{s}": PS.loc[ds][s].median() for ds in DATASETS for s in SPLITS})
PS.groupby("dataset").median().round(2)
''')
code(r'''
print(json.dumps(VALUES, indent=1, default=float)[:3000])
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
