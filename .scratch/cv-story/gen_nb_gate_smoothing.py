"""Generate notebooks/cv_gate_smoothing.ipynb (CV story part 2: Stages D-F = ensembles, quality gate, smoothing).
Usage: .venv/bin/python .scratch/cv-story/gen_nb_gate_smoothing.py notebooks/cv_gate_smoothing.ipynb"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from common import SETUP, STYLE, STYLE_MD

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CV story, part 2: ensembles, quality gate, smoothing (Stages D-F)

No new training. For three configs (final `dist_bootHL`, Stage B winner `mse_bootHL`, plain `mse` as baseline), per fold:

- **D** 3-seed ensemble (`evaluate.py --fold k --ckpt <3 seeds> --mc 20`) -> `cache/preds/ens_<config>_fold<k>.npz`
- **E** quality gate (`gate_report.py`): a per-window score; limit fitted on val, applied to test -> `cache/results/gate.csv`
- **F** every 2 s window of the val/test recordings (`predict_sequences.py`), then smoothing over time (`smooth_report.py`): median, Kalman + RTS, HMM (dist only); input gate before, output gate (posterior std) after; parameters tuned on val, scored on test -> `cache/results/smooth_ens_<config>_fold<k>.csv`

All test numbers below are means over the 5 test folds (each fold's subject-macro MAE), unless stated. Run `cv_selection.ipynb` first only if you want its keys in `values.json`; this notebook does not depend on it.
""")
code(SETUP)
md(STYLE_MD)
code(STYLE)
code(r'''
CFGS = ["dist_bootHL", "mse_bootHL", "mse"]


def load(tag, config, fold, seed):
    z = np.load(PREDS / f"{tag}.npz")
    d = pd.DataFrame({k: z[k] for k in ["subject", "split", "truth", "pred", "scores", "score_spread", "score_std", "score_mc"]})
    d["dataset"] = z["dataset_source"]
    d[["config", "fold", "seed"]] = config, fold, seed
    return d


E = pd.concat([load(f"ens_{c}_fold{k}", c, k, -1) for c in CFGS for k in range(5)], ignore_index=True)
S1 = pd.concat([load(f"{c}_fold{k}_seed{s}", c, k, s) for c in CFGS for k in range(5) for s in (42, 1, 2)], ignore_index=True)
for d in (E, S1):
    d["err"] = (d["pred"] - d["truth"]).abs()
print(len(E), "ensemble rows,", len(S1), "single-seed rows")
E.groupby("config")[["score_std", "score_spread", "score_mc"]].apply(lambda g: g.notna().mean()).round(2)
''')

md("""
## D. Seed ensemble

Ensemble = combine the 3 seeds (reg: mean prediction, spread = std across seeds; dist: average of the softmax curves). Test MAE per fold, single seeds averaged vs the ensemble.
""")
code(r'''
def fold_mae(d, by):
    t = d[d["split"] == "test"]
    s = subject_mae(t, by=by)
    return s.groupby(list(by))[["dataset", "mae"]].apply(macro)


single = fold_mae(S1, ("config", "fold", "seed")).groupby(["config", "fold"]).mean().unstack("fold")
ens = fold_mae(E, ("config", "fold")).unstack("fold")
D = pd.DataFrame({"single_seed": single.mean(1), "ensemble": ens.mean(1)}).loc[CFGS]
D["gain"] = D["single_seed"] - D["ensemble"]
put("ensemble", **{f"single_{c}": D.loc[c, "single_seed"] for c in CFGS}, **{f"ens_{c}": D.loc[c, "ensemble"] for c in CFGS},
    **{f"gain_{c}": D.loc[c, "gain"] for c in CFGS})
display(pd.concat({"single seed (mean of 3)": single, "ensemble": ens}).round(2))
D.round(3)
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(0.75))
x = np.arange(len(CFGS))
ax.bar(x - 0.2, D["single_seed"], 0.4, color=COLORS["neutral"], edgecolor=COLORS["muted"], label="single seed (mean of 3)")
ax.bar(x + 0.2, D["ensemble"], 0.4, color=[COLORS[c] for c in CFGS], label="3-seed ensemble")
for xi, (c, r) in zip(x, D.iterrows()):
    ax.annotate(f"-{r['gain']:.2f}", (xi + 0.2, r["ensemble"]), xytext=(0, 4), textcoords="offset points", ha="center")
ax.set_xticks(x, [LABELS[c] for c in CFGS])
ax.set_ylabel("test MAE (BPM), mean over folds")
ax.set_ylim(0, D.values[:, :2].max() * 1.3)
ax.set_title("Seed ensemble: a small, free gain for every config")
ax.legend(loc="upper center", ncols=2)
save(fig, "ensemble")
''')

md("""
## E. Quality gate

A score per window says how much to trust the prediction; windows above a limit are dropped. Scores:
- `score_std`: std of the dist head's softmax curve (dist only)
- `score_spread`: disagreement of the 3 seeds
- `score_mc`: std over 20 MC-dropout passes
- `prominence`, `skew`: signal-quality scores computed from the PPG window itself (no model)

**coverage** mode: limit = 90 % quantile of the val scores (keep 90 %). **mae** mode: loosest limit with val MAE ≤ 5 BPM; the val MAE of the ensembles is already below 5, so this mode keeps (almost) everything and is not shown further.
AURC = area under the risk-coverage curve (mean MAE over all coverage levels, lower = better ranking); ρ = Spearman(score, error).
""")
code(r'''
G = pd.read_csv(RESULTS / "gate.csv")
Gc = G[G["mode"] == "coverage"]
agg = Gc.groupby(["config", "score"])[["test_coverage", "test_mae_all", "test_mae_passed", "test_aurc", "test_rho",
                                       "test_mae@80%", "test_mae@70%"]].mean()
print("mae-mode test coverage (mean):", G[G["mode"] == "mae"].groupby("config")["test_coverage"].mean().round(3).to_dict())
best = agg.loc["dist_bootHL"].sort_values("test_aurc").index[0]
put("gate", best_score=best, **{f"{c}_{s}_{m}": agg.loc[(c, s), m] for c, s in agg.index
                                for m in ["test_mae_all", "test_mae_passed", "test_aurc", "test_rho", "test_coverage", "test_mae@80%"]})
agg.round(3)
''')
code(r'''
SCORES = ["score_std", "score_mc", "score_spread", "prominence", "skew"]
cov_cols = ["test_mae@100%", "test_mae@90%", "test_mae@80%", "test_mae@70%"]
cov_x = [1.0, 0.9, 0.8, 0.7]
fig, axes = plt.subplots(1, 2, figsize=figsize(1), sharey=True)
for ax, c in zip(axes, ["dist_bootHL", "mse"]):
    g = Gc[Gc["config"] == c].groupby("score")[cov_cols].mean()
    for s in SCORES:
        if s in g.index:
            ax.plot(cov_x, g.loc[s, cov_cols], marker="o", color=COLORS[s], ls=SCORE_LS[s], label=s)
    ax.invert_xaxis()
    ax.set_xlabel("coverage (share of windows kept)")
    ax.set_title(LABELS[c])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
axes[0].set_ylabel("test MAE of kept windows (BPM)")
axes[0].legend()
fig.suptitle("Quality gate: dropping the least trusted windows; dist curve std ranks best")
save(fig, "gate_curves")
''')
code(r'''
# where does the gate cut? coverage per BPM bin, coverage mode, dist_bootHL
bins_cov = [c for c in G.columns if c.startswith("cov_") and c[4:].isdigit()]
cb = Gc[Gc["config"] == "dist_bootHL"].groupby("score")[bins_cov].mean().loc[["score_std", "score_mc", "prominence"]]
fig, ax = plt.subplots(figsize=figsize(0.8))
x = np.arange(len(bins_cov))
for i, s in enumerate(cb.index):
    ax.bar(x + (i - 1) * 0.27, cb.loc[s], 0.27, color=COLORS[s], label=s)
ax.axhline(0.9, color=COLORS["muted"], ls="--", lw=STYLE["lw_thin"], label="90 % target")
ax.set_xticks(x, BIN_LABELS, rotation=45)
ax.set_ylim(0, 1.25)
ax.set_xlabel("ECG truth (BPM bin)"); ax.set_ylabel("share kept (test)")
ax.set_title("dist + boot: the gate drops mostly high-BPM windows")
ax.legend(loc="upper center", ncols=4)
save(fig, "gate_bins")
put("gate", **{f"cov_bin_{s}_{b}": cb.loc[s, b] for s in cb.index for b in bins_cov})
cb.round(3)
''')
code(r'''
# pooled risk-coverage curves of the model scores (from the preds files), test windows of all folds
def risk_curve(score, err):
    o = np.argsort(score, kind="stable")
    e = np.asarray(err)[o]
    n = np.arange(1, len(e) + 1)
    return n / len(e), np.cumsum(e) / n


fig, ax = plt.subplots(figsize=figsize(0.9))
t = E[(E["split"] == "test") & (E["config"] == "dist_bootHL")]
for s in ["score_std", "score_mc", "score_spread"]:
    cov, risk = risk_curve(t[s].to_numpy(), t["err"].to_numpy())
    ax.plot(cov, risk, color=COLORS[s], label=s)
oracle = risk_curve(t["err"].to_numpy(), t["err"].to_numpy())
ax.plot(*oracle, color=COLORS["ideal"], ls=":", label="oracle (sort by true error)")
ax.set_xlim(0.3, 1); ax.set_ylim(0, None)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("coverage"); ax.set_ylabel("MAE of kept windows (BPM), window-pooled")
ax.set_title("dist + boot, risk-coverage (test, all folds pooled)")
ax.legend()
save(fig, "risk_coverage")
''')

md("""
## F. Smoothing over time

Predictions every 2 s along each recording. Smoothers: running **median** (k windows), **Kalman + RTS** (random-walk HR, measurement noise from the window score, process noise from the train labels' step std), **HMM** over the 191 BPM bins (dist only, softmax as emission). Input gate: windows above the coverage limit get no weight (bridged if the gap ≤ 30 s). Output gate: drop windows whose posterior std is above the 90 % val quantile.

`smooth_report.py` scores every labelled window of the val/test recordings (not the ≤ 2000-per-subject cap of the preds files), so its "raw" MAE differs slightly from the ensemble MAE above.
""")
code(r'''
F = pd.concat([pd.read_csv(p) for p in sorted(RESULTS.glob("smooth_ens_*.csv"))], ignore_index=True)
PIPES = [("raw", "none", "none", "raw"), ("raw", "coverage", "none", "raw + in gate"),
         ("median", "none", "none", "median"), ("kalman", "none", "none", "Kalman"),
         ("hmm", "none", "none", "HMM"), ("kalman", "coverage", "coverage", "Kalman + gates"),
         ("hmm", "coverage", "coverage", "HMM + gates")]
rows = []
for c in CFGS:
    for sm, ig, og, name in PIPES:
        f = F[(F["config"] == c) & (F["smoother"] == sm) & (F["in_gate"] == ig) & (F["out_gate"] == og)]
        if len(f):
            assert len(f) == 5, (c, name, len(f))
            rows.append({"config": c, "pipeline": name, "coverage": f["coverage"].mean(), "mae": f["mae_macro"].mean(),
                         "mae_sd": f["mae_macro"].std(), "lag_mae": f["lag_mae"].mean(),
                         "params": ", ".join(sorted(set(f["params"])))})
P = pd.DataFrame(rows).set_index(["config", "pipeline"])
put("smooth", **{f"{c}_{p}_{m}".replace(" ", "_").replace("+", "plus"): P.loc[(c, p), m]
                 for c, p in P.index for m in ["coverage", "mae", "mae_sd", "lag_mae"]})
P.round(3)
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(1))
MK = {"raw": "o", "raw + in gate": "o", "median": "D", "Kalman": "s", "HMM": "^", "Kalman + gates": "s", "HMM + gates": "^"}
for c in CFGS:
    p = P.loc[c]
    ax.plot(p["coverage"], p["mae"], ls="none", color=COLORS[c], alpha=0.3)
    for name, r in p.iterrows():
        filled = "gate" in name
        ax.scatter(r["coverage"], r["mae"], s=STYLE["marker"], marker=MK[name], color=COLORS[c] if filled else COLORS["surface"],
                   edgecolor=COLORS[c], linewidth=1.5, zorder=3)
        if c == "dist_bootHL":
            dy = 6 if name.startswith("HMM") else -12  # HMM and Kalman sit on top of each other
            ax.annotate(name, (r["coverage"], r["mae"]), xytext=(8, dy), textcoords="offset points")
for c in CFGS:
    ax.scatter([], [], color=COLORS[c], label=LABELS[c])
ax.invert_xaxis()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("coverage (filled = gated)"); ax.set_ylabel("test MAE (BPM), mean over folds")
ax.set_title("Smoothing: Kalman/HMM + gates reach ~3.1 BPM at ~86 % coverage (dist + boot)")
ax.legend(loc="upper right")
save(fig, "smooth_pareto")
''')
code(r'''
# per-bin test MAE: raw vs Kalman + gates
bcols = [f"mae_{lo}" for lo in BIN_EDGES[:-1]]
fig, ax = plt.subplots(figsize=figsize(0.85))
x = np.arange(len(bcols))
for c in ["dist_bootHL", "mse"]:
    for (sm, ig, og), ls, lab in [(("raw", "none", "none"), "--", "raw"), (("kalman", "coverage", "coverage"), "-", "Kalman + gates")]:
        f = F[(F["config"] == c) & (F["smoother"] == sm) & (F["in_gate"] == ig) & (F["out_gate"] == og)]
        ax.plot(x, f[bcols].mean(), ls=ls, marker="o", color=COLORS[c], label=f"{LABELS[c]}, {lab}")
ax.set_xticks(x, BIN_LABELS, rotation=45)
ax.set_xlabel("ECG truth (BPM bin)"); ax.set_ylabel("test MAE (BPM), window-pooled")
ax.set_title("Smoothing helps the middle; the high-BPM tail stays the weak spot")
ax.legend()
save(fig, "smooth_bins")
''')

md("""
### Example recording

Re-runs the tuned Kalman + gates pipeline of one fold (same functions as `smooth_report.py`) on one test recording: the recording with the largest label range among the test recordings of fold 0. Shaded = windows dropped by the output gate.
""")
code(r'''
from bpm.quality import fit_limit
from bpm.split import fold_split, load_folds
from smooth_report import FOLDS_PATH, labelled_rows, smooth, train_q

CFG, FOLD = "dist_bootHL", 0
seq_dir = ROOT / "cache" / "sequences" / f"ens_{CFG}_fold{FOLD}"
seqs = []
for p in sorted(seq_dir.glob("*.npz")):
    z = np.load(p)
    seqs.append({"name": p.stem, **{k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}})
row = F[(F["config"] == CFG) & (F["fold"] == FOLD) & (F["smoother"] == "kalman") & (F["in_gate"] == "coverage") & (F["out_gate"] == "coverage")].iloc[0]
params = eval(row["params"])  # dict literal written by smooth_report
z = np.load(PREDS / f"ens_{CFG}_fold{FOLD}.npz")
v = z["split"] == "val"
in_limit = fit_limit(z["scores"][v], np.abs(z["pred"] - z["truth"])[v], z["subject"][v], z["dataset_source"][v])[0]
groups, _ = load_folds(FOLDS_PATH)
q = train_q(set(fold_split(groups, FOLD)["train"]))
res = [smooth(s, "kalman", params, in_limit, q, None, 30) for s in seqs]
val = labelled_rows(seqs, res, "val")
c = val["covered"]
out_limit = fit_limit(val["std"][c], val["err"][c], val["subject"][c], val["dataset"][c])[0]
test = labelled_rows(seqs, res, "test")
passed = test["covered"] & ~(test["std"] > out_limit)
from bpm.quality import balanced_mae
mae_check = balanced_mae(test["err"], test["subject"], test["dataset"], passed)[0]
print(f"params {params}, in_limit {in_limit:.3f}, out_limit {out_limit:.3f}; test MAE {mae_check:.3f} vs csv {row['mae_macro']:.3f}")
assert abs(mae_check - row["mae_macro"]) < 1e-6
''')
code(r'''
def rec_stats(seq, r):
    out, std, cov = r
    lab = np.isfinite(seq["label"])
    kept = lab & cov & ~(std > out_limit)
    e_raw = np.abs(seq["bpm"].astype(float) - seq["label"])[lab].mean()
    return lab.sum(), kept.sum() / max(lab.sum(), 1), e_raw, np.abs(out - seq["label"])[kept].mean()


# a recording that shows the gate at work: 300+ labelled windows, 70-95 % kept, raw MAE near the test mean,
# largest raw - (Kalman + gates) gain among those
cand = []
for i, (seq, r) in enumerate(zip(seqs, res)):
    if seq["split"] != "test":
        continue
    n, kept, e_raw, e_k = rec_stats(seq, r)
    if n >= 300 and 0.7 <= kept <= 0.95 and 3 <= e_raw <= 8:
        cand.append((e_raw - e_k, i))
assert cand, "no recording matches the example filter"
i = max(cand)[1]
s = seqs[i]
out, std, cov = res[i]
lab = np.isfinite(s["label"])
span = slice(max(np.flatnonzero(lab)[0] - 15, 0), np.flatnonzero(lab)[-1] + 15)  # labelled part +- 30 s
s = {k: (v[span] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(lab) else v) for k, v in s.items()}
out, std, cov = out[span], std[span], cov[span]
tmin = s["idx"] * 2 / 60  # window index * 2 s stride -> minutes
gated = ~(cov & ~(std > out_limit))
fig, ax = plt.subplots(figsize=figsize(1.1))
for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, gated.astype(int), 0]) == d) for d in (1, -1)]):
    ax.axvspan(tmin[a], tmin[min(b, len(tmin) - 1)], color=COLORS["bad"], alpha=STYLE["alpha"], lw=0)
ax.plot(tmin, s["bpm"].astype(float), color=COLORS["raw"], lw=STYLE["lw_thin"], alpha=0.7, label="raw ensemble (every 2 s)")
ax.plot(tmin, out, color=COLORS["kalman"], label="Kalman + RTS")
ax.plot(tmin, s["label"], color=COLORS["truth"], lw=STYLE["lw_thin"], label="ECG truth")
ax.fill_between([], [], color=COLORS["bad"], alpha=STYLE["alpha"], label="dropped by gate")
lab = np.isfinite(s["label"])
e_raw = np.abs(s["bpm"].astype(float) - s["label"])[lab].mean()
e_k = np.abs(out - s["label"])[lab & ~gated].mean()
ax.set_xlabel("time in recording (min)"); ax.set_ylabel("BPM")
ax.set_title(f"{s['name']} ({s['dataset_source']}, test fold {FOLD}): raw MAE {e_raw:.1f}, Kalman + gates MAE {e_k:.1f} on {(~gated[lab]).mean():.0%} kept")
ax.legend(loc="upper left", ncols=2)
save(fig, "smooth_example")
put("smooth", example=s["name"], example_raw=e_raw, example_kalman=e_k)
''')

md("""
## The whole pipeline in one number per step

dist + boot, test, mean over folds. Steps 1-2 from the capped preds files (Stage D), steps 3-6 from the smoothing report (all labelled windows).
""")
code(r'''
c = "dist_bootHL"
steps = pd.DataFrame([
    ("single model", D.loc[c, "single_seed"], 1.0),
    ("3-seed ensemble", D.loc[c, "ensemble"], 1.0),
    ("ensemble, all windows", P.loc[(c, "raw"), "mae"], 1.0),
    ("+ Kalman smoothing", P.loc[(c, "Kalman"), "mae"], P.loc[(c, "Kalman"), "coverage"]),
    ("+ in/out gate", P.loc[(c, "Kalman + gates"), "mae"], P.loc[(c, "Kalman + gates"), "coverage"]),
], columns=["step", "mae", "coverage"])
fig, ax = plt.subplots(figsize=figsize(0.8))
x = np.arange(len(steps))
ax.bar(x, steps["mae"], color=[COLORS["neutral"]] * 2 + [COLORS[c]] * 3, edgecolor=COLORS["muted"])
for xi, r in steps.iterrows():
    ax.annotate(f"{r['mae']:.2f}" + (f"\n{r['coverage']:.0%} kept" if r["coverage"] < 0.999 else ""),
                (xi, r["mae"]), xytext=(0, 4), textcoords="offset points", ha="center")
ax.axvline(1.5, color=COLORS["muted"], ls=":", lw=STYLE["lw_thin"])
ax.set_xticks(x, steps["step"])
ax.set_ylim(0, steps["mae"].max() * 1.25)
ax.set_ylabel("test MAE (BPM)")
ax.set_title("dist + boot: from one model to the full pipeline")
save(fig, "pipeline_steps")
put("pipeline", **{r["step"]: r["mae"] for _, r in steps.iterrows()}, final_coverage=steps["coverage"].iloc[-1])
steps.round(3)
''')
code(r'''
print(json.dumps({k: VALUES[k] for k in ["ensemble", "pipeline"]}, indent=1, default=float))
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
