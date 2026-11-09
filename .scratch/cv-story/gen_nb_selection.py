"""Generate notebooks/cv_selection.ipynb (CV story part 1: data, protocol, Stages A-C, final pick).
Usage: .venv/bin/python .scratch/cv-story/gen_nb_selection.py notebooks/cv_selection.ipynb"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
from common import SETUP, STYLE, STYLE_MD

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# CV story, part 1: model selection (Stages A-C)

Recomputes every number and figure of the deck slides on data, protocol, Stages A-C and the final config choice, straight from the run outputs. Nothing is retrained.

Inputs: `cache/preds/<tag>.npz` (one per run: val + test windows of its fold, `truth`, `pred`), `cache/runs/*.json`, `cache/results/summary.csv` (`scripts/aggregate_runs.py`), `cache/windows/`, `cache/folds.json`.
Outputs: figures `cache/deck/cv_story/*.png` and the cited numbers `cache/deck/cv_story/values.json` (the deck generator reads both; part 2 `cv_gate_smoothing.ipynb` adds its own keys).

Selection logic is the library's (`bpm.aggregate.decide`), not re-implemented here.
""")
code(SETUP)
md(STYLE_MD)
code(STYLE)

md("""
## Load all single-seed runs

One row per (run, window): 9 configs x 5 folds x 3 seeds, val and test windows of each run's fold. `err = |pred - truth|`.
""")
code(r'''
STAGES = {
    "A": ["huber10", "mae", "mse"],
    "B": ["mse_bootHL", "mse_bootHL_lds0.5", "mse_bootHL_lds1"],
    "C": ["dist_bootHL", "dist_bootHL_lds0.5", "dist_bootHL_lds1"],
}
CONFIGS = [c for s in STAGES.values() for c in s]

parts = []
for p in sorted(PREDS.glob("*.npz")):
    parsed = parse_tag(p.stem)
    if parsed is None or parsed[2] is None:  # skip ensembles (part 2)
        continue
    z = np.load(p)
    parts.append(pd.DataFrame({"config": parsed[0], "fold": parsed[1], "seed": parsed[2], "split": z["split"],
                               "dataset": z["dataset_source"], "subject": z["subject"],
                               "truth": z["truth"].astype(float), "pred": z["pred"].astype(float)}))
W = pd.concat(parts, ignore_index=True)
W["err"] = (W["pred"] - W["truth"]).abs()
W["bin"] = bin_of(W["truth"])
runs = pd.DataFrame([json.loads(p.read_text()) for p in sorted((ROOT / "cache" / "runs").glob("*.json"))])
runs["config"] = runs["tag"].map(lambda t: parse_tag(t)[0])

assert set(W["config"]) == set(CONFIGS), set(W["config"]) ^ set(CONFIGS)
check = W.groupby("config")[["fold", "seed"]].apply(lambda g: len(g.drop_duplicates()))
assert (check == 15).all(), check
assert (runs["rc"] == 0).all() and (runs["eval_rc"] == 0).all()
print(f"{len(W):,} window rows, {W.groupby(['config', 'fold', 'seed']).ngroups} runs")
runs.groupby("config")[["best_epoch", "stop_epoch", "collapsed"]].agg(["mean", "min", "max"]).round(1)
''')

md("""
## 1. Data

Windows and subjects per sub-dataset as cached (`cache/windows/`, up to 2000 windows per subject), and the BPM label distribution. The label distribution is the root of "the problem" further down: most windows sit at 60-90 BPM, the high bins are thin.
""")
code(r'''
folds = json.loads((ROOT / "cache" / "folds.json").read_text())
rows = []
for p in sorted((ROOT / "cache" / "windows").glob("*.npz")):
    z = np.load(p)
    rows.append(pd.DataFrame({"subject": p.stem, "dataset": z["dataset_source"], "truth": z["labels"].astype(float)}))
L = pd.concat(rows, ignore_index=True)
L["dataset"] = L["dataset"].replace({"multisite": "multisite-ppg", "wrist-finger": "wrist-finger-ppg"})
L["group"] = L["subject"].map(lambda s: folds[s]["group"])
L["bin"] = bin_of(L["truth"])

overview = L.groupby("dataset").agg(subjects=("subject", "nunique"), windows=("truth", "size"),
                                    bpm_mean=("truth", "mean"), bpm_sd=("truth", "std"),
                                    bpm_min=("truth", "min"), bpm_max=("truth", "max")).round(1)
share = pd.crosstab(L["bin"], L["dataset"], normalize="columns").mul(100).round(1)
put("data", subjects=int(overview["subjects"].sum()), windows=int(overview["windows"].sum()),
    **{f"subjects_{d}": int(overview.loc[d, "subjects"]) for d in overview.index},
    **{f"windows_{d}": int(overview.loc[d, "windows"]) for d in overview.index},
    share_60_90=float(L["truth"].between(60, 90, inclusive="left").mean() * 100),
    share_110plus=float((L["truth"] >= 110).mean() * 100))
display(overview)
display(share)
pd.crosstab(L["group"], L["dataset"], values=L["subject"], aggfunc="nunique").rename_axis("fold group")
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(1))
counts = pd.crosstab(L["bin"], L["dataset"])
x = np.arange(len(counts))
for i, d in enumerate(DATASETS):
    ax.bar(x + (i - 0.5) * 0.4, counts[d], 0.4, label=d, color=COLORS[d])
ax.set_xticks(x, counts.index)
ax.set_xlabel("ECG label (BPM bin)")
ax.set_ylabel("cached windows")
ax.set_title("Label distribution: most windows at 60-90 BPM, high bins are thin")
ax.legend()
save(fig, "data_labels")
''')

md("""
## 2. Protocol

- 5 stratified subject groups per sub-dataset (`cache/folds.json`). Fold k: test = group k, val = group (k+1) % 5, train = rest. Every subject is in test exactly once.
- 3 seeds per config and fold (42, 1, 2): seed changes init, shuffle and augmentation only.
- **Decision** (`decide`): on **val** windows only. Per subject: median over seeds of the subject MAE. Key = mean over subjects of the gap to the best config, then mean over the two sub-datasets (paired). SE = subject bootstrap. Tie set = configs within 1 SE of the best; the winner is the tie-set config with the lowest bin-balanced val MAE (mean over BPM bins).
- **Global** decision pools the val sets of all 5 folds. **LTGO** check (leave-test-group-out): for test group k, decide without the fold whose val is group k, so the decision never saw those subjects.
- Greedy stages: A picks the loss (L*), B adds bootstrap + LDS to L*, C swaps the head to dist; final = better of B and C winner. Test numbers (`oof`) are reported after the choice, they never enter it.
""")

md("""
## 3. Stage A: which loss?

Huber δ10, MAE, MSE; reg head, no bootstrap, LDS off.
""")
code(r'''
def decisions(cands):
    out = {"global": decide(W, range(5), candidates=cands)}
    for k in range(5):
        out[f"k{k}"] = decide(W, ltgo_folds(k), candidates=cands)
    return out


def verdict(dec):
    return pd.DataFrame({name: {"winner": d.index[d["win"]][0], "tie set": ", ".join(d.index[d["tie"]])}
                         for name, d in dec.items()}).T


decA = decisions(STAGES["A"])
winA = decA["global"].index[decA["global"]["win"]][0]
gA = decA["global"]
put("stage_a", winner=winA, ltgo_agree=int(sum(d.index[d["win"]][0] == winA for k, d in decA.items() if k != "global")),
    **{f"key_{c}": gA.loc[c, "key"] for c in gA.index}, **{f"se_{c}": gA.loc[c, "se"] for c in gA.index},
    **{f"binbal_{c}": gA.loc[c, "bin_bal"] for c in gA.index})
display(gA[["key", "se", "tie", "bin_bal", "win"]].round(3))
verdict(decA)
''')
code(r'''
def plot_keys(ax, d, title):
    d = d.sort_values("key")
    y = np.arange(len(d))[::-1]
    for yi, (c, r) in zip(y, d.iterrows()):
        ax.errorbar(r["key"], yi, xerr=r["se"], fmt=MARKERS[lds_of(c)], ms=9, color=COLORS[c], capsize=4,
                    mec=COLORS["truth"] if r["win"] else COLORS[c], mew=2 if r["win"] else 1)
    ax.axvline(0, color=COLORS["ideal"], lw=STYLE["lw_thin"], ls="--")
    ax.set_yticks(y, [LABELS[c] + ("  (winner)" if d.loc[c, "win"] else "") for c in d.index])
    ax.set_xlabel("gap to best config, val MAE (BPM), ± 1 SE")
    ax.set_title(title)


fig, ax = plt.subplots(figsize=figsize(0.6))
plot_keys(ax, gA, "Stage A: MSE is best; Huber δ10 and MAE are 0.1 BPM behind")
save(fig, "stage_a_keys")
''')
code(r'''
# per fold: seed-median val subject-macro MAE per config, to show how much folds differ vs configs
def fold_macro(cfgs, split="val"):
    v = W[(W["split"] == split) & W["config"].isin(cfgs)]
    s = subject_mae(v, by=("config", "fold", "seed"))
    return (s.groupby(["config", "fold", "seed"])[["dataset", "mae"]].apply(macro)
             .groupby(["config", "fold"]).median().unstack("fold"))


fmA = fold_macro(STAGES["A"])
fig, ax = plt.subplots(figsize=figsize(0.8))
for c in STAGES["A"]:
    ax.plot(range(5), fmA.loc[c], marker="o", color=COLORS[c], label=LABELS[c])
ax.set_xticks(range(5), [f"fold {k}" for k in range(5)])
ax.set_ylabel("val MAE (BPM), seed median")
ax.set_title("Per fold: folds differ by ~1 BPM, losses by ~0.1-0.3 BPM")
ax.legend()
save(fig, "stage_a_folds")
fmA.round(2)
''')

md("""
## 4. The problem: pull to the middle

From the Stage A winner (MSE, reg head), test windows of all folds and seeds (each subject appears 3 times, once per seed). A regressor trained on a label distribution peaked at 60-90 BPM shrinks extreme labels toward the middle: high labels get under-predicted, low labels over-predicted. Measures: bias per BPM bin (mean of pred - truth), OLS slope of pred on truth (1 = no shrinkage), MAE in the 120+ bins.
""")
code(r'''
def problem_stats(cfg, split="test"):
    t = W[(W["config"] == cfg) & (W["split"] == split)]
    g = t.assign(bias=t["pred"] - t["truth"]).groupby("bin", observed=False)
    per_bin = pd.DataFrame({"n": g.size(), "bias": g["bias"].mean(), "mae": g["err"].mean()})
    slope, icpt = np.polyfit(t["truth"], t["pred"], 1)
    return per_bin, slope, icpt


pbA, slopeA, icptA = problem_stats(winA)
hi = W[(W["config"] == winA) & (W["split"] == "test") & (W["truth"] >= 120)]
put("problem", slope=slopeA, intercept=icptA, mae_120plus=hi["err"].mean(), bias_120plus=(hi["pred"] - hi["truth"]).mean(),
    n_120plus=len(hi), mae_70_80=pbA.loc["70-80", "mae"], bias_40_60=pbA.loc["40-60", "bias"])
print(f"slope {slopeA:.3f}, intercept {icptA:.1f}; 120+ BPM: MAE {hi['err'].mean():.1f}, bias {(hi['pred'] - hi['truth']).mean():+.1f}, n={len(hi)}")
pbA.round(2)
''')
code(r'''
t = W[(W["config"] == winA) & (W["split"] == "test")]
fig, axes = plt.subplots(1, 2, figsize=figsize(1))
ax = axes[0]
hb = ax.hexbin(t["truth"], t["pred"], gridsize=60, bins="log", cmap=DENSITY_CMAP, mincnt=1, extent=(35, 185, 35, 185))
lim = np.array([35, 185])
ax.plot(lim, lim, color=COLORS["ideal"], ls="--", lw=STYLE["lw_thin"], label="pred = truth")
ax.plot(lim, slopeA * lim + icptA, color=COLORS["bad"], label=f"fit, slope {slopeA:.2f}")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("ECG truth (BPM)"); ax.set_ylabel("predicted (BPM)")
ax.set_title("MSE: predictions vs truth (test)")
ax.legend(loc="upper left")
fig.colorbar(hb, ax=ax, label="windows (log)")
ax = axes[1]
x = np.arange(len(pbA))
ax.bar(x, pbA["bias"], color=[COLORS["bad"] if b < 0 else COLORS["truth"] for b in pbA["bias"]])
ax.axhline(0, color=COLORS["muted"], lw=STYLE["lw_thin"])
ax.set_xticks(x, pbA.index, rotation=45)
ax.set_xlabel("ECG truth (BPM bin)"); ax.set_ylabel("mean pred - truth (BPM)")
ax.set_title("Bias per BPM bin")
save(fig, "problem")
''')

md("""
## 5. Stage B: fix the tails with data (bootstrap) and weights (LDS)

L* = MSE plus time-warped bootstrap windows in the low and high bins (`HL`), and label-distribution-smoothing loss weights with power p ∈ {0, 0.5, 1} (p = 0 = off). Decided among the 3 B configs only.
""")
code(r'''
decB = decisions(STAGES["B"])
winB = decB["global"].index[decB["global"]["win"]][0]
gB = decB["global"]
put("stage_b", winner=winB, ltgo_agree=int(sum(d.index[d["win"]][0] == winB for k, d in decB.items() if k != "global")),
    **{f"key_{c}": gB.loc[c, "key"] for c in gB.index}, **{f"se_{c}": gB.loc[c, "se"] for c in gB.index})
display(gB[["key", "se", "tie", "bin_bal", "win"]].round(3))
verdict(decB)
''')
code(r'''
def per_bin_mae(cfgs, split="val"):
    v = W[(W["split"] == split) & W["config"].isin(cfgs)]
    # seed median of the window-pooled per-bin MAE
    return (v.groupby(["config", "seed", "bin"], observed=False)["err"].mean()
             .groupby(["config", "bin"], observed=False).median().unstack("bin"))


def plot_bins(ax, cfgs, split, title):
    m = per_bin_mae(cfgs, split)
    x = np.arange(len(BIN_LABELS))
    for c in cfgs:
        ax.plot(x, m.loc[c, BIN_LABELS], marker=MARKERS[lds_of(c)], ls=LINESTYLES[lds_of(c)], color=COLORS[c], label=LABELS[c])
    ax.set_xticks(x, BIN_LABELS, rotation=45)
    ax.set_xlabel("ECG truth (BPM bin)"); ax.set_ylabel(f"{split} MAE (BPM), seed median")
    ax.set_title(title)
    ax.legend()
    return m


fig, ax = plt.subplots(figsize=figsize(1))
mB = plot_bins(ax, ["mse", *STAGES["B"]], "val", "Stage B: bootstrap + LDS lower the tails, raise the middle")
save(fig, "stage_b_bins")
mB.round(2)
''')

md("""
## 6. Stage C: distribution head

Same bootstrap and LDS grid with the `dist` head: 191 1-BPM bins (30-220), Gaussian-target cross-entropy, BPM = expectation of the softmax. Decided among the 3 C configs, then final = better of the B and C winner.
""")
code(r'''
decC = decisions(STAGES["C"])
winC = decC["global"].index[decC["global"]["win"]][0]
decF = decisions([winB, winC])
final = decF["global"].index[decF["global"]["win"]][0]
gC, gF = decC["global"], decF["global"]
gAll = decide(W, range(5))  # all 9 together, for the overview figure
put("stage_c", winner=winC, ltgo_agree=int(sum(d.index[d["win"]][0] == winC for k, d in decC.items() if k != "global")),
    **{f"key_{c}": gC.loc[c, "key"] for c in gC.index}, **{f"se_{c}": gC.loc[c, "se"] for c in gC.index})
put("final", config=final, ltgo_agree=int(sum(d.index[d["win"]][0] == final for k, d in decF.items() if k != "global")),
    gap_B_vs_C=gF.loc[winB, "key"] if final == winC else gF.loc[winC, "key"],
    se_B_vs_C=gF.loc[winB, "se"] if final == winC else gF.loc[winC, "se"],
    **{f"all_key_{c}": gAll.loc[c, "key"] for c in gAll.index}, **{f"all_se_{c}": gAll.loc[c, "se"] for c in gAll.index},
    **{f"all_binbal_{c}": gAll.loc[c, "bin_bal"] for c in gAll.index})
display(gC[["key", "se", "tie", "bin_bal", "win"]].round(3))
display(verdict(decC))
display(verdict(decF))
gAll[["key", "se", "bin_bal", "bin_bal_sd"]].sort_values("key").round(3)
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(1.15))
for c, r in gAll.iterrows():
    win = c == final
    ax.errorbar(r["key"], r["bin_bal"], xerr=r["se"], yerr=r["bin_bal_sd"], fmt=MARKERS[lds_of(c)], ms=10 if win else 8,
                color=COLORS[c], capsize=3, mec=COLORS["truth"] if win else COLORS[c], mew=2 if win else 1)
    ax.annotate(LABELS[c] + ("  (final)" if win else ""), (r["key"], r["bin_bal"]), xytext=(8, 4),
                textcoords="offset points", color=COLORS["text"], fontweight="bold" if win else "normal")
ax.set_xlabel("overall: gap to best config, val MAE (BPM)  [decides]")
ax.set_ylabel("bin-balanced val MAE (BPM)  [tails]")
ax.set_title("All 9 configs: LDS buys tail accuracy with overall error; dist + boot is best overall")
save(fig, "tradeoff")
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(1))
pb = {c: problem_stats(c)[0] for c in ["mse", winB, winC]}
x = np.arange(len(BIN_LABELS))
for i, c in enumerate(pb):
    ax.bar(x + (i - 1) * 0.27, pb[c]["bias"], 0.27, color=COLORS[c], label=f"{LABELS[c]} (slope {problem_stats(c)[1]:.2f})")
ax.axhline(0, color=COLORS["muted"], lw=STYLE["lw_thin"])
ax.set_xticks(x, BIN_LABELS, rotation=45)
ax.set_xlabel("ECG truth (BPM bin)"); ax.set_ylabel("mean pred - truth (BPM), test")
ax.set_title("Bias per bin: the stage winners shrink the pull to the middle")
ax.legend()
save(fig, "bias_winners")
put("problem", **{f"slope_{c}": problem_stats(c)[1] for c in pb},
    **{f"mae_120plus_{c}": W[(W["config"] == c) & (W["split"] == "test") & (W["truth"] >= 120)]["err"].mean() for c in pb})
pd.concat({c: v[["bias", "mae"]] for c, v in pb.items()}, axis=1).round(2)
''')

md("""
## 7. Test results (out-of-fold, after the choice)

Each seed pools its 5 test folds = every subject once; subject-macro balanced MAE, mean ± std over the 3 seeds. 95 % CI of the difference to MSE by subject bootstrap; Wilcoxon signed-rank per sub-dataset on paired per-subject MAE, Holm-adjusted over configs (all from `cache/results/summary.csv`, `bpm.aggregate.vs_baseline`).
""")
code(r'''
S = pd.read_csv(RESULTS / "summary.csv", index_col="config").loc[CONFIGS]
cols = ["oof_mean", "oof_std", "oof_ci_lo", "oof_ci_hi", "diff_ci_lo", "diff_ci_hi",
        "p_multisite-ppg_holm", "p_wrist-finger-ppg_holm"]
T = S[cols].copy()
# cross-check: recompute oof_mean from W
chk = (subject_mae(W[W["split"] == "test"], by=("config", "seed")).groupby(["config", "seed"])[["dataset", "mae"]]
       .apply(macro).groupby("config").mean())
assert np.allclose(chk.loc[CONFIGS], T["oof_mean"]), (chk, T["oof_mean"])
put("test", **{f"oof_{c}": T.loc[c, "oof_mean"] for c in CONFIGS}, **{f"oof_std_{c}": T.loc[c, "oof_std"] for c in CONFIGS},
    **{f"diff_lo_{c}": T.loc[c, "diff_ci_lo"] for c in CONFIGS}, **{f"diff_hi_{c}": T.loc[c, "diff_ci_hi"] for c in CONFIGS},
    **{f"p_ms_{c}": T.loc[c, "p_multisite-ppg_holm"] for c in CONFIGS}, **{f"p_wf_{c}": T.loc[c, "p_wrist-finger-ppg_holm"] for c in CONFIGS})
T.round(3)
''')
code(r'''
fig, ax = plt.subplots(figsize=figsize(1))
d = T.sort_values("oof_mean")
y = np.arange(len(d))[::-1]
for yi, (c, r) in zip(y, d.iterrows()):
    ax.errorbar(r["oof_mean"], yi, xerr=[[r["oof_mean"] - r["oof_ci_lo"]], [r["oof_ci_hi"] - r["oof_mean"]]],
                fmt=MARKERS[lds_of(c)], ms=9, color=COLORS[c], capsize=4,
                mec=COLORS["truth"] if c == final else COLORS[c], mew=2 if c == final else 1)
ax.set_yticks(y, [LABELS[c] for c in d.index])
ax.set_xlabel("test MAE (BPM), subject-macro, out-of-fold, 95 % CI")
ax.set_title("Test: dist + boot lowest; the CIs overlap, so the margin is small")
save(fig, "test_oof")
''')
md("""
## Summary values (what the deck cites)
""")
code(r'''
print(json.dumps({k: VALUES[k] for k in ["data", "stage_a", "problem", "stage_b", "stage_c", "final"]}, indent=1, default=float)[:4000])
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, sys.argv[1])
