"""Offline smoothing report (run plan Method 5) for one config's fold sequences from predict_sequences.py.
Pipeline: ensemble (in the sequences) -> input gate (limit fitted on the capped val preds) -> smoother
(gated windows: R = inf / flat emission / left out of the median) -> output gate on the posterior std
(limit refitted on val sequence windows). Smoother parameters tuned on val, scored on test.
Writes cache/results/smooth.csv."""
import argparse
import csv
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bpm.aggregate import BIN_EDGES
from bpm.quality import balanced_mae, fit_limit
from bpm.smoothing import estimate_q, hmm_forward_backward, kalman_rts, median_smooth, segments
from bpm.split import fold_split, load_folds
from bpm.windowing import STRIDE_S, WINDOW_S
from evaluate import FOLDS_PATH, HR_CURVE_DIR, MANIFEST_PATH

OUT_PATH = ROOT / "cache" / "results" / "smooth.csv"
GRID = {"median": {"k": [3, 5, 9, 15]},
        "kalman": {"m": [1, 2, 4], "c": [0.5, 1, 2, 4]},
        "kalman_const": {"m": [1, 2, 4], "c": [2, 4, 8, 16]},  # no model score: R = c^2
        "hmm": {"m": [1, 2, 4]}}
LAG_WINDOWS = 15  # 30 s


def train_q(train_subjects):
    """Step std of fold-train 10 s-mean labels at 2 s step, per sub-dataset (estimate_q)."""
    series, n_files = {}, 0
    with open(MANIFEST_PATH) as f:
        for row in csv.DictReader(f):
            path = HR_CURVE_DIR / f"{Path(row['file_path']).stem}.npy"
            if row["global_subject_id"] not in train_subjects or not path.exists():
                continue
            curve = np.load(path)
            if len(curve) < WINDOW_S:
                continue
            labels = sliding_window_view(curve, WINDOW_S)[::STRIDE_S].mean(axis=1)
            series.setdefault(row["sub_dataset"], []).extend([*labels, np.nan])  # NaN: no step across files
            n_files += 1
    q = {d: estimate_q(np.array(v)) for d, v in series.items()}
    print(f"q from {n_files} fold-train HR curves: " + ", ".join(f"{d} {v:.3f}" for d, v in q.items()))
    return q


def smooth(seq, name, params, in_limit, q, log_prior, max_gated_s):
    """-> (bpm, posterior std or None, covered) for one recording."""
    bpm, score = seq["bpm"].astype(float), seq["score"].astype(float)
    ok = ~(score > in_limit)  # NaN score passes
    if name == "raw":
        return bpm, None, ok
    out, std, covered = np.full(len(bpm), np.nan), np.full(len(bpm), np.nan), ok.copy()
    s_q = q[seq["dataset_source"]]
    for sl in segments(seq["idx"]):
        b, o = bpm[sl], ok[sl]
        if name == "median":
            out[sl] = median_smooth(b, params["k"], o)
        elif name.startswith("kalman"):
            r = (params["c"] * score[sl]) ** 2 if name == "kalman" else np.full(len(b), params["c"] ** 2)
            r = np.where(o, r, np.inf)
            mean, var = kalman_rts(b, r, (params["m"] * s_q) ** 2)
            out[sl], std[sl] = mean, np.sqrt(var)
        else:
            probs = np.exp(seq["logits"][sl].astype(float))
            _, out[sl], std[sl] = hmm_forward_backward(probs, params["m"] * s_q, log_prior=log_prior, ok=o)
        # gated runs longer than max_gated_s stay uncovered (bridging that long is not trusted);
        # a segment with no passed window has nothing to bridge from (Kalman 0 BPM, HMM flat)
        run_id = np.cumsum(np.r_[True, o[1:] != o[:-1]])
        run_len = np.bincount(run_id)[run_id] * STRIDE_S
        covered[sl] = o | ((run_len <= max_gated_s) & o.any())
    covered &= np.isfinite(out)
    return out, (std if name != "median" else None), covered


def labelled_rows(seqs, results, split):
    """Concatenate labelled windows of one split -> dict of arrays."""
    cols = {k: [] for k in ("err", "subject", "dataset", "covered", "std", "truth")}
    for seq, (out, std, cov) in zip(seqs, results):
        lab = (seq["split"] == split) & np.isfinite(seq["label"])
        if not lab.any():
            continue
        cols["err"].append(np.abs(out - seq["label"])[lab])
        cols["subject"].append(np.full(lab.sum(), seq["subject"]))
        cols["dataset"].append(np.full(lab.sum(), seq["dataset_source"]))
        cols["covered"].append(cov[lab])
        cols["std"].append((std if std is not None else np.full(len(out), np.nan))[lab])
        cols["truth"].append(seq["label"][lab])
    return {k: np.concatenate(v) for k, v in cols.items()}


def lag_error(seqs, results, split, out_limit):
    """shortcut: one jump per recording (largest 30 s label change within one segment), mean |err| over the
    next 30 s of passed windows (covered, posterior std <= out_limit) in that segment; mean over recordings.
    Upgrade to several jumps per recording if too noisy."""
    errs = []
    for seq, (out, std, cov) in zip(seqs, results):
        if seq["split"] != split:
            continue
        lab = seq["label"].astype(float)
        passed = cov if std is None else cov & ~(std > out_limit)
        best = None  # (jump, segment start, t within segment); never across a gap
        for sl in segments(seq["idx"]):
            ls = lab[sl]
            if len(ls) <= LAG_WINDOWS:
                continue
            jump = np.abs(ls[LAG_WINDOWS:] - ls[:-LAG_WINDOWS])
            if np.isfinite(jump).any() and (best is None or np.nanmax(jump) > best[0]):
                best = (np.nanmax(jump), sl.start, np.nanargmax(jump) + LAG_WINDOWS, sl.stop)
        if best is None:
            continue
        _, start, t, stop = best
        after = slice(start + t, min(start + t + LAG_WINDOWS, stop))
        e = np.abs(out[after] - lab[after])[passed[after] & np.isfinite(lab[after])]
        if len(e):
            errs.append(e.mean())
    return float(np.mean(errs)) if errs else np.nan


def metrics(r, passed):
    e, s, d, truth = r["err"], r["subject"], r["dataset"], r["truth"]
    m = {"coverage": passed.mean()}
    m["mae_macro"], m["subjects_out"] = balanced_mae(e, s, d, passed)
    m["mae_window"] = float(np.mean([e[passed & (d == x)].mean() for x in np.unique(d) if (passed & (d == x)).any()]))
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        rows = passed & (truth >= lo) & (truth < hi)
        m[f"mae_{lo}"] = e[rows].mean() if rows.any() else np.nan
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=Path, required=True, help="cache/sequences/<config id>_fold<k>_seed<s>/ or ens_<config id>_fold<k>/")
    parser.add_argument("--preds", type=Path, help="capped preds npz of the same ensemble (input gate limit)")
    parser.add_argument("--max-gated-s", type=float, default=30, help="longest gated run the smoother may bridge")
    args = parser.parse_args()
    config, fold = re.fullmatch(r"(?:ens_)?(.+)_fold(\d+)(?:_seed\d+)?", args.seq.name).groups()
    fold = int(fold)
    seqs = []
    for p in sorted(args.seq.glob("*.npz")):
        z = np.load(p)
        seqs.append({k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files})
    if not seqs:
        sys.exit(f"no sequences in {args.seq}")
    is_dist = "logits" in seqs[0]
    has_score = np.isfinite(np.concatenate([s["score"] for s in seqs]).astype(float)).all()

    groups, _ = load_folds(FOLDS_PATH)
    train = set(fold_split(groups, fold)["train"])
    assert not train & {s["subject"] for s in seqs}, "sequence subject in fold train"
    q = train_q(train)
    log_prior = None  # HMM emission = plain softmax (run plan decision 2026-11-07: the prior floor breaks it)

    in_gates = {"none": np.inf}
    if args.preds is not None and has_score:
        z = np.load(args.preds)
        v = z["split"] == "val"
        if np.isfinite(z["scores"][v]).all():
            for mode in ("coverage", "mae"):
                in_gates[mode] = fit_limit(z["scores"][v], np.abs(z["pred"] - z["truth"])[v],
                                           z["subject"][v], z["dataset_source"][v], mode=mode)[0]
    smoothers = ["median", "kalman" if has_score else "kalman_const"] + (["hmm"] if is_dist else [])

    rows = []
    for gate_name, limit in in_gates.items():
        for name in ["raw", *smoothers]:
            grid = [dict(zip(GRID[name], v)) for v in product(*GRID[name].values())] if name != "raw" else [{}]
            best = None
            for params in grid:  # tune on val: subject-macro balanced MAE of covered windows
                res = [smooth(s, name, params, limit, q, log_prior, args.max_gated_s) for s in seqs]
                val = labelled_rows(seqs, res, "val")
                score = balanced_mae(val["err"], val["subject"], val["dataset"], val["covered"])[0]
                if best is None or score < best[0]:
                    best = (score, params, res, val)
            _, params, res, val = best
            test = labelled_rows(seqs, res, "test")
            out_gates = {"none": np.inf}
            if name != "raw" and name != "median":
                for mode in ("coverage", "mae"):
                    c = val["covered"]
                    out_gates[mode] = fit_limit(val["std"][c], val["err"][c], val["subject"][c],
                                                val["dataset"][c], mode=mode)[0]
            for out_name, out_limit in out_gates.items():
                passed = test["covered"] & ~(test["std"] > out_limit)
                rows.append({"config": config, "fold": fold, "smoother": name, "params": params,
                             "in_gate": gate_name, "out_gate": out_name, "val_mae_macro": best[0],
                             **metrics(test, passed),
                             "lag_mae": lag_error(seqs, res, "test", out_limit)})
    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    show = ["smoother", "params", "in_gate", "out_gate", "val_mae_macro", "mae_macro", "mae_window",
            "coverage", "subjects_out", "lag_mae"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
