"""Quality-gate report: per preds file (cache/preds/*.npz from evaluate.py --fold), fit each score's gate
limit on val rows, apply it to test; writes cache/results/gate.csv."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.aggregate import BIN_EDGES, parse_tag
from bpm.quality import aurc, balanced_mae, fit_limit, gate, skewness, spearman_ci, spectral_prominence

CACHE_DIR = ROOT / "cache" / "windows"
OUT_PATH = ROOT / "cache" / "results" / "gate.csv"
MODEL_SCORES = ("score_spread", "score_std", "score_mc")
COVERAGES = (1.0, 0.9, 0.8, 0.7)


def signal_scores(z):
    """skew / prominence scores for every preds row, from the cached windows (same order as evaluate.py)."""
    subj = z["subject"]
    order = [s for i, s in enumerate(subj) if i == 0 or subj[i - 1] != s]
    sk, pr = [], []
    for s in order:
        d = np.load(CACHE_DIR / f"{s}.npz")
        for split in [sp for sp in ("val", "test") if ((subj == s) & (z["split"] == sp)).any()]:
            assert np.array_equal(d["labels"], z["truth"][(subj == s) & (z["split"] == split)]), s
            sk.append(skewness(d["windows"]))
            pr.append(spectral_prominence(d["windows"]))
    return {"skew": np.concatenate(sk), "prominence": np.concatenate(pr)}


def report(path, n_boot):
    config, fold, seed = parse_tag(path.stem)
    z = np.load(path)
    err = np.abs(z["pred"] - z["truth"]).astype(float)
    scores = signal_scores(z) | {k: z[k] for k in MODEL_SCORES if k in z.files and np.isfinite(z[k]).all()}
    val, test = z["split"] == "val", z["split"] == "test"
    cols = lambda rows: (err[rows], z["subject"][rows], z["dataset_source"][rows])
    rows_out = []
    for name, score in scores.items():
        score = np.asarray(score, float)
        for mode in ("coverage", "mae"):
            limit, info = fit_limit(score[val], *cols(val), mode=mode)
            e, s, d = cols(test)
            st, passed = score[test], gate(score[test], limit)
            row = {"preds": path.stem, "config": config, "fold": fold, "seed": seed, "score": name, "mode": mode,
                   "limit": limit, "infeasible": info["infeasible"], "val_coverage": info["coverage"],
                   "val_mae_passed": info["mae"], "test_coverage": passed.mean(),
                   "test_mae_all": balanced_mae(e, s, d)[0]}
            row["test_mae_passed"], row["test_subjects_out"] = balanced_mae(e, s, d, passed)
            row["test_mae_suppressed"] = balanced_mae(e, s, d, ~passed)[0] if (~passed).any() else np.nan
            row["test_aurc"] = aurc(st, e)
            row["test_rho"], row["test_rho_lo"], row["test_rho_hi"] = spearman_ci(st, e, s, n_boot)
            for c in COVERAGES:
                row[f"test_mae@{c:.0%}"] = balanced_mae(e, s, d, st <= np.quantile(st, c))[0]
            for ds in np.unique(d):
                row[f"cov_{ds}"] = passed[d == ds].mean()
            truth = z["truth"][test]
            for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
                inside = (truth >= lo) & (truth < hi)
                row[f"cov_{lo}"] = passed[inside].mean() if inside.any() else np.nan
            rows_out.append(row)
    return rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, nargs="+", required=True, help="cache/preds/*.npz (CV fold files)")
    parser.add_argument("--n-boot", type=int, default=200, help="subject-bootstrap draws for the Spearman CI")
    args = parser.parse_args()
    bad = [p for p in args.preds if parse_tag(p.stem) is None]
    if bad:
        sys.exit(f"not CV preds files: {bad}")
    df = pd.DataFrame([r for p in args.preds for r in report(p, args.n_boot)])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    show = ["preds", "score", "mode", "val_coverage", "test_coverage", "test_mae_all", "test_mae_passed",
            "test_mae_suppressed", "test_aurc", "test_rho", "infeasible"]
    print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {OUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
