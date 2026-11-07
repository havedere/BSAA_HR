"""Per-recording window sequences for smoothing: every 2 s-stride window of the fold's val+test recordings
within +-5 min of a labelled window, ensemble-predicted -> cache/sequences/<config id>_fold<k>_seed<s>/ (one ckpt) or ens_<config id>_fold<k>/ (several),
<file stem>.npz per recording; same naming as cache/preds/."""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bpm.edf_io import read_channels
from bpm.preprocessing import bandpass_filter
from bpm.split import fold_split, load_folds
from bpm.windowing import STRIDE_S, WINDOW_S, all_windows
from evaluate import FOLDS_PATH, HR_CURVE_DIR, MANIFEST_PATH, ckpt_config, load_model, predict

DATA_ROOT = ROOT / "harmonized_ecg_ppg_databases"
OUT_DIR = ROOT / "cache" / "sequences"
SCOPE_S = 300  # keep windows within +-5 min of a labelled window
CHUNK = 2048  # windows per forward pass (a full 18 h file at once does not fit on the GPU)


def in_scope(idx, labels, scope_s=SCOPE_S):
    """Mask of windows whose start lies within scope_s of a labelled window's start."""
    t, lab_t = idx * STRIDE_S, idx[~np.isnan(labels)] * STRIDE_S
    if not len(lab_t):
        return np.zeros(len(idx), bool)
    pos = np.searchsorted(lab_t, t)
    left = np.abs(t - lab_t[np.clip(pos - 1, 0, len(lab_t) - 1)])
    right = np.abs(lab_t[np.clip(pos, 0, len(lab_t) - 1)] - t)
    return np.minimum(left, right) <= scope_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--ckpt", type=Path, nargs="+", required=True)
    parser.add_argument("--files", nargs="*", help="only these file stems (smoke runs)")
    args = parser.parse_args()

    config, seed = ckpt_config(args.ckpt, args.fold)
    # like cache/preds/: a single-ckpt smoke must never land in an ensemble's dir
    name = f"{config}_fold{args.fold}_seed{seed}" if seed is not None else f"ens_{config}_fold{args.fold}"
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, _ = load_folds(FOLDS_PATH)
    split = fold_split(groups, args.fold)
    split_of = {s: name for name in ("val", "test") for s in split[name]}
    models = [load_model(c) for c in args.ckpt]
    dist = models[0].head_type == "dist"

    with open(MANIFEST_PATH) as f:
        rows = [r for r in csv.DictReader(f) if r["global_subject_id"] in split_of]
    if args.files:
        rows = [r for r in rows if Path(r["file_path"]).stem in args.files]
    missing, n_files, n_windows, t_read, t_pred = [], 0, 0, 0.0, 0.0
    for row in rows:
        stem = Path(row["file_path"]).stem
        curve_path = HR_CURVE_DIR / f"{stem}.npy"
        if not curve_path.exists():
            missing.append(stem)
            continue
        curve = np.load(curve_path)
        if not np.isfinite(curve).any():
            print(f"{stem}: no labelled windows, skipped")
            continue
        t0 = time.perf_counter()
        signals, fs = read_channels(DATA_ROOT / row["file_path"], ["PPG"])
        t_read += time.perf_counter() - t0
        windows, idx, labels = all_windows({"PPG": bandpass_filter(signals["PPG"], fs)}, curve,
                                           fs, WINDOW_S, STRIDE_S)
        keep = in_scope(idx, labels)
        windows, idx, labels = windows[keep].astype(np.float32), idx[keep], labels[keep]
        if not len(idx):
            print(f"{stem}: no labelled windows, skipped")
            continue
        t0 = time.perf_counter()
        parts = [predict(models, {"windows": windows[i:i + CHUNK]}, with_logp=dist)
                 for i in range(0, len(idx), CHUNK)]
        t_pred += time.perf_counter() - t0
        out = {
            "idx": idx.astype(np.int32), "label": labels.astype(np.float32),
            "bpm": np.concatenate([p["pred"] for p in parts]).astype(np.float16),
            "score": np.concatenate([p["scores"] for p in parts]).astype(np.float16),
            "subject": row["global_subject_id"], "dataset_source": row["sub_dataset"],
            "split": split_of[row["global_subject_id"]],
        }
        if dist:
            out["logits"] = np.concatenate([p["logp"] for p in parts]).astype(np.float16)
        np.savez(out_dir / f"{stem}.npz", **out)
        n_files, n_windows = n_files + 1, n_windows + len(idx)
        print(f"{stem}: {len(idx)} windows ({(~np.isnan(labels)).sum()} labelled)")

    print(f"wrote {n_files} recordings to {out_dir}; {n_windows} windows, "
          f"{n_windows / max(t_pred, 1e-9):.0f} windows/s predict, EDF read {t_read:.1f} s")
    if missing:
        print(f"WARNING: {len(missing)} recording(s) without cache/hr_curves/<file>.npy, skipped: {missing}")


if __name__ == "__main__":
    main()
