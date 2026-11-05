import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.cache import read_ppg_and_hr
from bpm.data_loading import load_windows_for_subjects, subjects_by_dataset_from_manifest
from bpm.edf_io import read_channels
from bpm.manifest import MULTISITE_DIR, WRISTFINGER_DIR
from bpm.preprocessing import bandpass_filter
from bpm.split import subject_wise_split
from bpm.timewarp import plan_bootstrap, ppg_hr_agrees, warp_window

DATA_ROOT = ROOT / "harmonized_ecg_ppg_databases"
MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
HR_CURVE_DIR = ROOT / "cache" / "hr_curves"  # delete when ground-truth code changes
OUT_DIR = ROOT / "cache" / "bootstrap"

TARGET_PER_BIN = 500
L_BINS = [(40, 50), (50, 60)]
H_BINS = [(lo, lo + 10) for lo in range(110, 190, 10)]
# wrist-finger test stays below 98 BPM and its sources top out at 113, so no H there
BINS = {MULTISITE_DIR: L_BINS + H_BINS, WRISTFINGER_DIR: L_BINS}
SEED = 42


def hr_curve_for(row):
    path = HR_CURVE_DIR / f"{Path(row['file_path']).stem}.npy"
    if path.exists():
        return np.load(path)
    # same unusable-second rules (bad beats, flat raw PPG) as the window cache
    _, curve, _ = read_ppg_and_hr(DATA_ROOT / row["file_path"])
    np.save(path, curve)
    return curve


def ppg_ok_for(row, curve):
    signals, fs = read_channels(DATA_ROOT / row["file_path"], ["PPG"])
    return ppg_hr_agrees(bandpass_filter(signals["PPG"], fs), curve, fs)


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    split = subject_wise_split(subjects_by_dataset_from_manifest(MANIFEST_PATH), seed=42)
    train = sorted(split["train"])
    real = load_windows_for_subjects(train, CACHE_DIR)

    HR_CURVE_DIR.mkdir(parents=True, exist_ok=True)
    for letter in "HL":
        (OUT_DIR / letter).mkdir(parents=True, exist_ok=True)
        # drop earlier runs so the loader never mixes stale windows in
        for stale in (OUT_DIR / letter).glob("*.npz"):
            stale.unlink()

    rng = np.random.default_rng(SEED)
    out = defaultdict(lambda: ([], []))  # (letter, subject) -> (windows, labels)
    for sub_dataset, bins in BINS.items():
        labels = real["labels"][real["dataset_source"] == sub_dataset]
        real_counts = {b: int(((labels >= b[0]) & (labels < b[1])).sum()) for b in bins}
        quotas = {b: max(0, TARGET_PER_BIN - n) for b, n in real_counts.items()}
        files = [r for r in rows if r["sub_dataset"] == sub_dataset and r["global_subject_id"] in split["train"]]

        print(f"{sub_dataset}: HR curves for {len(files)} train files")
        curves = [hr_curve_for(r) for r in files]
        l_quotas = {b: q for b, q in quotas.items() if b[1] <= 100}
        h_quotas = {b: q for b, q in quotas.items() if b[0] >= 100}
        picks, stats = plan_bootstrap(curves, l_quotas, rng)
        if h_quotas:
            # H only warps sources whose PPG pulse matches the ECG label (L already agrees ~95%).
            # shortcut: L and H track reuse separately, so a start in the 69-86 BPM overlap
            # can be used up to 2x max_reuse; merge into one call if that matters.
            source_ok = [ppg_ok_for(r, c) for r, c in zip(files, curves)]
            h_picks, h_stats = plan_bootstrap(curves, h_quotas, rng, source_ok=source_ok)
            picks += h_picks
            stats.update(h_stats)

        by_file = defaultdict(list)
        for file_idx, start, factor, label in picks:
            by_file[file_idx].append((start, factor, label))
        for file_idx, file_picks in sorted(by_file.items()):
            row = files[file_idx]
            signals, fs = read_channels(DATA_ROOT / row["file_path"], ["PPG"])
            ppg = bandpass_filter(signals["PPG"], fs)
            for start, factor, label in file_picks:
                window, _ = warp_window(ppg, curves[file_idx], fs, start, factor)
                letter = "L" if label < 100 else "H"
                ws, ls = out[(letter, row["global_subject_id"])]
                ws.append(window)
                ls.append(label)

        print(f"  {'bin':>9} {'real':>5} {'target':>6} {'made':>5} {'unique':>6}")
        for (lo, hi), (got, quota, unique) in stats.items():
            short = "  SHORT" if got < quota else ""
            print(f"  {lo:>4}-{hi:<4} {real_counts[(lo, hi)]:>5} {quota:>6} {got:>5} {unique:>6}{short}")

    sources = {r["global_subject_id"]: r["sub_dataset"] for r in rows}
    for (letter, subject), (ws, ls) in sorted(out.items()):
        np.savez(
            OUT_DIR / letter / f"{subject}.npz",
            windows=np.stack(ws).astype(np.float32),
            labels=np.array(ls, dtype=np.float32),
            dataset_source=np.full(len(ls), sources[subject]),
        )
    for letter in "HL":
        n = sum(len(ls) for (lt, _), (_, ls) in out.items() if lt == letter)
        print(f"{letter}: {n} windows in {OUT_DIR / letter}")


if __name__ == "__main__":
    main()
