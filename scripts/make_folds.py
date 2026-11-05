import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bpm.data_loading import load_subject_npz, subjects_by_dataset_from_manifest
from bpm.manifest import MULTISITE_DIR, WRISTFINGER_DIR
from bpm.split import assign_groups, load_folds

MANIFEST_PATH = ROOT / "cache" / "manifest.csv"
CACHE_DIR = ROOT / "cache" / "windows"
OUT_PATH = ROOT / "cache" / "folds.json"
K = 5
# stratification score = share of the subject's windows >= this BPM (no wrist-finger window reaches 130)
SCORE_BPM = {MULTISITE_DIR: 130, WRISTFINGER_DIR: 100}
TAIL_BPM = 140


def main():
    subjects_by_dataset = subjects_by_dataset_from_manifest(MANIFEST_PATH)
    labels, score = {}, {}
    for name, subjects in subjects_by_dataset.items():
        for s in subjects:
            data = load_subject_npz(s, CACHE_DIR)
            if data is None:
                raise FileNotFoundError(f"no cached windows for {s}; run scripts/build_cache.py")
            labels[s] = data["labels"]
            score[s] = float(np.mean(labels[s] >= SCORE_BPM[name]))

    groups = assign_groups(subjects_by_dataset, score, k=K, seed=42)
    with open(OUT_PATH, "w") as f:
        json.dump({s: {"group": groups[s], "score": score[s]} for s in sorted(groups)}, f, indent=1)
    _, folds_hash = load_folds(OUT_PATH)

    print(f"wrote {len(groups)} subjects to {OUT_PATH}")
    for name, subjects in sorted(subjects_by_dataset.items()):
        counts = [sum(groups[s] == g for s in subjects) for g in range(K)]
        print(f"{name}: subjects per group {counts}")
    ms = subjects_by_dataset[MULTISITE_DIR]
    top = sorted(ms, key=lambda s: (-score[s], s))[:K]
    print(f"multisite top {K} by >={SCORE_BPM[MULTISITE_DIR]} share: "
          + ", ".join(f"{s} (g{groups[s]}, {score[s]:.3f})" for s in top))
    tail = [sum(int(np.sum(labels[s] >= TAIL_BPM)) for s in groups if groups[s] == g) for g in range(K)]
    print(f"windows >={TAIL_BPM} BPM per group: {tail}")
    print(f"folds_hash {folds_hash}")


if __name__ == "__main__":
    main()
