import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpm.cache import build_subject_cache

DATA_ROOT = Path(__file__).resolve().parent.parent / "harmonized_ecg_ppg_databases"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "cache" / "manifest.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "cache" / "windows"


def main():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    by_subject = defaultdict(list)
    for row in rows:
        row["file_path"] = DATA_ROOT / row["file_path"]
        by_subject[row["global_subject_id"]].append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for subject_id, subject_rows in sorted(by_subject.items()):
        result = build_subject_cache(subject_rows)
        if result is None:
            print(f"{subject_id}: no usable windows, skipped")
            continue
        np.savez(
            OUT_DIR / f"{subject_id}.npz",
            windows=result["windows"],
            labels=result["labels"],
            dataset_source=result["dataset_source"],
        )
        print(f"{subject_id}: {len(result['labels'])} windows")


if __name__ == "__main__":
    main()
