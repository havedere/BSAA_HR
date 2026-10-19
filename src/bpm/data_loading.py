import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .channels import CHANNEL_ORDER

HAS_MOTION_CHANNEL = CHANNEL_ORDER.index("has_motion")


def subjects_by_dataset_from_manifest(manifest_path):
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    by_dataset = defaultdict(set)
    for row in rows:
        by_dataset[row["sub_dataset"]].add(row["global_subject_id"])
    return {k: sorted(v) for k, v in by_dataset.items()}


def load_subject_npz(subject_id, cache_dir):
    path = Path(cache_dir) / f"{subject_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    windows = data["windows"]
    has_motion = windows[:, HAS_MOTION_CHANNEL, 0]
    return {
        "windows": windows,
        "labels": data["labels"],
        "dataset_source": data["dataset_source"],
        "has_motion": has_motion,
    }


def load_windows_for_subjects(subject_ids, cache_dir):
    parts = [load_subject_npz(sid, cache_dir) for sid in subject_ids]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("no cached windows found for the given subjects")
    return {
        key: np.concatenate([p[key] for p in parts])
        for key in ("windows", "labels", "dataset_source", "has_motion")
    }
