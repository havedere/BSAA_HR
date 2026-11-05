import hashlib
import json
from pathlib import Path

import numpy as np


def subject_wise_split(subjects_by_dataset, seed=42):
    rng = np.random.default_rng(seed)
    train, val, test = set(), set(), set()

    for dataset_name in sorted(subjects_by_dataset):
        subjects = list(subjects_by_dataset[dataset_name])
        rng.shuffle(subjects)

        n = len(subjects)
        if n < 3:
            raise ValueError(
                f"{dataset_name!r} has only {n} subject(s); need >=3 for a "
                "non-empty train/val/test split"
            )
        n_val = max(1, round(0.15 * n))
        n_test = max(1, round(0.15 * n))
        n_train = n - n_val - n_test

        train.update(subjects[:n_train])
        val.update(subjects[n_train : n_train + n_val])
        test.update(subjects[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def assign_groups(subjects_by_dataset, strat_score, k=5, seed=42):
    """Stratified subject -> group (0..k-1), dealt per sub-dataset from strata of k."""
    groups = {}
    for dataset_name in sorted(subjects_by_dataset):
        rng = np.random.default_rng(seed)
        ranked = sorted(subjects_by_dataset[dataset_name], key=lambda s: (-strat_score[s], s))
        for start in range(0, len(ranked), k):
            stratum = ranked[start : start + k]
            rng.shuffle(stratum)
            for g, subject in enumerate(stratum):
                groups[subject] = g
    return groups


def fold_split(groups, fold, k=5):
    """Fold k: test = group k, val = group (k+1) % k, train = the rest."""
    val_group = (fold + 1) % k
    split = {"train": set(), "val": set(), "test": set()}
    for subject, g in groups.items():
        name = "test" if g == fold else "val" if g == val_group else "train"
        split[name].add(subject)
    return split


def load_folds(path):
    """(subject -> group, folds_hash) from folds.json; hash = sha256 of the file bytes."""
    raw = Path(path).read_bytes()
    groups = {s: v["group"] for s, v in json.loads(raw).items()}
    return groups, hashlib.sha256(raw).hexdigest()
