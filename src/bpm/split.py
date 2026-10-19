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
