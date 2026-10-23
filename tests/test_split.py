from bpm.split import subject_wise_split


def test_min_one_subject_per_split_per_dataset():
    subjects_by_dataset = {
        "exercise": [f"exercise_s{i}" for i in range(1, 9)],  # 8 subjects
    }

    result = subject_wise_split(subjects_by_dataset, seed=42)

    exercise_ids = set(subjects_by_dataset["exercise"])
    train, val, test = result["train"], result["val"], result["test"]

    assert train | val | test == exercise_ids
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(val) == 1
    assert len(test) == 1
    assert len(train) == 6


def test_unions_across_sub_datasets_and_is_deterministic():
    subjects_by_dataset = {
        "exercise": [f"exercise_s{i}" for i in range(1, 9)],
        "wristfinger": [f"wristfinger_{i:02d}" for i in range(1, 28)],
    }
    all_ids = {sid for ids in subjects_by_dataset.values() for sid in ids}

    result1 = subject_wise_split(subjects_by_dataset, seed=42)
    result2 = subject_wise_split(subjects_by_dataset, seed=42)

    for split in ("train", "val", "test"):
        assert result1[split] == result2[split]

    union = result1["train"] | result1["val"] | result1["test"]
    assert union == all_ids
    assert result1["train"].isdisjoint(result1["val"])
    assert result1["train"].isdisjoint(result1["test"])
