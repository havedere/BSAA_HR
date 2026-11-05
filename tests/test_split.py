import json

from bpm.split import assign_groups, fold_split, load_folds, subject_wise_split


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


def _cv_setup():
    subjects_by_dataset = {
        "multisite": [f"m{i:02d}" for i in range(20)],
        "wristfinger": [f"w{i:02d}" for i in range(27)],
    }
    score = {s: i / 20 for i, s in enumerate(subjects_by_dataset["multisite"])}
    score.update({s: 0.0 for s in subjects_by_dataset["wristfinger"]})  # all tied
    return subjects_by_dataset, score


def test_folds_disjoint_and_each_subject_test_and_val_once():
    subjects_by_dataset, score = _cv_setup()
    groups = assign_groups(subjects_by_dataset, score)
    all_ids = set(groups)
    test_count = dict.fromkeys(all_ids, 0)
    val_count = dict.fromkeys(all_ids, 0)
    for fold in range(5):
        split = fold_split(groups, fold)
        assert split["train"] | split["val"] | split["test"] == all_ids
        assert split["train"].isdisjoint(split["val"])
        assert split["train"].isdisjoint(split["test"])
        assert split["val"].isdisjoint(split["test"])
        for s in split["test"]:
            test_count[s] += 1
        for s in split["val"]:
            val_count[s] += 1
    assert set(test_count.values()) == {1}
    assert set(val_count.values()) == {1}


def test_groups_balanced_per_dataset():
    subjects_by_dataset, score = _cv_setup()
    groups = assign_groups(subjects_by_dataset, score)
    sizes = {
        name: sorted(sum(groups[s] == g for s in ids) for g in range(5))
        for name, ids in subjects_by_dataset.items()
    }
    assert sizes == {"multisite": [4] * 5, "wristfinger": [5, 5, 5, 6, 6]}


def test_stable_on_ties_and_input_order():
    subjects_by_dataset, score = _cv_setup()
    reversed_input = {k: v[::-1] for k, v in subjects_by_dataset.items()}
    assert assign_groups(subjects_by_dataset, score) == assign_groups(reversed_input, score)


def test_multisite_top_k_in_different_groups():
    subjects_by_dataset, score = _cv_setup()
    groups = assign_groups(subjects_by_dataset, score)
    top5 = sorted(subjects_by_dataset["multisite"], key=lambda s: -score[s])[:5]
    assert {groups[s] for s in top5} == set(range(5))


def test_load_folds_returns_groups_and_file_hash(tmp_path):
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({"a": {"group": 0, "score": 0.1}, "b": {"group": 3, "score": 0.0}}))
    groups, folds_hash = load_folds(path)
    assert groups == {"a": 0, "b": 3}
    assert len(folds_hash) == 64
    assert load_folds(path)[1] == folds_hash
