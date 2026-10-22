from pathlib import Path

from bpm.manifest import build_manifest


def fake_header_reader(file_path):
    return {"n_samples": 1000, "fs": 100.0}


def test_build_manifest_namespaces_subjects_and_extracts_metadata(tmp_path: Path):
    exercise_dir = tmp_path / "wrist-ppg-during-exercise-1.0.0"
    exercise_dir.mkdir()
    (exercise_dir / "s1_walk.edf").touch()
    (exercise_dir / "s3_high_resistance_bike.edf").touch()

    multisite_dir = tmp_path / "multisite-ppg"
    multisite_dir.mkdir()
    (multisite_dir / "P10_earring_0.edf").touch()
    (multisite_dir / "P10_earring_1.edf").touch()

    wristfinger_dir = tmp_path / "wrist-finger-ppg"
    wristfinger_dir.mkdir()
    (wristfinger_dir / "01_0.edf").touch()

    records = build_manifest(tmp_path, header_reader=fake_header_reader)

    by_path = {r["file_path"]: r for r in records}
    assert len(records) == 5

    r = by_path["wrist-ppg-during-exercise-1.0.0/s1_walk.edf"]
    assert r["sub_dataset"] == "wrist-ppg-during-exercise-1.0.0"
    assert r["global_subject_id"] == "exercise_s1"
    assert r["raw_subject_id"] == "s1"
    assert r["activity"] == "walk"
    assert r["n_samples"] == 1000
    assert r["duration_sec"] == 10.0

    r = by_path["wrist-ppg-during-exercise-1.0.0/s3_high_resistance_bike.edf"]
    assert r["activity"] == "high_resistance_bike"

    r = by_path["multisite-ppg/P10_earring_0.edf"]
    assert r["sub_dataset"] == "multisite-ppg"
    assert r["global_subject_id"] == "multisite_P10"
    assert r["raw_subject_id"] == "P10"
    assert r["site"] == "earring"
    assert r["trial"] == "0"

    r = by_path["wrist-finger-ppg/01_0.edf"]
    assert r["sub_dataset"] == "wrist-finger-ppg"
    assert r["global_subject_id"] == "wristfinger_01"
    assert r["raw_subject_id"] == "01"
    assert r["trial"] == "0"


def test_build_manifest_ignores_missing_subject(tmp_path: Path):
    exercise_dir = tmp_path / "wrist-ppg-during-exercise-1.0.0"
    exercise_dir.mkdir()
    (exercise_dir / "s1_walk.edf").touch()

    records = build_manifest(tmp_path, header_reader=fake_header_reader)
    subject_ids = {r["global_subject_id"] for r in records}
    assert "exercise_s7" not in subject_ids
