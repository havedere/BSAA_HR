import re
import warnings
from pathlib import Path

MULTISITE_DIR = "multisite-ppg"
WRISTFINGER_DIR = "wrist-finger-ppg"

_MULTISITE_RE = re.compile(r"^(P\d+)_([a-z]+)_(\d+)\.edf$")
_WRISTFINGER_RE = re.compile(r"^(\d+)_(\d+)\.edf$")


def build_manifest(root, header_reader):
    # wrist-ppg-during-exercise-1.0.0 is deliberately not scanned, see README.md
    records = []
    records += _scan_multisite(Path(root) / MULTISITE_DIR, header_reader)
    records += _scan_wristfinger(Path(root) / WRISTFINGER_DIR, header_reader)
    # relative to `root` so the manifest survives moving the data folder
    for r in records:
        r["file_path"] = str(Path(r["file_path"]).relative_to(root))
    return records


def _scan_multisite(dir_path, header_reader):
    records = []
    for file_path in sorted(dir_path.glob("*.edf")):
        m = _MULTISITE_RE.match(file_path.name)
        if not m:
            warnings.warn(f"skipping unrecognized filename: {file_path}")
            continue
        raw_subject_id, site, trial = m.group(1), m.group(2), m.group(3)
        records.append(
            _base_record(file_path, header_reader)
            | {
                "sub_dataset": MULTISITE_DIR,
                "global_subject_id": f"multisite_{raw_subject_id}",
                "raw_subject_id": raw_subject_id,
                "site": site,
                "trial": trial,
            }
        )
    return records


def _scan_wristfinger(dir_path, header_reader):
    records = []
    for file_path in sorted(dir_path.glob("*.edf")):
        m = _WRISTFINGER_RE.match(file_path.name)
        if not m:
            warnings.warn(f"skipping unrecognized filename: {file_path}")
            continue
        raw_subject_id, trial = m.group(1), m.group(2)
        records.append(
            _base_record(file_path, header_reader)
            | {
                "sub_dataset": WRISTFINGER_DIR,
                "global_subject_id": f"wristfinger_{raw_subject_id}",
                "raw_subject_id": raw_subject_id,
                "trial": trial,
            }
        )
    return records


def _base_record(file_path, header_reader):
    header = header_reader(file_path)
    return {
        "file_path": str(file_path),
        "n_samples": header["n_samples"],
        "duration_sec": header["n_samples"] / header["fs"],
    }
