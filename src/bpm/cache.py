import numpy as np

from .edf_io import read_channels
from .ground_truth import ecg_to_hr_curve
from .manifest import MULTISITE_DIR
from .preprocessing import bandpass_filter
from .windowing import make_windows

WINDOW_S, STRIDE_S = 10, 2
MULTISITE_CAP_PER_SUBJECT = 2000
CAP_SEED = 42


def process_file(file_path):
    signals, fs = read_channels(file_path, ["ECG", "PPG"])

    hr_curve = ecg_to_hr_curve(signals["ECG"], fs)
    windows, labels_arr = make_windows(
        {"PPG": bandpass_filter(signals["PPG"], fs)},
        hr_curve,
        fs,
        window_s=WINDOW_S,
        stride_s=STRIDE_S,
    )
    return windows, labels_arr


def build_subject_cache(subject_rows, seed=CAP_SEED):
    sub_dataset = subject_rows[0]["sub_dataset"]

    all_windows, all_labels = [], []
    for row in subject_rows:
        windows, labels = process_file(row["file_path"])
        if not len(windows):
            continue
        # cap per-file too: a single multisite file can run up to ~18hr and
        # produce tens of thousands of windows, which would otherwise spike
        # memory well past the eventual per-subject cap below.
        if sub_dataset == MULTISITE_DIR and len(windows) > MULTISITE_CAP_PER_SUBJECT:
            rng = np.random.default_rng(seed)
            keep = rng.choice(len(windows), size=MULTISITE_CAP_PER_SUBJECT, replace=False)
            windows, labels = windows[keep], labels[keep]
        all_windows.append(windows)
        all_labels.append(labels)

    if not all_windows:
        return None

    windows = np.concatenate(all_windows).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.float32)

    if sub_dataset == MULTISITE_DIR and len(windows) > MULTISITE_CAP_PER_SUBJECT:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(windows), size=MULTISITE_CAP_PER_SUBJECT, replace=False)
        windows, labels = windows[keep], labels[keep]

    dataset_source = np.full(len(windows), sub_dataset)
    return {
        "windows": windows,
        "labels": labels,
        "dataset_source": dataset_source,
    }
