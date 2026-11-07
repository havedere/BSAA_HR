import numpy as np

from .edf_io import read_channels
from .ground_truth import ecg_quality_mask, ecg_to_hr_curve
from .manifest import MULTISITE_DIR
from .preprocessing import bandpass_filter
from .windowing import STRIDE_S, WINDOW_S, make_windows

MULTISITE_CAP_PER_SUBJECT = 2000
CAP_SEED = 42


def read_ppg_and_hr(file_path):
    """Raw PPG, 1 Hz HR curve (NaN where unusable) and fs for one file."""
    signals, fs = read_channels(file_path, ["ECG", "PPG"])

    hr_curve = ecg_to_hr_curve(signals["ECG"], fs)
    hr_curve[ecg_quality_mask(signals["ECG"], fs)] = np.nan
    # constant raw PPG (sensor off / padding) is unusable even when ECG is live
    sec = int(fs)
    ppg_secs = signals["PPG"][: len(hr_curve) * sec].reshape(len(hr_curve), sec)
    hr_curve[np.ptp(ppg_secs, axis=1) == 0] = np.nan
    return signals["PPG"], hr_curve, fs


def process_file(file_path):
    ppg, hr_curve, fs = read_ppg_and_hr(file_path)
    windows, labels_arr = make_windows(
        {"PPG": bandpass_filter(ppg, fs)},
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
        # cap per file too, so one long multisite file can't blow up memory
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
