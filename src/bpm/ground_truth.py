import neurokit2 as nk
import numpy as np

_HR_MIN, _HR_MAX = 30.0, 220.0
_RR_RATIO_LIMIT = 1.2


def _neurokit2_r_peak_detector(ecg, fs):
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info["ECG_R_Peaks"])


def ecg_to_hr_curve(ecg, fs, r_peak_detector=_neurokit2_r_peak_detector):
    n_1hz = int(len(ecg) // fs)
    curve = np.full(n_1hz, np.nan)

    peak_samples = r_peak_detector(ecg, fs)
    peak_times = np.asarray(peak_samples) / fs
    if len(peak_times) < 2:
        return curve

    rr = np.diff(peak_times)
    inst_hr = 60.0 / rr
    anchor_times = peak_times[1:]

    t = np.arange(n_1hz, dtype=float)
    in_domain = (t >= anchor_times[0]) & (t <= anchor_times[-1])
    curve[in_domain] = np.interp(t[in_domain], anchor_times, inst_hr)

    bad = (inst_hr < _HR_MIN) | (inst_hr > _HR_MAX)
    ratio = rr[1:] / rr[:-1]
    bad[1:] |= (ratio > _RR_RATIO_LIMIT) | (ratio < 1 / _RR_RATIO_LIMIT)

    # NaN every integer second inside a bad span [peak_i, peak_i+1] (inclusive) via
    # a diff array; one full-length mask per bad beat was O(beats x seconds)
    lo = np.clip(np.ceil(peak_times[:-1][bad]).astype(int), 0, n_1hz)
    hi = np.clip(np.floor(peak_times[1:][bad]).astype(int) + 1, 0, n_1hz)
    keep = lo < hi
    cover = np.zeros(n_1hz + 1, dtype=int)
    np.add.at(cover, lo[keep], 1)
    np.add.at(cover, hi[keep], -1)
    curve[np.cumsum(cover[:-1]) > 0] = np.nan

    return curve
