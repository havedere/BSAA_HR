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

    for i in range(len(rr)):
        bad = inst_hr[i] < _HR_MIN or inst_hr[i] > _HR_MAX
        if i > 0:
            ratio = rr[i] / rr[i - 1]
            bad = bad or ratio > _RR_RATIO_LIMIT or ratio < 1 / _RR_RATIO_LIMIT
        if bad:
            span_start, span_end = peak_times[i], peak_times[i + 1]
            curve[(t >= span_start) & (t <= span_end)] = np.nan

    return curve
