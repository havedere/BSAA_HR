import neurokit2 as nk
import numpy as np

_HR_MIN, _HR_MAX = 30.0, 220.0
_RR_RATIO_LIMIT = 1.2
# bSQI (Li/Clifford): fraction of beats two detectors agree on, per window
_BSQI_WIN_S, _BSQI_TOL_S, _BSQI_MIN = 10, 0.15, 0.8
_FLAT_FRAC = 0.1


def _neurokit2_r_peak_detector(ecg, fs):
    _, info = nk.ecg_peaks(ecg, sampling_rate=fs, method="pantompkins1985")
    return np.asarray(info["ECG_R_Peaks"])


def ecg_quality_mask(ecg, fs):
    """True per 1 Hz second where the ECG can't serve as ground truth.

    Pan-Tompkins fires a steady fake beat train on lead-off ramps/plateaus and
    double-counts on noisy ECG; the RR checks in ecg_to_hr_curve pass both.
    Flags 10 s windows where Pan-Tompkins and neurokit disagree (bSQI < 0.8)
    and seconds whose cleaned ECG is near flat versus the recording median.
    """
    n = int(len(ecg) // fs)
    sec = int(fs)
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)

    ptp = np.ptp(cleaned[: n * sec].reshape(n, sec), axis=1)
    bad = ptp <= _FLAT_FRAC * np.median(ptp)

    p1 = _neurokit2_r_peak_detector(ecg, fs) / fs
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="neurokit")
    p2 = np.asarray(info["ECG_R_Peaks"]) / fs
    for a in range(0, n, _BSQI_WIN_S):
        b = a + _BSQI_WIN_S
        x = p1[(p1 >= a) & (p1 < b)]
        y = p2[(p2 >= a) & (p2 < b)]
        if len(x) < 2 or len(y) < 2:
            bad[a:b] = True
            continue
        j = np.clip(np.searchsorted(y, x), 1, len(y) - 1)
        matched = np.sum(np.minimum(np.abs(y[j] - x), np.abs(y[j - 1] - x)) <= _BSQI_TOL_S)
        if matched / (len(x) + len(y) - matched) < _BSQI_MIN:
            bad[a:b] = True
    return bad


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
