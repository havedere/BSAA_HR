import math

import numpy as np

from bpm.ground_truth import ecg_to_hr_curve


def _fake_detector(peak_times_sec):
    def detector(ecg, fs):
        return np.round(np.array(peak_times_sec) * fs).astype(int)

    return detector


def test_constant_rr_interpolates_flat_and_nans_before_first_anchor():
    fs = 100.0
    ecg = np.zeros(1000)  # 10s
    peak_times = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # RR=1s -> HR=60 throughout

    curve = ecg_to_hr_curve(ecg, fs, r_peak_detector=_fake_detector(peak_times))

    assert len(curve) == 10
    # first peak (t=1) has no RR interval before it, so the curve starts at t=2
    assert math.isnan(curve[0])
    assert math.isnan(curve[1])
    for t in range(2, 10):
        assert curve[t] == 60.0


def test_bad_rr_interval_nans_its_time_span():
    fs = 100.0
    ecg = np.zeros(1000)  # 10s
    # RR=1s except one 4s gap (HR=15, out of range) between t=4 and t=8
    peak_times = [1, 2, 3, 4, 8, 9]

    curve = ecg_to_hr_curve(ecg, fs, r_peak_detector=_fake_detector(peak_times))

    assert math.isnan(curve[0])
    assert math.isnan(curve[1])
    assert curve[2] == 60.0
    assert curve[3] == 60.0
    assert math.isnan(curve[5])
    assert math.isnan(curve[6])
    assert math.isnan(curve[7])
    assert math.isnan(curve[9])
