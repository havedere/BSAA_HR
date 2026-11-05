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


def test_adjacent_and_overlapping_bad_spans_nan_inclusive_bounds():
    fs = 100.0
    ecg = np.zeros(2000)  # 20s
    # RR=1s, then two consecutive 3s gaps (HR=20, both bad, sharing t=7),
    # then a 0.25s RR (ratio + range bad) spanning 11..11.25, then RR=1s
    peak_times = [1, 2, 3, 4, 7, 10, 10.25, 11.25, 12.25, 13.25, 14.25]

    curve = ecg_to_hr_curve(ecg, fs, r_peak_detector=_fake_detector(peak_times))

    assert curve[2] == 60.0 and curve[3] == 60.0
    # spans [4,7] and [7,10] are inclusive on both ends
    for t in range(4, 11):
        assert math.isnan(curve[t])
    # spans [10,10.25] (HR=240) and [10.25,11.25] (ratio=4) cover t=11
    assert math.isnan(curve[11])
    # [11.25,12.25] follows a 1s RR (ratio 1) -> good again
    assert curve[12] == 60.0
    assert curve[13] == 60.0
    assert curve[14] == 60.0


def _sim_ecg(duration_s, bpm, fs=100):
    import neurokit2 as nk

    return nk.ecg_simulate(duration=duration_s, sampling_rate=fs, heart_rate=bpm, random_state=0)


def test_quality_mask_keeps_clean_fast_ecg_either_polarity():
    from bpm.ground_truth import ecg_quality_mask

    ecg = _sim_ecg(60, 150)
    for sign in (1, -1):
        mask = ecg_quality_mask(sign * ecg, 100.0)
        assert len(mask) == 60
        assert mask[10:50].mean() == 0.0


def test_quality_mask_flags_ramp_and_flatline_segments():
    from bpm.ground_truth import ecg_quality_mask

    ecg = _sim_ecg(90, 70)
    ecg[3000:6000] = np.linspace(-1.0, 1.0, 3000)  # lead-off drift, s30-60
    mask = ecg_quality_mask(ecg, 100.0)
    assert mask[32:58].all()
    assert not mask[5:25].any()
    assert not mask[65:85].any()

    assert ecg_quality_mask(np.zeros(3000), 100.0).all()
