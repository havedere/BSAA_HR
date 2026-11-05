import numpy as np

from bpm.preprocessing import bandpass_filter
from bpm.timewarp import plan_bootstrap, ppg_hr_agrees, source_label, warp_window
from bpm.windowing import make_windows

FS = 100.0


def _sine_ppg(seconds, hz):
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * hz * t)


def test_warp_scales_frequency_and_label():
    ppg = _sine_ppg(60, 1.0)
    hr = np.full(60, 60.0)

    window, label = warp_window(ppg, hr, FS, start_s=10, factor=1.5)

    assert window.shape == (1, 1000)
    assert np.isclose(label, 90.0)
    spectrum = np.abs(np.fft.rfft(window[0]))
    freqs = np.fft.rfftfreq(1000, 1 / FS)
    assert np.isclose(freqs[spectrum.argmax()], 1.5)


def test_factor_one_matches_make_windows():
    rng = np.random.default_rng(0)
    t = np.arange(int(60 * FS)) / FS
    raw = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t) + 0.1 * rng.normal(size=len(t))
    ppg = bandpass_filter(raw, FS)
    hr = 70 + 5 * np.sin(np.arange(60) / 7.0)

    windows, labels = make_windows({"PPG": ppg}, hr, FS, window_s=10, stride_s=2)
    window, label = warp_window(ppg, hr, FS, start_s=10, factor=1.0)

    assert label == labels[5]  # window i=5 starts at 10 s
    assert np.corrcoef(window[0], windows[5, 0])[0, 1] > 0.99


def test_ramp_hr_label():
    hr = 60 + 0.5 * np.arange(100, dtype=float)
    # source times 10 + 1.2k, k=0..9: mean time 15.4 s, HR 67.7
    assert np.isclose(source_label(hr, 10, 1.2), 1.2 * 67.7)


def test_no_out_of_band_power_after_strong_compression():
    rng = np.random.default_rng(1)
    ppg = bandpass_filter(rng.normal(size=int(60 * FS)), FS)
    hr = np.full(60, 80.0)

    window, _ = warp_window(ppg, hr, FS, start_s=10, factor=1.6)

    power = np.abs(np.fft.rfft(window[0])) ** 2
    freqs = np.fft.rfftfreq(1000, 1 / FS)
    assert power[freqs > 10].sum() / power.sum() < 0.01


def test_rejects_nan_span_and_span_past_end():
    ppg = _sine_ppg(60, 1.0)
    hr = np.full(60, 60.0)
    hr[15] = np.nan

    assert warp_window(ppg, hr, FS, start_s=10, factor=1.0) is None
    assert warp_window(ppg, np.full(60, 60.0), FS, start_s=55, factor=1.0) is None


def test_plan_respects_bins_and_reuse_cap():
    hr_curves = [np.full(40, 80.0)]  # few valid starts, can't fill the quota
    quotas = {(120, 130): 1000, (40, 50): 0}

    picks, stats = plan_bootstrap(hr_curves, quotas, np.random.default_rng(0), max_reuse=5)

    got, quota, unique = stats[(120, 130)]
    assert got == len(picks) < quota
    assert got <= 5 * unique
    assert all(120 <= label < 130 for _, _, _, label in picks)
    starts = [s for _, s, _, _ in picks]
    assert max(starts.count(s) for s in set(starts)) <= 5
    assert stats[(40, 50)][0] == 0


def test_ppg_hr_agrees_per_start_second():
    ppg = bandpass_filter(_sine_ppg(60, 1.0), FS)
    hr = np.full(60, 60.0)
    hr[40] = np.nan

    ok = ppg_hr_agrees(ppg, hr, FS)

    assert len(ok) == 51  # one per integer start with a full 10 s window
    assert ok[:31].all()  # windows not touching the NaN second
    assert not ok[31:41].any()
    assert not ppg_hr_agrees(ppg, np.full(60, 120.0), FS).any()  # label 2x PPG rate


def test_plan_skips_sources_masked_out():
    hr_curves = [np.full(40, 80.0), np.full(40, 80.0)]
    source_ok = [np.zeros(31, bool), np.ones(31, bool)]

    picks, _ = plan_bootstrap(hr_curves, {(120, 130): 50}, np.random.default_rng(0), source_ok=source_ok)

    assert picks and all(file_idx == 1 for file_idx, _, _, _ in picks)
