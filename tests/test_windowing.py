import numpy as np

from bpm.windowing import make_windows


def test_drops_windows_touching_nan_and_labels_mean_hr():
    fs = 10.0
    window_s, stride_s = 2, 1
    # 10s of data, one channel, values = sample index (so window content is checkable)
    channels = {"PPG": np.arange(100, dtype=float)}
    hr_curve = np.arange(60, 70, dtype=float)  # 10 values, seconds 0..9
    hr_curve[5] = np.nan

    windows, labels = make_windows(channels, hr_curve, fs, window_s=window_s, stride_s=stride_s)

    # 9 candidate windows (i=0..8); windows touching second 5 (i=4,5) dropped
    assert windows.shape == (7, 1, 20)
    assert labels.shape == (7,)

    # first kept window: i=0, samples [0,20), label = mean(hr_curve[0:2])
    np.testing.assert_allclose(windows[0, 0], np.arange(0, 20, dtype=float))
    assert labels[0] == np.mean(hr_curve[0:2])

    # last kept window: i=8, samples [80,100), label = mean(hr_curve[8:10])
    np.testing.assert_allclose(windows[-1, 0], np.arange(80, 100, dtype=float))
    assert labels[-1] == np.mean(hr_curve[8:10])


def test_drops_trailing_partial_window():
    fs = 10.0
    channels = {"PPG": np.arange(25, dtype=float)}  # 2.5s, not a clean multiple
    hr_curve = np.array([60.0, 61.0])  # only 2 full seconds available

    windows, labels = make_windows(channels, hr_curve, fs, window_s=2, stride_s=1)

    # only one full 2s window fits: i=0, samples[0,20)
    assert windows.shape == (1, 1, 20)
    assert labels[0] == np.mean(hr_curve[0:2])
