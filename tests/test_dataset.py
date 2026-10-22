import numpy as np

from bpm.dataset import zscore_window


def test_zscores_each_channel_independently():
    window = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )

    normed = zscore_window(window)

    for row in normed:
        assert np.isclose(row.mean(), 0.0, atol=1e-8)
        assert np.isclose(row.std(), 1.0, atol=1e-8)


def test_skips_zscore_for_zero_std_channel_avoiding_nan():
    window = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],  # flat-lined channel
        ]
    )

    normed = zscore_window(window)

    assert not np.isnan(normed).any()
    np.testing.assert_array_equal(normed[1], window[1])


def test_zscores_batch_same_as_each_window():
    rng = np.random.default_rng(0)
    batch = rng.normal(5.0, 3.0, size=(4, 3, 50))
    batch[:, 2] = 1.0  # flat-lined channel

    np.testing.assert_allclose(
        zscore_window(batch), np.stack([zscore_window(w) for w in batch])
    )
