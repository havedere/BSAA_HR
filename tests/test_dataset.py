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


def test_sampler_weights_equal_per_tag_without_synthetic():
    from bpm.dataset import sampler_weights

    tags = np.array(["a"] * 3 + ["b"] * 1)
    w = sampler_weights(tags)

    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3, 1.0])


def test_sampler_caps_synthetic_share_per_tag():
    from bpm.dataset import sampler_weights

    tags = np.array(["a"] * 10 + ["b"] * 4)
    synth = np.array([False] * 5 + [True] * 5 + [False] * 3 + [True] * 1)
    w = sampler_weights(tags, synth, max_synth_share=0.2)

    a, b = tags == "a", tags == "b"
    assert np.isclose(w[a].sum(), 1.0) and np.isclose(w[b].sum(), 1.0)
    assert np.isclose(w[a & synth].sum(), 0.2)  # natural 50%, capped to 20%
    assert np.isclose(w[b & synth].sum(), 0.2)  # natural 25%, capped to 20%
