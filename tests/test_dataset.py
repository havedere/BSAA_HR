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


def _wmean(w, s):
    return np.sum(w * s) / np.sum(s)


def test_lds_uniform_labels_give_weight_one():
    from bpm.dataset import lds_weights

    labels = np.repeat(np.arange(30, 221), 3).astype(float)
    s = np.ones(len(labels))
    w = lds_weights(labels, 1.0, s)

    np.testing.assert_allclose(w, 1.0)


def test_lds_rare_label_gets_larger_weight():
    from bpm.dataset import lds_weights

    labels = np.array([70.0] * 50 + [150.0] * 2)
    w = lds_weights(labels, 1.0, np.ones(len(labels)))

    assert w[-1] > w[0]


def test_lds_clipped_and_mean_one_after_final_renorm():
    from bpm.dataset import lds_weights

    rng = np.random.default_rng(0)
    labels = np.concatenate([rng.normal(70, 5, 2000), [200.0, 40.0]])
    s = rng.uniform(0.5, 2.0, len(labels))
    w = lds_weights(labels, 1.0, s, clip=20.0)

    assert w.max() <= 20.0 + 1e-6
    assert np.isclose(_wmean(w, s), 1.0)


def test_lds_power_zero_gives_all_ones():
    from bpm.dataset import lds_weights

    labels = np.array([60.0, 60.0, 61.0, 180.0])
    w = lds_weights(labels, 0.0, np.array([1.0, 2.0, 3.0, 4.0]))

    np.testing.assert_allclose(w, 1.0)


def test_lds_unit_free_under_scaled_sample_weights():
    from bpm.dataset import lds_weights

    labels = np.array([60.0, 62.0, 65.0, 90.0, 140.0])
    s = np.array([1.0, 2.0, 1.0, 0.5, 1.0])

    np.testing.assert_allclose(
        lds_weights(labels, 0.5, s), lds_weights(labels, 0.5, s * 1e-4)
    )


def test_window_dataset_returns_weight_only_when_given():
    import torch

    from bpm.dataset import WindowDataset

    windows = np.random.default_rng(0).normal(size=(2, 1, 8))
    labels = np.array([60.0, 70.0])

    assert len(WindowDataset(windows, labels)[0]) == 2
    x, y, w = WindowDataset(windows, labels, weights=np.array([0.5, 2.0]))[1]
    assert w.dtype == torch.float32 and w.item() == 2.0
