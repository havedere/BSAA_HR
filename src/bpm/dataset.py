import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, WeightedRandomSampler

_NOISE_STD = 0.05


def zscore_window(window):
    """Z-score each channel over time (last axis); works on one window or a batch."""
    mean = window.mean(axis=-1, keepdims=True)
    std = window.std(axis=-1, keepdims=True)
    nonzero = std > 0
    # zero-std channels (flat-lined signal) pass through unchanged instead of NaN
    return np.where(nonzero, (window - mean) / np.where(nonzero, std, 1), window)


def _augment(window, rng):
    return window + rng.normal(0, _NOISE_STD, size=window.shape)


class WindowDataset(Dataset):
    def __init__(self, windows, labels, augment=False, seed=42, weights=None):
        # z-score once here; per-item z-scoring is too slow with num_workers=0
        self.windows = zscore_window(windows)
        self.labels = labels
        self.weights = weights
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.augment:
            window = _augment(window, self._rng)
        item = (
            torch.from_numpy(window).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )
        if self.weights is None:
            return item
        return item + (torch.tensor(self.weights[idx], dtype=torch.float32),)


def lds_weights(labels, power, sample_weights, sigma=2.0, clip=20.0):
    """LDS loss weights (Yang et al. 2021): inverse Gaussian-smoothed label density
    to the power p, sample-weighted mean 1, clipped at `clip`."""
    s = np.asarray(sample_weights, float)
    # 1-BPM bins 30..220; labels outside fall in the edge bins
    bins = np.clip(np.rint(np.asarray(labels, float)).astype(int), 30, 220) - 30
    density = gaussian_filter1d(np.bincount(bins, weights=s, minlength=191), sigma, mode="nearest")
    w = density[bins] ** -power
    w /= np.sum(w * s) / np.sum(s)
    # one clip+renorm can push the max back over clip, so repeat until it holds
    while w.max() > clip * (1 + 1e-12):
        w = np.minimum(w, clip)
        w /= np.sum(w * s) / np.sum(s)
    return w


def sampler_weights(dataset_tags, is_synthetic=None, max_synth_share=0.2):
    """Each tag gets total weight 1. Synthetic windows get at most
    max_synth_share of their tag's weight, real windows the rest."""
    tags = np.asarray(dataset_tags)
    synth = np.zeros(len(tags), bool) if is_synthetic is None else np.asarray(is_synthetic, bool)
    weights = np.empty(len(tags))
    for tag in np.unique(tags):
        in_tag = tags == tag
        n_synth, n_total = (in_tag & synth).sum(), in_tag.sum()
        share = min(max_synth_share, n_synth / n_total)
        weights[in_tag & ~synth] = (1 - share) / (n_total - n_synth)
        if n_synth:
            weights[in_tag & synth] = share / n_synth
    return weights


def make_weighted_sampler(dataset_tags, is_synthetic=None, max_synth_share=0.2):
    weights = sampler_weights(dataset_tags, is_synthetic, max_synth_share)
    # epoch length stays the real-window count, so bootstrap runs compare to best.pt
    n_real = len(weights) if is_synthetic is None else int((~np.asarray(is_synthetic, bool)).sum())
    return WeightedRandomSampler(weights, num_samples=n_real, replacement=True)
