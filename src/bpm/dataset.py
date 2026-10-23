import numpy as np
import torch
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
    def __init__(self, windows, labels, augment=False, seed=42):
        # z-score once here; per-item z-scoring is too slow with num_workers=0
        self.windows = zscore_window(windows)
        self.labels = labels
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.augment:
            window = _augment(window, self._rng)
        return (
            torch.from_numpy(window).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def make_weighted_sampler(dataset_tags):
    tags = np.asarray(dataset_tags)
    counts = {tag: (tags == tag).sum() for tag in np.unique(tags)}
    weights = np.array([1.0 / counts[tag] for tag in tags])
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
