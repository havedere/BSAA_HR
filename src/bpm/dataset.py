import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .channels import CHANNEL_ORDER

_NOISE_STD = 0.05
_ACC_DROPOUT_P = 0.2
_PPG = CHANNEL_ORDER.index("PPG")
_ACC_X, _ACC_Y, _ACC_Z = (CHANNEL_ORDER.index(c) for c in ("ACC_X", "ACC_Y", "ACC_Z"))


def zscore_window(window):
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    normed = window.copy()
    nonzero = std[:, 0] > 0
    normed[nonzero] = (window[nonzero] - mean[nonzero]) / std[nonzero]
    return normed


def _augment(window, has_motion, rng):
    window = window.copy()
    window[_PPG] += rng.normal(0, _NOISE_STD, size=window.shape[1])
    if has_motion:
        for acc_ch in (_ACC_X, _ACC_Y, _ACC_Z):
            if rng.random() < _ACC_DROPOUT_P:
                window[acc_ch] = 0.0
            else:
                window[acc_ch] += rng.normal(0, _NOISE_STD, size=window.shape[1])
    return window


class WindowDataset(Dataset):
    def __init__(self, windows, labels, has_motion, augment=False, seed=42):
        self.windows = windows
        self.labels = labels
        self.has_motion = has_motion
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        window = zscore_window(self.windows[idx])
        if self.augment:
            window = _augment(window, self.has_motion[idx], self._rng)
        return (
            torch.from_numpy(window).float(),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def make_weighted_sampler(dataset_tags):
    tags = np.asarray(dataset_tags)
    counts = {tag: (tags == tag).sum() for tag in np.unique(tags)}
    weights = np.array([1.0 / counts[tag] for tag in tags])
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
