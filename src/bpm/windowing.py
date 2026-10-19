import numpy as np


def make_windows(channels, hr_curve, fs, window_s=10, stride_s=2):
    window_samples = int(window_s * fs)
    stride_samples = int(stride_s * fs)
    channel_names = list(channels)
    total_samples = len(channels[channel_names[0]])

    n_candidates = (total_samples - window_samples) // stride_samples + 1

    kept_windows, kept_labels = [], []
    for i in range(max(n_candidates, 0)):
        start = i * stride_samples
        end = start + window_samples
        sec_start, sec_end = i * stride_s, i * stride_s + window_s
        if sec_end > len(hr_curve):
            continue
        hr_slice = hr_curve[sec_start:sec_end]
        if np.isnan(hr_slice).any():
            continue
        window = np.stack([channels[name][start:end] for name in channel_names])
        kept_windows.append(window)
        kept_labels.append(np.mean(hr_slice))

    if not kept_windows:
        return (
            np.empty((0, len(channel_names), window_samples)),
            np.empty((0,)),
        )
    return np.stack(kept_windows), np.array(kept_labels)
