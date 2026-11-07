import numpy as np

WINDOW_S, STRIDE_S = 10, 2  # single source; cache, sequences and smoothing import these


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
            np.empty((0, len(channel_names), window_samples), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return np.stack(kept_windows).astype(np.float32), np.array(kept_labels, dtype=np.float32)


def all_windows(channels, hr_curve, fs, window_s=10, stride_s=2):
    """Every candidate window -> (windows (N, C, window_samples), idx int32, labels float32);
    start_s = stride_s * idx; label NaN where the 1 Hz HR curve is NaN or ends.
    windows is a read-only strided view of the signal (multi-day files would not fit as a copy):
    index the rows you need, then cast. The labelled subset equals make_windows."""
    window_samples, stride_samples = int(window_s * fs), int(stride_s * fs)
    x = np.stack([np.asarray(c) for c in channels.values()])
    n = max((x.shape[1] - window_samples) // stride_samples + 1, 0)
    idx = np.arange(n, dtype=np.int32)
    windows = np.lib.stride_tricks.sliding_window_view(x, window_samples, axis=1)[:, ::stride_samples][:, :n]
    curve = np.asarray(hr_curve, float)
    curve = np.pad(curve, (0, max(0, (n - 1) * stride_s + window_s - len(curve))), constant_values=np.nan)
    labels = np.lib.stride_tricks.sliding_window_view(curve, window_s)[idx * stride_s].mean(axis=1)
    return windows.transpose(1, 0, 2), idx, labels.astype(np.float32)
