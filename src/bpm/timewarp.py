"""Time-warp bootstrap: synthetic low/high BPM windows from real PPG.

A factor f squeezes (f > 1) or stretches (f < 1) a source span of
window_s * f seconds into one window_s window, so the label is f times the
source HR.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .preprocessing import bandpass_filter

PAD_S = 3  # extra output seconds per side, so the post-warp filter's edge effects are cropped off


def source_label(hr_curve, start_s, factor, window_s=10, pad_s=PAD_S):
    """Label of the warped window starting at start_s, or None if unusable.

    Samples the 1 Hz curve once per output second (source times start_s + f*k),
    so f=1 at an integer start equals windowing.make_windows' mean(hr[s:s+10]).
    """
    end_s = start_s + window_s * factor
    if start_s - pad_s * factor < 0 or end_s + pad_s * factor > len(hr_curve):
        return None
    t = start_s + factor * np.arange(window_s)
    span_end = max(int(np.ceil(end_s)), int(np.ceil(t[-1])) + 1)
    if np.isnan(hr_curve[int(np.floor(start_s)) : span_end]).any():
        return None
    return float(factor * np.interp(t, np.arange(len(hr_curve)), hr_curve).mean())


def warp_window(ppg, hr_curve, fs, start_s, factor, window_s=10, pad_s=PAD_S):
    """(window (1, window_s*fs) float32, label) or None.

    `ppg` is the bandpassed full-file signal. It is filtered again after the
    warp, because warping moves the 0.5-8 Hz band (to 0.8-12.8 Hz at f=1.6) and
    the out-of-band energy would let the model spot synthetic windows.
    """
    label = source_label(hr_curve, start_s, factor, window_s, pad_s)
    if label is None:
        return None
    n_out, n_pad = int(window_s * fs), int(pad_s * fs)
    src = start_s * fs + factor * np.arange(-n_pad, n_out + n_pad)
    lo, hi = int(np.floor(src[0])), int(np.ceil(src[-1])) + 1
    if lo < 0 or hi > len(ppg):
        return None
    warped = np.interp(src - lo, np.arange(hi - lo), ppg[lo:hi])
    window = bandpass_filter(warped, fs)[n_pad : n_pad + n_out]
    return window[None].astype(np.float32), label


def plan_bootstrap(hr_curves, quotas, rng, f_min=0.7, f_max=1.6, max_reuse=5, window_s=10):
    """Pick (file_idx, start_s, factor, label) per bin from 1 Hz HR curves only.

    quotas: {(lo, hi): n}. Candidates are integer-second starts pooled over all
    curves; each start is used at most max_reuse times across all bins, and a
    bin that runs out of candidates stays short instead of being padded.
    Returns (picks, stats) with stats[(lo, hi)] = (generated, quota, unique_starts).
    """
    file_idx, starts, hr10 = [], [], []
    for i, hr in enumerate(hr_curves):
        if len(hr) < window_s:
            continue
        m = sliding_window_view(hr, window_s).mean(axis=1)
        s = np.flatnonzero(~np.isnan(m))
        file_idx.append(np.full(len(s), i))
        starts.append(s)
        hr10.append(m[s])
    file_idx = np.concatenate(file_idx) if file_idx else np.empty(0, int)
    starts = np.concatenate(starts) if starts else np.empty(0, int)
    hr10 = np.concatenate(hr10) if hr10 else np.empty(0)
    reuse = np.zeros(len(hr10), dtype=int)

    picks, stats = [], {}
    for (lo, hi), quota in sorted(quotas.items()):
        cand = np.flatnonzero((hr10 >= lo / f_max) & (hr10 <= hi / f_min))
        rng.shuffle(cand)
        got, used = 0, set()
        for _ in range(max_reuse):
            for c in cand:
                if got >= quota:
                    break
                if reuse[c] >= max_reuse:
                    continue
                # f range that maps this start's 10 s mean HR into [lo, hi)
                f = rng.uniform(max(lo / hr10[c], f_min), min(hi / hr10[c], f_max))
                label = source_label(hr_curves[file_idx[c]], starts[c], f, window_s)
                if label is None or not lo <= label < hi:
                    continue
                reuse[c] += 1
                used.add(c)
                got += 1
                picks.append((int(file_idx[c]), int(starts[c]), float(f), label))
            if got >= quota:
                break
        stats[(lo, hi)] = (got, quota, len(used))
    return picks, stats
