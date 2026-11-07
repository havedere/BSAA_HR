# 13 predict_sequences.py (+ windowing.all_windows)

Status: done
Blocked by: 10
Spec: Step 0B; run plan Method 5.

## Scope
- `windowing.all_windows(channels, hr_curve, fs, window_s=10, stride_s=2)` -> windows, idx, labels (NaN when unlabelled); labelled subset identical to `make_windows`.
- `scripts/predict_sequences.py --fold k --ckpt a.pt [b.pt ...]`: fold val+test recordings; HR curve from `cache/hr_curves/<file>.npy` (missing -> skip + report); bandpass full PPG; keep windows within ±5 min of a labelled window; ensemble predict (head-aware, reuse evaluate helpers).
- Output `cache/sequences/<config id>_fold<k>/<file stem>.npz`: idx int32 (start_s = 2·idx), bpm float16, score float16, label float32, logits float16 (dist only), subject, dataset_source, split.
- Prints windows/s and EDF read time.

## Verify
Test: all_windows labelled subset == make_windows. Smoke on one multisite + one wrist-finger recording: labels match, start times exact.

## Notes (2026-11-07)
- `all_windows` returns the windows as a read-only strided view (index, then cast): multisite has files up to 801,765 s (P4_earring_5, 9.3 days, 0 labelled s), whose full float32 copy is 1.6 GB.
- Recordings whose HR curve has no finite second are skipped before the EDF read.
- dist `logits` = log of the ensemble-mean probs (float16); `predict(..., with_logp=True)`.
- Smoke fold 0, dist seed 1: P9_earring_11 (99 windows, 11 labelled), 09_0 (115 / 59), P9_watch_0 (12,933 / 6,259; 2.9 s wall, 31.8k windows/s); labels == `make_windows`, label of idx i == mean(curve[2i:2i+10]).
- Audit fix 2026-11-07: output dir named like `cache/preds/`: `<config>_fold<k>_seed<s>/` (one ckpt) or `ens_<config>_fold<k>/` (several), so a single-ckpt smoke can't mix into an ensemble's dir. Smoke dir renamed to `cache/sequences/dist_fold0_seed1/`.
