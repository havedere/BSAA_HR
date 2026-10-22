# CNN-BiLSTM BPM Estimation — Plan

Status: locked, no implementation yet.

## Goal

Estimate heart rate (BPM) from PPG using a CNN-BiLSTM, trained against ECG-derived ground truth, across all three sub-datasets in `harmonized_ecg_ppg_databases/` (see `CLAUDE.md` for dataset details).

## 1. Ground truth generation

- R-peak detection via `neurokit2`, `ecg_peaks(method="pantompkins1985")`, on the ECG channel of every file.
- Compute instantaneous HR from R-R intervals, resample to a fixed-rate HR curve (1 Hz) aligned with the PPG timeline.
- QC: drop a window if any RR interval implies HR outside 30-220 BPM, or consecutive-RR ratio exceeds 1.2x (ectopic/misdetection heuristic). Exercise-set ECG will be noisiest under motion.

## 2. Windowing

- Fixed window length 10s at 100 Hz (1000 samples), sliding stride 2s.
- Label per window = mean HR over the window (from step 1).
- Keep subject ID per window for the split.

## 3. Input channels (ACC mismatch)

- Only `wrist-ppg-during-exercise-1.0.0` has `ACC_X/Y/Z`; `multisite-ppg` and `wrist-finger-ppg` don't.
- Decision: include ACC as model input (needed for exercise data — without it, motion artifact makes that sub-dataset near-unworkable).
- Handling: feed PPG + ACC_X/Y/Z as input channels; for the two ACC-less sub-datasets, zero-fill the ACC channels and add a binary "has_motion" flag channel. One model, not two variants.
- Gated/separate ACC-encoder branch considered and rejected for now (zero-fill is simpler, ships now); revisit only if exercise-set error is clearly ACC-starved.

## 4. Preprocessing

- Bandpass filter PPG only (~0.5-8 Hz HR band), 4th-order Butterworth, zero-phase (`scipy` `sosfiltfilt`), applied once on the full continuous per-file signal before windowing. ECG is NOT filtered this way — it's ground-truth-only (never a model input, see §3), and `neurokit2`'s R-peak pipeline applies its own appropriate internal cleaning (~5-15 Hz, QRS-preserving); squashing ECG to the 0.5-8 Hz HR band first would attenuate the QRS complex and hurt R-peak accuracy.
- Per-window z-score normalization (per-channel), skipped whenever a channel's window std is 0 (covers zero-filled ACC channels and the `has_motion` flag — avoids divide-by-zero NaN).
- Z-score applied at `Dataset.__getitem__` time, not baked into the cached `.npz` — cache holds filtered-but-unnormalized windows so changing normalization later doesn't require re-running R-peak detection/filtering/windowing.

### Augmentation (train split only, applied after z-score)

- Gaussian noise, std=0.05 (z-scored units): on `PPG` always; on `ACC_X/Y/Z` only when `has_motion=1` (real ACC, exercise set only). Never applied to the zero-filled ACC sentinel channels or `has_motion` itself — those are sentinels, not signal.
- ACC channel dropout: independently per channel (X/Y/Z), p=0.2, zero out one at a time. Exercise-set only (`has_motion=1`); no-op elsewhere since those channels are already 0.
- Time-jitter considered and rejected: the 2s-stride/10s-window scheme (§2) already gives 80% overlap between adjacent cached windows, i.e. near-continuous time-shifted views of the same signal — explicit jitter would be redundant and would force reworking the windowing/cache (pad+crop) for little extra benefit.
- Val/test: no augmentation.
- No resampling needed — all three sub-datasets confirmed 100 Hz.
- EDF signal reading via a small numpy plain-EDF reader (originally `pyedflib`, see pipeline-decisions.md §1).
- Windows precomputed to disk, one `.npz` per subject (windows + labels + has_motion flag) — not windowed on-the-fly in the training `Dataset`, to avoid redoing R-peak detection/filtering every epoch.

## 5. Split

- Subject-wise: no subject's windows cross train/val/test.
- Per sub-dataset, split its own subjects 70/15/15 independently, then union the three splits. Guarantees proportional sub-dataset representation without a stratification library.
- `multisite-ppg` capped at 2,000 windows/subject (random sample if over), so its ~18hr files don't dominate window count vs the other two sub-datasets.

## 6. Model: CNN-BiLSTM

- CNN front end: Conv1D x3 (filters 32→64→128, kernel 5, stride 1, padding `'same'`, each + BatchNorm + ReLU + MaxPool2) over the multi-channel window. 1000→125 timesteps after 3 pools.
- BiLSTM(hidden=64, 1 layer) on top of the CNN feature sequence → temporal context both directions. Output mean-pooled across all 125 timesteps (matches "mean HR over window" label semantics), not last-timestep-only.
- Dense(32) → Dense(1) head, dropout 0.3 before the final dense. Single BPM regression output per window.
- Loss: Huber (delta=5 BPM) — robust to ECG-QC noise/outliers, smoother gradient near zero than MAE.
- Framework: PyTorch. Training: Adam lr=1e-3 + `ReduceLROnPlateau` (monitor val MAE, factor=0.5, patience=5, min_lr=1e-5), batch=64, up to 100 epochs, early stopping on val MAE (patience=10). Runs on CUDA when available (host RTX 3080), CPU fallback.
- Checkpointing: save weights whenever val MAE improves (`best.pt`); final evaluation loads `best.pt`, not the loop's last epoch (early-stop patience runs 10 epochs past the best one).
- Train batches drawn via `WeightedRandomSampler`, weighted so each sub-dataset contributes roughly equally per epoch regardless of total window count — otherwise the sub-dataset with the most windows (`multisite-ppg`, capped but still largest) would dominate training.

## 7. Evaluation

- MAE, RMSE in BPM, broken down per sub-dataset (exercise vs multisite vs wrist-finger — difficulty differs a lot). Computed per-subject first, then macro-averaged across subjects within each sub-dataset — avoids the number being skewed by whichever test subject happens to have the most windows (same reasoning as the training sampler, §6).
- Secondary combined metric: equal-weighted average of the three sub-dataset MAEs. Reported alongside, never replacing, the per-sub-dataset breakdown.
- Bland-Altman agreement plot, overall and per sub-dataset: ECG-derived vs predicted, standard in HR-estimation literature.

## 8. Known risks

- `multisite-ppg` files run up to ~18hr — capped at 2,000 windows/subject (see §5) to avoid dataset imbalance.
- Exercise-set ECG under heavy motion may give unreliable R-peaks — ground truth itself is noisiest exactly where PPG needs the most help.
- Exercise sub-dataset has only 9 subjects — thin for its own subject-wise test slice. Accepted as-is, documented as a limitation; no augmentation/oversampling applied (would risk masking the real per-sub-dataset difficulty signal).

## Dependencies

Packages (`requirements.txt`, host `.venv`, default PyPI torch with bundled CUDA): `neurokit2`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `torch`.

## Decisions made so far

- Combine all three sub-datasets (not train on one alone).
- Use ACC channels, with zero-fill + flag for datasets missing them.
- Subject-wise split, not random-window split.
- PyTorch over TensorFlow/Keras.
- numpy plain-EDF reader (replaced `pyedflib`) for EDF reading, `neurokit2` (Pan-Tompkins) for R-peak detection.
- Precompute windows to disk rather than windowing on-the-fly.
