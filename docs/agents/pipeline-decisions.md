# Pipeline Decisions — Step-by-Step Log

Companion to `plan.md`. `plan.md` holds the locked high-level plan; this file is the detailed, append-only decision log per pipeline stage, filled in as we walk through implementation. Each stage: decisions made + why, open items (none left pending once a stage is closed).

Status: pipeline walkthrough complete (stages 0-10) + stage 11 (augmentation) added afterward. No open items.

---

## 0. File discovery & indexing — CLOSED

- Fixed a stale fact in `CLAUDE.md`: `wrist-ppg-during-exercise-1.0.0` has 8 subjects (`s1`-`s6`, `s8`, `s9`), not 9 — `s7` has no file. Caught by listing actual filenames.
- Global subject IDs are namespaced by sub-dataset tag: `exercise_s1`, `wristfinger_01`, `multisite_P10`. Raw ids collide in format across sub-datasets (`s1`/`01`/`P1`), and the subject-wise split (`plan.md` §5) needs one unambiguous ID space.
- Build `manifest.csv` once via a script (columns: global_subject_id, sub_dataset, raw_subject_id, file_path, activity/site/trial, duration_sec, n_samples), regenerated only if raw data changes. Single source of truth for every later stage (ground truth, windowing, split, cap) instead of re-discovering files inline each run.

## 1. EDF reading — CLOSED

- Checked actual EDF headers (one file/sub-dataset): channel labels are clean exact strings (`ECG`, `PPG`, `ACC_X/Y/Z`), but physical units differ by sub-dataset — ECG `mV` (exercise) / `uV` (multisite) / unitless (wrist-finger); PPG `mV` (exercise) / unitless (other two).
- No unit conversion: rely on the already-planned per-window z-score (`plan.md` §4) to wash out scale differences. Converting unitless channels to mV/µV would need fabricated calibration.
- Read via `pyedflib` physical-value signals, channels matched by exact label string. Fail loudly (raise) if an expected channel is missing from a file — catches a malformed file immediately instead of producing a wrong-shaped window downstream.

## 2. Ground truth generation — CLOSED

- R-peak → 1Hz HR curve: linear interpolation between consecutive R-peak instantaneous-HR values. Step/hold rejected — creates artificial jumps biasing the window-mean label toward whichever beat lands first.
- Edge regions (before first / after last R-peak in a recording): marked NaN on the HR curve, not extrapolated. No ECG support there to justify a fabricated value; regions are seconds-wide, negligible loss.
- QC propagation: a flagged bad RR interval (`plan.md` §1 rule) NaNs the HR curve over that interval's time span — same mechanism as edge gaps. One rule downstream ("window touches any NaN → drop") instead of two separate QC paths; never silently interpolates across a flagged bad beat.

## 3. Windowing — CLOSED

- Bandpass filter (`plan.md` §4) runs once on the full continuous per-file signal, before cutting into windows — not per-window. Avoids redundant recomputation across 80%-overlapping windows and edge-effect distortion from filtering a short (1000-sample) input.
- Trailing partial window (file length not an exact multiple of stride) is dropped, not zero-padded. Zero-padding would feed a partially-fake signal under a label still averaged from real HR.
- Random cap sampling (`multisite-ppg`, 2,000 windows/subject) uses a fixed global seed (42), documented in the manifest/config, so the exact window subset is reproducible across runs.

## 4. ACC channel assembly — CLOSED

- Caught a bug risk: per-window z-score (`plan.md` §4) on a constant-0 channel (zero-filled ACC when absent) divides by `std=0` → NaN. Fix: skip z-score whenever a channel's window std is 0 (covers zero-filled ACC and the `has_motion` flag generally, no hardcoded channel list).
- `has_motion` is a constant-value channel broadcast across the full window length (`1000` samples, all `1.0` or `0.0`), stacked as a 5th input channel — same tensor shape as the signal channels, no special-cased forward pass.
- Model input = 5 channels: `PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`, `has_motion`. `ECG` is ground-truth-only, never fed to the model (would make prediction trivial/circular).

## 5. Preprocessing (caching) — CLOSED

- Caught a plan error: `plan.md` §4 said to bandpass-filter ECG too. Dropped — ECG is ground-truth-only (never a model input, stage 4), and `neurokit2`'s R-peak pipeline has its own appropriate internal cleaning (~5-15 Hz, QRS-preserving); a 0.5-8 Hz HR-band filter on ECG first would attenuate the QRS complex and hurt R-peak accuracy. Only PPG gets the 0.5-8Hz bandpass.
- PPG filter: 4th-order Butterworth, zero-phase (`scipy.signal.sosfiltfilt`), applied once per whole file before windowing (consistent with stage 3's filter-before-window decision).
- Cache stores filtered-but-not-z-scored windows + labels + `has_motion` + subject metadata per `.npz`. Z-score applied at `Dataset.__getitem__` time — keeps the expensive steps (R-peak detection, filtering, windowing) out of the loop if normalization strategy changes later.
- `plan.md` §4 updated to reflect this.

## 6. Split — CLOSED

- Minimum-1-subject guarantee per split per sub-dataset: strict 70/15/15 rounding could zero out val or test for the 8-subject exercise set. Shrink train if needed to keep val/test ≥1. Exercise set: train=6, val=1, test=1.
- One global fixed seed (42) for all pipeline randomness (cap sampling, subject-shuffle-before-split) — documented once, not tracked per-source.

## 7. Dataset/DataLoader — CLOSED

- Total dataset size ~1.3GB (multisite capped ~800MB + wrist-finger ~430MB + exercise ~80MB, windows `(5,1000)` float32 ≈20KB each) — small enough to load fully.
- Each split loads its per-subject `.npz` files fully into RAM at `Dataset.__init__` (concatenated arrays), not memory-mapped/lazy. CPU-only training gets no benefit from memory-mapping; avoids disk I/O bottleneck once z-score is the only per-batch work.
- `DataLoader(num_workers=0)` — no worker processes. Data's already in RAM, per-sample work (z-score) is trivial, nothing to overlap.

## 8. Model architecture — CLOSED

- Conv1D padding: `'same'` on all 3 conv blocks — only `MaxPool(2)`×3 shrinks the sequence (1000→125 timesteps into the BiLSTM). Clean arithmetic, no edge-sample loss.
- BiLSTM(hidden=64) output collapsed to one vector via mean-pool across all 125 timesteps (not last-timestep). Matches the label's own semantics ("mean HR over the window", `plan.md` §2) — last-timestep-only would bias toward the window's end.

## 9. Training loop — CLOSED

- `ReduceLROnPlateau`: monitor val MAE, factor=0.5, patience=5, min_lr=1e-5. Patience is half early-stop's (10), so LR drops once before the run gives up.
- Checkpointing: save `best.pt` whenever val MAE improves; final evaluation loads `best.pt`, not the loop's last epoch — early-stop patience=10 means training runs 10 epochs past the best one.
- Caught a real imbalance risk: per-subject cap (stage 3) only bounds `multisite-ppg`'s extreme per-file blowup, not the overall sub-dataset window-count gap (exercise: ~thousands, wrist-finger: ~20k, multisite: up to ~40k capped). Fix: train batches drawn via `WeightedRandomSampler` so each sub-dataset contributes ~equally per epoch, not proportional to its raw window count — otherwise the largest sub-dataset would quietly dominate gradients, undermining the whole point of combining all three.
- `plan.md` §6 updated with all of the above.

## 10. Evaluation — CLOSED

- Per-sub-dataset MAE/RMSE computed per-subject first, then macro-averaged across subjects within each sub-dataset — not pooled window-level — same reasoning as the training `WeightedRandomSampler` (stage 9): avoids one long-recording test subject skewing the reported number.
- Added a secondary combined metric: equal-weighted average of the three sub-dataset MAEs. Reported alongside, never replacing, the per-sub-dataset breakdown (which matters more — difficulty differs a lot per sub-dataset).
- `plan.md` §7 updated with both.

## 11. Augmentation (train split only) — CLOSED

- Light set, applied after z-score, train split only: Gaussian noise (std=0.05, z-scored units) on `PPG` always and on `ACC_X/Y/Z` only when `has_motion=1`; ACC channel dropout (p=0.2 per channel, exercise-set only).
- Never touches the zero-filled ACC sentinel channels or `has_motion` itself — those encode "no data here", and noising/dropping them would corrupt that meaning.
- Time-jitter rejected: redundant given the existing 2s-stride/10s-window 80% overlap (already near-continuous time-shifted views in the cache); would also force reworking the windowing/cache (pad+crop) for little benefit.
- Heavier augmentation (warping, cutout) deferred until baseline numbers exist to compare against.
- `plan.md` §4 updated with a new "Augmentation" subsection.
