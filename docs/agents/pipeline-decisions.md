# Pipeline Decisions — Step-by-Step Log

Companion to `plan.md`. `plan.md` holds the locked high-level plan; this file is the detailed, append-only decision log per pipeline stage, filled in as we walk through implementation. Each stage: decisions made + why, open items (none left pending once a stage is closed).

Status: pipeline walkthrough complete (stages 0-10) + stage 11 (augmentation) + stage 12 (time-warp bootstrap) + stage 13 (5-fold CV, LDS) added afterward. No open items.

---

## 0. File discovery & indexing — CLOSED

- Fixed a stale fact in `CLAUDE.md`: `wrist-ppg-during-exercise-1.0.0` has 8 subjects (`s1`-`s6`, `s8`, `s9`), not 9 — `s7` has no file. Caught by listing actual filenames.
- Global subject IDs are namespaced by sub-dataset tag: `exercise_s1`, `wristfinger_01`, `multisite_P10`. Raw ids collide in format across sub-datasets (`s1`/`01`/`P1`), and the subject-wise split (`plan.md` §5) needs one unambiguous ID space.
- Build `manifest.csv` once via a script (columns: global_subject_id, sub_dataset, raw_subject_id, file_path, activity/site/trial, duration_sec, n_samples), regenerated only if raw data changes. Single source of truth for every later stage (ground truth, windowing, split, cap) instead of re-discovering files inline each run.

## 1. EDF reading — CLOSED

- Checked actual EDF headers (one file/sub-dataset): channel labels are clean exact strings (`ECG`, `PPG`, `ACC_X/Y/Z`), but physical units differ by sub-dataset — ECG `mV` (exercise) / `uV` (multisite) / unitless (wrist-finger); PPG `mV` (exercise) / unitless (other two).
- No unit conversion: rely on the already-planned per-window z-score (`plan.md` §4) to wash out scale differences. Converting unitless channels to mV/µV would need fabricated calibration.
- Read physical-value signals via a small numpy plain-EDF reader in `edf_io.py` (replaced `pyedflib`: no Python 3.14 wheel, and the host can't build it; verified identical to `pyedflib` output to ~1e-12), channels matched by exact label string. Fail loudly (raise) if an expected channel is missing from a file — catches a malformed file immediately instead of producing a wrong-shaped window downstream.

## 2. Ground truth generation — CLOSED

- R-peak → 1Hz HR curve: linear interpolation between consecutive R-peak instantaneous-HR values. Step/hold rejected — creates artificial jumps biasing the window-mean label toward whichever beat lands first.
- Edge regions (before first / after last R-peak in a recording): marked NaN on the HR curve, not extrapolated. No ECG support there to justify a fabricated value; regions are seconds-wide, negligible loss.
- QC propagation: a flagged bad RR interval (`plan.md` §1 rule) NaNs the HR curve over that interval's time span — same mechanism as edge gaps. One rule downstream ("window touches any NaN → drop") instead of two separate QC paths; never silently interpolates across a flagged bad beat.
- ECG quality gate (2026-11-04, `ground_truth.ecg_quality_mask`, applied in `cache.read_ppg_and_hr`). On multisite, Pan-Tompkins fires a steady fake ~180 BPM beat train on lead-off ramps/plateaus and double-counts on noisy ECG. The RR checks pass both because the fake RR is regular. Gate: NaN a 10 s window when bSQI (Pan-Tompkins vs neurokit beats matched within 150 ms) is below 0.8, and NaN seconds whose cleaned-ECG peak-to-peak is at most 0.1x the recording median. Wrist-finger keeps 99.7% of valid seconds. Rejected: `ecg_quality(averageQRS)`, which is min-max rescaled per recording, so ramps scored higher than clean ECG. Also rejected: `templatematch` at 0.3, which dropped 71% of windows. Most remaining multisite PPG/label disagreement comes from motion-corrupted PPG with a correct ECG label, not from bad labels (checked on raw ECG; `.scratch/ecg-label-doubling/spec.md`).

## 3. Windowing — CLOSED

- Bandpass filter (`plan.md` §4) runs once on the full continuous per-file signal, before cutting into windows — not per-window. Avoids redundant recomputation across 80%-overlapping windows and edge-effect distortion from filtering a short (1000-sample) input.
- Trailing partial window (file length not an exact multiple of stride) is dropped, not zero-padded. Zero-padding would feed a partially-fake signal under a label still averaged from real HR.
- Random cap sampling (`multisite-ppg`, 2,000 windows/subject) uses a fixed global seed (42), documented in the manifest/config, so the exact window subset is reproducible across runs.

## 4. ACC channel assembly — CLOSED

> **Superseded (2026-10-22):** the exercise sub-dataset is excluded from the project and the model input is PPG only — no ACC channels, no `has_motion` flag. See `README.md`. Text below kept as the original decision record.

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
- Each split loads its per-subject `.npz` files fully into RAM at `Dataset.__init__` (concatenated arrays), not memory-mapped/lazy. Avoids a disk I/O bottleneck. Z-scoring is done once, vectorized, at `__init__`, so per-item work is just augmentation.
- `DataLoader(num_workers=0)` — no worker processes. Data's already in RAM, per-sample work (augmentation only) is trivial, nothing to overlap.

## 8. Model architecture — CLOSED

- Conv1D padding: `'same'` on all 3 conv blocks — only `MaxPool(2)`×3 shrinks the sequence (1000→125 timesteps into the BiLSTM). Clean arithmetic, no edge-sample loss.
- BiLSTM(hidden=64) output collapsed to one vector via mean-pool across all 125 timesteps (not last-timestep). Matches the label's own semantics ("mean HR over the window", `plan.md` §2) — last-timestep-only would bias toward the window's end.

## 9. Training loop — CLOSED

> **Superseded in part (2026-11-06):** early stopping and the LR scheduler now run on window-pooled balanced val MAE, and early-stop patience is 20 (scheduler patience stays 5). See stage 13.

- `ReduceLROnPlateau`: monitor val MAE, factor=0.5, patience=5, min_lr=1e-5. Patience is half early-stop's (10), so LR drops once before the run gives up.
- Loss is switchable for experiments: `train.py --loss huber|mae|mse --delta <BPM>` (default Huber, delta=5). Model selection stays on val MAE for every loss. Only the default run writes `best.pt`; others write `best_<tag>.pt` (tag `mae` / `mse` / `huber<delta>`, plus `_boot<letters>`), and `evaluate.py --ckpt <path>` scores them into `cache/plots/<ckpt stem>/`, so the baseline is never overwritten.
- Checkpointing: save `best.pt` whenever val MAE improves; final evaluation loads `best.pt`, not the loop's last epoch — early-stop patience=10 means training runs 10 epochs past the best one.
- Caught a real imbalance risk: per-subject cap (stage 3) only bounds `multisite-ppg`'s extreme per-file blowup, not the overall sub-dataset window-count gap (exercise: ~thousands, wrist-finger: ~20k, multisite: up to ~40k capped). Fix: train batches drawn via `WeightedRandomSampler` so each sub-dataset contributes ~equally per epoch, not proportional to its raw window count — otherwise the largest sub-dataset would quietly dominate gradients, undermining the whole point of combining all three.
- `plan.md` §6 updated with all of the above.

## 10. Evaluation — CLOSED

- Per-sub-dataset MAE/RMSE computed per-subject first, then macro-averaged across subjects within each sub-dataset — not pooled window-level — same reasoning as the training `WeightedRandomSampler` (stage 9): avoids one long-recording test subject skewing the reported number.
- Added a secondary combined metric: equal-weighted average of the three sub-dataset MAEs. Reported alongside, never replacing, the per-sub-dataset breakdown (which matters more — difficulty differs a lot per sub-dataset).
- `plan.md` §7 updated with both.

## 11. Augmentation (train split only) — CLOSED

> **Superseded (2026-10-22):** PPG-only input now, so augmentation is Gaussian noise on PPG only; ACC noise/dropout removed with the exercise sub-dataset (see `README.md`).

- Light set, applied after z-score, train split only: Gaussian noise (std=0.05, z-scored units) on `PPG` always and on `ACC_X/Y/Z` only when `has_motion=1`; ACC channel dropout (p=0.2 per channel, exercise-set only).
- Never touches the zero-filled ACC sentinel channels or `has_motion` itself — those encode "no data here", and noising/dropping them would corrupt that meaning.
- Time-jitter rejected: redundant given the existing 2s-stride/10s-window 80% overlap (already near-continuous time-shifted views in the cache); would also force reworking the windowing/cache (pad+crop) for little benefit.
- Heavier augmentation (warping, cutout) deferred until baseline numbers exist to compare against.
- `plan.md` §4 updated with a new "Augmentation" subsection.

## 12. Time-warp bootstrap (train split only, opt-in) — CLOSED

- Problem: train labels sit at 60-100 BPM. Multisite train has 161 windows below 50 BPM and only 254 windows at 130+ BPM, while multisite test reaches 191 BPM. Wrist-finger train covers 56-113 BPM only.
- Fix: synthetic windows by time-warping real PPG. A factor f in [0.7, 1.6] maps a 10·f s source span onto one 10 s window; label = f × source HR, sampled once per output second (f=1 at an integer start equals the `make_windows` label exactly). Opt-in via `train.py --bootstrap H|L|HL`.
- Ranges: `L` = bins 40-60, both sub-datasets. `H` = bins 110-190, multisite only (wrist-finger test stays below 98 BPM; its sources top out at 113, so its high bins would be near-duplicates).
- Amount: each 10-BPM bin topped up to 500 windows per sub-dataset. Each source start is reused at most 5 times across all bins; short bins are reported, not padded.
- Sampler: synthetic windows get at most 20% of their sub-dataset's weight, and the epoch length stays at the real-window count, so runs compare fairly with `best.pt`. Without the cap, top-up made wrist-finger about 50% synthetic.
- Re-filter after warp (pad 3 s per side, bandpass, crop the centre): warping moves the 0.5-8 Hz band (to 0.8-12.8 Hz at f=1.6), and the model could spot synthetic high-BPM windows by out-of-band energy alone.
- Built from raw EDF, not the window cache: f=1.6 needs 16 s of source plus padding, and capped multisite windows have no neighbours. HR curves are cached in `cache/hr_curves/`.
- Train subjects only, so no val/test leakage. Early stopping stays on overall val MAE; val MAE at 110+ BPM is logged for information, and `evaluate.py` prints MAE per 10-BPM bin.
- Caveat: a uniform warp also scales systole and pulse shape, while real high HR mostly shortens diastole. The synthetic tail is an approximation; judge it by per-bin test MAE against `best.pt`.
- PPG source filter for `H` (2026-11-05): `H` only warps source starts where `timewarp.ppg_hr_agrees` holds (spectral PPG peak in 0.6-3.5 Hz within 10% of the 10 s ECG label). Multisite PPG is often motion-corrupted under a correct label, and warping such a source teaches high BPM on a window with no matching pulse. With the filter, `H` agreement within 10 BPM went from ~25% to 69-89% per bin. `L` is unfiltered (already ~95%). Real windows are not filtered: noisy PPG with a correct label stays as a realistic hard example.

## 13. Subject-wise 5-fold CV + LDS (2026-11-06) — CLOSED

Method definitions, selection rules and statistics: `.scratch/model-run-plan/spec.md` (Metrics, Method 1, Method 2, Selection).

- Why CV: the fixed split holds only 3 multisite subjects each in val and test, val has no window >= 140 BPM while test has 88, and 6 MSE seeds spread wider (test MAE 6.08 ± 0.79) than the whole 20-config table. 5 folds put every subject in test once.
- Folds: per sub-dataset, subjects sorted by (−score, subject_id), cut into strata of 5, shuffled within each stratum (`np.random.default_rng(42)`, one per sub-dataset), dealt to groups 0-4 (`split.assign_groups`). Score = share of the subject's windows >= 130 BPM (multisite) or >= 100 BPM (wrist-finger, which never reaches 130). Spreads the few high-BPM subjects across groups. Multisite 4 per group; wrist-finger 6/6/5/5/5.
- Rotating val: fold k test = group k, val = group (k+1) mod 5, train = the other 3 (`split.fold_split`). Every subject is val once and test once.
- Folds computed once by `scripts/make_folds.py` into `cache/folds.json` ({subject: {group, score}}), read by `train.py`, `build_bootstrap.py`, `evaluate.py` and `aggregate_runs.py`. Rebuilding `cache/windows/` does not reshuffle. `split.load_folds` returns `folds_hash` = sha256 of the file bytes; every run sidecar stores it, and `aggregate_runs.py` warns about runs with another hash.
- The fixed split (`subject_wise_split`) stays the default without `--fold`.
- Fitted quantities use the fold's train subjects only: bootstrap quotas and windows (`build_bootstrap.py --fold k` -> `cache/bootstrap/fold<k>/`), LDS density. Under `--fold`, wrist-finger also gets H bins [110,130): its labels reach 113 BPM, and the old exclusion came from the fixed test set.
- Early stopping + LR scheduler: window-pooled balanced val MAE (mean of the two sub-datasets' window-pooled MAEs), for every run, fold or not. Pooled val MAE would let the sub-dataset with more val windows decide.
- Patience 20 (was 10): with patience 10 vs scheduler patience 5 a run got only one LR halving, 2 runs collapsed (best epoch <= 1), and bootstrap runs often peaked late (epoch 30-32). Scheduler patience stays 5.
- Seeds: `--seed` seeds init/shuffle/augment, not the split. CV runs use 3 seeds per config; the selection key takes the median over seeds, so one collapsed seed (best epoch <= 2) cannot move it.
- Metrics: selection = subject-macro balanced MAE (mean over sub-datasets of the mean per-subject MAE). Tie-break only = bin-balanced MAE (window-pooled per bin, bins [40,60), [60,70), ..., [120,130), [130,∞), mean over bins with >= 20 windows). Window-pooled balanced MAE, RMSE, per-bin and per-dataset MAE are reported, not used to select.
- Leave-test-group-out selection: the decision for test group k uses only val errors of folds j ≠ k-1 (fold j's val is group j+1), so group k never selects its own config. Global selection over all val is reported too, labelled optimistic. Tests vs baseline (Huber delta 5, no bootstrap, LDS off): Wilcoxon signed-rank on per-subject OOF test MAE per sub-dataset, Holm across configs. Logic in `src/bpm/aggregate.py`.
- LDS (Yang et al., ICML 2021): `dataset.lds_weights`. Density = histogram of the fold's real train labels in 1-BPM bins over 30..220, each window counted with its sampler weight, smoothed with a Gaussian (σ = 2 BPM). Weight = density^-p, normalised to sampler-weighted mean 1, then clip at 20 and renormalise, repeated until max <= 20 (a single clip+renorm left max 25-27). Real windows only; synthetic bootstrap windows get weight 1. Loss = mean(w · per-window loss); val unweighted. `--lds-power p`, p ∈ {0 (off), 0.5, 1}.
- Outputs: sidecar `cache/runs/<tag>.json` per run; `evaluate.py --fold k` writes per-window val+test predictions to `cache/preds/` (one ckpt or a mean-BPM seed ensemble, `scores` = std across models); `aggregate_runs.py` writes `cache/results/runs.csv` and `summary.csv`. No Bland-Altman plots under `--fold` (would be about 1,200 PNGs).

## 14. Output heads, quality gate, offline smoothing (2026-11-07) — CODE DONE, RUNS PENDING

Method definitions: `.scratch/model-run-plan/spec.md` (Methods 3-5). Code: Step 0B of `.scratch/cv-code-impl/spec.md`.

- Heads (`CNNBiLSTM(head=...)`): only the last `Linear(32, n)` changes, so old reg checkpoints load unchanged and `evaluate.py` reads the head from that layer's shape. `dist`: 191 1-BPM bins 30..220, target = Gaussian (σ 2 BPM) over bin centres, cross-entropy; BPM = expected value within ±10 bins of the argmax (robust to a second, harmonic mode). `gauss`: mean + log-variance, beta-NLL with β = 0.5 (Seitzer et al. 2022) so high-variance windows do not stop learning the mean; log-variance clamped to [-10, 10].
- Ensembles: reg mean BPM; dist averages the per-model softmax (after the optional logit adjustment) and decodes once; gauss = mixture mean / variance.
- Logit adjustment `--tau 1`: prior = (Gaussian-smoothed fold-train label counts + 1) normalised. Smoke finding: the +1 floor makes unseen bins (150-220) ~7 logits more attractive, 30% of fold-0 windows then decode > 20 BPM off. Dropped 2026-11-07: dist is evaluated with τ = 0 only, no label prior anywhere (run plan "Decisions after Step 0B smokes"); the flag stays, default 0.
- Quality gate (`src/bpm/quality.py`): every score is "higher = worse", a window passes when score <= limit. Signal scores: −skewness, 1 − spectral prominence (0.6-3.5 Hz, ±0.1 Hz around the peak). Model scores: ensemble spread, dist curve std / gauss σ, MC-dropout std. Limit fitted on val, applied unchanged to test: coverage mode (val quantile) or mae mode (largest limit with passed balanced MAE <= 5, coverage fallback when infeasible). Subjects with no passed window are left out of the MAE and counted.
- Sequences (`predict_sequences.py`): all 2 s-stride windows within ±5 min of a labelled window, so the smoother sees context the capped window cache drops. `windowing.all_windows` returns a strided view (multisite files reach 9.3 days).
- Smoothing (`src/bpm/smoothing.py`, `smooth_report.py`): segments split at gaps > 30 s. Median (gated windows left out), Kalman + RTS random walk (R = (c·score)², gated R = ∞), HMM forward-backward over the 191 bins (emission = plain softmax, no prior division: decided 2026-11-07, see run plan "Decisions after Step 0B smokes"; Gaussian transition banded at ±5σ_T plus a 1e-3 uniform floor, gated windows flat). q / σ_T = m × MAD·1.4826 of fold-train label steps per sub-dataset. Gated runs longer than `--max-gated-s` (30 s) stay uncovered. Smoke fold 0: with the prior the HMM was broken by the same floor issue as τ = 1 (val 9.1-14.3); without it 3.5, level with Kalman.
