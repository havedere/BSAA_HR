# Handoff: wrong high-BPM ECG labels (likely R-peak double-counting)

Status: done (retrain pending, user runs it)

## Problem
Many multisite labels at 110+ BPM look about 2x the true heart rate. This affects the real train/val/test labels and the time-warp `H` bootstrap built from them. `L` bootstrap is fine.

Evidence (2026-11-04). Method: dominant FFT peak of the cached bandpassed PPG window in the 0.6-3.5 Hz band, compared to the label:
- Real multisite windows with label >= 110 (n=1460): PPG peak / label quantiles p10..p90 = 0.30, 0.34, 0.45, 0.91, 1.01. 27% of windows fall in 0.4-0.6. The median PPG rate is less than half the ECG rate.
- Real multisite, agreement within 10 BPM, by bin: 60: 62%, 70: 53%, 80: 42%, 90: 33%, 100: 26%. Wrist-finger tails: 98%.
- Synthetic `H` (`cache/bootstrap/H`, n=3261): 25% agree. Synthetic `L`: 96%.
- Hypothesis: Pan-Tompkins (`neurokit2`, `src/bpm/ground_truth.py:_neurokit2_r_peak_detector`) counts T-waves or noise as R-peaks on multisite ECG (free-living recordings, long). Not yet verified on raw ECG.

The existing QC (RR ratio limit 1.2, 30-220 BPM) misses a consistent doubling. Doubled beats alternate short/long RR, but a steady double-count can pass the ratio check.

## Goal
1. Confirm or reject the double-counting hypothesis on raw ECG.
2. Fix the labels at the source (the ground-truth stage), then rebuild the caches.
3. Re-check `H` bootstrap quality. If needed, add a source filter in `src/bpm/timewarp.py:plan_bootstrap` that only warps sources whose PPG peak agrees with the HR label.

## Suggested steps
1. **Confirm**: pick multisite train windows with label >= 140 and PPG peak/label ~0.5. Find their file and start time. The cache doesn't store those, so re-derive them by re-windowing one file via `bpm.cache.read_ppg_and_hr` + `make_windows`. Plot the raw ECG with the detected R-peaks over that span, and check whether the extra peaks are T-waves.
2. **Options to compare** (pick by agreement metric, not by eye):
   - Another detector: `nk.ecg_peaks(method="neurokit" | "engzeemod2012" | "hamilton2002")`. Clean ECG first with `nk.ecg_clean`.
   - Check ECG polarity per file. An inverted ECG makes T-waves dominant; flip it if the median R amplitude is negative.
   - A PPG cross-check QC: NaN out HR seconds where the PPG spectral HR disagrees by a ~2x factor. This drops windows instead of fixing them, so report the window loss per bin.
3. Keep `ecg_to_hr_curve(ecg, fs, r_peak_detector=...)`, which is the TDD seam (`tests/test_ground_truth.py`). Add a test with a synthetic ECG that has tall T-waves.
4. **Rebuild**, deleting the stale curves:
   - `build_cache.py` (about 4 min now)
   - `rm -r cache/hr_curves`
   - `build_bootstrap.py` (about 2 min)
5. **Verify**: rerun the agreement check above and expect the 110+ quantiles to move toward 1.0. Then retrain and compare:
   - `.venv/bin/python scripts/train.py --bootstrap HL`
   - `.venv/bin/python scripts/evaluate.py --ckpt cache/best_huber5_bootHL.pt` against `best.pt`
   - `evaluate.py` prints per-bin MAE. Note that `best.pt` must be retrained on the new labels for a fair comparison.

## Context / pointers
- Labels: `src/bpm/ground_truth.py` (R-peaks to 1 Hz HR curve, linear interpolation, NaN on bad RR). Flat-PPG masking: `src/bpm/cache.py:read_ppg_and_hr`.
- Bootstrap: `src/bpm/timewarp.py`, `scripts/build_bootstrap.py`. HR curves are cached in `cache/hr_curves/` and must be deleted when label code changes. Decisions are in `docs/agents/pipeline-decisions.md` §12.
- Multisite site files (earring/necklace/ring/watch) each have their own ECG (lengths differ), so labels are per file.
- Split: subject-wise, seed 42 (`src/bpm/split.py`). Test multisite reaches 191 BPM, so test labels are probably affected too and the current reported MAE may be misleading.
- Repo rules: plan before acting, no new deps without asking, use a branch for bigger changes, never commit or push unless asked. Another agent previously worked in the same checkout; check `git status`/`git branch` first.

## Comments
- 2026-11-04 (Claude): Hypothesis mostly rejected. In raw-ECG checks the bad labels come from non-ECG signal (lead-off ramp/plateau, steady fake ~180 BPM) and noisy-ECG double counting, not T-waves. Real tachycardia exists (P1_watch_1, P17_*) with correct labels and bad PPG. Fix: `ecg_quality_mask` (bSQI >= 0.8 per 10 s plus a flat-ECG check), applied in `read_ppg_and_hr`. On a 40-file sample the 140+ windows drop 96 -> 57, with no change below 140. Caches rebuilt. Cache-level agreement barely moved (multisite >=110 quantiles 0.30/0.35/0.46/0.92/1.01; H 25% -> ~25%), because the remaining disagreement is motion-corrupted PPG with a correct label. Multisite windows with PPG std >= 100 (37% of them) agree only 24%; near-flat PPG is negligible (30 windows). Open: PPG quality gate or H-source filter (goal 3), user decision.
- 2026-11-05 (Claude): User chose to filter H sources only; real windows stay untouched. `timewarp.ppg_hr_agrees` gives a per-start mask (PPG spectral peak within 10% of the label), and `plan_bootstrap(source_ok=...)` uses it for the H bins only in `build_bootstrap.py`. All quotas are filled with unique starts. H agreement within 10 BPM went from ~25% to 69-89% per bin (>=110 quantiles 0.57/0.97/0.99/1.01/1.03). L unchanged at 95%.
