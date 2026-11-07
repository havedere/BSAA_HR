# Handoff: data-analysis notebook (raw EDF to windows)

Status: in progress (data part done; next is more data plots or the model-analysis part)

## Goal
One large notebook, `notebooks/data_analysis.ipynb`, that shows the data pipeline visually: raw EDF, ECG to HR label, quality gates, PPG preprocessing, window creation, time-warp bootstrap and the label distribution. The user asked for data analysis only, no model analysis yet.

## State (2026-11-05)
- `notebooks/data_analysis.ipynb` exists, runs without errors (about 30 s, outputs saved). It is untracked and not committed.
- No changes to `src/` or `scripts/`.
- Generator: `.scratch/data-analysis-notebook/gen_nb.py` (nbformat; cells are strings). Rebuild and run:
  ```
  .venv/bin/python .scratch/data-analysis-notebook/gen_nb.py notebooks/data_analysis.ipynb
  cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace data_analysis.ipynb
  ```
  Edit either the generator or the notebook, not both. If you edit the notebook in Jupyter, the generator is out of date.
- A Jupyter Lab server was started on 127.0.0.1:8888 in the last session. It is probably gone now; restart it with `.venv/bin/jupyter lab notebooks/data_analysis.ipynb`.

## Notebook contents
1. **Style** cell at the top, the single source of truth for every plot. `COLORS` maps roles to colors from a colorblind-safe palette (ECG blue, PPG orange, bandpassed PPG green, rejected red, real blue, bootstrap orange). `STYLE` holds sizes, and `rcParams` are set from both. Rule: plot cells never use a raw hex color or font size. The user asked for this explicitly ("uniform look, good visible colors").
2. **Helpers**: `gate_masks(ecg, ppg, fs)` splits the merged library gates into one 1 Hz mask per gate: flat ECG, bSQI < 0.8, HR outside 30-220, RR ratio > 1.2, flat PPG, plus "outside beats". It reuses the `ground_truth` constants. An assert checks that the union of the masks equals the NaN seconds of `cache.read_ppg_and_hr`, so it fails if the library changes. `SCAN` holds the gate masks for all wrist-finger files plus a sample of 34 multisite files (2 per subject, 1800-20000 s).
3. **Data summary** table: per sub-dataset and pooled. Covers recordings, quality (from the gate scan), windows (possible vs cached, cap hits), BPM label stats, split and bootstrap counts.
4. Sections:
   1. Dataset overview.
   2. Raw EDF.
   3. Pipeline: R-peaks, HR curve, gate strip, bandpass with PSD, z-score.
   4. Windows: kept vs dropped, good window, one example per gate, bad-PPG/valid-ECG window, time-warp at f=0.7 and f=1.6.
   5. Gate rejection rates.
   6. BPM bars of train labels, real vs bootstrap, then labels per split.

## Findings to follow up (not fixed)
- **Pan-Tompkins peak lag**: the label detector marks beats about 50-70 ms after the R-peak, and the lag varies. Beat-to-beat RR-difference SD is 60-80 ms, against 20-35 ms for neurokit, measured on `wrist-finger-ppg/22_2.edf`, `01_0.edf` and `multisite-ppg/P20_watch_7.edf`. This makes the HR labels noisier and causes extra `RR ratio > 1.2` rejections. Candidate fix: label from neurokit peaks, or snap PT peaks to the local max of the cleaned ECG. This touches `ground_truth.py` (TDD seam), so the caches, `cache/hr_curves/` and `cache/ppg_ok/` must be rebuilt.
- **Multisite rejection**: 65% of seconds are rejected in the sample (bSQI alone 58%). Only 21% of the possible windows survive the gates, and 18 of 20 subjects still hit the 2000-window cap.
- **Gates miss baseline wander**: an ECG that is only baseline wander can pass every gate when both detectors lock onto the wander. The first automatic "valid ECG" pick was like this, so the bad-window search now also requires cleaned-ECG kurtosis > 5.
- **Flat-PPG gate is strict**: it only catches PPG that is exactly constant per second. Near-flat stretches pass.
- **Suspicious file durations**: `multisite-ppg/P3_{earring,necklace,ring}_7.edf` read as 966.9 h (about 40 days). The EDF header is probably wrong. This inflates the multisite recorded-time and candidate-window numbers. CLAUDE.md says files are at most ~18 h.

## Open / next
- Section 6 and the summary use the fixed split (`subject_wise_split`, seed 42). The CV folds (`cache/folds.json`, `split.fold_split`) were added in parallel. Consider a per-fold label-distribution plot.
- Bootstrap bars read `cache/bootstrap/{H,L}/` (fixed split). The per-fold bootstrap directories `cache/bootstrap/fold<k>/` are not shown.
- Model analysis is not started. The user wants it later, and `results.ipynb` already has a first version.
- Possibly add a CLAUDE.md command line for the data notebook (same form as the results notebook line).
- Nothing committed. The user decides whether to commit `notebooks/data_analysis.ipynb` (about 3.8 MB with outputs).
