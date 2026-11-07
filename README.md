# BSAA_HR

CNN-BiLSTM model that estimates heart rate (BPM) from PPG (photoplethysmography) windows, trained against ECG-derived ground truth. Design: `docs/agents/plan.md` and `docs/agents/pipeline-decisions.md`.

## Setup and run

Needs Python 3.14. Uses CUDA when available, otherwise CPU.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
.venv/bin/python scripts/build_manifest.py   # cache/manifest.csv
.venv/bin/python scripts/build_cache.py      # cache/windows/<subject>.npz (long-running)
.venv/bin/python scripts/make_folds.py       # cache/folds.json (5-fold CV subject groups)
.venv/bin/python scripts/build_bootstrap.py  # optional: cache/bootstrap/{H,L}/<subject>.npz (long-running)
.venv/bin/python scripts/train.py            # cache/best.pt (--bootstrap HL: cache/best_huber5_bootHL.pt)
.venv/bin/python scripts/evaluate.py         # MAE/RMSE per sub-dataset, cache/plots/
.venv/bin/python scripts/aggregate_runs.py   # CV runs: cache/results/{runs,summary}.csv
```

Raw data goes in `harmonized_ecg_ppg_databases/` (extracted from `harmonized_ecg_ppg_databases.zip`). `cache/` is generated, never checked in.

## Scripts

Run them in this order; each one reads what the previous one wrote. Re-run from `build_manifest.py` whenever the raw data or the windowing/label code changes. The scripts are thin entry points; the logic lives in `src/bpm/`.

### `scripts/build_manifest.py`

Lists every recording that will be used. It scans `multisite-ppg/` and `wrist-finger-ppg/`, reads the subject, site and trial from each filename, and reads the EDF header for the sample count and duration. It writes one row per file to `cache/manifest.csv`. File paths are stored relative to `harmonized_ecg_ppg_databases/`. Files with unexpected names are skipped with a warning. The exercise folder is not scanned (see below). Takes seconds.

### `scripts/build_cache.py`

Turns the recordings into training windows. Slow: it runs R-peak detection on every file, some up to 18 hours long. For each subject in the manifest, and each of that subject's files:

1. Read the ECG and PPG channels.
2. Find R-peaks in the ECG (`neurokit2`, Pan-Tompkins). Turn them into a heart-rate curve sampled once per second. Mark stretches as unusable when a beat implies a rate outside 30 to 220 BPM, or when two neighbouring beat intervals differ by more than 20%.
3. Bandpass-filter the PPG to 0.5 to 8 Hz, the heart-rate band.
4. Cut 10 s windows every 2 s. Drop any window that touches an unusable stretch, or any second where the raw PPG is constant (sensor off or padding: most of the multi-day multisite files are flat PPG while the ECG is still live). The label is the mean heart rate over the window.

Multisite subjects are capped at 2000 random windows (seed 42), so a few very long recordings don't dominate. Output: one `cache/windows/<subject>.npz` per subject, holding `windows` (N × 1 × 1000, float32), `labels` (BPM) and `dataset_source`. Subjects with no usable windows are skipped.

### `scripts/make_folds.py`

Splits the subjects into 5 groups for subject-wise cross-validation. Run once after `build_cache.py`; every `--fold` run reads the result, so rebuilding `cache/windows/` does not reshuffle the folds.

- Stratified per sub-dataset: each subject gets a score, the share of its windows at 130 BPM and above (multisite) or 100 BPM and above (wrist-finger, which never reaches 130). Subjects are sorted by score, cut into strata of 5, shuffled within each stratum (seed 42) and dealt to groups 0-4. So the subjects with the most high-BPM windows end up in different groups.
- Fold k: test = group k, validation = group (k+1) mod 5, train = the other 3 groups. Every subject is in test once and in validation once.
- Output: `cache/folds.json`, `{subject: {group, score}}`. Its sha256 (`folds_hash`) goes into every run sidecar. Prints subjects per group, the multisite top 5 and windows of 140 BPM and above per group.

### `scripts/build_bootstrap.py` (optional)

Makes synthetic train windows for the BPM ranges the data barely covers, by time-warping real PPG. A factor f (0.7 to 1.6) squeezes or stretches a 10·f s stretch of signal into one 10 s window, so the label is f times the source heart rate.

- Train subjects only, so nothing leaks into validation or test.
- `L`: bins 40-60 BPM, both sub-datasets. `H`: bins 110-190 BPM, multisite only (wrist-finger test stays below 98 BPM and its signals top out at 113 BPM).
- Each 10-BPM bin is topped up to 500 windows per sub-dataset, counting the real train windows. Each source position is used at most 5 times. Bins that run out of sources stay short and are flagged `SHORT` in the printed table, never padded with duplicates.
- The warped window is bandpass-filtered again, because warping shifts the 0.5-8 Hz band and the leftover high-frequency energy would give synthetic windows away.
- Slow: it runs R-peak detection on every train file. The HR curves are saved to `cache/hr_curves/` and the per-file PPG check to `cache/ppg_ok/`, so reruns are fast. Delete both folders when the ground-truth code changes.
- `--fold k` takes the train subjects and bin quotas from fold k of `cache/folds.json`. Under `--fold`, wrist-finger also gets the H bins 110-130 BPM (its labels reach 113 BPM; the full exclusion came from the fixed test set).

Output: `cache/bootstrap/H/<subject>.npz` and `cache/bootstrap/L/<subject>.npz` (with `--fold k`: `cache/bootstrap/fold<k>/{H,L}/`), same keys as `cache/windows/`.

### `scripts/train.py`

Trains the CNN-BiLSTM and keeps the best model.

- Splits subjects into train, validation and test (70/15/15 within each sub-dataset, seed 42). No subject appears in more than one split. `--fold k` uses fold k of `cache/folds.json` instead (see `make_folds.py`).
- Z-scores each window (mean 0, standard deviation 1) and adds small Gaussian noise to training windows.
- Samples training batches so both sub-datasets contribute equally, whatever their window counts.
- `--bootstrap H`, `L` or `HL` (`-bootstrap HL` also works) adds the windows from `build_bootstrap.py` (`cache/bootstrap/fold<k>/` under `--fold k`). Synthetic windows get at most 20% of each sub-dataset's sampling weight, and an epoch stays as long as the real-window count. The checkpoint gets a `_boot<letters>` suffix, so `best.pt` is not overwritten. Each epoch also prints the validation MAE on windows of 110 BPM and above.
- `--loss huber` (default), `--loss mae` or `--loss mse` picks the training loss; `--delta 10` sets the Huber delta in BPM (default 5, ignored for `mae` and `mse`). Early stopping and the LR schedule stay on the balanced validation MAE either way, so runs compare fairly.
- `--lds-power p` (default 0 = off) weights the loss per window by LDS (label-distribution smoothing): rare BPM values count more. Real train windows only; synthetic windows keep weight 1. Validation is unweighted.
- `--seed s` (default 42) seeds init, shuffling and augmentation; the split stays at seed 42.
- Huber loss by default, Adam (learning rate 1e-3, halved when validation MAE stalls), batch size 64, up to 100 epochs.
- Balanced validation MAE = mean of the two sub-datasets' window-pooled MAEs. Early stopping and the LR schedule use it.
- Stops after 20 epochs (`--patience`, was 10) without a better balanced validation MAE. The best model is saved to `cache/best.pt` for the default run (Huber, delta 5, no bootstrap). Other runs save to `cache/best_<tag>.pt`, tag = `<loss>[_boot<X>][_lds<p>][_fold<k>]_seed<s>` with loss `mae`, `mse` or `huber<delta>`. The seed is always in the tag under `--fold`, otherwise only when it is not 42 (e.g. `best_mse.pt`, `best_huber5_bootHL.pt`, `best_huber5_lds0.5_fold0_seed1.pt`).
- Every run writes a sidecar `cache/runs/<tag>.json`: the arguments, best and stop epoch, best validation value, whether it hit the epoch limit or collapsed (best epoch 2 or earlier), git hash and `folds_hash`.

Uses the GPU when CUDA is available. Prints the loss and MAE for every epoch.

### `scripts/evaluate.py`

Scores `cache/best.pt` on the test subjects. These are the same subjects as in training, because the split is rebuilt from the manifest with the same seed.

- Computes MAE and RMSE per subject, then averages over the subjects of each sub-dataset, so a subject with many windows doesn't outweigh the others.
- Prints the score per sub-dataset, plus a combined MAE that weights each sub-dataset equally.
- Prints the MAE per 10-BPM truth bin per sub-dataset, to show how the low and high tails do.
- `--ckpt cache/best_huber5_bootHL.pt` scores another checkpoint (default `cache/best.pt`); its plots go to `cache/plots/<checkpoint name>/`, so the baseline plots are kept.
- Saves Bland-Altman plots (prediction error against mean BPM) per sub-dataset and overall to `cache/plots/`.
- `--fold k --ckpt a.pt [b.pt ...]` predicts fold k's validation and test subjects and saves the per-window predictions (subject, dataset_source, split, truth, pred, scores), no plots. One checkpoint: `cache/preds/<tag>.npz`, `scores` is NaN. Several (seeds of one config): mean-BPM ensemble in `cache/preds/ens_<config id>_fold<k>.npz`, `scores` = std across models. Config id = tag without `_fold<k>_seed<s>`.

### `scripts/aggregate_runs.py`

Collects the CV runs: reads the sidecars in `cache/runs/` and the predictions in `cache/preds/`. The logic lives in `src/bpm/aggregate.py`.

- Per run: subject-macro balanced MAE (mean per-subject MAE per sub-dataset, averaged over the two), window-pooled balanced MAE, bin-balanced MAE, per split.
- Selection is leave-test-group-out: the decision for test group k only uses validation errors from folds whose validation group is not k. Ranking key = per-subject validation MAE, median over seeds, paired against the best config; ties within 1 SE (subject bootstrap) are broken by bin-balanced MAE. Also prints a global selection over all folds, labelled optimistic.
- Out-of-fold test MAE against the baseline (Huber delta 5, no bootstrap, no LDS): Wilcoxon signed-rank per sub-dataset, Holm-corrected across configs.
- Warns about runs trained on another `folds.json` and sidecars without predictions.
- Output: `cache/results/runs.csv` (one row per run or ensemble) and `cache/results/summary.csv` (one row per config).

### Archived fixed-split results

The results from before the CV switch (fixed 70/15/15 split, patience 10, early stopping on plain validation MAE) were moved to `cache/archive_fixedsplit/` in Step 1 of the run plan (2026-11-07): `cache/best_*.pt` (e.g. `best_mae.pt`, `best_huber10.pt`), `cache/logs/` and `cache/plots/` (e.g. `cache/plots/best_mae/`). After the move, those paths live under `cache/archive_fixedsplit/` (e.g. `cache/archive_fixedsplit/best_mae.pt`, `cache/archive_fixedsplit/plots/best_mae/`). `cache/best.pt` stays in place, because the notebook and the default `evaluate.py` load it. (It is currently missing and has to be regenerated with `train.py --patience 10`.) Smoke-test artefacts of the CV code (fold-0 checkpoints, sidecars, preds, sequences, results) are in `cache/archive_smoke/`.

## Data

Used: `multisite-ppg/` (20 subjects) and `wrist-finger-ppg/` (27 subjects). The model input is the bandpass-filtered PPG channel only; the ECG channel is used only to derive the BPM labels.

### Excluded: `wrist-ppg-during-exercise-1.0.0/`

The exercise sub-dataset (8 subjects, walk/run/bike, the only one with accelerometer channels) is not used anywhere in the project. `build_manifest` does not scan it, so no later step sees it.

Why:

- Most of its labels are unusable. BPM labels come from ECG R-peaks, and a window is dropped when the beat-to-beat intervals look implausible (see `ground_truth.py`). ECG recorded during running and cycling is noisy, so only 1255 of 3313 possible windows (38%) survived. For subject `s4` it was 3 of 142.
- The surviving windows are biased toward calm stretches. The high heart rates that make exercise data valuable are the ones that get dropped. No training subject kept a window above 136 BPM.
- Its test score was meaningless. With 8 subjects, the subject-wise split puts one subject in test. That was `s4`, whose 3 windows (about 174 BPM) gave an MAE of 92 BPM and dominated the combined score.
- Its accelerometer channels needed zero-filled ACC channels plus a `has_motion` flag for the other two sub-datasets. Without it, the model takes one PPG channel.

Bringing it back would need a more robust ECG label pipeline first (one that does not throw away high-heart-rate stretches during motion), then the ACC input channels again.

## Open issue: multi-day multisite recordings

Found 2026-10-22 during the audit; not investigated or fixed yet.

- Multisite files are typically a few hours long (median 4.5 h), but 99 of the 519 are longer than 18 h and some run for days (durations from the EDF headers, as recorded in `cache/manifest.csv`):

  | Files (all 4 sites each) | Declared duration |
  |---|---|
  | `P3_*_7.edf` | 966.9 h (40.3 days), 1.39 GB each |
  | `P4_*_5.edf` | 222.7 h (9.3 days) |
  | `P6_*_1.edf` | 121 to 143 h |
  | `P2_*_32.edf` | about 113 h |
  | `P2_*_31.edf`, `P14_*_7.edf`, `P15_*_3.edf` | 35 to 44 h |

- `P3_earring_7.edf` yields only 681 usable windows (about 23 minutes), so most of that file is probably padding or flat signal, not real recording. The others have not been checked.
- Subject P3 alone is 6.4 GB of the 14.2 GB multisite input, so these files dominate the `build_cache.py` run time.
- Processing one `P3_*_7.edf` file peaks at about 22 GB RAM (measured with `/usr/bin/time -v`). The cause is a float64 copy of the whole signal in `read_channels`, plus `ecg_to_hr_curve` and the window list in `make_windows`. On a 31 GB machine the cache build is close to running out of memory, and it can't be parallelised.

To check next: what the bulk of these files contains (zeros, a constant value, or real signal); whether the headers are wrong or the data is genuinely that long; how the source dataset documents these trials. Then decide whether to trim them to the recorded part, skip them, or stream them in chunks. The "up to ~18hr" figure in this README and in `CLAUDE.md` is wrong; update it once this is resolved.
