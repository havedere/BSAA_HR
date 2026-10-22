# BSAA_HR

CNN-BiLSTM model that estimates heart rate (BPM) from PPG (photoplethysmography) windows, trained against ECG-derived ground truth. Design: `docs/agents/plan.md` and `docs/agents/pipeline-decisions.md`.

## Setup and run

Needs Python 3.14. Uses CUDA when available, otherwise CPU.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest
.venv/bin/python scripts/build_manifest.py   # cache/manifest.csv
.venv/bin/python scripts/build_cache.py      # cache/windows/<subject>.npz (long-running)
.venv/bin/python scripts/train.py            # cache/best.pt
.venv/bin/python scripts/evaluate.py         # MAE/RMSE per sub-dataset, cache/plots/
```

Raw data goes in `harmonized_ecg_ppg_databases/` (extracted from `harmonized_ecg_ppg_databases.zip`). `cache/` is generated, never checked in.

## Scripts

Run them in this order; each one reads what the previous one wrote. Re-run from `build_manifest.py` whenever the raw data or the windowing/label code changes. The scripts are thin entry points; the logic lives in `src/bpm/`.

### `scripts/build_manifest.py`

Lists every recording that will be used. It scans `multisite-ppg/` and `wrist-finger-ppg/`, reads the subject, site and trial from each filename, and reads the EDF header for the sample count and duration. It writes one row per file to `cache/manifest.csv`. File paths are stored relative to `harmonized_ecg_ppg_databases/`. Files with unexpected names are skipped with a warning. The exercise folder is not scanned (see below). Takes seconds.

### `scripts/build_cache.py`

Turns the recordings into training windows. Slow: it runs R-peak detection on every file, some up to 18 hours long. For each subject in the manifest, and each of that subject's files:

1. Read the ECG and PPG channels.
2. Find R-peaks in the ECG (`neurokit2`, Pan-Tompkins). Turn them into a heart-rate curve sampled once per second. Mark stretches as unusable when a beat implies a rate outside 30–220 BPM, or when two neighbouring beat intervals differ by more than 20%.
3. Bandpass-filter the PPG to 0.5–8 Hz, the heart-rate band.
4. Cut 10 s windows every 2 s. Drop any window that touches an unusable stretch. The label is the mean heart rate over the window.

Multisite subjects are capped at 2000 random windows (seed 42), so a few very long recordings don't dominate. Output: one `cache/windows/<subject>.npz` per subject, holding `windows` (N × 1 × 1000, float32), `labels` (BPM) and `dataset_source`. Subjects with no usable windows are skipped.

### `scripts/train.py`

Trains the CNN-BiLSTM and keeps the best model.

- Splits subjects into train, validation and test (70/15/15 within each sub-dataset, seed 42). No subject appears in more than one split.
- Z-scores each window (mean 0, standard deviation 1) and adds small Gaussian noise to training windows.
- Samples training batches so both sub-datasets contribute equally, whatever their window counts.
- Huber loss, Adam (learning rate 1e-3, halved when validation MAE stalls), batch size 64, up to 100 epochs.
- Stops after 10 epochs without a better validation MAE. The best model is saved to `cache/best.pt`.

Uses the GPU when CUDA is available. Prints the loss and MAE for every epoch.

### `scripts/evaluate.py`

Scores `cache/best.pt` on the test subjects. These are the same subjects as in training, because the split is rebuilt from the manifest with the same seed.

- Computes MAE and RMSE per subject, then averages over the subjects of each sub-dataset, so a subject with many windows doesn't outweigh the others.
- Prints the score per sub-dataset, plus a combined MAE that weights each sub-dataset equally.
- Saves Bland-Altman plots (prediction error against mean BPM) per sub-dataset and overall to `cache/plots/`.

## Data

Used: `multisite-ppg/` (20 subjects) and `wrist-finger-ppg/` (27 subjects). The model input is the bandpass-filtered PPG channel only; the ECG channel is used only to derive the BPM labels.

### Excluded: `wrist-ppg-during-exercise-1.0.0/`

The exercise sub-dataset (8 subjects, walk/run/bike, the only one with accelerometer channels) is **not used anywhere** in the project. `build_manifest` does not scan it, so no later step sees it.

Why:

- **Most of its labels are unusable.** BPM labels come from ECG R-peaks, and a window is dropped when the beat-to-beat intervals look implausible (see `ground_truth.py`). ECG recorded during running and cycling is noisy, so only 1255 of 3313 possible windows (38%) survived. For subject `s4` it was 3 of 142.
- **The windows that survive are biased toward calm stretches.** The high heart rates that make exercise data valuable are exactly the ones that get dropped. No training subject kept a window above 136 BPM.
- **Its test score was meaningless.** With 8 subjects, the subject-wise split puts one subject in test. That was `s4`, whose 3 windows (about 174 BPM) gave an MAE of 92 BPM and dominated the combined score.
- **It forced extra machinery on everything else.** Supporting its accelerometer channels meant zero-filled ACC channels plus a `has_motion` flag for the other two sub-datasets. With it gone, the model takes one PPG channel.

Bringing it back would need a more robust ECG label pipeline first (one that does not throw away high-heart-rate stretches during motion), then the ACC input channels again.
