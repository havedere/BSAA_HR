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
