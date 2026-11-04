# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project

CNN-BiLSTM model to estimate heart rate (BPM) from PPG (photoplethysmography) signals, trained against ECG-derived ground truth. Implementation follows `docs/agents/plan.md` and `docs/agents/pipeline-decisions.md`.

# Tech Stack

Python 3.14, PyTorch (CUDA on the host RTX 3080, CPU fallback). `neurokit2` (R-peak detection), `scipy`/`numpy`/`matplotlib`. EDF reading is a small numpy plain-EDF reader in `src/bpm/edf_io.py` (`pyedflib` has no 3.14 wheel and this Bazzite host has no Python headers to build it).

Runs directly on the host (Bazzite, immutable Fedora) in a `.venv` (gitignored), no container. Default PyPI `torch` bundles CUDA, no extra index needed.

# Commands

- build: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- test: `.venv/bin/pytest`
- lint: none configured
- build manifest: `.venv/bin/python scripts/build_manifest.py` (writes `cache/manifest.csv`, paths relative to `harmonized_ecg_ppg_databases/`)
- build window cache: `.venv/bin/python scripts/build_cache.py` (writes `cache/windows/<subject>.npz`; long-running, walks all 684 EDF files, R-peak detection on up to ~18hr multisite recordings)
- build bootstrap (optional): `.venv/bin/python scripts/build_bootstrap.py` (time-warped train windows for low/high BPM, writes `cache/bootstrap/{H,L}/<subject>.npz`; long-running, R-peak detection on train files, HR curves cached in `cache/hr_curves/`)
- train: `.venv/bin/python scripts/train.py` (writes `cache/best.pt`); `--loss huber|mae` (default `huber`) and `--delta <BPM>` (Huber delta, default 5) pick the loss; `--bootstrap HL` (or `H`/`L`) adds the bootstrap windows. Non-default runs write `cache/best_<tag>.pt`, tag = `mae` or `huber<delta>` plus `_boot<letters>` (e.g. `best_huber10.pt`, `best_huber5_bootHL.pt`)
- evaluate: `.venv/bin/python scripts/evaluate.py` (prints per-sub-dataset MAE/RMSE, writes Bland-Altman plots to `cache/plots/`); `--ckpt <path>` scores another checkpoint (default `cache/best.pt`), plots then go to `cache/plots/<ckpt stem>/`
- results notebook: `.venv/bin/jupyter lab notebooks/results.ipynb` (test-set plots: Bland-Altman, ROC/AUROC, scatter, error breakdowns; re-execute headless with `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace results.ipynb`)

`cache/` is gitignored, regenerated from the raw EDF data, never checked in.

# Data

`harmonized_ecg_ppg_databases/` (extracted from `harmonized_ecg_ppg_databases.zip`) holds three EDF sub-datasets; only two are used (exercise is excluded, see `README.md`). All files are plain EDF (not EDF+), 100 Hz, no annotation channel, no CSV/ground-truth files anywhere. BPM labels must be derived at training time from ECG R-peaks.

- `wrist-ppg-during-exercise-1.0.0/`: excluded from the whole pipeline (not scanned by `build_manifest`; reasons in `README.md`). 19 files, 8 subjects (`s1`-`s6`, `s8`, `s9`; `s7` has no file). Channels: `ECG`, `PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`. Activities: walk/run/bike (low/high resistance), ~5-10 min per file. Only sub-dataset with accelerometer data, needed for motion-artifact handling during exercise.
- `multisite-ppg/`: 519 files, 20 subjects, 4 wearable sites (`earring`, `necklace`, `ring`, `watch`), 2 trials-ish per site per subject (filename pattern `P<id>_<site>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Free-living recordings, duration highly variable per file (~2500s to ~66000s / up to 18hr), so window-count imbalance across files/subjects is a real risk if not capped.
- `wrist-finger-ppg/`: 165 files, 27 subjects, 6 trials each (filename pattern `<subject>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Short controlled clips, ~125-420s per file.

EDF header layout per the EDF spec: 256-byte main header + 256-byte-per-signal header, see `src/bpm/edf_io.py`.

Model input is the bandpass-filtered PPG channel only (`in_channels=1`). The earlier ACC zero-fill + `has_motion` flag scheme was removed together with the exercise sub-dataset.

Split: subject-wise (no subject's windows cross train/val/test), 70/15/15 within each of the two used sub-datasets (`src/bpm/split.py`, seed 42; val/test each get `max(1, round(0.15 * n))` subjects) so the test set isn't dominated by one sub-dataset.

# Conventions

- `src/bpm/`: library code (manifest, EDF I/O, ground truth, windowing, preprocessing, dataset/model, split, caching). `scripts/`: thin CLI entry points over it.
- TDD seams (see `tests/`): `manifest.build_manifest`, `ground_truth.ecg_to_hr_curve`, `windowing.make_windows`, `split.subject_wise_split`, `dataset.zscore_window`, `model.CNNBiLSTM.forward`, `edf_io.read_channels`, `timewarp.warp_window`/`plan_bootstrap`, `dataset.sampler_weights`. Other modules (`preprocessing`, `cache`, `data_loading`) are thin glue, verified by smoke-running against real EDF files rather than unit tests.

## Agent skills

### Issue tracker

Issues tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: one root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
