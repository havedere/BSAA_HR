# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Next session TODO

Wire up GPU passthrough for training (currently CPU-only, per plan.md's locked decision):
- Host has an NVIDIA GPU; its device nodes (`/dev/nvidia0`, etc.) already reach this agent
  container, but the **nested** Docker daemon has no `nvidia-container-runtime` registered —
  confirmed via `docker run --gpus all ... nvidia-smi` failing with
  "could not select device driver \"\" with capabilities: [[gpu]]".
- To fix: install `nvidia-container-toolkit` for the nested daemon, add `runtime: nvidia`
  (or the `deploy.resources.reservations.devices` GPU block) to `docker-compose.yml`'s `dev`
  service, and switch `requirements.txt`'s torch install off the `https://download.pytorch.org/whl/cpu`
  index to a CUDA wheel matching the host driver's CUDA version.
- Not done yet: explicitly deferred by the user mid-session rather than wired up immediately,
  since plan.md locks CPU-only and a training run was already in flight.

# Project

CNN-BiLSTM model to estimate heart rate (BPM) from PPG (photoplethysmography) signals, trained against ECG-derived ground truth. Implementation follows `docs/agents/plan.md` and `docs/agents/pipeline-decisions.md`.

# Tech Stack

Python, PyTorch (CPU-only). `pyedflib` (EDF reading), `neurokit2` (R-peak detection), `scipy`/`numpy`/`scikit-learn`/`matplotlib`.

This agent container is Alpine/musl with no C toolchain — PyTorch ships no musllinux wheels, so all code runs inside the `dev` Docker Compose service (glibc, `python:3.12-slim`), not directly on the host container. `docker compose up -d dev` once, then `docker compose exec dev <cmd>`.

# Commands

- build: `docker compose build dev`
- test: `docker compose exec dev pytest`
- lint: n/a — none configured
- build manifest: `docker compose exec dev python3 scripts/build_manifest.py` (writes `cache/manifest.csv`)
- build window cache: `docker compose exec dev python3 scripts/build_cache.py` (writes `cache/windows/<subject>.npz`; long-running — walks all 703 EDF files, R-peak detection on up to ~18hr multisite recordings)
- train: `docker compose exec dev python3 scripts/train.py` (writes `cache/best.pt`)
- evaluate: `docker compose exec dev python3 scripts/evaluate.py` (prints per-sub-dataset MAE/RMSE, writes Bland-Altman plots to `cache/plots/`)

`cache/` is gitignored — regenerated from the raw EDF data, never checked in.

# Data

`harmonized_ecg_ppg_databases/` (extracted from `harmonized_ecg_ppg_databases.zip`) holds three EDF sub-datasets. All files are plain EDF (not EDF+), 100 Hz, no annotation channel, no CSV/ground-truth files anywhere — BPM labels must be derived at training time from ECG R-peaks.

- **`wrist-ppg-during-exercise-1.0.0/`** — 19 files, 8 subjects (`s1`-`s6`, `s8`, `s9`; `s7` has no file). Channels: `ECG`, `PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`. Activities: walk/run/bike (low/high resistance), ~5-10 min per file. Only sub-dataset with accelerometer data — needed for motion-artifact handling during exercise.
- **`multisite-ppg/`** — 519 files, 20 subjects, 4 wearable sites (`earring`, `necklace`, `ring`, `watch`), 2 trials-ish per site per subject (filename pattern `P<id>_<site>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Free-living recordings, duration highly variable per file (~2500s to ~66000s / up to 18hr) — window-count imbalance across files/subjects is a real risk if not capped.
- **`wrist-finger-ppg/`** — 165 files, 27 subjects, 6 trials each (filename pattern `<subject>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Short controlled clips, ~125-420s per file.

EDF headers parsed by hand (no `pyedflib`/`mne` installed) to get this info — see conversation history or re-derive via a minimal stdlib EDF-header parser if needed (256-byte main header + 256-byte-per-signal header, per the EDF spec).

Planned handling of the ACC mismatch: feed PPG + ACC_X/Y/Z as model input; for the two sub-datasets without ACC, zero-fill those channels and add a binary "has_motion" flag channel, rather than maintaining two separate model variants.

Planned split: subject-wise (no subject's windows cross train/val/test) across the combined pool from all three sub-datasets, roughly stratified by dataset-source/activity so the test set isn't dominated by one sub-dataset.

# Conventions

- `src/bpm/` — library code (manifest, EDF I/O, ground truth, windowing, channel assembly, preprocessing, dataset/model, split, caching). `scripts/` — thin CLI entry points over it.
- TDD seams (see `tests/`): `manifest.build_manifest`, `ground_truth.ecg_to_hr_curve`, `windowing.make_windows`, `split.subject_wise_split`, `dataset.zscore_window`, `model.CNNBiLSTM.forward`. Other modules (`edf_io`, `preprocessing`, `channels`, `cache`, `data_loading`) are thin glue, verified by smoke-running against real EDF files rather than unit tests.

## Agent skills

### Issue tracker

Issues tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: one root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
