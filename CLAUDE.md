# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Project

CNN-BiLSTM model to estimate heart rate (BPM) from PPG (photoplethysmography) signals, trained against ECG-derived ground truth. Implementation follows `docs/agents/plan.md` and `docs/agents/pipeline-decisions.md`.

# Tech Stack

Python 3.14, PyTorch (CUDA on the host RTX 3080, CPU fallback). `neurokit2` (R-peak detection), `scipy`/`numpy`/`matplotlib`. EDF reading is a small numpy plain-EDF reader in `src/bpm/edf_io.py` (`pyedflib` has no 3.14 wheel and this Bazzite host has no Python headers to build it).

Runs directly on the host (Bazzite, immutable Fedora) in a `.venv` (gitignored), no container. Default PyPI `torch` bundles CUDA, no extra index needed.

# Commands

- build: `/home/linuxbrew/.linuxbrew/bin/python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt` (Homebrew `python@3.14`: host `python3` is 3.15, which has no wheels for the pinned torch 2.14.1 / pandas 2.3.3, and neurokit2 0.2.13 needs pandas<3)
- test: `.venv/bin/pytest`
- lint: none configured
- build manifest: `.venv/bin/python scripts/build_manifest.py` (writes `cache/manifest.csv`, paths relative to `harmonized_ecg_ppg_databases/`)
- build window cache: `.venv/bin/python scripts/build_cache.py` (writes `cache/windows/<subject>.npz`; long-running, walks all 684 EDF files, R-peak detection on up to ~18hr multisite recordings)
- build folds: `.venv/bin/python scripts/make_folds.py` (stratified 5-fold subject groups from `cache/windows/`, writes `cache/folds.json` = {subject: {group, score}}; computed once, every `--fold` script reads it)
- build bootstrap (optional): `.venv/bin/python scripts/build_bootstrap.py` (time-warped train windows for low/high BPM, writes `cache/bootstrap/{H,L}/<subject>.npz`; long-running, R-peak detection on train files, HR curves cached in `cache/hr_curves/`, per-file PPG check cached in `cache/ppg_ok/`, delete both when ground-truth code changes); `--fold k` uses fold k's train subjects and writes `cache/bootstrap/fold<k>/{H,L}/` (wrist-finger also gets H bins [110,130) under `--fold`)
- train: `.venv/bin/python scripts/train.py` (writes `cache/best.pt`); `--loss huber|mae|mse` (default `huber`) and `--delta <BPM>` (Huber delta, default 5) pick the loss; `--bootstrap HL` (or `H`/`L`) adds the bootstrap windows. `--seed s` (default 42; init/shuffle/augment only, the split stays at seed 42), `--patience n` (early stop, default 20), `--lds-power p` (LDS loss weights, default 0 = off), `--fold k` (CV fold from `cache/folds.json`; default = fixed split). Early stop and LR schedule run on window-pooled balanced val MAE. Non-default runs write `cache/best_<tag>.pt`, tag = `<loss>[_boot<X>][_lds<p>][_fold<k>]_seed<s>`, loss = `mae`, `mse` or `huber<delta>`; the seed is always in the tag under `--fold`, otherwise only when != 42 (e.g. `best_huber10.pt`, `best_huber5_bootHL.pt`, `best_mse_lds0.5_fold0_seed1.pt`). Every run writes a sidecar `cache/runs/<tag>.json` (config, best/stop epoch, best val, git hash, folds_hash)
- evaluate: `.venv/bin/python scripts/evaluate.py` (prints per-sub-dataset MAE/RMSE, writes Bland-Altman plots to `cache/plots/`); `--ckpt <path>` scores another checkpoint (default `cache/best.pt`), plots then go to `cache/plots/<ckpt stem>/`. `--fold k --ckpt a.pt [b.pt ...]` predicts fold k's val+test subjects into `cache/preds/<tag>.npz` (one ckpt) or `cache/preds/ens_<config id>_fold<k>.npz` (seed ensemble: mean BPM, `scores` = std across models); no plots under `--fold`
- aggregate CV runs: `.venv/bin/python scripts/aggregate_runs.py` (reads `cache/runs/*.json` + `cache/preds/*.npz`, logic in `src/bpm/aggregate.py`; writes `cache/results/runs.csv`, `cache/results/summary.csv`, prints LTGO top-3/winner per test group)
- results notebook: `.venv/bin/jupyter lab notebooks/results.ipynb` (test-set plots: Bland-Altman, ROC/AUROC, scatter, error breakdowns; re-execute headless with `cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace results.ipynb`)

`cache/` is gitignored, regenerated from the raw EDF data, never checked in.

# Data

`harmonized_ecg_ppg_databases/` (extracted from `harmonized_ecg_ppg_databases.zip`) holds three EDF sub-datasets; only two are used (exercise is excluded, see `README.md`). All files are plain EDF (not EDF+), 100 Hz, no annotation channel, no CSV/ground-truth files anywhere. BPM labels must be derived at training time from ECG R-peaks.

- `wrist-ppg-during-exercise-1.0.0/`: excluded from the whole pipeline (not scanned by `build_manifest`; reasons in `README.md`). 19 files, 8 subjects (`s1`-`s6`, `s8`, `s9`; `s7` has no file). Channels: `ECG`, `PPG`, `ACC_X`, `ACC_Y`, `ACC_Z`. Activities: walk/run/bike (low/high resistance), ~5-10 min per file. Only sub-dataset with accelerometer data, needed for motion-artifact handling during exercise.
- `multisite-ppg/`: 519 files, 20 subjects, 4 wearable sites (`earring`, `necklace`, `ring`, `watch`), 2 trials-ish per site per subject (filename pattern `P<id>_<site>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Free-living recordings, duration highly variable per file (~2500s to ~66000s / up to 18hr), so window-count imbalance across files/subjects is a real risk if not capped.
- `wrist-finger-ppg/`: 165 files, 27 subjects, 6 trials each (filename pattern `<subject>_<trial>.edf`). Channels: `ECG`, `PPG` only (no ACC). Short controlled clips, ~125-420s per file.

EDF header layout per the EDF spec: 256-byte main header + 256-byte-per-signal header, see `src/bpm/edf_io.py`.

Model input is the bandpass-filtered PPG channel only (`in_channels=1`). The earlier ACC zero-fill + `has_motion` flag scheme was removed together with the exercise sub-dataset.

Split: subject-wise (no subject's windows cross train/val/test), 70/15/15 within each of the two used sub-datasets (`src/bpm/split.py`, seed 42; val/test each get `max(1, round(0.15 * n))` subjects) so the test set isn't dominated by one sub-dataset. This fixed split is the default. CV runs (`--fold k`) use `cache/folds.json` instead: 5 stratified subject groups per sub-dataset (`split.assign_groups`), fold k: test = group k, val = group (k+1) % 5, train = the rest (`split.fold_split`). `split.load_folds` returns the groups plus `folds_hash` (sha256 of the file bytes, stored in each run sidecar).

# Conventions

- `src/bpm/`: library code (manifest, EDF I/O, ground truth, windowing, preprocessing, dataset/model, split, caching). `scripts/`: thin CLI entry points over it.
- TDD seams (see `tests/`): `manifest.build_manifest`, `ground_truth.ecg_to_hr_curve`, `windowing.make_windows`, `split.subject_wise_split`, `dataset.zscore_window`, `model.CNNBiLSTM.forward`, `edf_io.read_channels`, `timewarp.warp_window`/`plan_bootstrap`, `dataset.sampler_weights`, `split.assign_groups`/`fold_split`, `dataset.lds_weights`, `aggregate` (metrics, LTGO decisions). Other modules (`preprocessing`, `cache`, `data_loading`) are thin glue, verified by smoke-running against real EDF files rather than unit tests.

## Agent skills

### Issue tracker

Issues tracked as local markdown files under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout: one root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
