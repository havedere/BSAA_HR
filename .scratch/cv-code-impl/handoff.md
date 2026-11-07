# Handoff: CV run plan, Step 0A done (2026-11-06)

> **Superseded 2026-11-07:** Steps 0A, 0B and 1 are done, the old fixed-split results and `best.pt` are deleted, and the run plan is frozen to the story design (135 runs, see `.scratch/model-run-plan/spec.md`, "Story design"). Next: Stage A.

Branch `feature/cv-run-plan` (off master, not pushed). Specs: `.scratch/cv-code-impl/spec.md` (code),
`.scratch/model-run-plan/spec.md` (runs, metrics, selection). Tickets: `.scratch/cv-code-impl/issues/01..07`, all `Status: done`.

## Done
Step 0A of the code spec, one commit per ticket, `.venv/bin/pytest` green (49 tests):

| Commit | Ticket | Result |
|---|---|---|
| 533e603 | 01 folds (`assign_groups`, `fold_split`, `load_folds`, `scripts/make_folds.py`) | `cache/folds.json` built; 47 subjects, multisite 4/group, wrist-finger 6/6/5/5/5; >=140 BPM windows per group 0/18/33/21/70 (as expected); folds_hash `a37b90f7...` |
| 2da5bde | 02 `dataset.lds_weights`, `WindowDataset(weights=)` | 6 tests |
| efe67c6 | 03 `train.py` `--seed/--patience/--fold/--lds-power`, balanced-val early stop, sidecar `cache/runs/<tag>.json` | smoke fold 0 ok (stop - best = 20) |
| d5cd602 | 04 `build_bootstrap.py --fold k` -> `cache/bootstrap/fold<k>/`, `cache/ppg_ok/` cache | fold 0 built (~1 h CPU): H 4044, L 1303 windows, no short bins, leakage assert held |
| ca6be93 | 05 `evaluate.py --fold k --ckpt a [b ...]` -> `cache/preds/*.npz` | 1- and 2-ckpt smoke ok |
| bc9951f | 06 `src/bpm/aggregate.py` + `scripts/aggregate_runs.py` | 11 tests incl. LTGO leak check |
| 8578c23, 794f03b | 07 docs (CLAUDE.md, README, pipeline-decisions stage 13), tickets marked done | |

The previously uncommitted `--seed` change in `train.py` is in efe67c6.
Left untouched (user's own WIP): `notebooks/results.ipynb`, `docs/hr_cnn_bilstm_results.odp`,
`notebooks/data_analysis.ipynb`, `.scratch/ecg-label-doubling/`.

## Open decisions (for the user)
1. **LDS stats off spec.** p=1, fold-0 train, sampler-weighted: clipped 1.46% (spec 0.5-1.2%), max 20.00 (ok),
   max/min 56.1 (spec 76-89). Unweighted, folds 0-4: clipped 0.42-1.11%, max/min 59-69. Possible causes:
   spec numbers computed differently, or edge handling (labels outside 30..220 go to edge bins; Gaussian smoothing
   `mode="nearest"`). Not investigated.
2. **Non-fold runs changed too.** Early stop/LR scheduler now on balanced val MAE with patience 20 for every run,
   including the fixed split. Old fixed-split runs used overall val MAE, patience 10. Keep or gate back?
3. **Aggregation choices made where the spec was silent** (confirm or change):
   - tie-set SE = each config's own paired-difference SE vs the argmin;
   - Wilcoxon / CIs combine seeds by mean of per-subject test MAE (ranking key uses the median);
   - CIs reported for the config value and for the difference to baseline;
   - Stage B restricted candidate set and "p = 0 in tie set wins" rule not in the CLI (only `decide(candidates=)`);
   - configs missing subjects are paired on shared subjects (check `n_subj`);
   - RMSE and per-bin MAE not yet in the CSVs (reported-only metrics).

## Cache state (gitignored)
- Smoke artefacts, **delete before Stage A** (`aggregate_runs.py` uses every config it finds):
  `cache/best_mse_lds0.5_fold0_seed{1,2}.pt`, `cache/runs/mse_lds0.5_fold0_seed{1,2}.json`,
  `cache/preds/mse_lds0.5_fold0_seed1.npz`, `cache/preds/ens_mse_lds0.5_fold0.npz`, `cache/results/`,
  `cache/logs/smoke_fold0_seed{1,2}.log`.
- Kept: `cache/folds.json`, `cache/bootstrap/fold0/`, `cache/ppg_ok/` (362 files), new HR curves in `cache/hr_curves/`.
- `cache/ppg_ok/` must be deleted together with `cache/hr_curves/` when ground-truth code changes.

## Next steps
1. Resolve the open decisions above.
2. Run plan Step 1: move `cache/best_*.pt`, `cache/logs/`, `cache/plots/` to `cache/archive_fixedsplit/`
   (`cache/best.pt` stays); then `build_bootstrap.py --fold k` for k = 1..4. Budget: fold 0 alone took ~1 h,
   not the 30-45 min total in the plan (later folds reuse cached HR curves / PPG checks, so they should be faster).
3. Stage A runner (throwaway bash in scratchpad): 20 configs x 5 folds x 3 seeds, train then `evaluate.py --fold`,
   then `aggregate_runs.py`.
4. Step 0B (heads, gate, smoothing) only after the Stage A/B report.

## Not verified
- Default (non-fold) `evaluate.py` and `build_bootstrap.py` not run, only read.
- `aggregate_runs.py` memory at ~300 runs (~6M rows in one table, likely a few GB), not measured.
