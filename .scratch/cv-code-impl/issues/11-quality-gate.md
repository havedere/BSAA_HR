# 11 quality.py + gate_report.py

Status: done
Blocked by: 10
Spec: Step 0B; run plan Method 4.

## Scope
- `src/bpm/quality.py`: `skewness(windows)` -> score `-skew`; `spectral_prominence(windows, fs=100)` (0.6-3.5 Hz power within ±0.1 Hz of the dominant peak / total band power) -> score `1 - prominence`; `fit_limit(score, err, subject, dataset, mode="coverage"|"mae", coverage=0.9, max_mae=5.0)` -> (limit, info incl. infeasible flag; mae mode falls back to coverage limit when infeasible; all pass -> max score); `gate(score, limit)` -> ok; `risk_coverage(score, err)` + `aurc`; `spearman_ci(score, err, subject, n_boot)` subject bootstrap.
- Subject-macro balanced MAE on passed windows; subjects with 0 passed windows left out and counted.
- `scripts/gate_report.py --preds <npz> [...]`: per fold, signal scores from `cache/windows`, model scores from preds; limit fitted on val rows, applied to test; reports risk-coverage/AURC, MAE at 100/90/80/70% coverage, passed vs suppressed MAE, coverage per bin and per sub-dataset; writes `cache/results/gate.csv`.

## Verify (TDD for quality.py)
Synthetic: known coverage limit; infeasible -> fallback; all-pass -> max; zero-passed subject excluded and counted; AURC of perfect score < random; prominence high for pure sine. Smoke gate_report on fold-0 preds: ~90% val coverage.

## Notes (2026-11-07)
- Every score is "higher = worse" (pass when score <= limit): skew score = -skewness, prominence score = 1 - prominence.
- `fit_limit` mae mode searches a 199-point coverage-quantile grid (shortcut comment in `quality.py`).
- `gate_report.py` takes the model scores `score_spread` / `score_std` / `score_mc` when finite; signal scores are recomputed from `cache/windows/<subject>.npz` in preds row order (asserted against `truth`). `--n-boot` (default 200) for the Spearman CI.
- Smoke fold 0 (mse ens of 2, dist seed 1, gauss seed 1): val coverage 0.900 in coverage mode for every score; suppressed MAE > passed MAE in all 22 rows. Best ranking: dist `score_std` (rho 0.70, test AURC 2.13 vs all-window MAE 4.51); gauss `score_std` rho -0.02 (smoke model, patience 2).
