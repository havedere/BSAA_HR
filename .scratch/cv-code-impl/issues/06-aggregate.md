# 06 aggregate_runs.py

Status: done
Blocked by: 03, 05
Spec: Step 0A.5; method: model-run-plan "Metrics" + "Selection, statistics, reporting".

## Scope
Reads `cache/runs/*.json` + `cache/preds/*.npz`; per-subject errors; ranking key (seed median, paired vs argmin,
mean over 2 sub-datasets, subject-bootstrap SE stratified by dataset, <3 seeds flagged); tie set within 1 SE; tie-break bin-balanced MAE;
LTGO decisions per k (only val of folds j != k-1) and global decision; Wilcoxon per sub-dataset vs baseline (Huber d5 basic, LDS off) + Holm;
subject-bootstrap CIs; collapse rate. Writes `cache/results/runs.csv`, `cache/results/summary.csv`.
Pure logic in an importable module so it is testable.

## Tests
Tiny synthetic preds set, incl. LTGO check: group k's val rows never influence decision k.
