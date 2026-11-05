# 01 Stratified 5-fold subject groups + folds.json

Status: ready-for-agent
Spec: `.scratch/cv-code-impl/spec.md` Step 0A.1; method: `.scratch/model-run-plan/spec.md` Method 1.

## Scope
- `src/bpm/split.py`: `assign_groups(subjects_by_dataset, strat_score, k=5, seed=42)` (strat_score required,
  dict subject -> score). Per sub-dataset own `np.random.default_rng(seed)`; sort by (-score, subject_id),
  strata of k, shuffle within stratum, deal to groups 0..k-1. Returns subject -> group.
- `fold_split(groups, fold, k=5)` -> {"train","val","test"} sets: test = group fold, val = (fold+1)%k, train = rest.
- Loader for `cache/folds.json` returning groups + `folds_hash` (sha of file bytes), used by tickets 03-06.
- `scripts/make_folds.py`: scores from `cache/windows` (multisite share >=130 BPM, wrist-finger share >=100),
  writes `cache/folds.json` (subject -> group + score). Prints per-group counts, >=140-BPM windows per group, hash.
- `subject_wise_split` unchanged.

## Tests (tests/test_split.py)
Disjoint; each subject test once and val once over folds; groups per dataset; stable on ties; multisite top-k by score in different groups.

## Verify
`.venv/bin/pytest`; run `make_folds.py`: 47 subjects, multisite 4/group, wrist-finger 6/6/5/5/5,
top 5 multisite by >=130 share in 5 different groups, >=140 windows per group expect 0/18/33/21/70.
