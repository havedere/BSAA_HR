# 04 build_bootstrap.py --fold

Status: ready-for-agent
Blocked by: 01
Spec: Step 0A.3a.

## Scope
- `--fold k`: train subjects and quotas from the fold; writes `cache/bootstrap/fold<k>/{H,L}/`; stale cleanup in that dir only.
- `ppg_hr_agrees` result cached per file in `cache/ppg_ok/<file>.npy`.
- Under `--fold` only, wrist-finger also gets H bins [110,130); BINS chosen per run (not module constant).
- Log unique source subjects per bin.
- Without `--fold`: unchanged output.
- Assert bootstrap source subjects subset of fold train.

## Verify
pytest; `--help`; a fold-0 run if time allows (long: missing HR curves), else a dry check that the subject list equals fold train.
