# 05 evaluate.py --fold, multi-ckpt ensemble, preds npz

Status: ready-for-agent
Blocked by: 01, 03
Spec: Step 0A.4.

## Scope
- `--fold k`: predict fold val and test subjects.
- `--ckpt` accepts several paths: ensemble = mean BPM, score = std across models.
- Write `cache/preds/<tag>.npz` (single) or `cache/preds/ens_<config id>_fold<k>.npz` (config id = tag without `_fold<k>_seed<s>`),
  fields: subject, dataset_source, split, truth, pred, scores.
- No Bland-Altman plots under `--fold`. Without `--fold` behaviour unchanged.

## Verify
`evaluate.py --fold 0` with 1 and 2 ckpts (from ticket 03 smoke) writes val+test rows.
