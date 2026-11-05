# 03 train.py: --fold, --patience, --lds-power, balanced-val early stop, sidecar

Status: ready-for-agent
Blocked by: 01, 02
Spec: Step 0A.3.

## Scope
- Keep the existing uncommitted `--seed`. Add `--patience` (default 20), `--fold k`, `--lds-power p` (default 0).
- `--fold`: split from `cache/folds.json` via ticket 01 loader; without `--fold` fixed split unchanged.
- Early stopping and `scheduler.step` on window-pooled balanced val MAE (mean of the two sub-datasets' window-pooled MAE on capped val);
  `run_epoch` returns per-window predictions for that. Log val with 4 decimals and a `saved` marker.
- LDS: weights on real train windows only (sample weights = sampler weights), synthetic bootstrap windows weight 1; loss = mean(w * per-window loss); val unweighted.
  Print clipped share, max, max/min.
- Tag `<loss>[_boot<X>][_lds<p>][_fold<k>]_seed<s>` (seed always present under --fold); checkpoint `cache/best_<tag>.pt`.
  Non-fold tags stay as today.
- Sidecar `cache/runs/<tag>.json`: config fields, fold, seed, patience, best_epoch, best_val, stop_epoch, hit_max_epochs,
  collapsed (best_epoch <= 2), git_hash, git_dirty, folds_hash.
- `--fold k --bootstrap X`: load from `cache/bootstrap/fold<k>/X/`.

## Verify
pytest; smoke `train.py --loss mse --fold 0 --seed 1 --lds-power 0.5`: sidecar complete; stop - best = patience unless hit_max_epochs;
balanced val 4 decimals. Also LDS stats at p=1 on fold-0 train (expect clipped 0.5-1.2%, max <= 20, max/min 76-89).
