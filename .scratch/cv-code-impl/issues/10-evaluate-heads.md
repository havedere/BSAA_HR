# 10 evaluate.py: heads, per-head ensembles, --tau, --mc

Status: done
Blocked by: 08
Spec: Step 0B (evaluate.py); run plan Methods 3-4.

## Scope
- Head inferred from the checkpoint's last-layer weight shape (1/191/2).
- Ensembles: reg = mean BPM, spread std (as today); dist = per-model logit adjustment -> softmax -> average probs -> decode; gauss = mixture mean / variance.
- `--tau 0|1` (dist only): prior from raw fold-train window counts: (Gaussian-smoothed counts σ=2 + 1) / (N + 191); tau=1 adds `_tau1` to the preds file name.
- `--mc N` (default 0): MC dropout, only `nn.Dropout` in train mode, BatchNorm eval, N passes; std of BPM -> `score_mc`.
- Preds npz keeps existing columns; `scores` = primary score (reg ensemble spread; dist curve std; gauss σ). Added: `score_spread`, `score_std`, `score_mc` (NaN where not applicable). `aggregate_runs.py` keeps working.
- Fixed-split path (no --fold) works for all heads.

## Verify
pytest; smoke on fold 0 with reg 1 and 2 ckpts (unchanged output), dist ckpt with tau 0/1, gauss ckpt, `--mc 20`.

## Notes (2026-11-07)
- `_tau1` goes into the config id (`dist_tau1_fold0_seed1.npz`, `ens_dist_tau1_fold0.npz`), not after the seed, so `aggregate.parse_tag` reads it as its own config instead of skipping the file.
- Smoke fold 0 (dist seed 1, patience 2): tau 0 val/test MAE multisite 6.40/7.16, wrist-finger 1.83/1.87; tau 1 multisite 35.6/38.3. Cause: the (counts + 1) floor gives unseen bins (150-220) log prior -10.4 vs -3.5 at the mode, so tau 1 lifts them ~7 logits; 30 % of windows decode > 20 BPM off, mostly to 150-220. Code follows the spec; tau 1 as specified is not usable on this smoke model (decision for run plan Method 3).
- reg preds unchanged vs the Step 0A version (max diff 0). `--mc 20` works for reg and gauss; MC std is large (13.7 BPM mean on reg ens) since the only dropout (p 0.3) sits right before the output layer.
- Decision 2026-11-07 (user): tau 1 dropped from the run plan; dist evaluated with tau 0 only. `--tau` stays in the code, default 0, unused.
