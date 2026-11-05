# Run plan: 5-fold subject CV + LDS + distribution head + quality gate + offline smoothing

Status: PRELIMINARY (2026-11-05), revised after review rounds 1-3. Not approved for execution.
No code changes, deletions or training runs until the user explicitly acknowledges a final version.
Code needed for these runs: `.scratch/cv-code-impl/spec.md` (Step 0A before Stage A, Step 0B before Stage C).

## Context
- Seed noise: 6 MSE seeds give combined test MAE 6.08 ± 0.79 (5.21-7.47), wider than the whole 20-config table.
- Split noise: test and val each hold only 3 multisite subjects (seed-42 split). Label share >=100 BPM:
  train 6.5%, val 5.2%, test 19.1%; >=140 BPM: train 54 windows, val 0, test 88. Val cannot see the tail that
  dominates test error, and test MAE hinges on which 3 subjects land there.
- The tail is a few subjects: all 142 windows >=140 BPM come from P1 (21), P3 (18), P17 (70), P18 (33);
  40-50 BPM: 138 windows, mostly P18 (117). Highest wrist-finger label: 113.3 BPM (subjects 12, 07).
- P3 alone is 4,466 of 9,874 multisite recording hours.
- Regression to the mean: 59% of train windows are 70-89 BPM; every model errs 33-38 BPM in the 120s bin.
- Early stopping (patience 10) vs LR scheduler (patience 5) allows one LR halving; 2 runs collapsed (best epoch
  <= 1, identical val 5.40), bootstrap runs often peak late (epoch 30-32).
Goal: a result that does not hinge on 3 test subjects or one seed, plus a direct fix attempt for the high-BPM tail.

## Metrics
- **Early stopping + LR scheduler:** window-pooled balanced val MAE = mean of the two sub-datasets'
  window-pooled MAEs on the fold's capped val (`val_data["dataset_source"]`, `scripts/train.py:84`).
- **Selection:** subject-macro balanced MAE = mean over the 2 sub-datasets of the mean per-subject MAE.
  Per-subject MAE = MAE of one subject's windows (each subject belongs to one sub-dataset).
- **Tie-break only:** bin-balanced MAE = window-pooled MAE per bin, windows of both sub-datasets pooled,
  then the unweighted mean over bins with >= 20 windows.
- **Reported, not used to select:** window-pooled balanced MAE, RMSE, per-bin MAE, per-dataset MAE.
- **Bins** (exact): [40,60), [60,70), [70,80), [80,90), [90,100), [100,110), [110,120), [120,130), [130,∞).
  Reports show windows and number of subjects per bin.

## Method 1: subject-wise 5-fold CV, rotating val (option A), stratified
- Per sub-dataset, own `np.random.default_rng(42)`. Stratification score (from `cache/windows`): multisite =
  share of the subject's windows >= 130 BPM; wrist-finger = share >= 100 BPM (no wrist-finger window reaches
  130). Sort by (−score, subject_id), cut into strata of 5, shuffle within each stratum with the rng, deal
  members to groups 0-4. Multisite 20 -> 4 per group; wrist-finger 27 -> 6/6/5/5/5.
- **Folds are computed once** by `scripts/make_folds.py` -> `cache/folds.json` (subject -> group + the scores
  used). Every script reads that file; sidecars store its hash. Rebuilding `cache/windows` does not reshuffle.
- Verified (review round 3, exact simulation): multisite top 5 = P17, P1, P3, P18, P7, each in a different group;
  >=140-BPM windows per group 0/18/33/21/70.
- **Remaining tail limits (documented, not fixable with 4 subjects):** val / train windows >=140 BPM per fold:
  f0 18/124, f1 33/91, f2 21/88, f3 70/51, f4 0/72 (fold 4 val has only 15 windows >=130); fold 0's test group has
  no window >=140. Per-fold tail numbers are reported, and tail conclusions rest on the pooled OOF only.
- Fold k: test = group k, val = group (k+1) mod 5, train = other 3 groups. Every subject is test once and val once.
  Per fold: multisite 12 train / 4 val / 4 test; wrist-finger ~16 / 5-6 / 5-6.
- The existing fixed split (`subject_wise_split`) stays the default when `--fold` is not given.
- **Fitted quantities use only the fold's train subjects:** bootstrap quotas and windows (per fold), LDS density,
  logit-adjustment prior, smoother q / σ_T. Gate limits and smoother parameters use the fold's val.

## Method 2: LDS loss weighting (Yang et al., ICML 2021, "Delving into Deep Imbalanced Regression")
- Density: histogram of the fold's real train labels in 1-BPM bins over 30..220 (same bins as the dist head),
  each window counted with its sampler weight, convolved with a Gaussian (σ = 2 BPM).
- Weights (real windows only): w = density^-p -> divide by its sampler-weighted mean -> repeat {clip at 20,
  renormalise to sampler-weighted mean 1} until max <= 20 (converges in a few iterations; a single clip+renorm
  left max 25-27, review round 3). Unit-free. Expected at p = 1 on real folds: 0.5-1.2% of windows clipped,
  max/min 76-89.
- Synthetic bootstrap windows: weight 1, outside the mean-1 constraint.
- Training loss = mean(w · per-window loss) over the sampled batch; val unweighted. `p` ∈ {0 (off), 0.5, 1}.

## Method 3: HR as distribution (classification head)
- Last layer `Linear(32, 1)` -> `Linear(32, 191)` (1-BPM bins 30..220); rest of `CNNBiLSTM` unchanged.
- Target: Gaussian centred on the label, σ = 2 BPM, normalised. Loss: `nn.CrossEntropyLoss` with probability
  targets, `reduction="none"`.
- Decode: expected value over bins within ±10 BPM of the argmax (primary); plain argmax also reported.
- Logit adjustment (inference-only): `logits − τ·log(prior)`, τ ∈ {0, 1}; prior from raw fold-train window
  counts: (Gaussian-smoothed counts + 1) / (N + 191) (191 bins, 67 with zero raw count on all data).
- Uncertainty: std of the curve.

## Method 4: uncertainty scores + quality gate (abstention)
Why: multisite windows with high-variance PPG (37%) agree with their (correct) ECG label only 24%; when the PPG has
no pulse no model can recover HR.
- **Scores**, each mapped to "higher = less trustworthy": `−skewness` (Elgendi 2016), `1 − spectral prominence`
  (0.6-3.5 Hz power within ±0.1 Hz of the dominant peak), seed-ensemble spread, dist curve std, gauss σ, MC
  dropout std (only `nn.Dropout` on, BatchNorm eval, 20 passes; single dropout before the last Linear).
- **Gate**: (`bpm`, `score`, `ok = score <= limit`); `ok = false` -> "value too uncertain"; raw `bpm` saved.
- **Limit fitted on the fold's val, applied to its test.** Operating points: 90% val coverage; largest limit with
  passed val subject-macro balanced MAE <= 5 BPM (none qualifies -> "infeasible", fall back to the coverage
  limit; all pass -> limit = max val score). Subjects with 0 passed windows are left out of the macro; their
  count is reported.
- **Checks:** calibration Spearman ρ(score, |error|) on val > 0 with subject-bootstrap 95% CI excluding 0;
  coverage per bin reported.
- **Report:** risk-coverage curve over all coverages + AURC; MAE at 100/90/80/70%; passed vs suppressed MAE;
  coverage per bin; per sub-dataset.

## Method 5: offline temporal smoothing (two-pass)
- Training unchanged. The model slides a 10 s window every 2 s; **scope: within ±5 min of labelled windows**
  (6.6% of multisite windows); reset across gaps > 30 s (flat PPG or outside scope).
- Smoothing runs on the **seed ensemble** of the chosen config, per fold: val group for tuning, test for scoring.
  Labels from `cache/hr_curves/*.npy` (complete only after all 5 bootstrap builds, Step 1).
- **Pipeline order (fixed):** ensemble -> input gate (Method 4 limit, from capped val) -> smoother (gated windows
  get R = ∞ / flat emission) -> output gate on the smoother's posterior std (Kalman posterior variance, HMM
  posterior std; limit refitted on val sequence windows, same two operating points).
- **Smoothers (all tried, chosen with the selection rule):** centred median (k windows); Kalman + RTS
  (`R_t = (c · s_t)²`, `s_t` = model score; reg head without score: constant R = c²); HMM forward-backward
  (dist): emission = softmax / prior (= τ=1 adjusted output; scaled likelihood, so the train prior is not
  multiplied in at every step), Gaussian transition σ_T per step with a uniform floor 1e-3, banded at
  ±5·σ_T·m (vectorised), floor added outside the band.
- **q / σ_T** from fold-train 10 s-mean labels at 2 s step (MAD·1.4826; measured multisite 0.94 BPM,
  wrist-finger 1.08) times a multiplier m ∈ {1, 2, 4} tuned on val (heavy tails: p99 |Δ| = 4.4σ). Also tuned
  on val: c, k, max gated run before "too uncertain" (30 s).
- **Scoring:** all labelled windows of the test recordings (uncapped); raw recomputed on the same windows; raw vs
  each smoother, with/without gates: subject-macro balanced MAE, window-pooled MAE, per-bin MAE, coverage, lag
  (error in the 30 s after the largest HR jumps).

## Selection, statistics, reporting (fixed before any run)
- **Leave-test-group-out (LTGO) selection.** For test group k, every decision uses only val errors of folds
  j ≠ k-1 (fold j's val is group j+1, so group k's val errors never enter decision k; verified in review
  round 3). Decisions are made 5 times (one per k); stages run the **union** of the per-k candidate lists.
  Residual, documented: models used in decision k were trained on group k (indirect only).
- **Decision unit:** decision k = argmin of the ranking key over a stated candidate set C_k, then the tie set and
  tie-break. The chosen config's components define "best": loss_k, boot_k, p_k, head_k.
  - Stage A: C_k = all 20 configs -> top-3_k = the 3 lowest keys.
  - Stage B: C_k = top-3_k × p {0, 0.5, 1} -> p_k (if any p = 0 config is in the tie set, p_k = 0, no tie-break).
  - Stage C: C_k = Stage-B winner_k plus the dist/gauss runs with (boot_k, p_k) -> head_k.
  - D/E/F: top 2 of the Stage C candidate set by key; smoother and gate score: argmin over their candidates.
- **Global selection** (all val pooled) also reported, labelled optimistic; gap to LTGO = selection optimism.
- **Loss:** loss_k is the loss of decision k's argmin config; reported "indistinguishable" if every loss occurs
  among the tie-set configs.
- **Ranking key:** per-subject val MAE, **median over the 3 seeds** (a collapsed seed cannot move it), paired vs
  the argmin config; key = mean over the 2 sub-datasets of the per-dataset mean paired difference; SE by
  bootstrap over subjects stratified by sub-dataset (cells with < 3 usable seeds: median of what exists,
  flagged). Tie set = configs within 1 SE of the argmin; tie-break: lowest bin-balanced val MAE (pooled val of
  the allowed folds), reported with seed spread and subject counts.
- **Tests vs baseline** (Huber δ5, basic, LDS off): paired per-subject OOF test MAE, Wilcoxon signed-rank
  **per sub-dataset** (multisite 20, wrist-finger 27 subjects), Holm across configs. CIs: subject bootstrap.
- **OOF value:** per seed (then mean ± std over seeds) and for the ensemble; both reported.
- **Collapsed runs** (best epoch <= 2): not rerun; excluded from decisions only through the seed median above,
  kept in all reports; collapse rate per config reported; medians reported next to means.
- **Final headline** (decided): LTGO-selected config -> 3-seed ensemble -> Method 5 pipeline; each layer
  reported separately: single model (mean over seeds), + ensemble, + input gate, + smoother, + output gate.

## Step 1: archive old results, build per-fold bootstrap
- Move `cache/best_*.pt`, `cache/logs/`, `cache/plots/` to `cache/archive_fixedsplit/` (~15 MB, source of the
  Context numbers). `cache/best.pt` stays in place (notebook and default `evaluate.py` still load it).
- `build_bootstrap.py --fold k` for k = 0..4: computes the 251 missing HR curves (5,259 h incl. P3; ~10 min at
  the measured rate) as their subjects enter train, plus the per-file PPG check (~15 min total, cached).
  HR curves are complete only after all 5 folds. Budget 30-45 min.

## Step 2: Stage A — loss × bootstrap under CV (~300 runs, ~11-15 h)
20 configs (Huber δ5/10/20, MAE, MSE × basic/+L/+H/+HL) × 5 folds × 3 seeds {42, 1, 2}, LDS off, head reg.
Measured basis: ~2.5 s/epoch + ~25 s overhead; patience 20 -> ~35-65 epochs -> 2-3 min per run incl. evaluate.
Runner: throwaway bash in scratchpad; train then evaluate `--fold`; continue past failures; one status line per
run. Order: seed-major, then fold, then config.

## Step 3: report Stage A; per-k top-3 lists (LTGO); Stage B runs their union U.
Pre-registered cap: if |U| > 5, run the 5 configs with the most per-k votes (ties: lower global key).

## Step 4: Stage B — LDS (30 × |U| runs: 90-150, ~4-7 h with the cap)
Union U × p {0.5, 1} × 5 folds × 3 seeds, paired to Stage A (same seeds). p_k per decision rule above.

## Step 5: Stage C — dist and gauss heads [after Step 0B]
2 heads × (union over k of {(boot_k, p_k)} ∪ {(basic, 0)}) × 5 folds × 3 seeds = 30-360 runs (~1.5-18 h);
pre-registered cap: at most 3 (boot, p) pairs, by per-k votes. dist evaluated with τ ∈ {0, 1}.

## Step 6: Stage D — seed ensemble (minutes)
Per k: top 2 configs of the Stage C candidate set by key; union run.

## Step 7: Stage E — quality gate (minutes)
`gate_report.py` on Stage D's configs; MC dropout only on these checkpoints.

## Step 8: Stage F — offline smoothing
Per k: chosen config's ensemble; Method 5 pipeline. GPU ~97k windows/s (measured in review); EDF reading not
measured yet (smoke test).

## Verification (runs)
- After stages: sidecar counts match the stage formulas, all rc = 0, collapse rate reported.
- Code-level checks: see `.scratch/cv-code-impl/spec.md`.

## Open for user
- Reviewer suggestion (round 1): drop Huber δ10/δ20, the gauss head and MC dropout (fixed-split val spread of
  the Huber variants 0.12 BPM). Kept for now: the user set the 20-config grid.
- Changes from review rounds 2-3 that alter earlier decisions: collapsed runs are not rerun (seed median in
  decisions instead); Step 0 is split so heads/gate/smoothing are coded only after Stage A/B results;
  wrist-finger gets H bins [110,130) under CV; run caps for Stages B and C.

## Deferred, documented for context (judged low value for now, 2026-11-05)
- **Extra augmentation / extra input channel** (#7): baseline wander, injected motion artifacts, cutout, a
  derivative (VPG) or spectral input channel. Modest expected gain; the main errors are label-imbalance and
  pulse-less PPG, which augmentation does not address. Revisit after CV results exist to compare against.
- **Bigger model**: only 33 train subjects (12 multisite per CV fold); capacity is not the bottleneck, data is.
  155k params already fit wrist-finger to ~1.2 BPM MAE.
- **LR / batch-size / architecture tuning**: meaningful only once comparisons are reliable (needs CV + seeds
  first); expected gains are smaller than the seed noise measured so far (±0.79 BPM).
- **Pretrained PPG foundation models** (e.g. PaPaGei, Pulse-PPG): new dependencies and weights, different input
  rates/lengths; out of scope unless the from-scratch model plateaus.
- **Viterbi smoother**: dropped in review round 1 (variant of HMM forward-backward).
- **Full nested CV** (inner CV per outer fold): would remove the residual LTGO leak; ~4× the runs.

No training run starts before the user acknowledges this plan; no push.
