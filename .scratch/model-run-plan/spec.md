# Run plan: 5-fold subject CV + LDS + distribution head + quality gate + offline smoothing

Status: FROZEN (user approval 2026-11-07). Revised after review rounds 1-3; grid cut to the story design
2026-11-07 (see "Story design"). No changes to the plan except bug/error fixes; anything else needs an explicit
user decision first.
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
  left max 25-27, review round 3). Unit-free. Expected at p = 1 on real folds (measured 2026-11-06 with
  the iterated rule): 1.1-1.5% of windows at the cap (0.8-1.0% sampler-weighted), max = 20, max/min 51-61.
  (Earlier 0.5-1.2% / 76-89 came from a single clip+renorm pass; bin edges and padding change < 0.1 pp.)
- Synthetic bootstrap windows: weight 1, outside the mean-1 constraint.
- Training loss = mean(w · per-window loss) over the sampled batch; val unweighted. `p` ∈ {0 (off), 0.5, 1}.

## Method 3: HR as distribution (classification head)
- Last layer `Linear(32, 1)` -> `Linear(32, 191)` (1-BPM bins 30..220); rest of `CNNBiLSTM` unchanged.
- Target: Gaussian centred on the label, σ = 2 BPM, normalised. Loss: `nn.CrossEntropyLoss` with probability
  targets, `reduction="none"`.
- Decode: expected value over bins within ±10 BPM of the argmax (primary); plain argmax also reported.
- ~~Logit adjustment (inference-only): `logits − τ·log(prior)`, τ ∈ {0, 1}; prior from raw fold-train window
  counts: (Gaussian-smoothed counts + 1) / (N + 191) (191 bins, 67 with zero raw count on all data).~~
  Dropped 2026-11-07: τ = 0 only, see "Decisions after Step 0B smokes".
- Uncertainty: std of the curve.

## Method 4: uncertainty scores + quality gate (abstention)
Why: multisite windows with high-variance PPG (37%) agree with their (correct) ECG label only 24%; when the PPG has
no pulse no model can recover HR.
- **Scores**, each mapped to "higher = less trustworthy": `−skewness` (Elgendi 2016), `1 − spectral prominence`
  (0.6-3.5 Hz power within ±0.1 Hz of the dominant peak), seed-ensemble spread, dist curve std, gauss σ, MC
  dropout std (only `nn.Dropout` on, BatchNorm eval, 20 passes; single dropout before the last Linear).
  gauss σ: dropped with the gauss head (story design 2026-11-07).
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
  (dist; emission changed to plain softmax 2026-11-07, see "Decisions after Step 0B smokes"): emission = softmax / prior (= τ=1 adjusted output; scaled likelihood, so the train prior is not
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
- **Decision unit:** decision = argmin of the ranking key over a stated candidate set, then the tie set and
  tie-break. **Story design (2026-11-07): one winner per step from the global decision** (all val folds pooled),
  so every stage trains one config line; the 5 LTGO decisions are reported next to it as a check (disagreement
  = selection uncertainty, gap = selection optimism).
  - Stage A: candidates Huber δ10, MAE, MSE (basic) -> L* (best loss).
  - Stage B: L* + HL × p {0, 0.5, 1} -> p_B (if a p = 0 config is in the tie set, p_B = 0, no tie-break).
  - Stage C: dist + HL × p {0, 0.5, 1} -> p_C (same rule).
  - Final: argmin of {Stage B winner, Stage C winner} -> head. D/E/F run on both winners; smoother and gate
    score: argmin over their candidates.
- **Loss:** loss_k is the loss of decision k's argmin config; reported "indistinguishable" if every loss occurs
  among the tie-set configs.
- **Ranking key:** per-subject val MAE, **median over the 3 seeds** (a collapsed seed cannot move it), paired vs
  the argmin config; key = mean over the 2 sub-datasets of the per-dataset mean paired difference; SE by
  bootstrap over subjects stratified by sub-dataset (cells with < 3 usable seeds: median of what exists,
  flagged). Tie set = configs within 1 SE of the argmin; tie-break: lowest bin-balanced val MAE (pooled val of
  the allowed folds), reported with seed spread and subject counts.
- **Tests vs baseline** (MSE, basic, LDS off; was Huber δ5 before the story design): paired per-subject OOF test MAE, Wilcoxon signed-rank
  **per sub-dataset** (multisite 20, wrist-finger 27 subjects), Holm across configs. CIs: subject bootstrap.
- **OOF value:** per seed (then mean ± std over seeds) and for the ensemble; both reported.
- **Collapsed runs** (best epoch <= 2): not rerun; excluded from decisions only through the seed median above,
  kept in all reports; collapse rate per config reported; medians reported next to means.
- **Final headline** (decided): LTGO-selected config -> 3-seed ensemble -> Method 5 pipeline; each layer
  reported separately: single model (mean over seeds), + ensemble, + input gate, + smoother, + output gate.

## Step 1: archive old results, build per-fold bootstrap
- Done 2026-11-07: old results first archived, then deleted on user request (with the fold-0 smoke artefacts and the fixed-split bootstrap). Original step: move `cache/best_*.pt`, `cache/logs/`, `cache/plots/` to `cache/archive_fixedsplit/` (~15 MB, source of the
  Context numbers). `cache/best.pt` dropped 2026-11-07: every run writes `best_<tag>.pt`, `evaluate.py --ckpt` is required; only the notebooks still refer to it (fixed later).
- `build_bootstrap.py --fold k` for k = 0..4: computes the 251 missing HR curves (5,259 h incl. P3; ~10 min at
  the measured rate) as their subjects enter train, plus the per-file PPG check (~15 min total, cached).
  HR curves are complete only after all 5 folds. Budget 30-45 min.

## Story design (2026-11-07, user decision)
Uni project: one red thread through the training, each stage answers one question. 135 runs, ~5.5 h GPU
(was ~480-540 runs, 20-30 h). Every config: 5 folds × 3 seeds {42, 1, 2}.

| Stage | Question | Configs | Runs |
|---|---|---|---|
| A basic | which loss? | Huber δ10, MAE, MSE; no bootstrap, LDS off, reg | 45 |
| problem | pull to mid values | none (per-bin bias and pred-vs-truth slope from Stage A preds) | 0 |
| B fix: data + loss weights | does bootstrap / LDS close the gap? | L* + HL × p {0, 0.5, 1} | 45 |
| C fix: output | does a distribution head help? | dist + HL × p {0, 0.5, 1} | 45 |
| D-F reliability | ensemble, gate, smoothing | Stage B and C winners | 0 |

- Dropped: Huber δ5 and δ20, bootstrap L-only / H-only, the gauss head (training and gauss σ score).
- p = 0 in B and C isolates the bootstrap effect, so the story keeps its attribution: basic -> +bootstrap HL
  -> +LDS (B) -> dist head (C).
- Greedy by design: the best loss without bootstrap may not be the best with it; stated as a limitation.
- Baseline for the tests: MSE basic (`aggregate.BASELINE = "mse"`).

## Step 2: Stage A — basic losses (45 runs, ~2 h)
3 configs (Huber δ10, MAE, MSE; basic) × 5 folds × 3 seeds, LDS off, head reg.
Measured basis: ~2.5 s/epoch + ~25 s overhead; patience 20 -> ~35-65 epochs -> 2-3 min per run incl. evaluate.
Runner: throwaway bash in scratchpad; train then evaluate `--fold`; continue past failures; one status line per
run. Order: seed-major, then fold, then config.

## Step 3: report Stage A -> L*; the problem (per-bin bias, pred-vs-truth slope, error in the 120s bin).

## Step 4: Stage B — L* + bootstrap HL × LDS p {0, 0.5, 1} (45 runs, ~2 h) -> p_B.

## Step 5: Stage C — dist head + bootstrap HL × LDS p {0, 0.5, 1} (45 runs, ~2 h) -> p_C
`--head dist` ignores `--loss`. dist evaluated with τ = 0 only (τ = 1 dropped 2026-11-07).

## Step 6: Stage D — seed ensemble (minutes)
3-seed ensemble of the Stage B and Stage C winners, per fold.

## Step 7: Stage E — quality gate (minutes)
`gate_report.py` on Stage D's ensembles; MC dropout (evaluation only) on these checkpoints.

## Step 8: Stage F — offline smoothing
Per fold: final config's ensemble (and the other winner for comparison); Method 5 pipeline. GPU ~97k windows/s (measured in review); EDF reading not
measured yet (smoke test).

## Verification (runs)
- After stages: sidecar counts match the stage formulas, all rc = 0, collapse rate reported.
- Code-level checks: see `.scratch/cv-code-impl/spec.md`.

## Open for user
- Resolved 2026-11-07 by the story design: grid cut to 135 runs, gauss head dropped. MC dropout kept as an
  evaluation-only score (no training cost).
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

## Decisions after Step 0B smokes (2026-11-07)
- **HMM emission = model softmax, no prior division** (supersedes "emission = softmax / prior" in Method 5).
  The label prior (Gaussian-smoothed fold-train counts + 1, normalised) gives never-seen bins (150-220 BPM)
  log prior -10.4 vs -3.5 at the mode; dividing by it lifts those bins ~7 logits at every window, and the
  transition model cannot pull the path back. Fold-0 smoke (dist, patience 2), val subject-macro MAE for
  m = 1/2/4: with prior 14.29 / 13.24 / 9.15, without 3.68 / 3.53 / 3.52 (Kalman 3.52). Cost: the train prior
  is multiplied in once per step (the reason for the scaled likelihood); accepted, since the transition model
  is the main effect.
- **τ = 1 logit adjustment dropped** (Method 3, Stage C): dist is evaluated with τ = 0 only, so no label prior
  is used anywhere in the pipeline. Same floor problem as the HMM: fold-0 smoke multisite val MAE 6.4 -> 35.6,
  30% of windows > 20 BPM off, mostly decoded to 150-220 BPM. Rare-BPM bias is left to bootstrap and LDS.
  `evaluate.py --tau` stays in the code (default 0), unused.
- **Architecture frozen (2026-11-07, user decision).** Model (CNN-BiLSTM, reg / dist / gauss heads), training,
  CV/selection, quality gate and smoothing pipeline stay as implemented in Steps 0A + 0B. No changes except
  bug fixes and error fixes; anything else needs an explicit user decision first.

Plan acknowledged and frozen by the user 2026-11-07; training runs may start. No push.
