# Code implementation plan: CV, LDS, heads, quality gate, smoothing

Status: PRELIMINARY (2026-11-05), revised after review rounds 1-3. Not approved for execution.
Progress: Step 0A code done and committed (2026-11-06). Step 0B code done, uncommitted (2026-11-07, tickets 08-15; the (counts + 1) label prior broke tau 1 and the HMM emission; both dropped 2026-11-07, see run plan "Decisions after Step 0B smokes").
No code changes, deletions or training runs until the user explicitly acknowledges a final version.
Method definitions, metrics and selection rules: `.scratch/model-run-plan/spec.md` (this plan implements them).
Order: Step 0A (needed for Stages A-B) now; Step 0B only after the Stage A/B report.
Branch `feature/cv-run-plan` off master; commit only on user go-ahead; no push.

## Step 0A: code for Stages A-B (branch `feature/cv-run-plan` off master; commit on user go-ahead)
1. `src/bpm/split.py`: `assign_groups(subjects_by_dataset, strat_score, k=5, seed=42)` (strat_score required)
   and `fold_split(groups, fold, k=5)` -> {"train","val","test"}. `scripts/make_folds.py` writes
   `cache/folds.json`; `train.py`, `evaluate.py`, `build_bootstrap.py`, `aggregate_runs.py` read it.
   Tests: disjoint; each subject test once and val once; per-dataset groups; stable order on ties; multisite
   top-k by score in different groups.
2. `src/bpm/dataset.py`: `lds_weights(labels, power, sample_weights, sigma=2.0, clip=20.0)` (order above).
   Tests: uniform -> 1; rare label larger; max <= 20 after the final renormalisation; sampler-weighted mean 1;
   p = 0 -> 1; unit-free (scaling the density does not change weights). `WindowDataset` optionally returns a
   weight. `sampler_weights` unchanged (round 3: a per-letter split would give +H and +HL different H doses).
3. `scripts/train.py`: keep uncommitted `--seed`; add `--patience` (default 20), `--fold k`, `--lds-power p`.
   Early stopping and `scheduler.step` on window-pooled balanced val MAE; `run_epoch` returns per-window
   predictions for that. Log val with 4 decimals and a `saved` marker. JSON sidecar `cache/runs/<tag>.json`
   {config fields, fold, seed, patience, best_epoch, best_val, stop_epoch, hit_max_epochs, collapsed,
   git_hash, git_dirty, folds_hash}; the runner adds `rc` (a crashed run has no sidecar from train.py, so the
   runner writes a stub with rc). Tag: `<loss>[_boot<X>][_lds<p>][_fold<k>]_seed<s>` (seed always under
   `--fold`); checkpoint `cache/best_<tag>.pt`; config id = tag without `_fold<k>_seed<s>`.
   With `--fold k --bootstrap X`: load `cache/bootstrap/fold<k>/X/`.
3a. `scripts/build_bootstrap.py --fold k`: train subjects and quotas from the fold; writes
   `cache/bootstrap/fold<k>/{H,L}/`; stale cleanup in that dir; `ppg_hr_agrees` cached per file in
   `cache/ppg_ok/<file>.npy`. Under `--fold` only, wrist-finger also gets H bins [110,130) (labels reach
   113 BPM; the old full exclusion was based on the fixed test set) - `BINS` branches on the fold
   (`scripts/build_bootstrap.py:29` is a module constant today). Adds ~1,000 wrist-finger H windows, so
   wrist-finger MAE under +H is expected to shift. Logs unique source subjects per bin (P3 dominates multisite
   hours). Without `--fold`: unchanged.
4. `scripts/evaluate.py`: `--fold k`; predicts the fold's val and test subjects; `--ckpt` accepts several paths
   (reg ensemble = mean BPM, spread = std); preds `cache/preds/<tag>.npz` or
   `cache/preds/ens_<config id>_fold<k>.npz` (subject, dataset_source, split, truth, pred, scores).
   No Bland-Altman plots under `--fold` (would be ~1,200 PNGs in Stage A); plots only for final configs.
5. `scripts/aggregate_runs.py`: reads sidecars + preds; per-subject errors, LTGO and global decisions per k,
   ranking key, tie-break, Wilcoxon per sub-dataset + Holm, subject bootstrap, collapse rate; writes
   `cache/results/runs.csv`, `cache/results/summary.csv`. Test on a tiny synthetic preds set (incl. an LTGO
   check: group k's val rows never influence decision k).
6. Docs: `CLAUDE.md`, `README.md` (also the archived `best_mae.pt`, `best_huber10.pt`, `cache/plots/best_mae/`
   paths -> `cache/archive_fixedsplit/`), `docs/agents/pipeline-decisions.md` (CV + stratification, folds.json,
   metrics, patience, LDS). `docs/model-and-training.md` left as is (fixed later with deck/notebook; its old
   paths noted in the "fixed later" list).

## Step 0B: code for Stages C-F (implemented after the Stage A/B report, same branch)
- `src/bpm/model.py`: `CNNBiLSTM(head="reg"|"dist"|"gauss")`, `gaussian_targets`, `decode_bpm(logits, window=10,
  log_prior=None, tau=0)`. Tests: shapes; target sums to 1 and peaks at label; decode of a target = label; with
  the prior floor no inf/NaN; τ = 0 equals plain decode.
- `scripts/train.py --head dist|gauss` (tag part `dist` / `gauss`); gauss loss `GaussianNLLLoss(mean, target,
  var=exp(logvar))` with β-NLL (β = 0.5, Seitzer et al. 2022).
- `scripts/evaluate.py`: head from the last layer's shape; dist ensemble = per-model logit adjustment ->
  softmax -> average -> decode; gauss ensemble = mixture mean/variance; `--mc 20`.
- `src/bpm/quality.py` + `scripts/gate_report.py`; `src/bpm/smoothing.py` (`median_smooth`, `kalman_rts`,
  `hmm_forward_backward`, `estimate_q`) + `scripts/predict_sequences.py` (±5 min scope; `idx` int32, start_s =
  2·idx; bpm/score float16; label float32; dist logits float16) + `scripts/smooth_report.py`. Tests as in
  review round 1 (outlier pulled back, r = ∞ bridged, 2× harmonic, estimate_q on known random walk of 10 s
  means) plus: HMM with a uniform prior equals plain forward-backward on the softmax.
- Docs for heads, gate, smoothing.

## Verification (code)
- `.venv/bin/pytest` green after Step 0A (and again after 0B).
- Folds (`make_folds.py` output): union of test groups = all 47 subjects; each subject val once; multisite top 5
  by >=130 share in 5 different groups; >=140 windows per group printed (expect 0/18/33/21/70); wrist-finger
  groups 6/6/5/5/5; all scripts report the same folds_hash.
- Smoke: `train.py --loss mse --fold 0 --seed 1 --lds-power 0.5`: sidecar complete; stop − best = 20 unless
  `hit_max_epochs`; balanced val logged with 4 decimals; `evaluate.py --fold 0` with 1 and 2 ckpts writes val+test.
- Leakage: per fold, bootstrap subjects ⊆ fold train; quotas, LDS density, prior and q from fold-train subjects
  only (asserted in code); aggregator LTGO test passes.
- LDS on real fold-0 train labels: clipped share, max and max/min printed (expect 1.1-1.5% at the cap,
  max = 20, max/min 51-61 at p = 1; measured 2026-11-06, see run plan Method 2).
- 0B smokes: dist/gauss train on fold 0; gate keeps ≈90% of val in coverage mode, suppressed MAE > passed MAE;
  smoothing on one multisite + one wrist-finger recording (labels match `make_windows`, start times exact,
  windows/s and EDF read time printed).
