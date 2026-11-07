# 14 smooth_report.py

Status: done
Blocked by: 11, 12, 13
Spec: Step 0B; run plan Method 5.

## Scope
- Reads `cache/sequences/<config>_fold<k>/`; pipeline order fixed: ensemble -> input gate (Method 4 limit fitted on capped val preds) -> smoother (gated: r = inf / flat emission) -> output gate on posterior std (limit refitted on val sequence windows, both operating points).
- Smoothers: median (k), Kalman (R = (c·s)², const c² without score), HMM (dist). q / σ_T from fold-train 10 s labels (`estimate_q`) × m ∈ {1, 2, 4}; m, c, k, max gated run (default 30 s) tuned on val; scored on test.
- Scoring on all labelled windows of test recordings: raw vs each smoother, with/without gates: subject-macro balanced MAE, window-pooled MAE, per-bin MAE, coverage, lag (error in the 30 s after the largest HR jumps). Writes `cache/results/smooth.csv`.

## Verify
Smoke on fold-0 sequences from ticket 13.

## Notes (2026-11-07)
- CLI: `smooth_report.py --seq cache/sequences/<config>_fold<k> --preds <capped preds npz of the same ensemble> [--max-gated-s 30]`. Input gate variants none / coverage / mae (from the preds `scores`, val rows); output gate on posterior std (Kalman, HMM; not median/raw).
- Max gated run is a fixed CLI value, not tuned: tuning it on val MAE alone always picks the shortest run. Gated runs longer than it stay uncovered.
- Grids: median k {3,5,9,15}; Kalman m {1,2,4} × c {0.5,1,2,4} (const R without score: c {2,4,8,16}); HMM m {1,2,4}. Objective: val subject-macro balanced MAE on covered windows.
- Lag: one jump per test recording (largest 30 s label change), mean |err| of the next 15 passed windows, mean over recordings (shortcut comment).
- Smoke fold 0, dist seed 1 (153 recordings, 198,597 windows; 117 recordings without HR curve until Step 1): q multisite 1.006, wrist-finger 1.062. Test macro MAE raw 4.91, median 4.78, Kalman 4.54 (4.04 at 0.88 coverage with output gate), HMM 11.90.
- HMM is broken by the spec'd emission = softmax / prior (same floor issue as tau 1, ticket 10): val macro 9.1-14.3 with the prior, 3.52-3.68 without (m 1-4), i.e. on par with Kalman. Decision for run plan Method 5.
- Runtime 53 s (7 min CPU, numpy threads) for one config-fold.
- Decision 2026-11-07 (user, run plan "Decisions after Step 0B smokes"): HMM emission = plain softmax; `smooth_report.py` passes no prior. Rerun fold 0: HMM val 3.518 (m 4), test macro 4.65 (Kalman 4.54).
- Audit fixes 2026-11-07 (tests in `tests/test_smooth_report.py`): a segment with no passed window is never covered (Kalman gave 0 BPM, HMM a flat ~125 BPM there); lag jump searched within one segment, never across a gap; `--seq` accepts both dir names. Fold-0 numbers unchanged.
