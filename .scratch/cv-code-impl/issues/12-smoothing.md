# 12 smoothing.py: median, Kalman+RTS, HMM forward-backward, estimate_q

Status: done
Blocked by: none
Spec: Step 0B; run plan Method 5.

## Scope
- `median_smooth(bpm, k, ok=None)`: centred median over k windows, gated windows excluded.
- `kalman_rts(bpm, r, q)`: random walk, measurement var r per window (np.inf = missing), returns (mean, posterior var).
- `hmm_forward_backward(probs, sigma_t, floor=1e-3, log_prior=None)`: emission = probs / prior (scaled likelihood), Gaussian transition σ_T over 1-BPM bins, banded at ±5·σ_T, uniform floor; flat emission for gated windows; returns posterior (T, 191), mean, std. Vectorised.
- `estimate_q(labels_2s)`: MAD·1.4826 of first differences.
- Segment helper: split at gaps > 30 s.

## Verify (TDD)
Outlier pulled back; r = inf bridged; 2× harmonic outlier corrected by HMM; estimate_q on random walk of known step; HMM with uniform prior = plain forward-backward on the softmax.

## Notes (2026-11-07)
- `estimate_q` returns the step std (BPM per 2 s window); Kalman q = (m · s)², HMM σ_T = m · s.
- Harmonic test uses a realistic emission (0.9 mass at 140, 0.1 at 70): a pure peak at 140 has exactly zero likelihood at 70, and no transition prior can recover from that.
- `segments(idx, max_gap_s=30)` is the segment helper.
