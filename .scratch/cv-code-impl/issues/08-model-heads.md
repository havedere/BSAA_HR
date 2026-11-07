# 08 model.py: reg / dist / gauss heads, gaussian_targets, decode_bpm

Status: done
Blocked by: none
Spec: Step 0B (model.py); run plan Method 3.

## Scope
- `CNNBiLSTM(head="reg"|"dist"|"gauss")`: only the last `Linear(32, n)` changes (reg 1, dist 191 = 1-BPM bins 30..220, gauss 2 = mean, logvar). Default `reg`: old checkpoints (`best.pt`, `best_*.pt`) load unchanged.
- Output: reg `(B,)` BPM; dist `(B, 191)` logits; gauss `(B, 2)`.
- `gaussian_targets(labels, sigma=2.0)`: (B, 191) Gaussian over bin centres, each row sums to 1.
- `decode_bpm(logits, window=10, log_prior=None, tau=0)`: logits - tau*log_prior, softmax, expected value over bins within ±window of the argmax (renormalised). Plain argmax also available (`argmax_bpm`). No inf/NaN with a log prior from (counts + 1)/(N + 191).
- `dist_std(logits)`: std of the softmax curve (uncertainty score).

## Verify (TDD)
Shapes per head; target sums to 1 and peaks at label; decode(log of target) = label (±0.05); prior floor -> finite; tau=0 equals plain decode; old reg state_dict loads into `CNNBiLSTM()`.
