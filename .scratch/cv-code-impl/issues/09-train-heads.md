# 09 train.py --head dist|gauss

Status: done
Blocked by: 08
Spec: Step 0B (train.py); run plan Method 3.

## Scope
- `--head reg|dist|gauss` (default reg; reg behaviour and tags unchanged).
- dist: `nn.CrossEntropyLoss(reduction="none")` with `gaussian_targets` probability targets; BPM for MAE / early stopping via `decode_bpm` (tau 0).
- gauss: β-NLL, β = 0.5 (Seitzer et al. 2022): `GaussianNLLLoss(mean, target, var=exp(logvar), reduction="none") * var.detach()**0.5`; BPM = mean.
- LDS weights and bootstrap apply unchanged (per-window loss · w).
- Tag: head replaces the loss part: `dist[_boot<X>][_lds<p>][_fold<k>]_seed<s>`, `gauss...`; `--loss/--delta` ignored for these heads. Sidecar gets `head`.

## Verify
pytest; smoke `train.py --head dist --fold 0 --seed 1` and `--head gauss` (can stop early with small --patience); sidecar has head, tag correct.
