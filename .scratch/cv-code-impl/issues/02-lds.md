# 02 LDS loss weights

Status: ready-for-agent
Spec: Step 0A.2; method: model-run-plan Method 2.

## Scope
- `src/bpm/dataset.py`: `lds_weights(labels, power, sample_weights, sigma=2.0, clip=20.0)`.
  Density = histogram of labels, 1-BPM bins 30..220, each window counted with its sample weight, Gaussian-smoothed (sigma BPM).
  w = density^-p -> divide by sample-weighted mean -> repeat {clip at `clip`, renormalise to sample-weighted mean 1} until max <= clip.
- `WindowDataset` optionally takes per-window weights and then returns (x, y, w); without weights behaviour unchanged.
- `sampler_weights` unchanged.

## Tests (tests/test_dataset.py)
uniform labels -> 1; rare label gets larger weight; max <= 20 after final renorm; sample-weighted mean 1; p=0 -> all 1;
unit-free (scaling density does not change weights).
