# CV story: notebooks + deck (2026-11-08)

Deck and notebooks that tell the CV run story (Stages A-F of `.scratch/model-run-plan/spec.md`) from the run outputs. Nothing is retrained.

- `notebooks/cv_selection.ipynb`: data, protocol, Stages A-C, "the problem" (pull to the middle), final pick, test results.
- `notebooks/cv_gate_smoothing.ipynb`: Stages D-F (seed ensembles, quality gate, smoothing) for `dist_bootHL`, `mse_bootHL`, `mse`.
- Both write figures + cited numbers to `cache/deck/cv_story/` (`*.png`, `values.json`).
- `notebooks/final_results.ipynb` (`gen_nb_final.py`): final OOF results of `dist_bootHL` with the shipped pipeline (ensemble -> input gate -> Kalman, no output gate): table, scatter, Bland-Altman, per-bin, per-subject, error CDF. Writes `cache/plots/final/` and `cache/results/final_dist_bootHL.csv`. Needs `cache/sequences/ens_dist_bootHL_fold*/`.
- `gen_deck.py` builds `cache/deck/cv_story/cv_story.fodp` from those, then LibreOffice (flatpak) converts it to `docs/cv_story.odp` (layout and colors of `docs/hr_cnn_bilstm_results.odp`, to be merged) and `cache/deck/cv_story/cv_story.pdf`. Every number on a slide comes from `values.json`.

Generators own the notebooks (edit `gen_nb_*.py`, not the `.ipynb`). Rebuild:
```
.venv/bin/python .scratch/cv-story/gen_nb_selection.py notebooks/cv_selection.ipynb
.venv/bin/python .scratch/cv-story/gen_nb_gate_smoothing.py notebooks/cv_gate_smoothing.ipynb
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace cv_selection.ipynb \
  && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace cv_gate_smoothing.ipynb && cd ..
.venv/bin/python .scratch/cv-story/gen_deck.py
```
Inputs needed: `cache/preds/` (135 runs + 15 `ens_*`), `cache/runs/`, `cache/results/{summary,gate}.csv`, `cache/results/smooth_ens_*.csv`, `cache/sequences/ens_dist_bootHL_fold0/`, `cache/windows/`, `cache/folds.json`.

Checks built into the notebooks: run count and rc per config, OOF test MAE recomputed vs `summary.csv`, the example-trace pipeline reproduces the `smooth_report.py` test MAE of its fold exactly.
