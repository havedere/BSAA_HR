#!/usr/bin/env bash
# Stages D-F (no training): per config and fold, D = 3-seed ensemble preds with MC dropout (evaluate.py --mc 20),
# F = sequences + smoothing (predict_sequences.py, smooth_report.py); then E = one gate_report.py over all ens files.
# Resumable: a step whose output exists is skipped. Continues past failures. Usage: run_stage_def.sh [--dry-run]
set -u
ROOT=/var/home/julian/git/BSAA_HR
PY=$ROOT/.venv/bin/python
DRY=${1:-}
CONFIGS=(dist_bootHL mse_bootHL mse)
SEEDS=(42 1 2)
cd "$ROOT" || exit 1
LOG=cache/logs/stage_def; mkdir -p "$LOG"
step() {  # step <name> <output> <cmd...>
  local name=$1 out=$2; shift 2
  if [ -e "$out" ]; then echo "$name skip (done)"; return; fi
  if [ "$DRY" = --dry-run ]; then echo "$name: $*"; return; fi
  local t0=$(date +%s); "$@" > "$LOG/$name.log" 2>&1; local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s"
}
for cfg in "${CONFIGS[@]}"; do
 for fold in 0 1 2 3 4; do
  ck=(); for s in "${SEEDS[@]}"; do ck+=("cache/best_${cfg}_fold${fold}_seed${s}.pt"); done
  ens=ens_${cfg}_fold${fold}
  step "D_$ens" "cache/preds/$ens.npz" "$PY" scripts/evaluate.py --fold "$fold" --ckpt "${ck[@]}" --mc 20
  step "Fseq_$ens" "cache/sequences/$ens" "$PY" scripts/predict_sequences.py --fold "$fold" --ckpt "${ck[@]}"
  step "Fsmooth_$ens" "cache/results/smooth_$ens.csv" \
    "$PY" scripts/smooth_report.py --seq "cache/sequences/$ens" --preds "cache/preds/$ens.npz"
 done
done
ens_files=(); for cfg in "${CONFIGS[@]}"; do for fold in 0 1 2 3 4; do ens_files+=("cache/preds/ens_${cfg}_fold${fold}.npz"); done; done
step E_gate cache/results/gate.csv "$PY" scripts/gate_report.py --preds "${ens_files[@]}"
