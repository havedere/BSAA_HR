#!/usr/bin/env bash
# Stage B runner (story design): L* = mse + bootstrap HL x LDS p {0, 0.5, 1} x 5 folds x 3 seeds, seed-major, then fold, then config.
# Per run: train.py --fold k, then evaluate.py --fold k. Continues past failures. Resumable: a run whose
# sidecar has rc 0 and whose preds file exists is skipped. Usage: run_stage_b.sh [--dry-run]
set -u
ROOT=/var/home/julian/git/BSAA_HR
PY=$ROOT/.venv/bin/python
DRY=${1:-}
LOSSES=("mse")
BOOTS=("HL")
SEEDS=(42 1 2)
LDS=(0 0.5 1)
cd "$ROOT" || exit 1
mkdir -p cache/logs/stage_b
total=$(( ${#SEEDS[@]} * 5 * ${#LOSSES[@]} * ${#BOOTS[@]} * ${#LDS[@]} )); i=0
for seed in "${SEEDS[@]}"; do
 for fold in 0 1 2 3 4; do
  for l in "${LOSSES[@]}"; do
   read -r loss delta <<< "$l"
   for boot in "${BOOTS[@]}"; do
   for p in "${LDS[@]}"; do
    i=$((i+1))
    tag=$([ "$loss" = huber ] && echo "huber$delta" || echo "$loss")
    [ -n "$boot" ] && tag+="_boot$boot"
    [ "$p" != 0 ] && tag+="_lds$p"
    tag+="_fold${fold}_seed${seed}"
    side=cache/runs/$tag.json; preds=cache/preds/$tag.npz; log=cache/logs/stage_b/$tag.log
    args=(--loss "$loss" --fold "$fold" --seed "$seed")
    [ "$loss" = huber ] && args+=(--delta "$delta")
    [ -n "$boot" ] && args+=(--bootstrap "$boot")
    [ "$p" != 0 ] && args+=(--lds-power "$p")
    if [ -f "$preds" ] && [ -f "$side" ] && "$PY" -c "import json,sys; sys.exit(json.load(open('$side')).get('rc',1) != 0)"; then
      echo "[$i/$total] $tag skip (done)"; continue
    fi
    if [ "$DRY" = --dry-run ]; then echo "[$i/$total] $tag: train.py ${args[*]}"; continue; fi
    t0=$(date +%s)
    "$PY" scripts/train.py "${args[@]}" > "$log" 2>&1; rc=$?
    erc=-1
    if [ $rc -eq 0 ]; then
      "$PY" scripts/evaluate.py --fold "$fold" --ckpt "cache/best_$tag.pt" >> "$log" 2>&1; erc=$?
    fi
    # rc into the sidecar; a crashed train.py wrote none, so write a stub
    "$PY" - "$side" "$tag" "$rc" "$erc" <<'PYEOF'
import json, sys, pathlib
p, tag, rc, erc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.loads(p.read_text()) if p.exists() and rc == 0 else {"tag": tag}
d.update(rc=rc, eval_rc=erc)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=1))
PYEOF
    best=$("$PY" -c "import json; d=json.load(open('$side')); print(f\"best_val={d.get('best_val', float('nan')):.4f} best_ep={d.get('best_epoch')} stop_ep={d.get('stop_epoch')}\")")
    echo "[$i/$total] $tag rc=$rc eval_rc=$erc $best $(( $(date +%s) - t0 ))s"
   done
   done
  done
 done
done
