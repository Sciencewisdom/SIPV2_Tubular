#!/bin/bash
# B3 eval: deterministic per-case eval (valid split, same protocol as R1 reference
# in outputs/road_percase_deterministic.json) for each ablation checkpoint.
set -e
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
for A in sobel stencil3 nogate; do
  CKPT=$(ls outputs/b3_road_${A}_seed42/*/checkpoints/checkpoint_final.pth 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "SKIP $A: no checkpoint"; continue; fi
  echo "=== eval $A: $CKPT ==="
  $PY scripts/eval_road_deterministic.py --checkpoint "$CKPT" --block_type sipv2_road \
    --output outputs/b3_eval_${A}.json
done
echo "B3 eval complete"
