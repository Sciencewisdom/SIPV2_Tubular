#!/bin/bash
# Official-clDice replication of the core sweep (audit section 9 finding).
#
# The historical clDice experiments used the 'crossed' variant, which deviates
# from Shit et al. 2021 (~0.066 clDice lower on trained DRIVE predictions).
# Internal comparisons remain valid (all runs used the same variant), but for
# external validity the core interaction result should be replicated with the
# reference formula: E1 vs E4 at lambda in {0, 0.1, 0.3}, DRIVE, seed 42,
# 200 epochs — the key cells of the P0-1 sweep.
#
# Run AFTER the fixed-code retrain chain (GPU queue).

set -u
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
LOG=outputs/official_cldice_sweep.log

for EXP in E1 E4; do
  for L in 0.0 0.1 0.3; do
    OUT=outputs/official_cldice_${EXP}_lam${L}_seed42
    echo "=== $EXP official-clDice lambda=$L ===" | tee -a "$LOG"
    if [ "$L" = "0.0" ]; then
      $PY scripts/train.py --exp $EXP --dataset DRIVE --seed 42 \
        --img_size 512 --batch_size 2 --epochs 200 --lr 0.0003 \
        --use_amp --grad_clip 1.0 --num_workers 4 \
        --output_dir "$OUT" 2>&1 | tail -3
    else
      $PY scripts/train.py --exp $EXP --dataset DRIVE --seed 42 \
        --img_size 512 --batch_size 2 --epochs 200 --lr 0.0003 \
        --use_amp --grad_clip 1.0 --num_workers 4 \
        --topo_loss cldice --cldice_lambda $L --cldice_warmup 20 \
        --loss_compat fixed \
        --output_dir "$OUT" 2>&1 | tail -3
    fi
  done
done

echo "$(date) official-clDice replication sweep complete" | tee -a "$LOG"
