#!/bin/bash
# ATW' RETRAIN (fixed code, post-56c5d56) — run AFTER run_fixed_retrain_chain.sh
# completes (needs fixed_road_r2_seed42 checkpoint).
#
# Regenerates the deprecated ATW results (audit doc section 9.3):
#   1. R1'/R2' extra seeds (0, 1) for the 3-seed paired tests
#   2. ATW seed42 lambda sweep {0.1, 0.15, 0.2, 0.3}  (5 ep, lr 5e-5, warmup 1,
#      resumed from R2' epoch49 — same protocol as the original run, see
#      outputs/road_experiments/...atw0.15/config.json)
#   3. ATW lambda=0.15 for seeds 0, 1
# JPR evaluation is a separate analysis step (scripts/eval_atw_percase_jpr.py).

set -u
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
LOG=outputs/atw_retrain.log

R2_CKPT=outputs/fixed_road_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth
if [ ! -f "$R2_CKPT" ]; then
  echo "$(date) ABORT: $R2_CKPT missing — run run_fixed_retrain_chain.sh first" | tee -a "$LOG"
  exit 1
fi

for S in 0 1; do
  echo "=== R1' seed$S (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed $S \
    --batch_size 8 --directions 16 --use_confidence_gate \
    --output_dir outputs/fixed_road_r1_seed$S 2>&1 | tail -3
  echo "=== R2' seed$S (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed $S \
    --batch_size 8 --directions 16 --use_confidence_gate \
    --use_cldice --cldice_lambda 0.3 --cldice_warmup 10 \
    --output_dir outputs/fixed_road_r2_seed$S 2>&1 | tail -3
done

for L in 0.1 0.15 0.2 0.3; do
  echo "=== ATW' seed42 lambda=$L (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train_road_atw.py --epochs 5 --seed 42 --batch_size 8 \
    --lr 5e-5 --atw_lambda $L --atw_warmup 1 \
    --resume "$R2_CKPT" \
    --output_dir outputs/fixed_atw_lam${L}_seed42 2>&1 | tail -3
done

for S in 0 1; do
  CKPT_S=outputs/fixed_road_r2_seed${S}/road_sipv2_road_crop512_bs8_ep50_seed${S}_cldice0.3/checkpoints/checkpoint_epoch49.pth
  if [ ! -f "$CKPT_S" ]; then echo "SKIP ATW seed$S: $CKPT_S missing" | tee -a "$LOG"; continue; fi
  echo "=== ATW' seed$S lambda=0.15 (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train_road_atw.py --epochs 5 --seed $S --batch_size 8 \
    --lr 5e-5 --atw_lambda 0.15 --atw_warmup 1 \
    --resume "$CKPT_S" \
    --output_dir outputs/fixed_atw_lam0.15_seed$S 2>&1 | tail -3
done

echo "$(date) ATW retrain complete — next: per-case JPR eval (eval_atw_percase_jpr.py / eval_atw_lambda_sweep_jpr.py)" | tee -a "$LOG"
