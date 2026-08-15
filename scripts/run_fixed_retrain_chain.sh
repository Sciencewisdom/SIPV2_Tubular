#!/bin/bash
# FIXED-CODE RETRAIN CHAIN (post UNet image-forwarding fix, commit 56c5d56)
#
# Background: UNet.forward only passed the input image to blocks when
# block_type == 'sipv2'. SIPV2FullBlock/SIPV2RoadBlock skip the diffusion
# branch when image is None, so ALL E5 (sipv2_full) and road (sipv2_road)
# results were produced with the diffusion branch disabled. E4 ('sipv2')
# and E0/E1/E1D/E2/E3 are unaffected.
#
# This chain regenerates every affected result with the fixed code:
#   Road:   R1' / R2' (bs8, seed42) + B3' real ablations (bs4: scharr/sobel/stencil3/nogate)
#           + deterministic per-case eval for each
#   Retinal: E5' DRIVE (seed42, +clDice variant), CHASE_DB1/HRF (seeds 0,1,42)
# ATW' (fine-tune from R2') is NOT included yet — add once R2' exists.
#
# Waits for the current B3/B2 chain to finish first ("B2 done" marker).

set -u  # NOTE: no -e; one failed run must not kill the whole chain
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
LOG=outputs/fixed_retrain_chain.log

echo "$(date) waiting for B3/B2 chain to finish..." | tee -a "$LOG"
while ! grep -q "B2 done" outputs/b3_b2_chain.log 2>/dev/null; do sleep 60; done
echo "$(date) previous chain done; starting fixed-code retrain" | tee -a "$LOG"

road_eval () {
  # $1 = run output dir, $2 = eval json name, extra args passed to eval
  local DIR=$1 NAME=$2; shift 2
  local CKPT
  CKPT=$(ls "$DIR"/*/checkpoints/checkpoint_final.pth 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "SKIP eval $NAME: no checkpoint" | tee -a "$LOG"; return; fi
  echo "=== eval $NAME ===" | tee -a "$LOG"
  $PY scripts/eval_road_deterministic.py --checkpoint "$CKPT" --block_type sipv2_road \
    "$@" --output "outputs/fixed_eval_${NAME}.json" 2>&1 | tail -2
}

echo "=== R1' sipv2_road bs8 (fixed code) ===" | tee -a "$LOG"
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 8 --directions 16 --use_confidence_gate \
  --output_dir outputs/fixed_road_r1_seed42 2>&1 | tail -3
road_eval outputs/fixed_road_r1_seed42 r1

echo "=== R2' sipv2_road + clDice 0.3 (fixed code) ===" | tee -a "$LOG"
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 8 --directions 16 --use_confidence_gate \
  --use_cldice --cldice_lambda 0.3 --cldice_warmup 10 \
  --output_dir outputs/fixed_road_r2_seed42 2>&1 | tail -3
road_eval outputs/fixed_road_r2_seed42 r2

for A in scharr sobel stencil3 nogate; do
  EXTRA=""
  case $A in
    sobel)    EXTRA="--grad_op sobel" ;;
    stencil3) EXTRA="--stencil 3" ;;
    nogate)   EXTRA="--no-use_confidence_gate" ;;
  esac
  echo "=== B3' $A bs4 (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
    --batch_size 4 --directions 16 $EXTRA \
    --output_dir outputs/fixed_b3_${A}_seed42 2>&1 | tail -3
  road_eval outputs/fixed_b3_${A}_seed42 b3_${A}
done

retinal () {
  # $1 = dataset, $2 = seed, $3 = output dir, extra args after
  local DS=$1 SEED=$2 OUT=$3; shift 3
  echo "=== E5' $DS seed$SEED (fixed code) ===" | tee -a "$LOG"
  $PY scripts/train.py --exp E5 --dataset "$DS" --seed "$SEED" \
    --img_size 512 --batch_size 2 --epochs 200 --lr 0.0003 \
    --use_amp --grad_clip 1.0 --num_workers 4 \
    --output_dir "$OUT" "$@" 2>&1 | tail -3
}

retinal DRIVE 42 outputs/fixed_E5_drive_seed42
retinal DRIVE 42 outputs/fixed_E5_cldice_drive_seed42 --use_cldice --cldice_lambda 0.3 --cldice_warmup 20
for S in 0 1 42; do
  retinal CHASE_DB1 $S outputs/fixed_E5_chasedb1_seed$S
  retinal HRF       $S outputs/fixed_E5_hrf_seed$S
done

echo "$(date) fixed-code retrain chain complete" | tee -a "$LOG"
echo "TODO: ATW' fine-tune from outputs/fixed_road_r2_seed42 (see scripts/train_road_atw.py)"
