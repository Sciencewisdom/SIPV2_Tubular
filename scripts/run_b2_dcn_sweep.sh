#!/bin/bash
# B2: DCNv2 control architecture (E1D, parameter-matched to E1 at 1.41M) under
# the same clDice weight sweep. Decides whether "structure-anchored" (SIP-v2)
# vs "freely learned directional" (DCN) vs "isotropic" (E1) propagation is the
# variable that flips the sign of the clDice effect.
# DRIVE, seed 42, 200 epochs, identical hyperparameters to the main sweep.
set -e
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
for LAMBDA in 0.0 0.1 0.3; do
  echo "=== B2: E1D + clDice lambda=$LAMBDA ==="
  EXTRA=""
  if [ "$LAMBDA" != "0.0" ]; then EXTRA="--use_cldice --cldice_lambda $LAMBDA"; fi
  $PY scripts/train.py --exp E1D --dataset DRIVE --seed 42 --epochs 200 \
    --batch_size 2 --img_size 512 $EXTRA \
    --output_dir outputs/b2_E1D_cldice${LAMBDA}_seed42
done
echo "B2 sweep complete"
