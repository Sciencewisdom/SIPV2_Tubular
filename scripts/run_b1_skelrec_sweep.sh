#!/bin/bash
# B1: skeleton-recall topology loss sweep (second topology loss) — DRIVE, seed 42
# E1 (DW) and E4 (SIP-v2 Min) at lambda in {0.1, 0.3}; lambda=0 baselines already exist.
set -e
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python
for EXP in E1 E4; do
  for LAMBDA in 0.1 0.3; do
    echo "=== B1: $EXP + skelrec lambda=$LAMBDA ==="
    $PY scripts/train.py --exp $EXP --dataset DRIVE --seed 42 --epochs 200 \
      --batch_size 2 --img_size 512 --topo_loss skelrec --cldice_lambda $LAMBDA \
      --output_dir outputs/b1_${EXP}_skelrec${LAMBDA}_seed42
  done
done
echo "B1 sweep complete"
