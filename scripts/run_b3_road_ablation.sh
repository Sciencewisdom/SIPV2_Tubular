#!/bin/bash
# B3: SIP-v2 Road component ablation (audit item B3) — isolates the contribution
# of the three road-specific modifications vs the directional-diffusion core.
# Same protocol as R1: sipv2_road, no clDice, 50 epochs, seed 42, bs 8.
set -e
cd "$(dirname "$0")/.."
PY=.venv_torch/Scripts/python

echo "=== B3a: Scharr -> Sobel ==="
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 8 --directions 16 --grad_op sobel \
  --output_dir outputs/b3_road_sobel_seed42

echo "=== B3b: 5x5 -> 3x3 stencil ==="
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 8 --directions 16 --stencil 3 \
  --output_dir outputs/b3_road_stencil3_seed42

echo "=== B3c: confidence gate off ==="
$PY scripts/train_road.py --block_type sipv2_road --epochs 50 --seed 42 \
  --batch_size 8 --directions 16 --no-use_confidence_gate \
  --output_dir outputs/b3_road_nogate_seed42

echo "B3 ablation complete"
