#!/bin/bash
# Run E5 (SIP-v2 Full) + clDice on DRIVE

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="outputs/E5_cldice_size512_bs2_seed42"
mkdir -p "$OUTPUT_DIR"

echo "Starting E5 + clDice on DRIVE..."

python3 scripts/train.py \
    --exp E5 \
    --dataset DRIVE \
    --data_root data/raw/DRIVE \
    --output_dir "$OUTPUT_DIR" \
    --img_size 512 \
    --batch_size 2 \
    --epochs 200 \
    --lr 0.0003 \
    --seed 42 \
    --use_amp \
    --grad_clip 1.0 \
    --num_workers 4 \
    --use_cldice \
    --cldice_lambda 0.3 \
    --cldice_warmup 20 \
    2>> "$OUTPUT_DIR/training.log"

echo "E5 + clDice training complete."
