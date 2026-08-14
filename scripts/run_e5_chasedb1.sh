#!/bin/bash
set -e
cd "$(dirname "$0")/.."
OUTPUT_DIR="outputs/chasedb1_E5"
mkdir -p "$OUTPUT_DIR"
python3 scripts/train.py \
    --exp E5 --dataset CHASE_DB1 --data_root data/raw/CHASE_DB1 \
    --output_dir "$OUTPUT_DIR" --img_size 512 --batch_size 2 \
    --epochs 200 --lr 0.0003 --seed 42 --use_amp --grad_clip 1.0 --num_workers 4
