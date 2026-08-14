#!/bin/bash
# Evaluate real road experiments deterministically (valid + test)
set -e

cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

OUTPUT_BASE="outputs/road_real_deterministic"
mkdir -p "$OUTPUT_BASE"

echo "=== Deterministic evaluation of real road experiments ==="

# R0 DW seed 42
CKPT="outputs/road_real_r0_seed42/road_dw_crop512_bs8_ep50_seed42/checkpoints/checkpoint_final.pth"
if [ -f "$CKPT" ]; then
    echo "Evaluating R0 valid..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type dw \
        --batch_size 1 \
        --split valid \
        --output "${OUTPUT_BASE}/R0_seed42_valid.json"
    echo "Evaluating R0 test..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type dw \
        --batch_size 1 \
        --split test \
        --output "${OUTPUT_BASE}/R0_seed42_test.json"
else
    echo "Missing: $CKPT"
fi

# R1 SIP-v2 Road seed 42
CKPT="outputs/road_real_r1_seed42/road_sipv2_road_crop512_bs8_ep50_seed42/checkpoints/checkpoint_final.pth"
if [ -f "$CKPT" ]; then
    echo "Evaluating R1 valid..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type sipv2_road \
        --batch_size 1 \
        --split valid \
        --output "${OUTPUT_BASE}/R1_seed42_valid.json"
    echo "Evaluating R1 test..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type sipv2_road \
        --batch_size 1 \
        --split test \
        --output "${OUTPUT_BASE}/R1_seed42_test.json"
else
    echo "Missing: $CKPT"
fi

# R2 SIP-v2 Road + clDice seed 42
CKPT="outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_final.pth"
if [ -f "$CKPT" ]; then
    echo "Evaluating R2 valid..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type sipv2_road \
        --batch_size 1 \
        --split valid \
        --output "${OUTPUT_BASE}/R2_seed42_valid.json"
    echo "Evaluating R2 test..."
    python3 scripts/eval_road_deterministic.py \
        --checkpoint "$CKPT" \
        --block_type sipv2_road \
        --batch_size 1 \
        --split test \
        --output "${OUTPUT_BASE}/R2_seed42_test.json"
else
    echo "Missing: $CKPT"
fi

echo "Done. Results in ${OUTPUT_BASE}/"
