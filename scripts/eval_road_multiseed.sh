#!/bin/bash
# Evaluate all multi-seed road checkpoints deterministically
set -e

cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

OUTPUT_BASE="outputs/road_multiseed_deterministic"
mkdir -p "$OUTPUT_BASE"

echo "=== Multi-seed deterministic evaluation ==="

for SEED in 42 0 1; do
    # R0
    CKPT="outputs/road_real_r0_seed${SEED}/road_dw_crop512_bs8_ep50_seed${SEED}/checkpoints/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        for SPLIT in valid test; do
            python3 scripts/eval_road_deterministic.py \
                --checkpoint "$CKPT" \
                --block_type dw \
                --batch_size 1 \
                --split "$SPLIT" \
                --output "${OUTPUT_BASE}/R0_seed${SEED}_${SPLIT}.json"
        done
    fi

    # R1
    CKPT="outputs/road_real_r1_seed${SEED}/road_sipv2_road_crop512_bs8_ep50_seed${SEED}/checkpoints/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        for SPLIT in valid test; do
            python3 scripts/eval_road_deterministic.py \
                --checkpoint "$CKPT" \
                --block_type sipv2_road \
                --batch_size 1 \
                --split "$SPLIT" \
                --output "${OUTPUT_BASE}/R1_seed${SEED}_${SPLIT}.json"
        done
    fi

    # R2
    CKPT="outputs/road_real_r2_seed${SEED}/road_sipv2_road_crop512_bs8_ep50_seed${SEED}_cldice0.3/checkpoints/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        for SPLIT in valid test; do
            python3 scripts/eval_road_deterministic.py \
                --checkpoint "$CKPT" \
                --block_type sipv2_road \
                --batch_size 1 \
                --split "$SPLIT" \
                --output "${OUTPUT_BASE}/R2_seed${SEED}_${SPLIT}.json"
        done
    fi
done

echo "Done. Results in ${OUTPUT_BASE}/"
