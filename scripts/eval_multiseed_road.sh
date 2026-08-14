#!/bin/bash
# Evaluate all multi-seed road checkpoints deterministically and run Wilcoxon tests
set -e

cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

OUTPUT_BASE="outputs/road_multiseed_deterministic"
mkdir -p "$OUTPUT_BASE"

echo "=== Multi-seed deterministic evaluation ==="

# R0 DW seeds
for seed in 0 1; do
    CKPT="outputs/road_dw_multiseed/road_dw_crop512_bs8_ep50_seed${seed}/checkpoints/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        echo "Evaluating R0 seed=$seed ..."
        python3 scripts/eval_road_deterministic.py \
            --checkpoint "$CKPT" \
            --block_type dw \
            --batch_size 1 \
            --output "${OUTPUT_BASE}/R0_seed${seed}.json"
    else
        echo "Missing: $CKPT"
    fi
done

# R1 SIP-v2 Road seeds
for seed in 0 1; do
    CKPT="outputs/road_sipv2_multiseed/road_sipv2_road_crop512_bs8_ep50_seed${seed}/checkpoints/checkpoint_final.pth"
    if [ -f "$CKPT" ]; then
        echo "Evaluating R1 seed=$seed ..."
        python3 scripts/eval_road_deterministic.py \
            --checkpoint "$CKPT" \
            --block_type sipv2_road \
            --batch_size 1 \
            --output "${OUTPUT_BASE}/R1_seed${seed}.json"
    else
        echo "Missing: $CKPT"
    fi
done

echo "=== Running Wilcoxon tests ==="
python3 scripts/wilcoxon_road_multiseed.py \
    --input_dir "$OUTPUT_BASE" \
    --output "${OUTPUT_BASE}/wilcoxon_results.json"

echo "Done. Results in ${OUTPUT_BASE}/"
