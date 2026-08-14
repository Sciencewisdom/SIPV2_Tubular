#!/bin/bash
# Multi-seed road validation for R0 (DW) and R1 (SIP-v2 Road)
cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

SEEDS=(0 1)
for SEED in "${SEEDS[@]}"; do
    # R0 DW
    python3 scripts/train_road.py \
        --block_type dw \
        --epochs 50 \
        --batch_size 8 \
        --seed $SEED \
        --output_dir outputs/road_dw_multiseed \
        > /tmp/road_dw_seed${SEED}.log 2>&1 &
    
    # R1 SIP-v2 Road
    python3 scripts/train_road.py \
        --block_type sipv2_road \
        --epochs 50 \
        --batch_size 8 \
        --seed $SEED \
        --output_dir outputs/road_sipv2_multiseed \
        > /tmp/road_sipv2_seed${SEED}.log 2>&1 &
done

echo "Launched 4 road experiments (R0/R1 × seed 0/1)"
echo "Logs: /tmp/road_*_seed*.log"
