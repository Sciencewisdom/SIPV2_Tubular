#!/bin/bash
set -e
cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

for SEED in 0 1; do
    echo "=== Seed $SEED ==="
    
    echo "R0 DW seed=$SEED ..."
    python scripts/train_road.py --block_type dw --epochs 50 --seed $SEED --batch_size 8 --output_dir outputs/road_real_r0_seed${SEED}
    
    echo "R1 SIP-v2 Road seed=$SEED ..."
    python scripts/train_road.py --block_type sipv2_road --epochs 50 --seed $SEED --batch_size 8 --directions 16 --use_confidence_gate --output_dir outputs/road_real_r1_seed${SEED}
    
    echo "R2 SIP-v2 + clDice seed=$SEED ..."
    python scripts/train_road.py --block_type sipv2_road --epochs 50 --seed $SEED --batch_size 8 --directions 16 --use_confidence_gate --use_cldice --cldice_lambda 0.3 --cldice_warmup 10 --output_dir outputs/road_real_r2_seed${SEED}
done

echo "All multi-seed experiments complete!"
