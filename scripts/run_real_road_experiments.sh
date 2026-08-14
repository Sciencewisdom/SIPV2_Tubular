#!/bin/bash
# Run real Massachusetts Roads experiments once data is fully downloaded
set -e

cd /root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular

# Verify data exists (check both .tif and .tiff extensions)
expected_train=1108
actual_train=$(ls data/raw/mass_roads/train/sat/*.tiff data/raw/mass_roads/train/sat/*.tif 2>/dev/null | wc -l)
actual_train_map=$(ls data/raw/mass_roads/train/map/*.tiff data/raw/mass_roads/train/map/*.tif 2>/dev/null | wc -l)

if [ "$actual_train" -lt "$expected_train" ] || [ "$actual_train_map" -lt "$expected_train" ]; then
    echo "Data not ready: train/sat=$actual_train, train/map=$actual_train_map (expected $expected_train each)"
    exit 1
fi

echo "Data ready. Starting experiments..."

# R0: DW baseline (seed 42)
echo "=== R0 DW seed=42 ==="
python scripts/train_road.py \
    --block_type dw \
    --epochs 50 \
    --seed 42 \
    --batch_size 8 \
    --output_dir outputs/road_real_r0_seed42

# R1: SIP-v2 Road (seed 42)
echo "=== R1 SIP-v2 Road seed=42 ==="
python scripts/train_road.py \
    --block_type sipv2_road \
    --epochs 50 \
    --seed 42 \
    --batch_size 8 \
    --directions 16 \
    --use_confidence_gate \
    --output_dir outputs/road_real_r1_seed42

# R2: SIP-v2 Road + clDice (seed 42)
echo "=== R2 SIP-v2 Road + clDice seed=42 ==="
python scripts/train_road.py \
    --block_type sipv2_road \
    --epochs 50 \
    --seed 42 \
    --batch_size 8 \
    --directions 16 \
    --use_confidence_gate \
    --use_cldice \
    --cldice_lambda 0.3 \
    --cldice_warmup 10 \
    --output_dir outputs/road_real_r2_seed42

echo "All experiments complete!"
