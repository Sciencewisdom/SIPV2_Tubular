#!/bin/bash
# Run all DRIVE experiments (E0-E4) for SIP-v2 Phase 1
# Optimized for RTX 4050 (6GB VRAM)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# DRIVE data path (adjust if needed)
DATA_ROOT="${PROJECT_DIR}/data/raw/DRIVE"

# Check if DRIVE data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DRIVE dataset not found at $DATA_ROOT"
    echo "Please download DRIVE from: https://drive.grand-challenge.org/"
    echo "Expected structure:"
    echo "  data/raw/DRIVE/training/images/*.tif"
    echo "  data/raw/DRIVE/training/1st_manual/*.gif"
    echo "  data/raw/DRIVE/training/mask/*.gif"
    echo "  data/raw/DRIVE/test/images/*.tif"
    echo "  data/raw/DRIVE/test/1st_manual/*.gif"
    echo "  data/raw/DRIVE/test/mask/*.gif"
    exit 1
fi

# Configuration for RTX 4050 (6GB VRAM)
IMG_SIZE=512
BATCH_SIZE=2
EPOCHS=200
LR=3e-4
SEED=42

echo "=========================================="
echo "SIP-v2 DRIVE Phase 1 Experiments"
echo "GPU: RTX 4050 (6GB VRAM)"
echo "Config: img_size=${IMG_SIZE}, batch_size=${BATCH_SIZE}, epochs=${EPOCHS}"
echo "=========================================="

# Experiment list
EXPERIMENTS=("E0" "E1" "E2" "E3" "E4")
EXP_NAMES=("U-Net" "DW" "IsoDiffusion" "OldSIP" "SIP-v2")

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NAME="${EXP_NAMES[$i]}"

    echo ""
    echo "=========================================="
    echo "Running: ${EXP} (${NAME})"
    echo "=========================================="

    python scripts/train.py \
        --exp "$EXP" \
        --data_root "$DATA_ROOT" \
        --img_size "$IMG_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --seed "$SEED" \
        --use_amp \
        --output_dir "outputs" \
        --save_freq 20

echo "${EXP} complete."
done

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
echo "Results in: outputs/"
echo ""
echo "Next steps:"
echo "  1. Check outputs/*/summary.json for best Dice"
echo "  2. Run tensorboard: tensorboard --logdir outputs"
echo "  3. Run visualization: python scripts/visualize_cases.py"
