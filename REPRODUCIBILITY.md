# SIP-v2 Reproducibility Guide

This document provides step-by-step instructions to reproduce all core results reported in the paper.

## Environment Setup

```bash
# Requirements
- Python 3.8+
- PyTorch 2.6.0+cu124 (or compatible CUDA version)
- GPU with ≥6GB VRAM (RTX 3090 48GB recommended for multi-experiment parallelism)

# Installation
pip install torch torchvision torchaudio
pip install scikit-image scipy numpy matplotlib tqdm tensorboard
```

## Datasets

### Retinal Vessels
1. **DRIVE**: Download from https://drive.grand-challenge.org/
   - Place in `data/raw/DRIVE/`
   - Structure: `training/images/`, `training/1st_manual/`, `test/images/`

2. **CHASE_DB1**: Download from https://blogs.kingston.ac.uk/retinal/chasedb1/
   - Place in `data/raw/CHASE_DB1/`

3. **HRF**: Download from https://www5.cs.fau.de/research/data/fundus-images/
   - Place in `data/raw/HRF/`

### Road Networks
4. **Massachusetts Roads**: Download from https://www.cs.toronto.edu/~vmnih/data/
   - Place in `data/raw/massachusetts_roads/`
   - Training: `train/`, Validation: `valid/`, Test: `test/`

## Phase 1: Retinal Vessel Baselines (E0-E5)

### E1 (DW CNN Baseline)
```bash
python scripts/train.py \
  --experiment E1 --dataset drive --img_size 512 --batch_size 2 \
  --epochs 200 --lr 3e-4 --seed 42 \
  --output_dir outputs/E1_size512_bs2_seed42
```

### E4 (SIP-v2 Min)
```bash
python scripts/train.py \
  --experiment E4 --dataset drive --img_size 512 --batch_size 2 \
  --epochs 200 --lr 3e-4 --seed 42 \
  --output_dir outputs/E4_size512_bs2_seed42
```

### E5 (SIP-v2 Full)
```bash
python scripts/train.py \
  --experiment E5 --dataset drive --img_size 512 --batch_size 2 \
  --epochs 200 --lr 3e-4 --seed 42 \
  --output_dir outputs/E5_size512_bs2_seed42
```

### Multi-seed Validation
Repeat each experiment with `--seed 0` and `--seed 1`.

## Phase 2: clDice Sensitivity Study (P0-1)

### E1 + clDice (varying λ_c)
```bash
for lambda in 0.0 0.1 0.3 0.5; do
  python scripts/train.py \
    --experiment E1 --dataset drive --img_size 512 --batch_size 2 \
    --epochs 200 --lr 3e-4 --seed 42 --cldice_weight $lambda \
    --output_dir outputs/E1_cldice${lambda}_size512_bs2_seed42
done
```

### E4 + clDice (varying λ_c)
```bash
for lambda in 0.0 0.1 0.3 0.5; do
  python scripts/train.py \
    --experiment E4 --dataset drive --img_size 512 --batch_size 2 \
    --epochs 200 --lr 3e-4 --seed 42 --cldice_weight $lambda \
    --output_dir outputs/E4_cldice${lambda}_size512_bs2_seed42
done
```

## Phase 3: Cross-Dataset Validation

### CHASE_DB1 and HRF
Replace `--dataset drive` with `--dataset chase_db1` or `--dataset hrf` in the Phase 1 commands.

## Phase 4: Road Network Validation

### R0 (DW Road Baseline)
```bash
python scripts/train_road.py \
  --model_type dw --epochs 50 --batch_size 4 --seed 42 \
  --output_dir outputs/road_real_r0_seed42
```

### R1 (SIP-v2 Road)
```bash
python scripts/train_road.py \
  --model_type sipv2_road --epochs 50 --batch_size 4 --seed 42 \
  --output_dir outputs/road_real_r1_seed42
```

### R2 (SIP-v2 Road + clDice)
```bash
python scripts/train_road.py \
  --model_type sipv2_road --use_cldice --cldice_weight 0.3 \
  --epochs 50 --batch_size 4 --seed 42 \
  --output_dir outputs/road_real_r2_seed42
```

### R3 (ATW Fine-tuning from R2)
```bash
python scripts/train_road_atw.py \
  --resume outputs/road_real_r2_seed42/checkpoints/checkpoint_epoch49.pth \
  --epochs 15 --lr 1e-4 --atw_lambda 0.15 --seed 42 \
  --output_dir outputs/road_atw_l015_seed42
```

### Multi-seed Road Experiments
Repeat with `--seed 0` and `--seed 1`.

## Phase 5: Diagnostic Analyses

### Gradient Conflict Analysis
```bash
python scripts/analyze_gradient_conflict.py \
  --checkpoint outputs/road_real_r2_seed42/checkpoints/checkpoint_epoch49.pth \
  --model_type sipv2_road --split valid

python scripts/analyze_gradient_conflict.py \
  --checkpoint outputs/road_real_r0_seed42/checkpoints/checkpoint_epoch49.pth \
  --model_type dw --split valid
```

### Coherence Visualization
```bash
python scripts/visualize_coherence.py \
  --data_root data/raw/massachusetts_roads/valid \
  --output paper_figures/coherence_visualization.png
```

### Feature-level Coherence Diagnostic
```bash
python scripts/diagnostic_feature_coherence.py \
  --checkpoint outputs/road_real_r2_seed42/checkpoints/checkpoint_epoch49.pth \
  --data_root data/raw/massachusetts_roads/valid
```

### DRIVE ATW Cross-Domain Validation
```bash
python scripts/train_drive_atw.py \
  --resume outputs/E5_size512_bs2_seed42/checkpoints/checkpoint_best.pth \
  --epochs 15 --lr 1e-4 --atw_lambda 0.15 --seed 42 \
  --output_dir outputs/drive_e5_atw0.15_seed42
```

## Evaluation

### Retinal Vessels (Test Set)
```bash
python scripts/eval_drive.py \
  --checkpoint outputs/E5_size512_bs2_seed42/checkpoints/checkpoint_best.pth \
  --dataset drive --split test
```

### Roads (Test Set, 49 cases)
```bash
python scripts/eval_road.py \
  --checkpoint outputs/road_real_r2_seed42/checkpoints/checkpoint_epoch49.pth \
  --model_type sipv2_road --split test --deterministic
```

### ATW Evaluation
```bash
python scripts/eval_road_atw_test.py \
  --checkpoint outputs/road_atw_l015_seed42/checkpoints/checkpoint_final.pth \
  --model_type sipv2_road --split test --deterministic
```

## Expected Runtime

| Experiment | GPU | Epochs | Runtime |
|------------|-----|--------|---------|
| DRIVE E1/E4/E5 | RTX 3090 | 200 | ~15 min |
| CHASE_DB1 E5 | RTX 3090 | 200 | ~20 min |
| HRF E5 | RTX 3090 | 200 | ~20 min |
| Roads R0/R1/R2 | RTX 3090 | 50 | ~10 min |
| ATW fine-tuning | RTX 3090 | 15 | ~3 min |
| Gradient conflict | RTX 3090 | — | ~2 min |

## Key Checkpoints

All checkpoints are saved as `checkpoint_best.pth` (best validation Dice) and `checkpoint_final.pth` (last epoch). For road experiments, `checkpoint_epoch49.pth` is used as the ATW fine-tuning source.

## Random Seeds

All experiments use fixed seeds: 42, 0, 1. Set via `--seed` argument or `set_seed()` function.

## Contact

For questions about reproduction, please open an issue in the GitHub repository or contact the corresponding author.
