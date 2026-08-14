# SIP-v2 Reproduction Guide

This document provides step-by-step instructions to reproduce all experiments reported in the SIP-v2 paper.

## Environment Setup

```bash
# Clone repository
git clone <repo-url>
cd SIPV2_Tubular

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sipv2; print('OK')"
```

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 6GB VRAM (tested on RTX 4050)
- **Recommended**: NVIDIA GPU with 12GB+ VRAM for faster training
- **Storage**: ~20GB free space (datasets + outputs)

## Datasets

Download datasets into `data/raw/`:

| Dataset | Source | Expected Path |
|---------|--------|---------------|
| DRIVE | https://drive.grand-challenge.org/ | `data/raw/DRIVE/` |
| CHASE_DB1 | https://blogs.kingston.ac.uk/retinal/chasedb1/ | `data/raw/CHASE_DB1/` |
| HRF | https://www5.cs.fau.de/research/data/fundus-images/ | `data/raw/HRF/` |
| Mass Roads | https://www.cs.toronto.edu/~vmnih/data/ | `data/raw/mass_roads/` |

## Phase 1: Ablation Study (Retinal Vessels)

### Q1-Q2: Baseline Comparison

```bash
bash scripts/run_drive_experiments.sh
```

This runs E0 (U-Net), E1 (DW), E2 (IsoDiffusion), E3 (OldSIP), E4 (SIP-v2 Min) on DRIVE.

### Q3: clDice Sensitivity Sweep

```bash
bash scripts/run_p01_cldice_sweep.sh
```

Sweeps clDice weight λ ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 1.0} for both E1 and E4.

### Q4: Full Architecture (Multi-dataset)

```bash
# DRIVE
bash scripts/run_e5_drive.sh

# CHASE_DB1
bash scripts/run_e5_chasedb1.sh

# HRF
bash scripts/run_e5_hrf.sh
```

### Q5: Multi-seed Stability

```bash
# Retinal datasets
bash scripts/run_e5_multiseed.sh
bash scripts/run_e5_chasedb1_multiseed.sh
bash scripts/run_hrf_e5_multiseed.sh
```

## Phase 2: Cross-Domain Validation (Road Networks)

### Synthetic Roads

```bash
# Generate synthetic road dataset (if needed)
python scripts/generate_synthetic_roads.py

# Train R0-R2 with 3 seeds
bash scripts/run_road_multiseed.sh
```

### Real Massachusetts Roads

```bash
# Download real roads dataset first
python scripts/download_mass_roads.py

# Train R0-R2 with 3 seeds
bash scripts/run_real_road_multiseed.sh
```

## Deterministic Evaluation

```bash
# Retinal experiments: evaluation is automatic during training
# Check outputs/*/summary.json for metrics

# Road experiments (deterministic center-crop evaluation)
python scripts/eval_road_deterministic.py \
    --checkpoint outputs/road_sipv2_50ep/.../checkpoint_final.pth \
    --block_type sipv2_road
```

## Phase 3: Interaction Generality Controls (Supplement S12/S13)

```bash
# B1: second topology loss (soft skeleton recall) sweep, E1 vs E4 on DRIVE
bash scripts/run_b1_skelrec_sweep.sh
python scripts/analyze_b1_skelrec.py            # -> Table S12 / Fig S12 numbers
python scripts/recompute_drive_breaks_b1.py     # branch-break counts for B1 runs

# B2: parameter-matched DCNv2 control (E1D) under the same clDice sweep
bash scripts/run_b2_dcn_sweep.sh

# B3: road-component ablation (Scharr/Sobel, stencil, isotropy gate)
bash scripts/run_b3_road_ablation.sh
bash scripts/run_b3_eval.sh                     # deterministic per-case evaluation
bash scripts/run_b3_scharr_bs4_control.sh       # bs4 Scharr control (batch-size confound)

# Joint analysis of B2+B3
python scripts/analyze_b2_b3.py

# GPU-memory-safe relay (runs B3 then B2 sequentially on a 12GB card)
bash scripts/run_b3_b2_chain.sh
```

## Statistical Analysis

```bash
# Wilcoxon tests on real road multi-seed results
python scripts/wilcoxon_road_real.py

# Wilcoxon tests on synthetic road results
python scripts/wilcoxon_road_multiseed.py

# Tensor alignment analysis
python scripts/analyze_tensor_alignment.py
```

## Figure Generation

```bash
# Generate clDice response curve (Figure 3)
python scripts/generate_p01_final_response.py

# Generate multi-seed stability plot (Figure 6)
python scripts/plot_multiseed_road.py

# Generate comprehensive paper figures
python scripts/generate_paper_figures.py
```

## Expected Runtime

| Experiment | Dataset | Epochs | GPU | Runtime |
|------------|---------|--------|-----|---------|
| E0-E4 | DRIVE | 200 | RTX 3090 | ~13 min each |
| E5 | DRIVE | 200 | RTX 3090 | ~15 min |
| E5 | CHASE_DB1 | 200 | RTX 3090 | ~15 min |
| E5 | HRF | 200 | RTX 3090 | ~25 min |
| R0-R2 (real roads) | Mass Roads | 50 | RTX 3090 | ~7 min each |

## Output Structure

```
outputs/
  {exp}_size{size}_bs{bs}_seed{seed}/
    checkpoints/          # model checkpoints
    logs/                 # tensorboard logs
    predictions/          # validation predictions
    summary.json          # final metrics
```

## Reproducing Paper Tables

| Table | Script / Command |
|-------|-----------------|
| Table 1 (Q1) | `scripts/run_drive_experiments.sh` → check E3 vs E4 |
| Table 2 (Q2) | `scripts/run_drive_experiments.sh` → check E2 vs E4 |
| Table 3 (Q3) | `scripts/run_p01_cldice_sweep.sh` → check λ sweep |
| Table 4 (Q4) | `scripts/run_e5_*.sh` → check E5 across datasets |
| Table 5 (Complexity) | `scripts/compare_experiments.py` |
| Table 6 (Q6 synthetic) | `scripts/train_road.py` |
| Table 7 (Q6 real) | `scripts/run_real_road_experiments.sh` |
| Table 8 (Q6 multiseed) | `scripts/run_real_road_multiseed.sh` |

## Contact

For reproduction issues, please open a GitHub issue or contact the authors.
