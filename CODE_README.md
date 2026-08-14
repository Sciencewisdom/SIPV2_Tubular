# SIP-v2 Code Documentation

## Quick Start

### Training
```bash
# DRIVE E5
python scripts/train.py --exp E5 --dataset DRIVE --seed 42

# CHASE_DB1 E5
python scripts/train.py --exp E5 --dataset CHASE_DB1 --seed 42

# HRF E5
python scripts/train.py --exp E5 --dataset HRF --seed 42
```

### Experiments
| Experiment | Description | Command |
|------------|-------------|---------|
| E0 | U-Net baseline | `--exp E0` |
| E1 | Depthwise separable | `--exp E1` |
| E2 | Isotropic diffusion | `--exp E2` |
| E3 | Old SIP (free tensor) | `--exp E3` |
| E4 | SIP-v2 Minimal | `--exp E4` |
| E5 | SIP-v2 Full | `--exp E5` |

### Key Files
- `sipv2/models/blocks_sipv2_full.py` - Full SIP-v2 block
- `sipv2/ops/structure_tensor.py` - Structure tensor computation
- `sipv2/ops/directional_diffusion.py` - Diffusion operator
- `scripts/train.py` - Main training script

### Output Structure
```
outputs/
  {exp}_size{size}_bs{bs}_seed{seed}/
    checkpoints/
    logs/
    predictions/
    summary.json
```
