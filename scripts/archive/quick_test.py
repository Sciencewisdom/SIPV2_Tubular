#!/usr/bin/env python3
"""
Quick integration test using synthetic data.
Verifies training pipeline works without requiring DRIVE dataset.
Optimized for RTX 4050 (6GB VRAM).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sipv2.models import build_experiment_model
from sipv2.losses import BCEDiceLoss
from sipv2.engine import train_one_epoch, validate
from sipv2.utils import set_seed


class SyntheticDataset(Dataset):
    """Synthetic retinal-like images for quick testing."""
    def __init__(self, n_samples=20, size=256):
        self.n_samples = n_samples
        self.size = size
        np.random.seed(42)
        self.images = []
        self.masks = []
        for _ in range(n_samples):
            # Random background with vessel-like lines
            img = np.random.rand(3, size, size).astype(np.float32) * 0.3
            mask = np.zeros((1, size, size), dtype=np.float32)
            # Add random lines
            n_lines = np.random.randint(3, 8)
            for _ in range(n_lines):
                y = np.random.randint(20, size - 20)
                x_start = np.random.randint(0, size // 2)
                x_end = np.random.randint(size // 2, size)
                thickness = np.random.randint(1, 4)
                cv2.line(mask[0], (x_start, y), (x_end, y), 1.0, thickness)
                # Add to green channel
                cv2.line(img[1], (x_start, y), (x_end, y), 0.8, thickness)
            self.images.append(img)
            self.masks.append(mask)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'image': torch.from_numpy(self.images[idx]),
            'mask': torch.from_numpy(self.masks[idx]),
            'fov_mask': torch.ones((1, self.size, self.size), dtype=torch.float32),
            'image_id': f'synth_{idx:03d}',
        }


def quick_test():
    print("=" * 60)
    print("SIP-v2 Quick Integration Test")
    print("=" * 60)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create synthetic data
    print("\nCreating synthetic dataset...")
    train_ds = SyntheticDataset(n_samples=16, size=256)
    val_ds = SyntheticDataset(n_samples=4, size=256)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    # Test each experiment
    experiments = ['E0', 'E1', 'E2', 'E3', 'E4']
    exp_names = ['U-Net', 'DW', 'IsoDiffusion', 'OldSIP', 'SIP-v2']
    block_types = ['conv', 'dw', 'iso', 'old_sip', 'sipv2']

    for exp, name, bt in zip(experiments, exp_names, block_types):
        print(f"\n{'='*60}")
        print(f"Testing {exp}: {name}")
        print(f"{'='*60}")

        model = build_experiment_model(
            exp,
            in_channels=3,
            num_classes=1,
            channels=[32, 64, 128, 256],
            blocks_per_stage=[2, 2, 2, 2],
            decoder_blocks=1,
        ).to(device)

        n_params = model.count_parameters()
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

        # Check forward pass
        dummy = torch.randn(1, 3, 256, 256).to(device)
        with torch.no_grad():
            if bt == 'sipv2':
                out = model(dummy, image=dummy)
            else:
                out = model(dummy)
        print(f"  Forward: {dummy.shape} -> {out.shape}")

        # Train 2 epochs
        criterion = BCEDiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(2):
            loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch=epoch, use_amp=True, grad_clip=1.0, block_type=bt
            )
            print(f"  Epoch {epoch}: loss={loss:.4f}")

        # Validate
        metrics = validate(
            model, val_loader, criterion, device,
            epoch=1, block_type=bt,
        )
        print(f"  Val Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("All experiments passed quick test!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Download DRIVE dataset")
    print("  2. Run: python scripts/prepare_drive.py")
    print("  3. Run: bash scripts/run_drive_experiments.sh")


if __name__ == '__main__':
    try:
        import cv2
    except ImportError:
        print("Installing opencv-python...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python'])
        import cv2
    quick_test()
