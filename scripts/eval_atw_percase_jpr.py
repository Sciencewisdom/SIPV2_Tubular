#!/usr/bin/env python3
"""Per-case JPR evaluation for ATW vs R2 across seeds — answers reviewer point on
the missing paired statistical test for the ATW +16.0% JPR claim.

Runs inference with existing checkpoints (R2 seeds {0,1,42}, ATW lambda=0.15
seeds {0,1,42}), dumps per-case JPR JSON, then reports:
  - paired Wilcoxon ATW vs R2 per seed and on seed-averaged per-case values
  - bootstrap 95% CI for the mean paired difference (10k resamples)

Requires torch + CUDA. Usage:
  python scripts/eval_atw_percase_jpr.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import wilcoxon

from sipv2.models import build_model
from sipv2.utils import load_checkpoint
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.metrics.junction_preservation import compute_junction_preservation

CKPTS = {
    ('R2', 0):  'outputs/road_real_r2_seed0/road_sipv2_road_crop512_bs8_ep50_seed0_cldice0.3/checkpoints/checkpoint_epoch49.pth',
    ('R2', 1):  'outputs/road_real_r2_seed1/road_sipv2_road_crop512_bs8_ep50_seed1_cldice0.3/checkpoints/checkpoint_epoch49.pth',
    ('R2', 42): 'outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth',
    ('ATW', 0):  'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed0_atw0.15/checkpoints/checkpoint_final.pth',
    ('ATW', 1):  'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed1_atw0.15/checkpoints/checkpoint_final.pth',
    ('ATW', 42): 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.15/checkpoints/checkpoint_final.pth',
}
OUT = 'outputs/atw_percase_jpr.json'


def eval_jpr(ckpt_path, test_loader, device, threshold=0.5):
    model = build_model(block_type='sipv2_road', in_channels=3, num_classes=1,
                        channels=[32, 64, 128, 256], blocks_per_stage=[2, 2, 2, 2],
                        decoder_blocks=1, directions=16, use_confidence_gate=True)
    model = model.to(device)
    load_checkpoint(model, None, ckpt_path, device)
    model.eval()
    per_case = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=os.path.basename(ckpt_path))):
            images = batch['image'].to(device)
            masks = batch['mask'][:, 0].cpu().numpy()
            probs = torch.sigmoid(model(images, image=images))[:, 0].cpu().numpy()
            for pred, mask in zip(probs, masks):
                jpr, _ = compute_junction_preservation(pred > threshold, mask > 0)
                per_case.append(None if np.isnan(jpr) else float(jpr))
    return per_case


def boot_ci(diff, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = MassachusettsRoadsDataset(root_dir='data/raw/mass_roads', split='test',
                                             crop_size=512, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                              num_workers=4, pin_memory=True)

    data = {}
    for (run, seed), path in CKPTS.items():
        if not os.path.exists(path):
            print(f'SKIP {run} seed{seed}: {path} missing')
            continue
        data[f'{run}_seed{seed}'] = eval_jpr(path, test_loader, device)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'saved {OUT}')

    # paired tests on cases where both sides have a JPR value
    for seed in (0, 1, 42):
        a, b = data.get(f'R2_seed{seed}'), data.get(f'ATW_seed{seed}')
        if not a or not b:
            continue
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        a2 = np.array([p[0] for p in pairs]); b2 = np.array([p[1] for p in pairs])
        p = wilcoxon(a2, b2).pvalue
        lo, hi = boot_ci(b2 - a2)
        print(f'seed {seed}: n={len(pairs)}, mean JPR R2={a2.mean():.3f} ATW={b2.mean():.3f}, '
              f'Δ={b2.mean()-a2.mean():+.4f}, wilcoxon p={p:.4f}, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]')

    # seed-averaged per-case test
    common = [s for s in (0, 1, 42) if f'R2_seed{s}' in data and f'ATW_seed{s}' in data]
    if len(common) == 3:
        n = len(data['R2_seed42'])
        avg_r2, avg_atw = [], []
        for i in range(n):
            xs = [data[f'R2_seed{s}'][i] for s in common]
            ys = [data[f'ATW_seed{s}'][i] for s in common]
            if all(v is not None for v in xs + ys):
                avg_r2.append(np.mean(xs)); avg_atw.append(np.mean(ys))
        a = np.array(avg_r2); b = np.array(avg_atw)
        p = wilcoxon(a, b).pvalue
        lo, hi = boot_ci(b - a)
        print(f'seed-averaged: n={len(a)}, Δ={b.mean()-a.mean():+.4f}, '
              f'wilcoxon p={p:.4f}, bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]')


if __name__ == '__main__':
    main()
