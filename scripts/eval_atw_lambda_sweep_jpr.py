#!/usr/bin/env python3
"""Per-case JPR for ATW lambda sweep checkpoints (B7): extends
eval_atw_percase_jpr.py to lambda in {0.1, 0.2} runs so we can build a
lambda x seed sensitivity table for the supplement.

Usage: python scripts/eval_atw_lambda_sweep_jpr.py
Output: outputs/atw_lambda_sweep_jpr.json
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from sipv2.models import build_model
from sipv2.utils import load_checkpoint
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.metrics.junction_preservation import compute_junction_preservation

CKPTS = {
    'ATW_l0.1_seed0':  'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed0_atw0.1/checkpoints/checkpoint_final.pth',
    'ATW_l0.1_seed42': 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.1/checkpoints/checkpoint_final.pth',
    'ATW_l0.2_seed42': 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.2/checkpoints/checkpoint_final.pth',
}
OUT = 'outputs/atw_lambda_sweep_jpr.json'


def eval_jpr(ckpt_path, test_loader, device, threshold=0.5):
    model = build_model(block_type='sipv2_road', in_channels=3, num_classes=1,
                        channels=[32, 64, 128, 256], blocks_per_stage=[2, 2, 2, 2],
                        decoder_blocks=1, directions=16, use_confidence_gate=True)
    model = model.to(device)
    load_checkpoint(model, None, ckpt_path, device)
    model.eval()
    per_case = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=os.path.basename(ckpt_path)):
            images = batch['image'].to(device)
            masks = batch['mask'][:, 0].cpu().numpy()
            probs = torch.sigmoid(model(images, image=images))[:, 0].cpu().numpy()
            for pred, mask in zip(probs, masks):
                jpr, _ = compute_junction_preservation(pred > threshold, mask > 0)
                per_case.append(None if np.isnan(jpr) else float(jpr))
    return per_case


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = MassachusettsRoadsDataset(root_dir='data/raw/mass_roads', split='test',
                                             crop_size=512, augment=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                              num_workers=0, pin_memory=True)
    data = {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: {path} missing')
            continue
        data[name] = eval_jpr(path, test_loader, device)
        arr = np.array([x for x in data[name] if x is not None])
        print(f'{name}: n={len(arr)}, mean JPR={arr.mean():.4f} +- {arr.std():.4f}')

    with open(OUT, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
