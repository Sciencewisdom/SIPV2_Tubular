#!/usr/bin/env python3
"""Robustness decay curves for SIP-v2 vs matched CNN — audit item A4.

Quantifies the robustness boundary of gradient anchoring: DRIVE validation
images (last 4 of the training split, the same split used for the break-count
analysis) are degraded with (a) additive Gaussian noise and (b) contrast
reduction at 5 levels each, and Dice / clDice / skeleton recall are recomputed
per case. Produces a JSON + a decay-curve figure for the supplement.

NOTE: DRIVE test ground truth is not publicly available in this repo, so this
runs on the 4-image validation split. That is sufficient for a *relative*
decay comparison (E1 vs E5 under identical degradations), which is what the
reviewer question asks for.

Requires torch + CUDA. Usage:
  python scripts/robustness_decay_curve.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from sipv2.models import build_experiment_model
from sipv2.utils import load_checkpoint
from sipv2.datasets.drive import DRIVEDataset
from sipv2.metrics.region import dice_score
from sipv2.metrics.skeleton import cl_dice_score, skeleton_recall
from torch.utils.data import DataLoader

CKPTS = {
    'E1': ('E1', 'outputs/E1_size512_bs2_seed42/checkpoints/checkpoint_best.pth'),
    'E5': ('E5', 'outputs/E5_size512_bs2_seed42/E5_size512_bs2_seed42/checkpoints/checkpoint_best.pth'),
}
NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.20]      # on [0,1] images
CONTRAST_SCALES = [1.0, 0.8, 0.6, 0.4, 0.2]       # around per-image mean
OUT_JSON = 'outputs/robustness_decay_drive.json'
OUT_FIG = 'paper_figures/robustness_decay_drive.png'


def degrade(images, kind, level, rng):
    """images: [B,3,H,W] float in [0,1]. Returns degraded copy."""
    x = images.clone()
    if kind == 'noise':
        x = x + torch.from_numpy(
            rng.normal(0, level, size=x.shape).astype(np.float32)).to(x.device)
    else:  # contrast: scale deviation from per-image mean
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = mean + level * (x - mean)
    return x.clamp(0, 1)


@torch.no_grad()
def eval_level(model, loader, device, kind, level, rng, threshold=0.5):
    per_case = []
    for batch in loader:
        images = batch['image'].to(device)
        masks = batch['mask'][:, 0].numpy()
        fovs = batch['fov_mask'][:, 0].numpy()
        images = degrade(images, kind, level, rng)
        probs = torch.sigmoid(model(images, image=images))[:, 0].cpu().numpy()
        for prob, mask, fov in zip(probs, masks, fovs):
            prob = prob * (fov > 0)  # FOV-masked metrics, matching paper protocol
            per_case.append({
                'dice': float(dice_score(prob, mask, threshold)),
                'cldice': float(cl_dice_score(prob, mask, threshold)),
                'skelrec': float(skeleton_recall(prob, mask, threshold)),
            })
    keys = per_case[0].keys()
    return {k: float(np.mean([c[k] for c in per_case])) for k in keys}, per_case


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = DRIVEDataset(root_dir=os.path.join('data', 'raw', 'DRIVE'),
                      split='val', img_size=512, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print(f'Validation cases: {len(ds)}')

    results = {}
    for name, (exp, ckpt) in CKPTS.items():
        assert os.path.exists(ckpt), f'missing checkpoint: {ckpt}'
        model = build_experiment_model(exp, in_channels=3, num_classes=1,
                                       channels=[32, 64, 128, 256],
                                       blocks_per_stage=[2, 2, 2, 2],
                                       decoder_blocks=1).to(device)
        load_checkpoint(model, None, ckpt, device)
        model.eval()
        results[name] = {'noise': {}, 'contrast': {}}
        for kind, levels in [('noise', NOISE_SIGMAS), ('contrast', CONTRAST_SCALES)]:
            for lv in tqdm(levels, desc=f'{name} {kind}'):
                rng = np.random.default_rng(42)  # same noise field across models/levels
                mean, per_case = eval_level(model, loader, device, kind, lv, rng)
                results[name][kind][str(lv)] = {'mean': mean, 'per_case': per_case}

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'wrote {OUT_JSON}')

    # Figure: 2 degradations x 3 metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('dice', 'Dice'), ('cldice', 'clDice'), ('skelrec', 'Skeleton recall')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (kind, levels, xlabel) in enumerate([
            ('noise', NOISE_SIGMAS, 'Gaussian noise σ'),
            ('contrast', CONTRAST_SCALES, 'Contrast scale')]):
        for col, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            for name, color in [('E1', '#0072B2'), ('E5', '#D55E00')]:
                ys = [results[name][kind][str(lv)]['mean'][key] for lv in levels]
                ax.plot(levels, ys, 'o-', color=color,
                        label={'E1': 'E1 DW-CNN', 'E5': 'E5 SIP-v2 Full'}[name])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    fig.suptitle('Robustness decay on DRIVE validation (n=4): matched CNN vs SIP-v2')
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    print(f'wrote {OUT_FIG}')


if __name__ == '__main__':
    main()
