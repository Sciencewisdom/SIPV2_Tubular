#!/usr/bin/env python3
"""Full-image sliding-window inference on Massachusetts Roads test set — audit item B6.

The paper's center-crop protocol evaluates 512x512 center crops; reviewers may
ask whether conclusions survive full-image evaluation. This script runs R0/R1/R2
(seed 42, final-epoch checkpoints) on all 49 full 1500x1500 test images with
overlapping 512 windows (stride 384, average pooling of overlapping logits),
then computes Dice / clDice / skeleton recall / gap recovery / JPR per image,
and writes a qualitative comparison figure for the supplement.

Requires torch (+CUDA recommended). Usage:
  python scripts/eval_road_fullimage.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sipv2.models import build_model
from sipv2.utils import load_checkpoint
from sipv2.metrics.region import dice_score
from sipv2.metrics.skeleton import cl_dice_score, skeleton_recall
from sipv2.metrics.road_topology import gap_recovery_rate
from sipv2.metrics.junction_preservation import compute_junction_preservation

CKPTS = {
    'R0': ('dw',        'outputs/road_real_r0_seed42/road_dw_crop512_bs8_ep50_seed42/checkpoints/checkpoint_epoch49.pth'),
    'R1': ('sipv2_road','outputs/road_real_r1_seed42/road_sipv2_road_crop512_bs8_ep50_seed42/checkpoints/checkpoint_epoch49.pth'),
    'R2': ('sipv2_road','outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth'),
}
WINDOW, STRIDE = 512, 384
OUT_JSON = 'outputs/road_fullimage_metrics.json'
OUT_FIG = 'paper_figures/road_fullimage_qualitative.png'
ROOT = 'data/raw/mass_roads/test'


def load_img(p):
    a = np.array(Image.open(p))
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return a.astype(np.float32) / 255.0


def load_mask(p):
    a = np.array(Image.open(p))
    if a.ndim == 3:
        a = a[..., 0]
    if a.max() > 1:
        a = a / 255.0
    return (a > 0.5).astype(np.float32)


@torch.no_grad()
def sliding_window_probs(model, img, device):
    """img: [H,W,3] float in [0,1] -> averaged probability map [H,W]."""
    h, w = img.shape[:2]
    ph = (WINDOW - h % STRIDE) % STRIDE
    pw = (WINDOW - w % STRIDE) % STRIDE
    imgp = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect')
    H, W = imgp.shape[:2]
    acc = torch.zeros(H, W, device=device)
    cnt = torch.zeros(H, W, device=device)
    xs = list(range(0, W - WINDOW + 1, STRIDE)) + [W - WINDOW]
    ys = list(range(0, H - WINDOW + 1, STRIDE)) + [H - WINDOW]
    xs, ys = sorted(set(xs)), sorted(set(ys))
    for y in ys:
        batch = []
        for x in xs:
            t = torch.from_numpy(imgp[y:y+WINDOW, x:x+WINDOW]).permute(2, 0, 1).float()
            batch.append(t)
        b = torch.stack(batch).to(device)
        p = torch.sigmoid(model(b, image=b))[:, 0]
        for i, x in enumerate(xs):
            acc[y:y+WINDOW, x:x+WINDOW] += p[i]
            cnt[y:y+WINDOW, x:x+WINDOW] += 1
    return (acc / cnt)[:h, :w].cpu().numpy()


def metrics_for(prob, mask, threshold=0.5):
    pred = prob > threshold
    jpr, _ = compute_junction_preservation(pred, mask > 0)
    return {
        'dice': float(dice_score(prob, mask, threshold)),
        'cldice': float(cl_dice_score(prob, mask, threshold)),
        'skelrec': float(skeleton_recall(prob, mask, threshold)),
        'gaprec': float(gap_recovery_rate(prob, mask, threshold)),
        'jpr': None if np.isnan(jpr) else float(jpr),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_paths = sorted(glob.glob(os.path.join(ROOT, 'sat', '*.tif*')))
    gt_paths = sorted(glob.glob(os.path.join(ROOT, 'map', '*.tif*')))
    assert len(img_paths) == len(gt_paths) == 49
    print(f'{len(img_paths)} test images, device={device}')

    results, gallery = {}, {}
    for name, (block, ckpt) in CKPTS.items():
        assert os.path.exists(ckpt), ckpt
        kwargs = dict(in_channels=3, num_classes=1, channels=[32, 64, 128, 256],
                      blocks_per_stage=[2, 2, 2, 2], decoder_blocks=1)
        if block == 'sipv2_road':
            kwargs.update(directions=16, use_confidence_gate=True)
        model = build_model(block_type=block, **kwargs).to(device)
        load_checkpoint(model, None, ckpt, device)
        model.eval()
        per = []
        for ip, gp in tqdm(list(zip(img_paths, gt_paths)), desc=name):
            img, mask = load_img(ip), load_mask(gp)
            prob = sliding_window_probs(model, img, device)
            m = metrics_for(prob, mask)
            m['image_id'] = os.path.basename(ip).split('.')[0]
            per.append(m)
            gallery.setdefault(os.path.basename(ip), {})[name] = prob
        results[name] = per
        keys = ['dice', 'cldice', 'skelrec', 'gaprec']
        means = {k: float(np.mean([m[k] for m in per])) for k in keys}
        jprs = [m['jpr'] for m in per if m['jpr'] is not None]
        means['jpr'] = float(np.mean(jprs))
        print(name, {k: round(v, 4) for k, v in means.items()})

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', OUT_JSON)

    # qualitative figure: 3 images x (input, GT, R0, R1, R2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids = [os.path.basename(p) for p in img_paths]
    picks = [ids[10], ids[25], ids[40]]
    fig, axes = plt.subplots(len(picks), 5, figsize=(20, 4 * len(picks)))
    for r, pid in enumerate(picks):
        i = ids.index(pid)
        img, mask = load_img(img_paths[i]), load_mask(gt_paths[i])
        axes[r, 0].imshow(img); axes[r, 0].set_ylabel(pid.split('.')[0])
        axes[r, 1].imshow(mask, cmap='gray')
        for c, name in enumerate(['R0', 'R1', 'R2']):
            axes[r, c+2].imshow(gallery[pid][name] > 0.5, cmap='gray')
    for ax, t in zip(axes[0], ['Input', 'GT', 'R0 (DW)', 'R1 (SIP-v2)', 'R2 (+clDice)']):
        ax.set_title(t)
    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print('wrote', OUT_FIG)


if __name__ == '__main__':
    main()
