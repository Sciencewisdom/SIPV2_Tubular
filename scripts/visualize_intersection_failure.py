#!/usr/bin/env python3
"""
Visualize intersection failure cases across R0/R1/R2.
Shows: original image, GT junctions, predicted junctions for each method.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from PIL import Image

from sipv2.models import build_model
from sipv2.utils import load_checkpoint
from sipv2.metrics.junction_preservation import skeleton_to_graph
from sipv2.datasets.mass_roads import get_mass_roads_loaders


CHECKPOINTS = {
    'R0 (DW)': 'outputs/road_real_r0_seed0/road_dw_crop512_bs8_ep50_seed0/checkpoints/checkpoint_best.pth',
    'R1 (SIP-v2)': 'outputs/road_real_r1_seed42/road_sipv2_road_crop512_bs8_ep50_seed42/checkpoints/checkpoint_epoch49.pth',
    'R2 (+clDice)': 'outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth',
}

COLORS = {
    'R0 (DW)': '#1f77b4',
    'R1 (SIP-v2)': '#ff7f0e',
    'R2 (+clDice)': '#2ca02c',
}


def load_model(ckpt_path, block_type, device):
    model = build_model(
        block_type=block_type,
        in_channels=3, num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=16,
        use_confidence_gate=True,
    )
    model = model.to(device)
    load_checkpoint(model, None, ckpt_path, device)
    model.eval()
    return model


def predict(model, image_tensor, device, block_type='dw'):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        if block_type in ('sipv2', 'sipv2_road'):
            logits = model(image_tensor, image=image_tensor)
        else:
            logits = model(image_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
    return prob


def get_junctions(mask, threshold=0.5):
    skel = skeletonize(mask > threshold)
    coords, degrees = skeleton_to_graph(skel)
    junctions = [(y, x) for (y, x), d in zip(coords, degrees) if d >= 3]
    return junctions


def visualize_case(image_np, mask_gt, probs_dict, case_id, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Intersection Failure Analysis: {case_id}', fontsize=14, fontweight='bold')

    methods = list(probs_dict.keys())

    # Row 0: Original + GT junctions
    ax = axes[0, 0]
    ax.imshow(image_np.transpose(1, 2, 0))
    ax.set_title('Input Image')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(mask_gt, cmap='gray')
    gt_junctions = get_junctions(mask_gt)
    for y, x in gt_junctions:
        ax.plot(x, y, 'ro', markersize=6, alpha=0.8)
    ax.set_title(f'GT Mask + Junctions (n={len(gt_junctions)})')
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(image_np.transpose(1, 2, 0))
    for y, x in gt_junctions:
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1, alpha=0.9)
    ax.set_title('GT Junctions on Image')
    ax.axis('off')

    # Row 1: R0, R1, R2 predictions with junctions
    for idx, method in enumerate(methods):
        ax = axes[1, idx]
        prob = probs_dict[method]
        ax.imshow(prob, cmap='jet', vmin=0, vmax=1)
        pred_junctions = get_junctions(prob)
        for y, x in pred_junctions:
            ax.plot(x, y, 'wo', markersize=5, alpha=0.7)
        # Count preserved junctions
        from scipy.spatial import cKDTree
        if len(pred_junctions) > 0 and len(gt_junctions) > 0:
            pred_tree = cKDTree(pred_junctions)
            preserved = sum(1 for tj in gt_junctions if pred_tree.query(tj, k=1)[0] <= 2)
        else:
            preserved = 0
        jpr = preserved / max(len(gt_junctions), 1)
        ax.set_title(f'{method}\nPred Junctions: {len(pred_junctions)} | JPR: {jpr:.2f}')
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'intersection_failure_{case_id}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = 'outputs/paper_figures'
    os.makedirs(out_dir, exist_ok=True)

    # Load models
    models = {}
    models['R0 (DW)'] = load_model(CHECKPOINTS['R0 (DW)'], 'dw', device)
    models['R1 (SIP-v2)'] = load_model(CHECKPOINTS['R1 (SIP-v2)'], 'sipv2_road', device)
    models['R2 (+clDice)'] = load_model(CHECKPOINTS['R2 (+clDice)'], 'sipv2_road', device)

    # Load test data
    _, val_loader = get_mass_roads_loaders(
        root_dir='data/raw/mass_roads',
        crop_size=512, batch_size=1, num_workers=2,
    )

    # Select cases with many junctions
    selected_cases = []
    for batch in val_loader:
        image_tensor = batch['image'][0]
        image = image_tensor.numpy()
        mask = batch['mask'][0, 0].numpy()
        gt_junctions = get_junctions(mask)
        if len(gt_junctions) >= 3:
            selected_cases.append((batch['image_id'][0], image, mask, image_tensor))
        if len(selected_cases) >= 6:
            break

    print(f"Visualizing {len(selected_cases)} cases with junctions...")
    for case_id, image, mask, image_tensor in selected_cases:
        probs = {}
        probs['R0 (DW)'] = predict(models['R0 (DW)'], image_tensor, device, 'dw')
        probs['R1 (SIP-v2)'] = predict(models['R1 (SIP-v2)'], image_tensor, device, 'sipv2_road')
        probs['R2 (+clDice)'] = predict(models['R2 (+clDice)'], image_tensor, device, 'sipv2_road')
        visualize_case(image, mask, probs, case_id, out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
