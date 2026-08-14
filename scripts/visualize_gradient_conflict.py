#!/usr/bin/env python3
"""Visualize gradient conflict maps for paper figures."""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def visualize_case(case_data, output_path):
    """Create a multi-panel figure showing gradient conflict for one case."""
    pred = np.array(case_data['pred_prob'])[0]
    mask = np.array(case_data['mask'])[0]
    grad_c = np.array(case_data['grad_cldice'])[0]
    grad_g = np.array(case_data['grad_gap'])[0]
    sign_conflict = np.array(case_data['sign_conflict_map'])[0]
    norm_conflict = np.array(case_data['normalized_conflict'])[0]

    # Define region of interest: dilated mask
    roi = binary_dilation(mask > 0, iterations=5)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: inputs and predictions
    ax = axes[0, 0]
    ax.imshow(mask, cmap='gray')
    ax.set_title('Ground Truth Mask')
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(pred, cmap='hot', vmin=0, vmax=1)
    ax.set_title('Prediction Probability')
    ax.axis('off')

    ax = axes[0, 2]
    vmax_c = max(np.abs(grad_c[roi]).max(), 1e-10)
    im = ax.imshow(grad_c * roi, cmap='RdBu_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_c, vmax=vmax_c))
    ax.set_title(r'$\nabla L_{clDice}$')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 3]
    vmax_g = max(np.abs(grad_g[roi]).max(), 1e-10)
    im = ax.imshow(grad_g * roi, cmap='RdBu_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_g, vmax=vmax_g))
    ax.set_title(r'$\nabla L_{gap}$')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: conflict analysis
    ax = axes[1, 0]
    conflict_mask = (sign_conflict < 0) & roi
    display = np.zeros((*sign_conflict.shape, 3))
    display[roi & (sign_conflict > 0)] = [0, 1, 0]  # agree: green
    display[roi & (sign_conflict < 0)] = [1, 0, 0]  # conflict: red
    display[roi & (sign_conflict == 0)] = [0.5, 0.5, 0.5]  # neutral: gray
    ax.imshow(display)
    ax.set_title('Sign Conflict (Red=Conflict)')
    ax.axis('off')

    ax = axes[1, 1]
    vmax_n = max(np.abs(norm_conflict[roi]).max(), 1e-10)
    im = ax.imshow(norm_conflict * roi, cmap='RdBu_r',
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_n, vmax=vmax_n))
    ax.set_title('Normalized Conflict')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    overlay = plt.cm.gray(pred)
    overlay[conflict_mask] = [1, 0, 0, 1]  # red overlay on conflict pixels
    ax.imshow(overlay)
    ax.set_title('Conflict Overlay on Pred')
    ax.axis('off')

    ax = axes[1, 3]
    overlay = plt.cm.gray(mask)
    overlay[conflict_mask] = [1, 0, 0, 1]
    ax.imshow(overlay)
    ax.set_title('Conflict Overlay on GT')
    ax.axis('off')

    fig.suptitle(f"Case: {case_data['image_id']} | "
                 f"cos_sim={case_data['global_cos_sim']:.3f} | "
                 f"L_cldice={case_data['loss_cldice']:.3f} | "
                 f"L_gap={case_data['loss_gap']:.3f}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/gradient_conflict/figures')
    parser.add_argument('--max_cases', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    for i, case in enumerate(data[:args.max_cases]):
        name = case['image_id'].replace('/', '_')
        out_path = os.path.join(args.output_dir, f'conflict_{name}.png')
        visualize_case(case, out_path)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
