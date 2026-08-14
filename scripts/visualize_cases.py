#!/usr/bin/env python3
"""
Visualize predictions from a trained model.
Usage: python scripts/visualize_cases.py --exp E4 --epoch best
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def visualize_predictions(exp_name, epoch='best', output_dir='outputs'):
    """Visualize predictions from saved npz files."""
    exp_dir = os.path.join(output_dir, f'{exp_name}_size512_bs2_seed42')
    pred_dir = os.path.join(exp_dir, 'predictions')

    if not os.path.exists(pred_dir):
        print(f'No predictions found at {pred_dir}')
        return

    npz_files = [f for f in os.listdir(pred_dir) if f.endswith('.npz')]
    if len(npz_files) == 0:
        print(f'No .npz files in {pred_dir}')
        return

    # Select a few cases
    npz_files = sorted(npz_files)[:5]

    fig, axes = plt.subplots(len(npz_files), 4, figsize=(16, 4 * len(npz_files)))
    if len(npz_files) == 1:
        axes = axes.reshape(1, -1)

    for i, npz_file in enumerate(npz_files):
        data = np.load(os.path.join(pred_dir, npz_file))
        pred_prob = data['pred_prob']
        gt = data['gt']
        fov = data['fov']

        # Threshold
        pred_binary = (pred_prob > 0.5).astype(np.float32)

        # Skeletons
        from skimage.morphology import skeletonize
        skel_gt = skeletonize(gt > 0) if gt.sum() > 0 else np.zeros_like(gt)
        skel_pred = skeletonize(pred_binary > 0) if pred_binary.sum() > 0 else np.zeros_like(pred_binary)

        axes[i, 0].imshow(pred_prob * fov, cmap='jet', vmin=0, vmax=1)
        axes[i, 0].set_title('Probability')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_binary * fov, cmap='gray')
        axes[i, 1].set_title('Prediction')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt * fov, cmap='gray')
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')

        # Overlay
        overlay = np.zeros((*gt.shape, 3))
        overlay[..., 0] = pred_binary * fov  # Red = prediction
        overlay[..., 1] = gt * fov  # Green = GT
        overlay[..., 2] = fov * 0.3  # Blue background
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (R=Pred, G=GT)')
        axes[i, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(exp_dir, 'visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Visualization saved to {save_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='E4', help='Experiment name')
    parser.add_argument('--epoch', type=str, default='best', help='Epoch to visualize')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    visualize_predictions(args.exp, args.epoch, args.output_dir)


if __name__ == '__main__':
    main()
