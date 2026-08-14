#!/usr/bin/env python3
"""
Gradient conflict analysis between topology loss (clDice) and gap recovery.
Produces spatial conflict maps for mechanism-level interpretation.
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy import ndimage

from sipv2.models import build_model
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.utils import set_seed, load_checkpoint
from torch.utils.data import DataLoader
from torch.amp import autocast


class DifferentiableGapRecovery(nn.Module):
    """Differentiable approximation of gap recovery rate."""
    def __init__(self, dilate_iter=2):
        super().__init__()
        self.dilate_iter = dilate_iter

    def _soft_skeletonize(self, x, num_iter=5):
        for _ in range(num_iter):
            x = x * F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

    def forward(self, pred_prob, target):
        skel_pred = self._soft_skeletonize(pred_prob)
        # Precompute dilated target skeleton (hard, detached)
        skel_target = skeletonize(target.squeeze().cpu().numpy() > 0)
        dilated = ndimage.binary_dilation(skel_target, iterations=self.dilate_iter)
        dilated = torch.from_numpy(dilated).float().to(pred_prob.device)
        dilated = dilated.view_as(pred_prob)

        recall = (skel_pred * dilated).sum() / (skel_pred.sum() + 1e-8)
        return 1.0 - recall  # Loss = 1 - gap_recovery


class CLDiceLossGrad(nn.Module):
    """clDice loss identical to sipv2.losses.cldice but here for clarity."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def _soft_skeletonize(self, x, num_iter=5):
        for _ in range(num_iter):
            x = x * F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

    def forward(self, pred_prob, target):
        skel_pred = self._soft_skeletonize(pred_prob)
        skel_target = self._soft_skeletonize(target.float())
        tprec = (skel_target * pred_prob).sum() / (skel_pred.sum() + self.smooth)
        tsens = (skel_pred * target.float()).sum() / (skel_target.sum() + self.smooth)
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


def compute_gradient_conflict(model, images, masks, device, block_type='dw'):
    """Compute gradient conflict maps for a single batch."""
    images = images.to(device)
    masks = masks.to(device)
    images.requires_grad = False
    masks.requires_grad = False

    model.eval()

    # Forward pass with gradient tracking on logits
    with autocast('cuda'):
        if block_type in ('sipv2', 'sipv2_road'):
            logits = model(images, image=images)
        else:
            logits = model(images)

    pred_prob = torch.sigmoid(logits)

    # Loss 1: clDice
    cldice_loss_fn = CLDiceLossGrad()
    loss_cldice = cldice_loss_fn(pred_prob, masks)

    # Loss 2: Differentiable gap recovery
    gap_loss_fn = DifferentiableGapRecovery()
    loss_gap = gap_loss_fn(pred_prob, masks)

    # Compute gradients w.r.t. logits (not pred_prob) to capture full backprop signal
    grad_cldice = torch.autograd.grad(loss_cldice, logits, retain_graph=True)[0]
    grad_gap = torch.autograd.grad(loss_gap, logits, retain_graph=True)[0]

    # Spatial conflict analysis
    # Per-pixel cosine similarity (treating each pixel as 1D vector)
    eps = 1e-8
    grad_cldice_flat = grad_cldice.view(grad_cldice.size(0), -1)
    grad_gap_flat = grad_gap.view(grad_gap.size(0), -1)

    # Global cosine similarity per image
    global_cos_sim = (grad_cldice_flat * grad_gap_flat).sum(dim=1) / (
        grad_cldice_flat.norm(dim=1) * grad_gap_flat.norm(dim=1) + eps
    )

    # Sign conflict map: negative where gradients oppose
    sign_conflict_map = torch.sign(grad_cldice) * torch.sign(grad_gap)

    # Magnitude of conflict: product is negative when they conflict
    conflict_magnitude = grad_cldice * grad_gap

    # Relative magnitude (normalized by L2 norm)
    norm_cldice = grad_cldice_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
    norm_gap = grad_gap_flat.norm(dim=1, keepdim=True).view(-1, 1, 1, 1)
    normalized_conflict = conflict_magnitude / (norm_cldice * norm_gap + eps)

    return {
        'loss_cldice': loss_cldice.item(),
        'loss_gap': loss_gap.item(),
        'global_cos_sim': global_cos_sim.detach().cpu().numpy(),
        'sign_conflict_map': sign_conflict_map.detach().cpu().numpy(),
        'conflict_magnitude': conflict_magnitude.detach().cpu().numpy(),
        'normalized_conflict': normalized_conflict.detach().cpu().numpy(),
        'grad_cldice': grad_cldice.detach().cpu().numpy(),
        'grad_gap': grad_gap.detach().cpu().numpy(),
        'pred_prob': pred_prob.detach().cpu().numpy(),
        'masks': masks.detach().cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--block_type', type=str, default='dw')
    parser.add_argument('--data_root', type=str, default='data/raw/mass_roads')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/gradient_conflict')
    parser.add_argument('--max_cases', type=int, default=10)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(
        block_type=args.block_type,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=16,
        use_confidence_gate=True,
    )
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    dataset = MassachusettsRoadsDataset(
        root_dir=args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        augment=False,
    )
    # Force deterministic center crop
    orig_random_crop = dataset._random_crop
    def deterministic_crop(img, mask, deterministic=False):
        return orig_random_crop(img, mask, deterministic=True)
    dataset._random_crop = deterministic_crop

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_results = []
    case_count = 0

    for batch in tqdm(loader, desc='Analyzing gradient conflict'):
        if case_count >= args.max_cases:
            break

        result = compute_gradient_conflict(
            model, batch['image'], batch['mask'], device, block_type=args.block_type
        )

        for i in range(result['pred_prob'].shape[0]):
            if case_count >= args.max_cases:
                break

            case = {
                'image_id': batch['image_id'][i],
                'loss_cldice': result['loss_cldice'],
                'loss_gap': result['loss_gap'],
                'global_cos_sim': float(result['global_cos_sim'][i]),
                'sign_conflict_map': result['sign_conflict_map'][i].tolist(),
                'conflict_magnitude': result['conflict_magnitude'][i].tolist(),
                'normalized_conflict': result['normalized_conflict'][i].tolist(),
                'grad_cldice': result['grad_cldice'][i].tolist(),
                'grad_gap': result['grad_gap'][i].tolist(),
                'pred_prob': result['pred_prob'][i].tolist(),
                'mask': result['masks'][i].tolist(),
            }
            all_results.append(case)
            case_count += 1

    output_path = os.path.join(args.output_dir, f'gradient_conflict_{args.block_type}.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary statistics
    cos_sims = [r['global_cos_sim'] for r in all_results]
    print(f"\nGradient Conflict Summary ({args.block_type}, {len(all_results)} cases):")
    print(f"  Global cosine similarity: mean={np.mean(cos_sims):.4f}, std={np.std(cos_sims):.4f}")
    print(f"  Range: [{np.min(cos_sims):.4f}, {np.max(cos_sims):.4f}]")

    # Per-pixel sign conflict statistics
    total_conflict_pixels = 0
    total_pixels = 0
    for r in all_results:
        scm = np.array(r['sign_conflict_map'])
        mask = np.array(r['mask'])
        # Only consider pixels inside the mask region (foreground + nearby background)
        # Use a dilated mask to include boundary region
        from scipy.ndimage import binary_dilation
        region = binary_dilation(mask[0] > 0, iterations=5)
        conflict = (scm[0] < 0) & region
        total_conflict_pixels += conflict.sum()
        total_pixels += region.sum()

    print(f"  Sign-conflict pixels: {total_conflict_pixels}/{total_pixels} ({100*total_conflict_pixels/total_pixels:.1f}%)")
    print(f"  Saved to {output_path}")


if __name__ == '__main__':
    main()
