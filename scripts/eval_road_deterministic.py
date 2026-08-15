#!/usr/bin/env python3
"""
Deterministic evaluation script for road extraction checkpoints.
Uses center crop (no randomness) for reproducible metrics.
Collects per-case metrics for statistical testing.
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast

from sipv2.models import build_model
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.metrics.region import pixel_metrics
from sipv2.metrics.skeleton import compute_all_skeleton_metrics
from sipv2.metrics.road_topology import compute_all_road_topology_metrics
from sipv2.utils import set_seed, load_checkpoint


def get_deterministic_val_loader(root_dir, crop_size=512, batch_size=1, num_workers=4, split="valid"):
    """Create deterministic validation loader with center crop."""
    dataset = MassachusettsRoadsDataset(
        root_dir=root_dir,
        split=split,
        crop_size=crop_size,
        augment=False,  # No augmentation
        normalize=True,
    )
    # Monkey-patch to force deterministic center crop
    orig_random_crop = dataset._random_crop
    def deterministic_crop(img, mask, deterministic=False):
        return orig_random_crop(img, mask, deterministic=True)
    dataset._random_crop = deterministic_crop
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def evaluate_checkpoint(checkpoint_path, args):
    """Evaluate a single checkpoint deterministically."""
    set_seed(42)  # Fixed seed for deterministic evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config if available
    output_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        block_type = config.get('block_type', args.block_type)
        crop_size = config.get('crop_size', args.crop_size)
        directions = config.get('directions', 16)
        use_confidence_gate = config.get('use_confidence_gate', True)
        grad_op = config.get('grad_op', 'scharr')
        stencil = config.get('stencil', 5)
    else:
        block_type = args.block_type
        crop_size = args.crop_size
        directions = args.directions
        use_confidence_gate = args.use_confidence_gate
        grad_op = getattr(args, 'grad_op', 'scharr')
        stencil = getattr(args, 'stencil', 5)
    
    model = build_model(
        block_type=block_type,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=directions,
        use_confidence_gate=use_confidence_gate,
        grad_op=grad_op,
        stencil=stencil,
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    loader = get_deterministic_val_loader(
        root_dir=args.data_root,
        crop_size=crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split=args.split,
    )
    
    all_cases = []
    all_pred_probs = []
    all_masks = []
    image_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            with autocast('cuda'):
                if block_type in ('sipv2', 'sipv2_road'):
                    outputs = model(images, image=images)
                else:
                    outputs = model(images)
            
            pred_prob = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            
            for i in range(pred_prob.shape[0]):
                all_pred_probs.append(pred_prob[i, 0])
                all_masks.append(masks_np[i, 0])
                image_ids.append(batch['image_id'][i])
    
    # Threshold scan for best dice
    thresholds = np.arange(0.05, 1.0, 0.05)
    best_dice_th = 0.5
    best_dice = 0.0
    for th in thresholds:
        dices = [pixel_metrics(p, m, threshold=th)['dice'] for p, m in zip(all_pred_probs, all_masks)]
        mean_dice = np.mean(dices)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_th = th
    
    # Per-case metrics
    for pred, mask, img_id in zip(all_pred_probs, all_masks, image_ids):
        pm = pixel_metrics(pred, mask, threshold=best_dice_th)
        skel = compute_all_skeleton_metrics(pred, mask, threshold=best_dice_th)
        road = compute_all_road_topology_metrics(pred, mask, threshold=best_dice_th)
        
        case = {
            'image_id': img_id,
            'dice': float(pm['dice']),
            'iou': float(pm['iou']),
            'cldice': float(skel['cldice']),
            'skel_recall': float(skel['skeleton_recall']),
            'apls': float(road['apls']),
            'connectivity': float(road['connectivity']),
            'gap_recovery': float(road['gap_recovery']),
        }
        all_cases.append(case)
    
    # Aggregate
    agg = {}
    for key in ['dice', 'iou', 'cldice', 'skel_recall', 'apls', 'connectivity', 'gap_recovery']:
        vals = [c[key] for c in all_cases]
        agg[key] = float(np.mean(vals))
        agg[key + '_std'] = float(np.std(vals))
    agg['best_threshold'] = float(best_dice_th)
    agg['n_cases'] = len(all_cases)
    
    return {
        'aggregate': agg,
        'cases': all_cases,
        'checkpoint': checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, default='data/raw/mass_roads')
    parser.add_argument('--block_type', type=str, default='dw')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    # NOTE: default 0 — the monkey-patched deterministic_crop closure is not
    # picklable, so spawn-mode workers (Windows default) crash eval outright.
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--directions', type=int, default=16)
    parser.add_argument('--use_confidence_gate', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    result = evaluate_checkpoint(args.checkpoint, args)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output}")
    
    # Print summary
    agg = result['aggregate']
    print(f"\nDeterministic Evaluation Results ({result['checkpoint']}):")
    print(f"  Cases: {agg['n_cases']}, Best threshold: {agg['best_threshold']:.2f}")
    print(f"  Dice: {agg['dice']:.4f} ± {agg['dice_std']:.4f}")
    print(f"  IoU:  {agg['iou']:.4f} ± {agg['iou_std']:.4f}")
    print(f"  clDice: {agg['cldice']:.4f} ± {agg['cldice_std']:.4f}")
    print(f"  SkelRec: {agg['skel_recall']:.4f} ± {agg['skel_recall_std']:.4f}")
    print(f"  APLS: {agg['apls']:.4f} ± {agg['apls_std']:.4f}")
    print(f"  Conn: {agg['connectivity']:.4f} ± {agg['connectivity_std']:.4f}")
    print(f"  GapRec: {agg['gap_recovery']:.4f} ± {agg['gap_recovery_std']:.4f}")


if __name__ == '__main__':
    main()
