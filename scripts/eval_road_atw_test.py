#!/usr/bin/env python3
"""Evaluate ATW (R3) and baselines (R0/R1/R2) on full Massachusetts Roads test set."""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize

from sipv2.models import build_model
from sipv2.utils import load_checkpoint
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.metrics.region import pixel_metrics
from sipv2.metrics.skeleton import compute_all_skeleton_metrics
from sipv2.metrics.road_topology import compute_all_road_topology_metrics
from sipv2.metrics.junction_preservation import compute_junction_preservation


CHECKPOINTS = {
    # Uniform protocol: final-epoch checkpoints only (checkpoint_best of road runs
    # can be early-stopped at epoch 4 — see supplement S5 note; never use it).
    'R0 (DW)': ('dw', 'outputs/road_real_r0_seed0/road_dw_crop512_bs8_ep50_seed0/checkpoints/checkpoint_final.pth'),
    'R1 (SIP-v2)': ('sipv2_road', 'outputs/road_real_r1_seed42/road_sipv2_road_crop512_bs8_ep50_seed42/checkpoints/checkpoint_epoch49.pth'),
    'R2 (+clDice)': ('sipv2_road', 'outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth'),
    'R3 (ATW λ=0.3)': ('sipv2_road', 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep15_seed42_atw0.3/checkpoints/checkpoint_final.pth'),
    'R3_v3 (ATW λ=0.15)': ('sipv2_road', 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.15/checkpoints/checkpoint_final.pth'),
    'R3_v4 (ATW λ=0.1)': ('sipv2_road', 'outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.1/checkpoints/checkpoint_final.pth'),
    'R3_v5 (ATW σ=0.5 λ=0.15)': ('sipv2_road', 'outputs/road_experiments_multiscale/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.15/checkpoints/checkpoint_final.pth'),
}


def load_model(block_type, ckpt_path, device):
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


def evaluate_model(name, model, test_loader, device, block_type, threshold=0.5):
    all_dice = []
    all_iou = []
    all_cldice = []
    all_skelrec = []
    all_apls = []
    all_gaprec = []
    all_jpr = []
    all_best_dice = []

    thresholds = np.arange(0.05, 1.0, 0.05)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=name):
            images = batch['image'].to(device)
            masks = batch['mask'][:, 0].cpu().numpy()

            if block_type in ('sipv2', 'sipv2_road'):
                logits = model(images, image=images)
            else:
                logits = model(images)
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy()

            for pred, mask in zip(probs, masks):
                # Best dice threshold scan
                best_d = 0.0
                for th in thresholds:
                    m = pixel_metrics(pred, mask, threshold=th)
                    if m['dice'] > best_d:
                        best_d = m['dice']
                all_best_dice.append(best_d)

                m = pixel_metrics(pred, mask, threshold=threshold)
                all_dice.append(m['dice'])
                all_iou.append(m['iou'])

                skel = compute_all_skeleton_metrics(pred, mask, threshold=threshold)
                all_cldice.append(skel['cldice'])
                all_skelrec.append(skel['skeleton_recall'])

                topo = compute_all_road_topology_metrics(pred, mask, threshold=threshold)
                all_apls.append(topo['apls'])
                all_gaprec.append(topo['gap_recovery'])

                jpr, _ = compute_junction_preservation(pred > threshold, mask > 0)
                if not np.isnan(jpr):
                    all_jpr.append(jpr)

    results = {
        # WARNING: per-case oracle threshold on the test set — test-set tuning.
        # Never cite this field; kept only for diagnostics.
        'oracle_best_dice_test_tuned': float(np.mean(all_best_dice)),
        'dice': float(np.mean(all_dice)),
        'iou': float(np.mean(all_iou)),
        'cldice': float(np.mean(all_cldice)),
        'skelrec': float(np.mean(all_skelrec)),
        'apls': float(np.mean(all_apls)),
        'gaprec': float(np.mean(all_gaprec)),
        'jpr': float(np.mean(all_jpr)) if all_jpr else 0.0,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/raw/mass_roads')
    parser.add_argument('--output', default='outputs/road_experiments/test_evaluation.json')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda')
    # Build test loader manually (get_mass_roads_loaders does not expose test)
    from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
    test_dataset = MassachusettsRoadsDataset(
        root_dir=args.data_root, split='test', crop_size=512, augment=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    all_results = {}
    for name, (block_type, ckpt_path) in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: {ckpt_path} not found")
            continue
        model = load_model(block_type, ckpt_path, device)
        res = evaluate_model(name, model, test_loader, device, block_type, args.threshold)
        all_results[name] = res
        print(f"\n{name}: {json.dumps(res, indent=2)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
