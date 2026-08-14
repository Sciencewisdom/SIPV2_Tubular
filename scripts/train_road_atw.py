#!/usr/bin/env python3
"""
Training script for Massachusetts Roads Dataset with Adaptive Topology Weighting (ATW).
R3 = SIP-v2 Road + ATW (adaptive clDice weighting by tensor coherence).
"""
import os
import sys
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sipv2.models import build_model
from sipv2.losses import BCEDiceLoss, BCEDiceCLDiceLoss, BCEDiceATWLoss
from sipv2.datasets import get_mass_roads_loaders
from sipv2.engine.train import train_one_epoch
from sipv2.utils import set_seed, save_checkpoint, load_checkpoint

from sipv2.metrics.region import MetricsTracker, pixel_metrics
from sipv2.metrics.skeleton import compute_all_skeleton_metrics
from sipv2.metrics.road_topology import compute_all_road_topology_metrics
from sipv2.metrics.junction_preservation import compute_junction_preservation
from skimage.morphology import skeletonize
import numpy as np
from tqdm import tqdm
from torch.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description='Train SIP-v2 Road + ATW')
    parser.add_argument('--data_root', type=str, default='data/raw/mass_roads')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs/road_experiments')
    parser.add_argument('--resume', type=str, default=None)
    # ATW-specific
    parser.add_argument('--use_atw', action='store_true', default=True)
    parser.add_argument('--atw_lambda', type=float, default=0.3)
    parser.add_argument('--atw_warmup', type=int, default=10)
    parser.add_argument('--atw_sigma', type=float, default=1.0)
    # Model
    parser.add_argument('--block_type', type=str, default='sipv2_road',
                        choices=['dw', 'sipv2', 'sipv2_road'])
    parser.add_argument('--directions', type=int, default=16)
    parser.add_argument('--use_confidence_gate', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_road_atw(model, val_loader, criterion, device, epoch, save_dir=None, block_type='sipv2_road'):
    """Validation with road topology metrics including junction preservation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_pred_probs = []
    all_masks = []
    all_images = []

    tracker = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            fov_masks = batch['fov_mask'].to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(images, image=images)
                loss = criterion(outputs, masks, fov_masks, images, epoch)

            total_loss += loss.item()
            num_batches += 1

            pred_prob = torch.sigmoid(outputs).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()

            tracker.update(outputs, masks, fov_masks, threshold=0.5)

            for i in range(pred_prob.shape[0]):
                all_pred_probs.append(pred_prob[i, 0])
                all_masks.append(masks_np[i, 0])
                all_images.append(batch['image'][i].cpu().numpy())

    metrics = tracker.compute()
    metrics['val_loss'] = total_loss / max(num_batches, 1)

    # Threshold scan
    thresholds = np.arange(0.05, 1.0, 0.05)
    best_dice_th = 0.5
    best_dice = 0.0
    for th in thresholds:
        dices = []
        for pred, mask in zip(all_pred_probs, all_masks):
            m = pixel_metrics(pred, mask, threshold=th)
            dices.append(m['dice'])
        mean_dice = np.mean(dices)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_th = th

    metrics['best_dice'] = best_dice
    metrics['best_dice_threshold'] = best_dice_th

    # Skeleton and topology metrics
    skel_list = []
    road_topo_list = []
    jpr_list = []
    for pred, mask in zip(all_pred_probs, all_masks):
        skel_list.append(compute_all_skeleton_metrics(pred, mask, threshold=best_dice_th))
        road_topo_list.append(compute_all_road_topology_metrics(pred, mask, threshold=best_dice_th))
        jpr, _ = compute_junction_preservation(pred > best_dice_th, mask > 0)
        if not np.isnan(jpr):
            jpr_list.append(jpr)

    for key in ['cldice', 'skeleton_recall', 'skeleton_precision', 'break_count']:
        vals = [m[key] for m in skel_list]
        metrics[f'skel_{key}'] = np.mean(vals)

    for key in ['apls', 'connectivity', 'gap_recovery']:
        vals = [m[key] for m in road_topo_list]
        metrics[f'road_{key}'] = np.mean(vals)

    if jpr_list:
        metrics['junction_preservation'] = np.mean(jpr_list)
    else:
        metrics['junction_preservation'] = 0.0

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    exp_name = f"road_{args.block_type}_atw_crop{args.crop_size}_bs{args.batch_size}_ep{args.epochs}_seed{args.seed}"
    if args.use_atw:
        exp_name += f"_atw{args.atw_lambda}"
    output_dir = os.path.join(args.output_dir, exp_name)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(log_dir)

    print(f"Loading Massachusetts Roads from: {args.data_root}")
    train_loader, val_loader = get_mass_roads_loaders(
        root_dir=args.data_root,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)} crops, Val: {len(val_loader.dataset)} crops")

    print(f"Building model: {args.block_type}")
    model = build_model(
        block_type=args.block_type,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=args.directions,
        use_confidence_gate=args.use_confidence_gate,
    )
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    if args.use_atw:
        criterion = BCEDiceATWLoss(
            bce_weight=1.0, dice_weight=1.0,
            atw_weight=args.atw_lambda,
            atw_warmup=args.atw_warmup,
            atw_sigma=args.atw_sigma,
        )
        print(f"Using ATW: lambda={args.atw_lambda}, warmup={args.atw_warmup}, sigma={args.atw_sigma}")
    else:
        criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_dice = 0.0
    resume_metrics = {}
    if args.resume:
        start_epoch, resume_metrics = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
        best_dice = resume_metrics.get('best_dice', 0.0)

    end_epoch = args.epochs
    if args.resume and start_epoch >= end_epoch:
        # Fine-tuning mode: add epochs on top of resumed epoch
        end_epoch = start_epoch + args.epochs

    print("\n" + "="*60)
    print(f"Starting ATW training from epoch {start_epoch} to {end_epoch}")
    print("="*60)

    metrics = resume_metrics  # fallback if loop doesn't run
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, use_amp=args.use_amp, grad_clip=args.grad_clip,
            block_type=args.block_type, use_atw=args.use_atw,
        )
        scheduler.step()

        metrics = validate_road_atw(
            model, val_loader, criterion, device,
            epoch=epoch, save_dir=None, block_type=args.block_type,
        )

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={metrics['val_loss']:.4f}")
        print(f"  Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        print(f"  BestDice={metrics['best_dice']:.4f} (th={metrics['best_dice_threshold']:.2f})")
        print(f"  clDice={metrics.get('skel_cldice', 0):.4f}, SkelRec={metrics.get('skel_skeleton_recall', 0):.4f}")
        print(f"  APLS={metrics.get('road_apls', 0):.4f}, Conn={metrics.get('road_connectivity', 0):.4f}, GapRec={metrics.get('road_gap_recovery', 0):.4f}")
        print(f"  JPR={metrics.get('junction_preservation', 0):.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/dice', metrics['dice'], epoch)
        writer.add_scalar('Metrics/best_dice', metrics['best_dice'], epoch)
        writer.add_scalar('Metrics/cldice', metrics.get('skel_cldice', 0), epoch)
        writer.add_scalar('Metrics/apls', metrics.get('road_apls', 0), epoch)
        writer.add_scalar('Metrics/gap_recovery', metrics.get('road_gap_recovery', 0), epoch)
        writer.add_scalar('Metrics/junction_preservation', metrics.get('junction_preservation', 0), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, save_path)

        if metrics['best_dice'] > best_dice:
            best_dice = metrics['best_dice']
            save_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            save_checkpoint(model, optimizer, epoch, metrics, save_path)
            print(f"  *** New best Dice: {best_dice:.4f} ***")

    save_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
    save_checkpoint(model, optimizer, epoch, metrics, save_path)
    writer.close()

    print("\n" + "="*60)
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
