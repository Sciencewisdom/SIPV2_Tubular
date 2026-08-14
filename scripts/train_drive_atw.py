#!/usr/bin/env python3
"""
Minimal DRIVE ATW fine-tuning script.
Fine-tunes E5 (SIP-v2 Full + clDice) with ATW on DRIVE.
"""
import os
import sys
import argparse
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sipv2.models import build_experiment_model
from sipv2.losses import BCEDiceATWLoss
from sipv2.datasets import get_drive_loaders
from sipv2.engine.train import train_one_epoch
from sipv2.utils import set_seed, save_checkpoint, load_checkpoint
from sipv2.metrics.region import MetricsTracker
from sipv2.metrics.junction_preservation import compute_junction_preservation
from skimage.morphology import skeletonize
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/raw/DRIVE')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs/drive_atw')
    parser.add_argument('--resume', type=str, required=True)
    # ATW
    parser.add_argument('--atw_lambda', type=float, default=0.15)
    parser.add_argument('--atw_warmup', type=int, default=5)
    parser.add_argument('--atw_sigma', type=float, default=1.0)
    return parser.parse_args()


def validate_drive_atw(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    tracker = MetricsTracker()
    all_jpr = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            fov = batch.get('fov_mask', torch.ones_like(masks)).to(device)

            logits = model(images, image=images)
            loss = criterion(logits, masks, fov, images, epoch)
            total_loss += loss.item()
            num_batches += 1

            tracker.update(logits, masks, fov, threshold=0.5)

            pred_prob = torch.sigmoid(logits).detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            for i in range(pred_prob.shape[0]):
                jpr, _ = compute_junction_preservation(pred_prob[i, 0] > 0.5, masks_np[i, 0] > 0)
                if not np.isnan(jpr):
                    all_jpr.append(jpr)

    metrics = tracker.compute()
    metrics['val_loss'] = total_loss / max(num_batches, 1)
    if all_jpr:
        metrics['jpr'] = float(np.mean(all_jpr))
    else:
        metrics['jpr'] = 0.0
    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    exp_name = f"drive_e5_atw{args.atw_lambda}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(log_dir)

    train_loader, test_loader = get_drive_loaders(
        root_dir=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = build_experiment_model('E5', in_channels=3, num_classes=1,
                                   channels=[32, 64, 128, 256],
                                   blocks_per_stage=[2, 2, 2, 2],
                                   decoder_blocks=1)
    model = model.to(device)
    print(f"Parameters: {model.count_parameters():,}")

    criterion = BCEDiceATWLoss(
        bce_weight=1.0, dice_weight=1.0,
        atw_weight=args.atw_lambda,
        atw_warmup=args.atw_warmup,
        atw_sigma=args.atw_sigma,
    )
    print(f"ATW: lambda={args.atw_lambda}, warmup={args.atw_warmup}")

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
        print(f"Resumed from epoch {start_epoch-1}")

    end_epoch = args.epochs
    if args.resume and start_epoch >= end_epoch:
        end_epoch = start_epoch + args.epochs

    metrics = resume_metrics
    for epoch in range(start_epoch, end_epoch):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, use_amp=args.use_amp, grad_clip=args.grad_clip,
            block_type='sipv2_full', use_atw=True,
        )
        scheduler.step()

        metrics = validate_drive_atw(model, test_loader, criterion, device, epoch)
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={metrics['val_loss']:.4f}")
        print(f"  Dice={metrics['dice']:.4f}, BestDice={metrics.get('best_dice', metrics['dice']):.4f}, JPR={metrics['jpr']:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/dice', metrics['dice'], epoch)
        writer.add_scalar('Metrics/jpr', metrics['jpr'], epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(model, optimizer, epoch, metrics,
                            os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))

        if metrics['dice'] > best_dice:
            best_dice = metrics['dice']
            save_checkpoint(model, optimizer, epoch, metrics,
                            os.path.join(checkpoint_dir, 'checkpoint_best.pth'))

    save_checkpoint(model, optimizer, epoch, metrics,
                    os.path.join(checkpoint_dir, 'checkpoint_final.pth'))
    writer.close()
    print(f"Done. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
