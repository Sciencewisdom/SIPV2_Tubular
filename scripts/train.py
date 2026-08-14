#!/usr/bin/env python3
"""
Main training script for SIP-v2 DRIVE experiments.
Optimized for RTX 4050 (6GB VRAM) + 10GB RAM.
"""
import os
import sys
import argparse
import json
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from sipv2.models import build_experiment_model
from sipv2.losses import BCEDiceLoss, BCEDiceCLDiceLoss
from sipv2.datasets import get_drive_loaders
from sipv2.engine import train_one_epoch, validate, validate_with_tensor_capture
from sipv2.utils import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train SIP-v2 on DRIVE')
    parser.add_argument('--exp', type=str, default='E0',
                        choices=['E0', 'E1', 'E2', 'E3', 'E4', 'E5'],
                        help='Experiment: E0=Conv, E1=DW, E2=Iso, E3=OldSIP, E4=SIP-v2')
    parser.add_argument('--dataset', type=str, default='DRIVE',
                        choices=['DRIVE', 'CHASE_DB1', 'HRF'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to dataset (auto-set if not provided)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (2 for 6GB VRAM)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Dataloader workers')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True,
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--use_cldice', action='store_true', default=False,
                        help='Use clDice loss (Phase 2)')
    parser.add_argument('--cldice_lambda', type=float, default=0.3,
                        help='clDice loss weight')
    parser.add_argument('--cldice_warmup', type=int, default=20,
                        help='Epochs before clDice activates')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output dirs
    exp_name = f"{args.exp}_size{args.img_size}_bs{args.batch_size}_seed{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    log_dir = os.path.join(output_dir, 'logs')
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Auto-set data_root
    if args.data_root is None:
        if args.dataset == 'DRIVE':
            args.data_root = os.path.join('data', 'raw', 'DRIVE')
        elif args.dataset == 'CHASE_DB1':
            args.data_root = os.path.join('data', 'raw', 'CHASE_DB1')
        elif args.dataset == 'HRF':
            args.data_root = os.path.join('data', 'raw', 'HRF')

    # Data loaders
    print(f"Loading {args.dataset} dataset from: {args.data_root}")
    if args.dataset == 'DRIVE':
        from sipv2.datasets import get_drive_loaders
        train_loader, test_loader = get_drive_loaders(
            root_dir=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.dataset == 'CHASE_DB1':
        from sipv2.datasets import get_chasedb1_loaders
        train_loader, test_loader = get_chasedb1_loaders(
            root_dir=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    elif args.dataset == 'HRF':
        from sipv2.datasets import get_hrf_loaders
        train_loader, test_loader = get_hrf_loaders(
            root_dir=args.data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    print(f"Train: {len(train_loader.dataset)} images, Test: {len(test_loader.dataset)} images")

    # Model
    print(f"Building model: {args.exp}")
    model = build_experiment_model(
        args.exp,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
    )
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Loss
    if args.use_cldice:
        criterion = BCEDiceCLDiceLoss(
            bce_weight=1.0, dice_weight=1.0,
            cldice_weight=args.cldice_lambda,
            cldice_warmup=args.cldice_warmup,
        )
        print(f'Using BCEDice + clDice (lambda={args.cldice_lambda}, warmup={args.cldice_warmup})')
    else:
        criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: cosine with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 10  # 10 epochs warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1

    # Training loop
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, use_amp=args.use_amp, grad_clip=args.grad_clip,
            block_type={'E0': 'conv', 'E1': 'dw', 'E2': 'iso', 'E3': 'old_sip', 'E4': 'sipv2', 'E5': 'sipv2_full'}[args.exp]
        )

        scheduler.step()

        # Validate every epoch
        bt = {'E0': 'conv', 'E1': 'dw', 'E2': 'iso', 'E3': 'old_sip', 'E4': 'sipv2', 'E5': 'sipv2_full'}[args.exp]
        should_save = (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1
        if args.exp in ('E4', 'E5'):
            metrics = validate_with_tensor_capture(
                model, test_loader, criterion, device,
                epoch=epoch,
                save_dir=pred_dir if should_save else None,
            )
        else:
            metrics = validate(
                model, test_loader, criterion, device,
                epoch=epoch,
                save_dir=pred_dir if should_save else None,
                block_type=bt,
            )

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}, val_loss={metrics['val_loss']:.4f}")
        print(f"  Dice={metrics['dice']:.4f}, IoU={metrics['iou']:.4f}")
        print(f"  Sens={metrics['sensitivity']:.4f}, Spec={metrics['specificity']:.4f}")
        print(f"  PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        print(f"  Best Dice={metrics['best_dice']:.4f} (th={metrics['best_dice_threshold']:.2f})")
        print(f"  clDice={metrics.get('skel_cldice', 0):.4f}, SkelRecall={metrics.get('skel_skeleton_recall', 0):.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', metrics['val_loss'], epoch)
        writer.add_scalar('Metrics/dice', metrics['dice'], epoch)
        writer.add_scalar('Metrics/iou', metrics['iou'], epoch)
        writer.add_scalar('Metrics/pr_auc', metrics['pr_auc'], epoch)
        writer.add_scalar('Metrics/best_dice', metrics['best_dice'], epoch)
        writer.add_scalar('Metrics/cldice', metrics.get('skel_cldice', 0), epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, save_path)

        # Save best
        if metrics['best_dice'] > best_dice:
            best_dice = metrics['best_dice']
            save_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
            save_checkpoint(model, optimizer, epoch, metrics, save_path)
            print(f"  *** New best Dice: {best_dice:.4f} ***")

    # Final save
    save_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
    save_checkpoint(model, optimizer, args.epochs - 1, metrics, save_path)

    writer.close()

    # Save summary (convert numpy types to python native)
    def _convert(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    summary = {
        'exp': args.exp,
        'best_dice': float(best_dice),
        'final_metrics': _convert(metrics),
        'config': vars(args),
    }
    # Save per-case metrics separately for statistical analysis
    per_case = {}
    for key in ['dice_per_case', 'skel_cldice_per_case', 'skel_skeleton_recall_per_case',
                'skel_break_count_per_case', 'skel_skeleton_precision_per_case']:
        if key in metrics:
            per_case[key] = [float(v) for v in metrics[key]]
    if per_case:
        summary['per_case'] = per_case
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print(f"Training complete. Best Dice: {best_dice:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
