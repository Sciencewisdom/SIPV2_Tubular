"""
Training loop with AMP support for limited VRAM.
"""
import os
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    use_amp=True,
    grad_clip=1.0,
    block_type='conv',
    use_atw=False,
):
    """Train for one epoch."""
    model.train()
    scaler = GradScaler() if use_amp else None

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)      # [B, 3, H, W]
        masks = batch['mask'].to(device, non_blocking=True)        # [B, 1, H, W]
        fov_masks = batch['fov_mask'].to(device, non_blocking=True)  # [B, 1, H, W]

        optimizer.zero_grad()

        # Forward with AMP
        if use_amp:
            with autocast('cuda'):
                if block_type in ('sipv2', 'sipv2_road', 'sipv2_full'):
                    outputs = model(images, image=images)
                else:
                    outputs = model(images)
                if use_atw:
                    loss = criterion(outputs, masks, fov_masks, images, epoch)
                else:
                    loss = criterion(outputs, masks, fov_masks, epoch)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if block_type in ('sipv2', 'sipv2_road', 'sipv2_full'):
                outputs = model(images, image=images)
            else:
                outputs = model(images)
            if use_atw:
                loss = criterion(outputs, masks, fov_masks, images, epoch)
            else:
                loss = criterion(outputs, masks, fov_masks, epoch)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
