#!/usr/bin/env python3
"""Visualize ATW loss weight map lambda(x) = lambda_base * coherence(x)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.losses.atw_loss import ATWLoss


def visualize(out_dir='outputs/paper_figures'):
    os.makedirs(out_dir, exist_ok=True)
    _, val_loader = get_mass_roads_loaders(root_dir='data/raw/mass_roads', crop_size=512, batch_size=4, num_workers=2)
    
    atw = ATWLoss(lambda_base=0.3, sigma=1.0).cuda()
    batch = next(iter(val_loader))
    images = batch['image'].cuda()
    masks = batch['mask'][:,0].cpu().numpy()
    
    with torch.no_grad():
        coherence = atw.compute_coherence(images).cpu().numpy()[:,0]
    
    weight = 0.3 * coherence
    images_np = images.cpu().numpy().transpose(0,2,3,1)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        img = (images_np[i] - images_np[i].min()) / (images_np[i].max() - images_np[i].min() + 1e-8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(masks[i], cmap='gray')
        axes[i, 1].set_title('GT Mask')
        axes[i, 1].axis('off')
        
        im = axes[i, 2].imshow(coherence[i], cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Coherence c(x) (mean={coherence[i].mean():.2f})')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
        
        im2 = axes[i, 3].imshow(weight[i], cmap='hot', vmin=0, vmax=0.3)
        axes[i, 3].set_title(f'ATW weight λ(x) (mean={weight[i].mean():.3f})')
        axes[i, 3].axis('off')
        plt.colorbar(im2, ax=axes[i, 3], fraction=0.046)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'atw_weight_map.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    visualize()
