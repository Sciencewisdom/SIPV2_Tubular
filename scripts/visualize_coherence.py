#!/usr/bin/env python3
"""Visualize structure tensor coherence maps for road images."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.ops.structure_tensor import StructureTensor


def visualize_coherence(out_dir='outputs/paper_figures'):
    os.makedirs(out_dir, exist_ok=True)
    _, val_loader = get_mass_roads_loaders(root_dir='data/raw/mass_roads', crop_size=512, batch_size=4, num_workers=2)
    st = StructureTensor(sigma=1.0).cuda()
    
    batch = next(iter(val_loader))
    images = batch['image'].cuda()
    masks = batch['mask'][:,0].numpy()
    
    with torch.no_grad():
        res = st(images)
        l1, l2 = res['lambda1'], res['lambda2']
        coherence = ((l1 - l2) / (l1 + l2 + 1e-6)).mean(dim=1).cpu().numpy()
    
    images_np = images.cpu().numpy()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        img = images_np[i].transpose(1,2,0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].set_title('GT Mask')
        axes[1, i].axis('off')
        
        im = axes[2, i].imshow(coherence[i], cmap='jet', vmin=0, vmax=1)
        axes[2, i].set_title(f'Coherence (mean={coherence[i].mean():.2f})')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i], fraction=0.046)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'coherence_visualization.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    visualize_coherence()
