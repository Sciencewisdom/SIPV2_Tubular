#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.ops.structure_tensor import StructureTensor
from sipv2.metrics.junction_preservation import skeleton_to_graph

st = StructureTensor(sigma=1.0).cuda()
_, val_loader = get_mass_roads_loaders(root_dir='data/raw/mass_roads', crop_size=512, batch_size=4, num_workers=2)

all_j = []
all_n = []

for batch in val_loader:
    images = batch['image'].cuda()
    masks = batch['mask'][:,0].numpy()
    with torch.no_grad():
        res = st(images)
        l1, l2 = res['lambda1'], res['lambda2']
        coherence = ((l1 - l2) / (l1 + l2 + 1e-6)).mean(dim=1).cpu().numpy()
    
    for i in range(coherence.shape[0]):
        skel = skeletonize(masks[i] > 0)
        coords, degrees = skeleton_to_graph(skel)
        junctions = [(y, x) for (y, x), d in zip(coords, degrees) if d >= 3]
        non_junction = [(y, x) for (y, x), d in zip(coords, degrees) if d < 3]
        
        for y, x in junctions:
            all_j.append(coherence[i, y, x])
        for y, x in non_junction:
            all_n.append(coherence[i, y, x])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(all_j, bins=30, range=(0,1), alpha=0.6, label=f'Junctions (n={len(all_j)})', color='red')
axes[0].hist(all_n, bins=30, range=(0,1), alpha=0.6, label=f'Straight (n={len(all_n)})', color='blue')
axes[0].set_xlabel('Coherence c(x)')
axes[0].set_ylabel('Count')
axes[0].set_title('Coherence Distribution')
axes[0].legend()

axes[1].hist([0.3*v for v in all_j], bins=30, range=(0,0.35), alpha=0.6, label='Junctions', color='red')
axes[1].hist([0.3*v for v in all_n], bins=30, range=(0,0.35), alpha=0.6, label='Straight', color='blue')
axes[1].axvline(0.15, color='green', linestyle='--', label='λ_base=0.15')
axes[1].set_xlabel('ATW weight λ(x) = 0.3 · c(x)')
axes[1].set_ylabel('Count')
axes[1].set_title('Effective Topology Weight')
axes[1].legend()

plt.tight_layout()
out_path = 'outputs/paper_figures/coherence_histogram.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
