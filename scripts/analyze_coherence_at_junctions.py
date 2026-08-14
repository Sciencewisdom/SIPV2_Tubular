#!/usr/bin/env python3
"""Analyze coherence values at GT junctions vs straight roads."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from skimage.morphology import skeletonize
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.ops.structure_tensor import StructureTensor
from sipv2.metrics.junction_preservation import skeleton_to_graph

st = StructureTensor(sigma=1.0).cuda()
_, val_loader = get_mass_roads_loaders(root_dir='data/raw/mass_roads', crop_size=512, batch_size=4, num_workers=2)

batch = next(iter(val_loader))
images = batch['image'].cuda()
masks = batch['mask'][:,0].numpy()

with torch.no_grad():
    res = st(images)
    l1, l2 = res['lambda1'], res['lambda2']
    coherence = ((l1 - l2) / (l1 + l2 + 1e-6)).mean(dim=1).cpu().numpy()

for i in range(4):
    skel = skeletonize(masks[i] > 0)
    coords, degrees = skeleton_to_graph(skel)
    junctions = [(y, x) for (y, x), d in zip(coords, degrees) if d >= 3]
    
    # Sample non-junction skeleton points
    non_junction = [(y, x) for (y, x), d in zip(coords, degrees) if d < 3]
    
    j_vals = [coherence[i, y, x] for y, x in junctions] if junctions else [0]
    n_vals = [coherence[i, y, x] for y, x in non_junction[:len(junctions)*3]] if non_junction else [0]
    
    print(f"Sample {i}: junctions={len(junctions)}, non_junc={len(non_junction)}")
    print(f"  Coherence at junctions: {np.mean(j_vals):.3f} ± {np.std(j_vals):.3f}")
    print(f"  Coherence on straight:  {np.mean(n_vals):.3f} ± {np.std(n_vals):.3f}")
    print(f"  Ratio (junc/straight):  {np.mean(j_vals)/np.mean(n_vals):.3f}" if np.mean(n_vals)>0 else "  N/A")
