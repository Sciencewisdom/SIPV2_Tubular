#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from skimage.morphology import skeletonize
from sipv2.datasets.mass_roads import get_mass_roads_loaders
from sipv2.ops.structure_tensor import StructureTensor
from sipv2.metrics.junction_preservation import skeleton_to_graph
from sipv2.models import build_model
from sipv2.utils import load_checkpoint

st = StructureTensor(sigma=1.0).cuda()
device = torch.device('cuda')

model = build_model('sipv2_road', in_channels=3, num_classes=1, channels=[32,64,128,256], blocks_per_stage=[2,2,2,2], decoder_blocks=1, directions=16, use_confidence_gate=True).to(device)
load_checkpoint(model, None, 'outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_epoch49.pth', device)
model.eval()

_, val_loader = get_mass_roads_loaders(root_dir='data/raw/mass_roads', crop_size=512, batch_size=4, num_workers=2)
batch = next(iter(val_loader))
images = batch['image'].cuda()
masks = batch['mask'][:,0].numpy()

with torch.no_grad():
    logits = model(images, image=images)
    pred_prob = torch.sigmoid(logits)
    
    # Image coherence
    res_img = st(images)
    l1, l2 = res_img['lambda1'], res_img['lambda2']
    coh_img = ((l1 - l2) / (l1 + l2 + 1e-6)).mean(dim=1).cpu().numpy()
    
    # Prediction coherence (treat prob map as grayscale image)
    res_pred = st(pred_prob)
    l1p, l2p = res_pred['lambda1'], res_pred['lambda2']
    coh_pred = ((l1p - l2p) / (l1p + l2p + 1e-6)).mean(dim=1).cpu().numpy()

for i in range(4):
    skel = skeletonize(masks[i] > 0)
    coords, degrees = skeleton_to_graph(skel)
    junctions = [(y, x) for (y, x), d in zip(coords, degrees) if d >= 3]
    straight = [(y, x) for (y, x), d in zip(coords, degrees) if d < 3]
    
    j_img = [coh_img[i, y, x] for y, x in junctions] if junctions else [0]
    s_img = [coh_img[i, y, x] for y, x in straight[:len(junctions)*3]] if straight else [0]
    j_pred = [coh_pred[i, y, x] for y, x in junctions] if junctions else [0]
    s_pred = [coh_pred[i, y, x] for y, x in straight[:len(junctions)*3]] if straight else [0]
    
    print(f"Sample {i}: junctions={len(junctions)}")
    print(f"  Image coh:  junc={np.mean(j_img):.3f}±{np.std(j_img):.3f}  straight={np.mean(s_img):.3f}±{np.std(s_img):.3f}  ratio={np.mean(j_img)/np.mean(s_img):.3f}")
    print(f"  Pred coh:   junc={np.mean(j_pred):.3f}±{np.std(j_pred):.3f}  straight={np.mean(s_pred):.3f}±{np.std(s_pred):.3f}  ratio={np.mean(j_pred)/np.mean(s_pred):.3f}")
