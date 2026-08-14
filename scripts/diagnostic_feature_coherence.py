#!/usr/bin/env python3
"""
Diagnostic script: compare image-level vs feature-level coherence at junctions vs straight segments.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

from sipv2.models import build_model
from sipv2.datasets import get_mass_roads_loaders
from sipv2.utils import set_seed
from sipv2.ops.structure_tensor import StructureTensor


def compute_coherence_map(tensor, sigma=1.0):
    """
    Compute structure tensor coherence from a tensor [B, C, H, W].
    Returns coherence [B, 1, H, W].
    """
    st = StructureTensor(sigma=sigma)
    if tensor.is_cuda:
        st = st.to(tensor.device)
    # StructureTensor expects image-like input; we'll compute per-channel and average
    # But first, reduce channels to a scalar representation
    if tensor.shape[1] > 1:
        # Option 1: L2 norm across channels
        gray = torch.norm(tensor, dim=1, keepdim=True)
    else:
        gray = tensor
    
    # Normalize to [0, 1] range for structure tensor stability
    gray_min = gray.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    gray_max = gray.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    gray = (gray - gray_min) / (gray_max - gray_min + 1e-8)
    
    out = st(gray)
    l1 = out['lambda1']
    l2 = out['lambda2']
    coherence = (l1 - l2) / (l1 + l2 + 1e-8)
    return coherence


def identify_skeleton_regions(mask):
    """
    Given binary mask [H, W], return junction and straight pixel coordinates.
    """
    skel = skeletonize(mask > 0)
    coords = np.argwhere(skel)
    if len(coords) == 0:
        return np.array([]), np.array([])
    
    # Compute degree of each skeleton pixel
    degrees = []
    for y, x in coords:
        patch = skel[max(0, y-1):y+2, max(0, x-1):x+2]
        degree = patch.sum() - 1  # exclude self
        degrees.append(degree)
    degrees = np.array(degrees)
    
    junctions = coords[degrees >= 3]
    straights = coords[degrees == 2]
    
    return junctions, straights


def sample_region_coherence(coherence, coords, scale_factor=1.0):
    """
    Sample coherence values at given coordinates.
    If coherence is downsampled, adjust coordinates accordingly.
    """
    h, w = coherence.shape[-2:]
    scaled = coords / scale_factor
    scaled = scaled.astype(np.int32)
    # Clamp to bounds
    scaled[:, 0] = np.clip(scaled[:, 0], 0, h - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, w - 1)
    values = coherence[0, 0, scaled[:, 0], scaled[:, 1]].cpu().numpy()
    return values


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = build_model(
        block_type='sipv2_road',
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=16,
        use_confidence_gate=True,
    )
    
    checkpoint_path = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/road_experiments/road_sipv2_road_atw_crop512_bs8_ep5_seed42_atw0.15/checkpoints/checkpoint_final.pth'
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Register hooks to extract encoder features
    features = {}
    def get_hook(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Stage 2 is index 2 in enc_blocks (after downsampling twice)
    # But we want features AFTER enc_blocks[i], before downsampling
    # Actually enc_blocks[i] processes features at stage i
    hooks = []
    for i in range(4):
        h = model.enc_blocks[i].register_forward_hook(get_hook(f'enc_{i}'))
        hooks.append(h)
    
    # Load validation data
    _, val_loader = get_mass_roads_loaders(
        root_dir='data/raw/mass_roads',
        crop_size=512,
        batch_size=4,
        num_workers=4,
    )
    
    results = {
        'image_junction': [],
        'image_straight': [],
        'feat0_junction': [], 'feat0_straight': [],
        'feat1_junction': [], 'feat1_straight': [],
        'feat2_junction': [], 'feat2_straight': [],
        'feat3_junction': [], 'feat3_straight': [],
    }
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Processing'):
            images = batch['image'].to(device)
            masks = batch['mask'].cpu().numpy()
            
            # Forward pass
            _ = model(images, image=images)
            
            for b in range(images.shape[0]):
                mask = masks[b, 0]
                junctions, straights = identify_skeleton_regions(mask)
                if len(junctions) == 0 or len(straights) == 0:
                    continue
                
                # Image-level coherence
                img_coherence = compute_coherence_map(images[b:b+1])
                img_j = sample_region_coherence(img_coherence, junctions, scale_factor=1.0)
                img_s = sample_region_coherence(img_coherence, straights, scale_factor=1.0)
                results['image_junction'].extend(img_j.tolist())
                results['image_straight'].extend(img_s.tolist())
                
                # Feature-level coherence for each stage
                for stage_idx in range(4):
                    feat = features[f'enc_{stage_idx}'][b:b+1]
                    scale = 2 ** stage_idx  # H/(2^stage_idx)
                    feat_coh = compute_coherence_map(feat)
                    fj = sample_region_coherence(feat_coh, junctions, scale_factor=scale)
                    fs = sample_region_coherence(feat_coh, straights, scale_factor=scale)
                    results[f'feat{stage_idx}_junction'].extend(fj.tolist())
                    results[f'feat{stage_idx}_straight'].extend(fs.tolist())
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Print statistics
    print("\n" + "="*70)
    print("Coherence comparison: Junctions vs Straight segments")
    print("="*70)
    
    def print_stats(name, j_vals, s_vals):
        if len(j_vals) == 0 or len(s_vals) == 0:
            return
        j_mean = np.mean(j_vals)
        j_std = np.std(j_vals)
        s_mean = np.mean(s_vals)
        s_std = np.std(s_vals)
        sep = s_mean - j_mean
        ratio = s_mean / (j_mean + 1e-8)
        print(f"{name:20s} | Junc: {j_mean:.3f}±{j_std:.3f} | Straight: {s_mean:.3f}±{s_std:.3f} | Sep: {sep:+.3f} | Ratio: {ratio:.2f}")
    
    print_stats("Image-level", results['image_junction'], results['image_straight'])
    for i in range(4):
        print_stats(f"Feature-stage{i}", results[f'feat{i}_junction'], results[f'feat{i}_straight'])
    
    # Save results
    out_dir = '/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/outputs/feature_coherence_diagnostic'
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'coherence_stats.npz'), **results)
    print(f"\nResults saved to {out_dir}/coherence_stats.npz")


if __name__ == '__main__':
    main()
