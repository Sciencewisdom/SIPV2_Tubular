#!/usr/bin/env python3
"""
Failure region localization: gradient conflict analysis by road geometry.
Identifies intersections, endpoints, straight segments, wide/narrow roads.
"""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree
from tqdm import tqdm

from sipv2.models import build_model
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.utils import set_seed
from torch.utils.data import DataLoader
from torch.amp import autocast
from analyze_gradient_conflict import compute_gradient_conflict, DifferentiableGapRecovery, CLDiceLossGrad


def identify_road_regions(mask, radius=10):
    """
    Identify road geometry regions from binary mask.
    Returns dict of region masks.
    """
    h, w = mask.shape
    skel = skeletonize(mask > 0)
    coords = np.argwhere(skel)

    regions = {
        'intersection': np.zeros_like(mask, dtype=bool),
        'endpoint': np.zeros_like(mask, dtype=bool),
        'straight': np.zeros_like(mask, dtype=bool),
        'wide': np.zeros_like(mask, dtype=bool),
        'narrow': np.zeros_like(mask, dtype=bool),
        'background_near': np.zeros_like(mask, dtype=bool),
    }

    if len(coords) < 3:
        return regions

    # Build graph
    coord_set = set(map(tuple, coords))
    G = nx.Graph()
    for y, x in coords:
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_coord = y + dy, x + dx
                if (ny, nx_coord) in coord_set:
                    G.add_edge((y, x), (ny, nx_coord))

    # Node degrees
    degree_map = {}
    for node in G.nodes():
        degree_map[node] = G.degree(node)

    # Intersections: degree >= 3
    inter_nodes = [n for n, d in degree_map.items() if d >= 3]
    # Endpoints: degree == 1
    end_nodes = [n for n, d in degree_map.items() if d == 1]
    # Straight: degree == 2
    straight_nodes = [n for n, d in degree_map.items() if d == 2]

    # Dilate intersection and endpoint regions
    yy, xx = np.mgrid[0:h, 0:w]
    for node in inter_nodes:
        cy, cx = node
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        regions['intersection'] |= dist <= radius
    for node in end_nodes:
        cy, cx = node
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        regions['endpoint'] |= dist <= radius
    for node in straight_nodes:
        cy, cx = node
        dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        regions['straight'] |= dist <= 3  # thinner band for straight

    # Distance transform for width analysis
    dt = ndimage.distance_transform_edt(mask > 0)
    if dt.max() > 0:
        median_dt = np.median(dt[dt > 0])
        regions['wide'] = (dt > median_dt) & (mask > 0)
        regions['narrow'] = (dt <= median_dt) & (mask > 0)

    # Background near road (dilated mask minus mask)
    dilated = ndimage.binary_dilation(mask > 0, iterations=radius)
    regions['background_near'] = dilated & ~(mask > 0)

    return regions


def analyze_region_conflict(model, dataset, device, block_type, max_cases=10):
    """Run gradient conflict analysis broken down by road region."""
    results = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=2, pin_memory=True)

    for idx, batch in enumerate(tqdm(loader, desc='Analyzing regions')):
        if idx >= max_cases:
            break

        # Gradient conflict
        gc = compute_gradient_conflict(
            model, batch['image'], batch['mask'], device, block_type=block_type
        )

        mask = batch['mask'][0, 0].cpu().numpy()
        pred = gc['pred_prob'][0, 0]
        sign_conflict = gc['sign_conflict_map'][0, 0]
        norm_conflict = gc['normalized_conflict'][0, 0]

        # Identify regions
        regions = identify_road_regions(mask, radius=10)

        case_result = {
            'image_id': batch['image_id'][0],
            'global_cos_sim': float(gc['global_cos_sim'][0]),
            'loss_cldice': gc['loss_cldice'],
            'loss_gap': gc['loss_gap'],
        }

        # Stats per region
        for region_name, region_mask in regions.items():
            if region_mask.sum() == 0:
                continue
            scm = sign_conflict[region_mask]
            ncm = norm_conflict[region_mask]

            conflict_ratio = (scm < 0).sum() / len(scm)
            mean_norm_conflict = ncm.mean()
            std_norm_conflict = ncm.std()

            case_result[region_name] = {
                'n_pixels': int(region_mask.sum()),
                'conflict_ratio': float(conflict_ratio),
                'mean_norm_conflict': float(mean_norm_conflict),
                'std_norm_conflict': float(std_norm_conflict),
            }

        results.append(case_result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--block_type', type=str, default='dw')
    parser.add_argument('--data_root', type=str, default='data/raw/mass_roads')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--max_cases', type=int, default=14)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(
        block_type=args.block_type,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1,
        directions=16,
        use_confidence_gate=True,
    )
    model = model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    dataset = MassachusettsRoadsDataset(
        root_dir=args.data_root,
        split=args.split,
        crop_size=args.crop_size,
        augment=False,
    )
    orig_random_crop = dataset._random_crop
    def deterministic_crop(img, mask, deterministic=False):
        return orig_random_crop(img, mask, deterministic=True)
    dataset._random_crop = deterministic_crop

    results = analyze_region_conflict(
        model, dataset, device, args.block_type, max_cases=args.max_cases
    )

    # Aggregate stats
    print(f"\n=== Failure Region Localization ({args.block_type}, {len(results)} cases) ===")
    region_names = ['intersection', 'endpoint', 'straight', 'wide', 'narrow', 'background_near']
    for rname in region_names:
        vals = [c[rname]['conflict_ratio'] for c in results if rname in c]
        if vals:
            print(f"{rname:20s}: conflict={np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
