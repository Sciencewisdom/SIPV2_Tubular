"""
Junction Preservation Rate (JPR) metric.

Measures what fraction of ground-truth junctions (degree >= 3 nodes in skeleton graph)
remain preserved (degree >= 3) in the predicted skeleton.

This is critical for road networks where clDice tends to over-thin intersections.
"""
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage
from scipy.spatial import cKDTree


def skeleton_to_graph(skel):
    """
    Convert binary skeleton to graph nodes and degrees.
    Returns list of (y, x) node coordinates and their degrees.
    """
    # Find skeleton pixels
    ys, xs = np.where(skel > 0)
    if len(ys) == 0:
        return [], []

    # 8-neighborhood kernel for counting neighbors
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(skel.astype(int), kernel, mode='constant', cval=0)
    neighbor_count = neighbor_count * skel

    # Nodes are skeleton pixels with degree != 2 (endpoints: 1, junctions: >=3)
    # Actually degree in graph sense: number of connected neighbors
    node_mask = (neighbor_count != 2) * skel
    node_ys, node_xs = np.where(node_mask > 0)

    coords = []
    degrees = []
    for y, x in zip(node_ys, node_xs):
        coords.append((y, x))
        degrees.append(int(neighbor_count[y, x]))

    return coords, degrees


def compute_junction_preservation(pred_mask, target_mask, tolerance=2):
    """
    Compute Junction Preservation Rate.

    Args:
        pred_mask: [H, W] binary predicted mask
        target_mask: [H, W] binary target mask
        tolerance: pixel distance for matching junctions
    Returns:
        jpr: float in [0, 1], or np.nan if no junctions in target
        info: dict with counts
    """
    pred_skel = skeletonize(pred_mask > 0)
    target_skel = skeletonize(target_mask > 0)

    # Extract target junctions (degree >= 3)
    target_coords, target_degrees = skeleton_to_graph(target_skel)
    target_junctions = [(y, x) for (y, x), d in zip(target_coords, target_degrees) if d >= 3]

    if len(target_junctions) == 0:
        return np.nan, {'target_junctions': 0, 'preserved': 0}

    # Extract predicted junctions (degree >= 3)
    pred_coords, pred_degrees = skeleton_to_graph(pred_skel)
    pred_junctions = [(y, x) for (y, x), d in zip(pred_coords, pred_degrees) if d >= 3]

    if len(pred_junctions) == 0:
        return 0.0, {'target_junctions': len(target_junctions), 'preserved': 0}

    # Match target junctions to predicted junctions via nearest neighbor
    pred_tree = cKDTree(pred_junctions)
    preserved = 0
    for tj in target_junctions:
        dist, _ = pred_tree.query(tj, k=1)
        if dist <= tolerance:
            preserved += 1

    jpr = preserved / len(target_junctions)
    info = {
        'target_junctions': len(target_junctions),
        'pred_junctions': len(pred_junctions),
        'preserved': preserved,
        'jpr': jpr,
    }
    return jpr, info


def compute_junction_preservation_batch(pred_masks, target_masks, tolerance=2):
    """
    Batch version.
    Args:
        pred_masks: [B, H, W] or [B, 1, H, W]
        target_masks: [B, H, W] or [B, 1, H, W]
    Returns:
        jpr_list: list of floats
    """
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[:, 0]
    if target_masks.ndim == 4:
        target_masks = target_masks[:, 0]

    jpr_list = []
    for p, t in zip(pred_masks, target_masks):
        jpr, _ = compute_junction_preservation(p, t, tolerance)
        jpr_list.append(jpr)
    return jpr_list
