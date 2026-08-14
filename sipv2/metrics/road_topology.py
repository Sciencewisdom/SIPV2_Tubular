"""
Road topology metrics: APLS, connectivity, gap recovery.
Optimized for speed during training validation.
"""
import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree


def mask_to_graph_fast(binary_mask):
    """
    Fast graph construction from binary skeleton.
    Uses 8-connectivity and unweighted edges for BFS speed.
    """
    skel = skeletonize(binary_mask > 0)
    if skel.sum() == 0:
        return nx.Graph(), skel

    coords = np.argwhere(skel)
    if len(coords) < 2:
        return nx.Graph(), skel

    # Build coordinate set for O(1) lookup
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

    return G, skel


def apls_score(pred, target, threshold=0.5, n_samples=50, tolerance=4):
    """
    APLS (Average Path Length Similarity) with tolerance-based node matching.
    
    Tolerance parameter allows nodes to match within a small spatial radius,
    which is crucial because skeletonization can shift by 1-2 pixels.
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    G_pred, _ = mask_to_graph_fast(pred_binary)
    G_target, _ = mask_to_graph_fast(target_binary)

    if len(G_target.nodes()) == 0 or len(G_pred.nodes()) == 0:
        return 0.0

    nodes_target = list(G_target.nodes())
    nodes_pred = list(G_pred.nodes())
    
    # Build KD-tree for fast nearest-neighbor lookup in prediction graph
    pred_coords = np.array(nodes_pred)
    pred_tree = cKDTree(pred_coords)
    
    rng = np.random.default_rng(42)

    scores = []
    n_target_nodes = len(nodes_target)
    
    # If graph is small, sample all pairs; otherwise sample n_samples
    max_samples = min(n_samples, n_target_nodes * (n_target_nodes - 1) // 2)
    
    attempts = 0
    max_attempts = n_samples * 3
    
    while len(scores) < max_samples and attempts < max_attempts:
        attempts += 1
        if n_target_nodes < 2:
            break
        s, t = rng.choice(n_target_nodes, size=2, replace=False)
        src = nodes_target[s]
        dst = nodes_target[t]

        try:
            len_target = nx.shortest_path_length(G_target, src, dst)
        except nx.NetworkXNoPath:
            continue

        # Find nearest nodes in prediction graph within tolerance
        src_dist, src_idx = pred_tree.query(src, k=1)
        dst_dist, dst_idx = pred_tree.query(dst, k=1)
        
        if src_dist > tolerance or dst_dist > tolerance:
            scores.append(0.0)
            continue
            
        src_pred = tuple(pred_coords[src_idx])
        dst_pred = tuple(pred_coords[dst_idx])

        try:
            len_pred = nx.shortest_path_length(G_pred, src_pred, dst_pred)
        except nx.NetworkXNoPath:
            scores.append(0.0)
            continue

        diff = abs(len_pred - len_target)
        score = max(0.0, 1.0 - diff / (len_target + 1e-8))
        scores.append(score)

    if len(scores) == 0:
        return 0.0
    return float(np.mean(scores))


def connectivity_score(pred, target, threshold=0.5):
    """Connectivity score based on component count ratio."""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    _, n_target = ndimage.label(target_binary)
    _, n_pred = ndimage.label(pred_binary)

    if n_target == 0:
        return 0.0
    score = max(0.0, 1.0 - abs(n_pred - 1) / n_target)
    return float(score)


def gap_recovery_rate(pred, target, threshold=0.5):
    """Gap recovery: pred skeleton overlap with dilated target skeleton."""
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    skel_target = skeletonize(target_binary > 0)
    skel_pred = skeletonize(pred_binary > 0)

    if skel_target.sum() == 0:
        return 0.0

    dilated = ndimage.binary_dilation(skel_target, iterations=2)
    recall = (skel_pred & dilated).sum() / (skel_pred.sum() + 1e-8)
    return float(recall)


def compute_all_road_topology_metrics(pred, target, threshold=0.5):
    """Compute all road topology metrics."""
    return {
        'apls': apls_score(pred, target, threshold),
        'connectivity': connectivity_score(pred, target, threshold),
        'gap_recovery': gap_recovery_rate(pred, target, threshold),
    }
