"""
Skeleton-based metrics: clDice, skeleton recall, branch breaks.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def extract_skeleton(binary_mask):
    """
    Extract skeleton from binary mask.
    Args:
        binary_mask: [H, W] binary numpy array
    Returns:
        skeleton: [H, W] binary skeleton
    """
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask)
    return skeletonize(binary_mask > 0)


def cl_dice_score(pred, target, threshold=0.5):
    """
    Compute clDice score.
    Args:
        pred: [H, W] probabilities or logits
        target: [H, W] binary
    Returns:
        clDice score
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        return 0.0

    # Skeletons
    skel_pred = extract_skeleton(pred_binary)
    skel_target = extract_skeleton(target_binary)

    if skel_pred.sum() == 0 or skel_target.sum() == 0:
        return 0.0

    # Precision: pred skeleton on target mask
    tprec = (skel_pred & target_binary).sum() / (skel_pred.sum() + 1e-8)
    # Sensitivity: target skeleton on pred mask
    tsens = (skel_target & pred_binary).sum() / (skel_target.sum() + 1e-8)

    return 2.0 * tprec * tsens / (tprec + tsens + 1e-8)


def skeleton_recall(pred, target, threshold=0.5):
    """
    Skeleton recall: what fraction of target skeleton is covered by prediction.
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    if target_binary.sum() == 0:
        return 0.0

    skel_target = extract_skeleton(target_binary)
    if skel_target.sum() == 0:
        return 0.0

    recall = (skel_target & pred_binary).sum() / skel_target.sum()
    return float(recall)


def skeleton_precision(pred, target, threshold=0.5):
    """
    Skeleton precision: what fraction of pred skeleton is on target.
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    if pred_binary.sum() == 0:
        return 0.0

    skel_pred = extract_skeleton(pred_binary)
    if skel_pred.sum() == 0:
        return 0.0

    precision = (skel_pred & target_binary).sum() / skel_pred.sum()
    return float(precision)


def count_branch_breaks(pred, target, threshold=0.5):
    """
    Count branch breaks by comparing connected components of skeletons.
    A simple proxy: difference in number of connected components.
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    skel_pred = extract_skeleton(pred_binary)
    skel_target = extract_skeleton(target_binary)

    if skel_pred.sum() == 0 or skel_target.sum() == 0:
        return {'pred_components': 0, 'target_components': 0, 'break_count': 0}

    _, n_pred = ndimage.label(skel_pred)
    _, n_target = ndimage.label(skel_target)

    return {
        'pred_components': n_pred,
        'target_components': n_target,
        'break_count': abs(n_pred - n_target),
    }


def thin_vessel_recall(pred, target, threshold=0.5):
    """
    Recall on thin vessels only (estimated by distance transform).
    Vessels with small distance transform values are thin.
    """
    pred_binary = (pred > threshold).astype(np.uint8)
    target_binary = target.astype(np.uint8)

    if target_binary.sum() == 0:
        return 0.0

    # Distance transform on target
    dist = ndimage.distance_transform_edt(target_binary > 0)

    # Thin vessels: distance < 2 pixels
    thin_mask = dist < 2
    if thin_mask.sum() == 0:
        return 0.0

    thin_recall = (pred_binary[thin_mask] & target_binary[thin_mask]).sum() / thin_mask.sum()
    return float(thin_recall)


def compute_all_skeleton_metrics(pred, target, threshold=0.5):
    """Compute all skeleton-based metrics."""
    return {
        'cldice': cl_dice_score(pred, target, threshold),
        'skeleton_recall': skeleton_recall(pred, target, threshold),
        'skeleton_precision': skeleton_precision(pred, target, threshold),
        'thin_vessel_recall': thin_vessel_recall(pred, target, threshold),
        **count_branch_breaks(pred, target, threshold),
    }
