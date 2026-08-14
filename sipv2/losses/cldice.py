"""
centerline Dice (clDice) loss for tubular structure segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


def soft_cldice(pred, target, skeleton_pred, skeleton_target, smooth=1e-6):
    """
    Soft clDice: overlap between pred and target skeletons.

    Args:
        pred: [B, 1, H, W] probabilities
        target: [B, 1, H, W] binary
        skeleton_pred: [B, 1, H, W] predicted skeleton (soft)
        skeleton_target: [B, 1, H, W] target skeleton (hard)
    """
    tprec = (skeleton_target * pred).sum() / (skeleton_pred.sum() + smooth)
    tsens = (skeleton_pred * target).sum() / (skeleton_target.sum() + smooth)
    return 1.0 - 2.0 * tprec * tsens / (tprec + tsens + smooth)


class CLDiceLoss(nn.Module):
    """
    clDice loss with skeleton extraction.
    For efficiency, we use morphological approximation.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def _soft_skeletonize(self, x, num_iter=5):
        """
        Soft skeletonization via iterative erosion/dilation.
        Simplified version for GPU.
        """
        # Use max pooling as approximation of morphological operations
        # This is a differentiable approximation
        for _ in range(num_iter):
            x = x * F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary
            mask: [B, 1, H, W] optional FOV mask
        """
        pred_prob = torch.sigmoid(pred)

        # Soft skeleton
        skel_pred = self._soft_skeletonize(pred_prob)
        skel_target = self._soft_skeletonize(target.float())

        if mask is not None:
            skel_pred = skel_pred * mask
            skel_target = skel_target * mask

        # clDice
        tprec = (skel_target * pred_prob).sum() / (skel_pred.sum() + self.smooth)
        tsens = (skel_pred * target.float()).sum() / (skel_target.sum() + self.smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice
