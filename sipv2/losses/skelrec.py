"""
Soft skeleton-recall loss — a second topology-aware objective (audit item B1).

Penalizes only missed ground-truth centerlines (recall direction), unlike
clDice which balances skeleton precision and recall. Used to test whether the
loss--architecture interaction generalizes beyond clDice.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice import DiceLoss


class SkeletonRecallLoss(nn.Module):
    """1 - soft skeleton recall: fraction of GT skeleton covered by pred."""

    def __init__(self, smooth=1e-6, skeleton_iter=5):
        super().__init__()
        self.smooth = smooth
        self.skeleton_iter = skeleton_iter

    @staticmethod
    def _soft_skeletonize(x, num_iter=5):
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
        skel_target = self._soft_skeletonize(target.float(), self.skeleton_iter)
        if mask is not None:
            pred_prob = pred_prob * mask
            skel_target = skel_target * mask
        recall = (skel_target * pred_prob).sum() / (skel_target.sum() + self.smooth)
        return 1.0 - recall


class BCEDiceSkelRecLoss(nn.Module):
    """BCE + Dice + lambda * skeleton-recall, same interface as BCEDiceCLDiceLoss."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, skelrec_weight=0.3, skelrec_warmup=20,
                 legacy_double_sigmoid=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.skelrec_weight = skelrec_weight
        self.skelrec_warmup = skelrec_warmup
        # See BCEDiceCLDiceLoss: legacy path double-applies sigmoid before DiceLoss.
        self.legacy_double_sigmoid = legacy_double_sigmoid
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        self.skelrec = SkeletonRecallLoss()

    def forward(self, pred, target, mask=None, epoch=0):
        # BCE (with mask) — identical to BCEDiceCLDiceLoss
        bce_loss = self.bce(pred, target)
        if mask is not None:
            bce_loss = bce_loss * mask
            bce_loss = bce_loss.sum() / (mask.sum() + 1e-6)
        else:
            bce_loss = bce_loss.mean()

        dice_loss = self.dice(torch.sigmoid(pred) if self.legacy_double_sigmoid else pred, target)

        skel_loss = 0.0
        if epoch >= self.skelrec_warmup:
            skel_loss = self.skelrec(pred, target, mask)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        if epoch >= self.skelrec_warmup:
            total = total + self.skelrec_weight * skel_loss

        return total
