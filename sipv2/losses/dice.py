"""
Dice loss and BCE + Dice combination.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary 0/1
            mask: [B, 1, H, W] optional FOV mask
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        if mask is not None:
            mask = mask.view(-1)
            pred = pred[mask > 0]
            target = target[mask > 0]

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice combined loss."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target, mask=None, epoch=None):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary 0/1
            mask: [B, 1, H, W] optional FOV mask
            epoch: ignored (for compatibility with BCEDiceCLDiceLoss)
        """
        # BCE
        bce_loss = self.bce(pred, target.float())
        if mask is not None:
            bce_loss = bce_loss * mask
            bce_loss = bce_loss.sum() / (mask.sum() + 1e-8)
        else:
            bce_loss = bce_loss.mean()

        # Dice
        dice_loss = self.dice(pred, target, mask)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
