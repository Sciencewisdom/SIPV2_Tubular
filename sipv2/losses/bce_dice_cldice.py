"""
Combined BCE + Dice + clDice loss with warmup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice import DiceLoss
from .cldice import CLDiceLoss


class BCEDiceCLDiceLoss(nn.Module):
    """BCE + Dice + optional clDice with warmup."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, cldice_weight=0.3, cldice_warmup=20,
                 cldice_variant='crossed', legacy_double_sigmoid=True):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.cldice_warmup = cldice_warmup
        # legacy_double_sigmoid: historical code passed torch.sigmoid(pred) into
        # DiceLoss, which applies sigmoid() again (double sigmoid). Default True
        # keeps historical comparability; False fixes it.
        self.legacy_double_sigmoid = legacy_double_sigmoid
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        self.cldice = CLDiceLoss(variant=cldice_variant)

    def forward(self, pred, target, mask=None, epoch=0):
        # BCE (with mask)
        bce_loss = self.bce(pred, target)
        if mask is not None:
            bce_loss = bce_loss * mask
            bce_loss = bce_loss.sum() / (mask.sum() + 1e-6)
        else:
            bce_loss = bce_loss.mean()

        # Dice (DiceLoss applies sigmoid internally; legacy path double-applies it)
        dice_loss = self.dice(torch.sigmoid(pred) if self.legacy_double_sigmoid else pred, target)

        # clDice (with warmup)
        cldice_loss = 0.0
        if epoch >= self.cldice_warmup:
            cldice_loss = self.cldice(pred, target, mask)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        if epoch >= self.cldice_warmup:
            total = total + self.cldice_weight * cldice_loss

        return total
