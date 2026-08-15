"""
Combined BCE + Dice + ATW (Adaptive Topology Weighting) loss.
"""
import torch.nn as nn
from .dice import DiceLoss
from .atw_loss import ATWLoss


class BCEDiceATWLoss(nn.Module):
    """BCE + Dice + optional ATW with warmup."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0,
                 atw_weight=0.3, atw_warmup=10,
                 atw_lambda_base=1.0, atw_sigma=1.0, atw_variant='crossed'):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.atw_weight = atw_weight
        self.atw_warmup = atw_warmup
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.dice = DiceLoss()
        # NOTE: atw_lambda_base defaults to 1.0 so that the effective topology
        # weight is atw_weight * c(x), matching the paper's w(x)=lambda_base*c(x)
        # (historical runs silently had an extra 0.5 factor from ATWLoss's
        # internal default lambda_base=0.5).
        self.atw = ATWLoss(lambda_base=atw_lambda_base, sigma=atw_sigma, variant=atw_variant)

    def forward(self, pred, target, mask=None, image=None, epoch=0):
        # BCE (with mask)
        bce_loss = self.bce(pred, target)
        if mask is not None:
            bce_loss = bce_loss * mask
            bce_loss = bce_loss.sum() / (mask.sum() + 1e-6)
        else:
            bce_loss = bce_loss.mean()

        # Dice (DiceLoss applies its own sigmoid; pass raw logits and FOV mask)
        dice_loss = self.dice(pred, target, mask)

        # ATW (with warmup)
        atw_loss = 0.0
        atw_info = {}
        if epoch >= self.atw_warmup and image is not None:
            atw_loss, atw_info = self.atw(pred, target, image)

        total = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        if epoch >= self.atw_warmup and image is not None:
            total = total + self.atw_weight * atw_loss

        # Attach info for logging
        total_dict = {
            'loss': total,
            'bce': bce_loss.detach(),
            'dice': dice_loss.detach(),
        }
        if atw_info:
            total_dict['atw_loss'] = atw_loss.detach()
            total_dict['coherence_mean'] = atw_info['coherence_mean']
            total_dict['coherence_std'] = atw_info['coherence_std']

        return total
