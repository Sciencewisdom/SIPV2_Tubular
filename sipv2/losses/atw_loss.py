"""
Adaptive Topology Weighting (ATW) loss.
Weights clDice by tensor coherence to reduce topology pressure at junctions
and low-confidence regions.

Form: lambda(x) = lambda_base * c(x)
where c(x) = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)
is the structure tensor coherence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.structure_tensor import StructureTensor


def soft_skeletonize(x, num_iter=5):
    """Differentiable soft skeletonization."""
    for _ in range(num_iter):
        x = x * F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x


class ATWLoss(nn.Module):
    """
    Adaptive Topology Weighting for topology-aware losses.

    High coherence (straight vessels/roads) -> full topology pressure.
    Low coherence (junctions, isotropic regions) -> reduced topology pressure.
    """

    def __init__(self, lambda_base=0.5, sigma=1.0, smooth=1e-6, skeleton_iter=5, use_prediction=False):
        super().__init__()
        self.lambda_base = lambda_base
        self.smooth = smooth
        self.skeleton_iter = skeleton_iter
        self.use_prediction = use_prediction
        self.structure_tensor = StructureTensor(sigma=sigma)

    def compute_coherence(self, image_or_pred):
        """
        Compute tensor coherence map from input image or prediction.

        Args:
            image_or_pred: [B, C, H, W] input image (e.g., RGB or grayscale) or prediction
        Returns:
            coherence: [B, 1, H, W] in [0, 1]
        """
        # Ensure structure tensor is on the same device as input
        if self.structure_tensor.gaussian_kernel.device != image_or_pred.device:
            self.structure_tensor = self.structure_tensor.to(image_or_pred.device)
        st = self.structure_tensor(image_or_pred)
        l1 = st['lambda1']  # [B, C, H, W]
        l2 = st['lambda2']
        # Per-channel coherence, then average
        coherence = (l1 - l2) / (l1 + l2 + self.smooth)  # [B, C, H, W]
        coherence = coherence.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        return coherence

    def forward(self, pred, target, image):
        """
        Args:
            pred: [B, 1, H, W] logits
            target: [B, 1, H, W] binary ground truth
            image: [B, C, H, W] input image for coherence computation (if not use_prediction)
        Returns:
            atw_loss: scalar
            info: dict with coherence stats for logging
        """
        pred_prob = torch.sigmoid(pred)

        # Compute coherence-weighted clDice
        if self.use_prediction:
            coherence = self.compute_coherence(pred_prob)
        else:
            coherence = self.compute_coherence(image)

        # Soft skeletons
        skel_pred = soft_skeletonize(pred_prob, self.skeleton_iter)
        skel_target = soft_skeletonize(target.float(), self.skeleton_iter)

        # Weighted precision and sensitivity
        tprec = (skel_target * pred_prob * coherence).sum() / \
                (skel_pred * coherence).sum().clamp_min(self.smooth)
        tsens = (skel_pred * target.float() * coherence).sum() / \
                (skel_target * coherence).sum().clamp_min(self.smooth)

        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        loss = self.lambda_base * (1.0 - cl_dice)

        info = {
            'coherence_mean': coherence.mean().item(),
            'coherence_std': coherence.std().item(),
            'coherence_min': coherence.min().item(),
            'coherence_max': coherence.max().item(),
        }
        return loss, info
