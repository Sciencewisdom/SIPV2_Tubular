"""
Relative norm clipping for diffusion outputs.
"""
import torch
import torch.nn as nn


def relative_norm_clip(diff, x, rho=0.3):
    """
    Clip diffusion output relative to input feature norm.

    Args:
        diff: [B, C, H, W] diffusion output
        x: [B, C, H, W] input features
        rho: clipping ratio

    Returns:
        clipped_diff: [B, C, H, W]
        scale: [B, 1, H, W] actual scale applied
    """
    diff_norm = diff.norm(dim=1, keepdim=True)
    x_norm = x.detach().norm(dim=1, keepdim=True) + 1e-6
    max_norm = rho * x_norm
    scale = torch.clamp(max_norm / (diff_norm + 1e-6), max=1.0)
    clipped = diff * scale
    return clipped, scale
