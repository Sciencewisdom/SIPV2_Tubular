"""
Isotropic diffusion block.
T = alpha * I, where alpha is learned.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IsoDiffusionBlock(nn.Module):
    """
    Isotropic diffusion: T = alpha * I.
    Proves the benefit is from anisotropy, not just smoothing.
    """
    def __init__(self, channels, rho=0.3, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus(),
        )
        self.react = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.rho = rho

    def forward(self, x):
        u = self.norm(x)
        alpha = self.alpha_head(u) + 1e-4  # [B, 1, H, W]

        # 4-neighbor Laplacian
        lap = (torch.roll(u, 1, -1) + torch.roll(u, -1, -1) +
               torch.roll(u, 1, -2) + torch.roll(u, -1, -2) - 4 * u)
        diff = alpha * lap

        # Clip
        diff_norm = diff.norm(dim=1, keepdim=True)
        u_norm = u.detach().norm(dim=1, keepdim=True) + 1e-6
        max_norm = self.rho * u_norm
        scale = torch.clamp(max_norm / (diff_norm + 1e-6), max=1.0)
        diff = diff * scale

        rea = self.react(u)
        return x + diff + rea
