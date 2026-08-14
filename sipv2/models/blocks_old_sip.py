"""
Old SIPBlock (from SIPNet v1) — freely learned structure tensor.
Reproduces the failure mode: tensor becomes globally uniform.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..ops import directional_diffusion, relative_norm_clip


class OldSIPBlock(nn.Module):
    """
    Old SIPBlock: free tensor prediction.
    StructureTensorEstimator predicts theta, lambda_par, lambda_perp.
    """
    def __init__(self, channels, c_mid=16, rho=0.3, **kwargs):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        # Tensor estimator
        self.reduce = nn.Conv2d(channels, c_mid, kernel_size=1)
        self.dw = nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, groups=c_mid)
        self.head = nn.Conv2d(c_mid, 3, kernel_size=1)
        # Reaction
        self.react = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.rho = rho

    def _predict_tensor(self, x):
        z = self.head(F.gelu(self.dw(self.reduce(x))))
        theta = math.pi * torch.tanh(z[:, 0:1])
        lam_par = F.softplus(z[:, 1:2]) + 1e-4
        lam_perp = lam_par * torch.sigmoid(-z[:, 2:3]) + 1e-4

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        t11 = lam_par * cos_t * cos_t + lam_perp * sin_t * sin_t
        t12 = (lam_par - lam_perp) * cos_t * sin_t
        t22 = lam_par * sin_t * sin_t + lam_perp * cos_t * cos_t
        return {'t11': t11, 't12': t12, 't22': t22}

    def forward(self, x):
        u = self.norm(x)
        T = self._predict_tensor(u)
        diff = directional_diffusion(u, T, directions=8)
        diff, scale = relative_norm_clip(diff, u, rho=self.rho)
        rea = self.react(u)
        return x + diff + rea
