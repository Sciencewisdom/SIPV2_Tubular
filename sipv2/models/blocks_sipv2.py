"""
SIP-v2 minimal block.
Gradient-anchored structure tensor with learned diffusion strength.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops import (
    compute_structure_tensor,
    build_diffusion_tensor_from_structure,
    directional_diffusion,
    relative_norm_clip,
)


class SIPV2Block(nn.Module):
    """
    SIP-v2 minimal block.

    Y = X + Phi(X) + beta * Diff(X, T_anchored)

    where:
    - Phi(X): reaction branch (CNN)
    - T_anchored: structure tensor from image/feature gradients
    - beta: small learnable residual weight (init ~0.047)
    - Diff: directional diffusion with output clipping
    """

    def __init__(
        self,
        channels,
        stage_id=0,
        rho=0.3,
        beta_init=-3.0,
        lambda_max=1.0,
        lambda_min=1e-4,
        directions=8,
        tensor_sigma=1.0,
    ):
        super().__init__()
        self.channels = channels
        self.stage_id = stage_id
        self.rho = rho
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.directions = directions

        # --- Reaction branch ---
        # GroupNorm -> 1x1 expand -> GELU -> 3x3 DW -> GELU -> 1x1 project
        hidden = channels * 2
        self.react_norm = nn.GroupNorm(min(8, channels), channels)
        self.react_pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.react_act = nn.GELU()
        self.react_dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.react_norm_dw = nn.GroupNorm(min(8, hidden), hidden)
        self.react_pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

        # --- Structure tensor computation (image gradient) ---
        self.st = compute_structure_tensor
        self.tensor_sigma = tensor_sigma

        # --- Lambda strength modulation ---
        # Predict lambda_parallel and lambda_perp ratio from features
        self.lambda_norm = nn.GroupNorm(min(8, channels), channels)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1),  # [lambda_par_logit, lambda_ratio_logit]
        )

        # --- Beta: small learnable residual weight ---
        self.beta_logit = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def _reaction(self, x):
        u = self.react_norm(x)
        out = self.react_act(self.react_pw1(u))
        out = self.react_act(self.react_norm_dw(self.react_dw(out)))
        out = self.react_pw2(out)
        return out

    def _compute_diffusion(self, x, image):
        """
        Compute directional diffusion with anchored structure tensor.

        Args:
            x: [B, C, H, W] features
            image: [B, 3, H, W] original RGB image (may differ from feature resolution)
        Returns:
            diff: [B, C, H, W]
            aux: dict with tensor info
        """
        # Resize image to match feature resolution if needed
        _, _, h, w = x.shape
        if image.shape[-2:] != (h, w):
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)

        # Use green channel for structure tensor
        if image.shape[1] == 3:
            green = image[:, 1:2]  # [B, 1, H, W]
        else:
            green = image

        # Compute structure tensor from image gradients
        st = self.st(green, sigma=self.tensor_sigma)

        # Predict lambda strengths from features
        u = self.lambda_norm(x)
        z = self.lambda_head(u)  # [B, 2, H, W]

        lambda_par = self.lambda_max * torch.sigmoid(z[:, 0:1])
        # lambda_perp = lambda_par * sigmoid(-ratio_logit) + eps
        lambda_perp = self.lambda_min + (lambda_par - self.lambda_min) * torch.sigmoid(-z[:, 1:2])

        # Build diffusion tensor from structure directions + learned strengths
        T = build_diffusion_tensor_from_structure(st, lambda_par, lambda_perp)

        # Directional diffusion
        diff = directional_diffusion(x, T, directions=self.directions)

        # Clip
        diff, scale = relative_norm_clip(diff, x, rho=self.rho)

        # Compute ratio for logging
        ratio = lambda_par / (lambda_perp + 1e-8)

        aux = {
            'lambda_par': lambda_par.detach(),
            'lambda_perp': lambda_perp.detach(),
            'ratio': ratio.detach(),
            'theta_tangent': st['theta2'].detach(),
            'scale': scale.detach(),
            'diff_norm': diff.norm(dim=1, keepdim=True).detach(),
        }
        return diff, aux

    def forward(self, x, image=None):
        """
        Args:
            x: [B, C, H, W]
            image: [B, 3, H, W] original image (needed for structure tensor)
        Returns:
            y: [B, C, H, W]
            aux: dict or None
        """
        rea = self._reaction(x)

        aux = None
        if image is not None:
            diff, aux = self._compute_diffusion(x, image)
            beta = torch.sigmoid(self.beta_logit)
            y = x + rea + beta * diff
        else:
            # Fallback: no diffusion if image not provided
            y = x + rea

        return y, aux


class SIPV2BlockWrapper(nn.Module):
    """
    Wrapper to make SIPV2Block compatible with nn.Sequential usage.
    By default discards aux output; set return_aux=True to capture tensor info.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = SIPV2Block(*args, **kwargs)
        self.return_aux = False

    def forward(self, x, image=None):
        y, aux = self.block(x, image)
        if self.return_aux:
            # Attach aux as an attribute for external capture
            self._last_aux = aux
        return y
