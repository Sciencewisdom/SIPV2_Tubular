"""
SIP-v2 Road block.
Modifications for road extraction:
  - Scharr gradient (better rotational symmetry)
  - 5x5 directional stencil (larger gap bridging)
  - Orientation confidence gate (isotropic fallback in unstructured regions)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.structure_tensor import StructureTensor, compute_structure_tensor
from ..ops.scharr import scharr_gradients
from ..ops.directional_diffusion import directional_diffusion
from ..ops.directional_diffusion_road import (
    directional_diffusion_5x5,
    isotropic_diffusion_5x5,
    build_diffusion_tensor_from_structure,
)
from ..ops.norm_clip import relative_norm_clip


def isotropic_diffusion_3x3(x, alpha):
    """5-point Laplacian isotropic diffusion (B3 stencil ablation)."""
    out = 0.0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out = out + torch.roll(x, shifts=(dy, dx), dims=(-2, -1)) - x
    return alpha * out


class StructureTensorScharr(nn.Module):
    """Structure tensor using Scharr gradients instead of Sobel."""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        kernel_size = max(3, int(6 * sigma) | 1)
        self.kernel_size = kernel_size
        kernel = self._gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)

    def _gaussian_kernel_2d(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.outer(g)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, image):
        B, C, H, W = image.shape
        Ix, Iy = scharr_gradients(image)

        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        padding = self.kernel_size // 2
        gaussian_kernel = self.gaussian_kernel
        if C > 1:
            gaussian_kernel = gaussian_kernel.expand(C, 1, -1, -1)
            groups = C
        else:
            groups = 1

        Jxx = F.conv2d(Ixx, gaussian_kernel, padding=padding, groups=groups)
        Jxy = F.conv2d(Ixy, gaussian_kernel, padding=padding, groups=groups)
        Jyy = F.conv2d(Iyy, gaussian_kernel, padding=padding, groups=groups)

        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        discriminant = trace * trace - 4 * det
        discriminant = torch.clamp(discriminant, min=1e-8)
        sqrt_disc = torch.sqrt(discriminant)

        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0

        theta_eig = 0.5 * torch.atan2(2 * Jxy, Jyy - Jxx)
        v2x = torch.cos(theta_eig)
        v2y = torch.sin(theta_eig)
        v1x = -v2y
        v1y = v2x

        theta2 = theta_eig
        theta1 = theta_eig + torch.where(theta_eig < 0,
                                         torch.tensor(torch.pi, device=theta_eig.device),
                                         torch.tensor(-torch.pi, device=theta_eig.device))

        return {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'theta1': theta1,
            'theta2': theta2,
            'v1x': v1x, 'v1y': v1y,
            'v2x': v2x, 'v2y': v2y,
            'Jxx': Jxx, 'Jxy': Jxy, 'Jyy': Jyy,
        }


def compute_structure_tensor_scharr(image, sigma=1.0):
    """Functional API."""
    module = StructureTensorScharr(sigma=sigma)
    if image.is_cuda:
        module = module.to(image.device)
    return module(image)


class SIPV2RoadBlock(nn.Module):
    """
    SIP-v2 block adapted for road extraction.

    Key differences from retinal SIP-v2:
    1. Scharr gradients (better for wide/blurry edges)
    2. 5x5 directional stencil (larger gap bridging)
    3. Orientation confidence gate c = (l1 - l2) / (l1 + l2)
       D_total = c * D_aniso + (1 - c) * D_iso
    """

    def __init__(
        self,
        channels,
        stage_id=0,
        rho=0.3,
        beta_init=-3.0,
        lambda_max=1.0,
        lambda_min=1e-4,
        directions=16,
        tensor_sigma=1.5,
        use_confidence_gate=True,
        grad_op='scharr',
        stencil=5,
    ):
        super().__init__()
        self.channels = channels
        self.stage_id = stage_id
        self.rho = rho
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.directions = directions
        self.tensor_sigma = tensor_sigma
        self.use_confidence_gate = use_confidence_gate
        # Ablation switches (B3): gradient operator and diffusion stencil size
        assert grad_op in ('scharr', 'sobel')
        assert stencil in (3, 5)
        self.grad_op = grad_op
        self.stencil = stencil

        # Reaction branch
        hidden = channels * 2
        self.react_norm = nn.GroupNorm(min(8, channels), channels)
        self.react_pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.react_act = nn.GELU()
        self.react_dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.react_norm_dw = nn.GroupNorm(min(8, hidden), hidden)
        self.react_pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

        # Lambda strength modulation
        self.lambda_norm = nn.GroupNorm(min(8, channels), channels)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        # Isotropic fallback alpha (for confidence gate)
        if use_confidence_gate:
            self.iso_norm = nn.GroupNorm(min(8, channels), channels)
            self.iso_head = nn.Sequential(
                nn.Conv2d(channels, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1),
                nn.Softplus(),
            )

        # Beta residual weight
        self.beta_logit = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def _reaction(self, x):
        u = self.react_norm(x)
        out = self.react_act(self.react_pw1(u))
        out = self.react_act(self.react_norm_dw(self.react_dw(out)))
        out = self.react_pw2(out)
        return out

    def _compute_diffusion(self, x, image):
        _, _, h, w = x.shape
        if image.shape[-2:] != (h, w):
            image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)

        # Use all channels (RGB) averaged for road images
        if image.shape[1] == 3:
            gray = image.mean(dim=1, keepdim=True)
        else:
            gray = image

        # Structure tensor (Scharr by default; Sobel for the B3 ablation)
        if self.grad_op == 'sobel':
            st = compute_structure_tensor(gray, sigma=self.tensor_sigma)
        else:
            st = compute_structure_tensor_scharr(gray, sigma=self.tensor_sigma)

        # Predict lambda strengths
        u = self.lambda_norm(x)
        z = self.lambda_head(u)
        lambda_par = self.lambda_max * torch.sigmoid(z[:, 0:1])
        lambda_perp = self.lambda_min + (lambda_par - self.lambda_min) * torch.sigmoid(-z[:, 1:2])

        # Build diffusion tensor
        T = build_diffusion_tensor_from_structure(st, lambda_par, lambda_perp)

        # Anisotropic diffusion (5x5 stencil by default; 3x3 for the B3 ablation)
        if self.stencil == 3:
            diff_aniso = directional_diffusion(x, T, directions=8)
        else:
            diff_aniso = directional_diffusion_5x5(x, T, directions=self.directions)
        diff_aniso, scale_aniso = relative_norm_clip(diff_aniso, x, rho=self.rho)

        # Confidence gate
        if self.use_confidence_gate:
            l1 = st['lambda1']
            l2 = st['lambda2']
            c = (l1 - l2) / (l1 + l2 + 1e-8)  # [B, 1, H, W], in [0, 1]

            # Isotropic fallback
            u_iso = self.iso_norm(x)
            alpha = self.iso_head(u_iso) + 1e-4
            if self.stencil == 3:
                diff_iso = isotropic_diffusion_3x3(x, alpha)
            else:
                diff_iso = isotropic_diffusion_5x5(x, alpha)
            diff_iso, scale_iso = relative_norm_clip(diff_iso, x, rho=self.rho)

            diff = c * diff_aniso + (1.0 - c) * diff_iso
        else:
            diff = diff_aniso
            c = None

        ratio = lambda_par / (lambda_perp + 1e-8)

        aux = {
            'lambda_par': lambda_par.detach(),
            'lambda_perp': lambda_perp.detach(),
            'ratio': ratio.detach(),
            'theta_tangent': st['theta2'].detach(),
            'confidence': c.detach() if c is not None else None,
        }
        return diff, aux

    def forward(self, x, image=None):
        rea = self._reaction(x)
        aux = None
        if image is not None:
            diff, aux = self._compute_diffusion(x, image)
            beta = torch.sigmoid(self.beta_logit)
            y = x + rea + beta * diff
        else:
            y = x + rea
        return y, aux


class SIPV2RoadBlockWrapper(nn.Module):
    """Wrapper for nn.Sequential compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = SIPV2RoadBlock(*args, **kwargs)
        self.return_aux = False

    def forward(self, x, image=None):
        y, aux = self.block(x, image)
        if self.return_aux:
            self._last_aux = aux
        return y
