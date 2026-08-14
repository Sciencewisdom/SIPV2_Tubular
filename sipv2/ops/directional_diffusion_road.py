"""
Directional diffusion with 5x5 stencil and confidence gating for road extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 5x5 directional stencil (16 directions)
DIRS_16 = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]


def directional_diffusion_5x5(x, T_components, directions=16):
    """
    Directional diffusion with 5x5 structure tensor.

    Args:
        x: [B, C, H, W] feature map
        T_components: dict with 't11', 't12', 't22' each [B, 1, H, W]
        directions: 8 (3x3) or 16 (5x5)

    Returns:
        diff: [B, C, H, W] diffusion output
    """
    t11 = T_components['t11']
    t12 = T_components['t12']
    t22 = T_components['t22']

    dirs = DIRS_16 if directions == 16 else [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    out = 0.0

    for dx, dy in dirs:
        x_nb = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        denom = dx * dx + dy * dy
        w = (t11 * dx * dx + 2 * t12 * dx * dy + t22 * dy * dy) / denom
        out = out + w * (x_nb - x)

    return out


def isotropic_diffusion_5x5(x, alpha):
    """
    Isotropic diffusion using 5x5 Laplacian-like stencil.
    Used as fallback when orientation confidence is low.

    Args:
        x: [B, C, H, W]
        alpha: [B, 1, H, W] scalar diffusivity

    Returns:
        diff: [B, C, H, W]
    """
    # 5x5 normalized Laplacian (approximation)
    # Use a larger kernel for isotropic smoothing
    kernel = torch.tensor([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, -12, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ], dtype=torch.float32).view(1, 1, 5, 5) / 12.0

    if x.is_cuda:
        kernel = kernel.to(x.device)

    B, C, H, W = x.shape
    diff_list = []
    for c in range(C):
        xc = x[:, c:c+1]
        lap = F.conv2d(xc, kernel, padding=2)
        diff_list.append(lap)
    diff = torch.cat(diff_list, dim=1)
    return alpha * diff


def build_diffusion_tensor_from_structure(st_output, lambda_par, lambda_perp):
    """
    Build diffusion tensor T from structure tensor directions and learned lambdas.
    (Same as original, kept here for self-containment)
    """
    v2x = st_output['v2x']
    v2y = st_output['v2y']
    v1x = st_output['v1x']
    v1y = st_output['v1y']

    t11 = lambda_par * v2x * v2x + lambda_perp * v1x * v1x
    t12 = lambda_par * v2x * v2y + lambda_perp * v1x * v1y
    t22 = lambda_par * v2y * v2y + lambda_perp * v1y * v1y

    return {'t11': t11, 't12': t12, 't22': t22}
