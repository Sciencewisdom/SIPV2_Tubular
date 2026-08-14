"""
Directional diffusion operator using structure tensor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Direction sets
DIRS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_8 = [(1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]


def directional_diffusion(x, T_components, directions=8):
    """
    Directional diffusion with structure tensor.

    Args:
        x: [B, C, H, W] feature map
        T_components: dict with 't11', 't12', 't22' each [B, 1, H, W]
        directions: 4 or 8

    Returns:
        diff: [B, C, H, W] diffusion output
    """
    t11 = T_components['t11']
    t12 = T_components['t12']
    t22 = T_components['t22']

    dirs = DIRS_8 if directions == 8 else DIRS_4
    out = 0.0

    for dx, dy in dirs:
        x_nb = torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        denom = dx * dx + dy * dy
        # Weight: d^T T d / |d|^2
        w = (t11 * dx * dx + 2 * t12 * dx * dy + t22 * dy * dy) / denom
        out = out + w * (x_nb - x)

    return out


def build_diffusion_tensor_from_structure(st_output, lambda_par, lambda_perp):
    """
    Build diffusion tensor T from structure tensor directions and learned lambdas.

    Args:
        st_output: dict from StructureTensor with v2x, v2y (tangent direction)
        lambda_par: [B, 1, H, W] - strength along tangent
        lambda_perp: [B, 1, H, W] - strength along normal

    Returns:
        dict with t11, t12, t22
    """
    v2x = st_output['v2x']  # tangent x
    v2y = st_output['v2y']  # tangent y
    v1x = st_output['v1x']  # normal x
    v1y = st_output['v1y']  # normal y

    # T = lambda_par * v_t v_t^T + lambda_perp * v_n v_n^T
    t11 = lambda_par * v2x * v2x + lambda_perp * v1x * v1x
    t12 = lambda_par * v2x * v2y + lambda_perp * v1x * v1y
    t22 = lambda_par * v2y * v2y + lambda_perp * v1y * v1y

    return {'t11': t11, 't12': t12, 't22': t22}
