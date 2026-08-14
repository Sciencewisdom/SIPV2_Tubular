"""
Scharr gradient operators for structure tensor computation.
Better rotational symmetry than Sobel, more stable for wide/blurry edges.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScharrOperator(nn.Module):
    """Scharr gradient operator for 2D images."""

    def __init__(self):
        super().__init__()
        # Scharr kernels (3x3, optimized for rotational invariance)
        kernel_x = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] grayscale or multi-channel image
        Returns:
            Ix, Iy: [B, C, H, W] gradients
        """
        B, C, H, W = x.shape
        if C > 1:
            ix_list, iy_list = [], []
            for c in range(C):
                xc = x[:, c:c+1]
                ix = F.conv2d(xc, self.kernel_x, padding=1)
                iy = F.conv2d(xc, self.kernel_y, padding=1)
                ix_list.append(ix)
                iy_list.append(iy)
            Ix = torch.cat(ix_list, dim=1)
            Iy = torch.cat(iy_list, dim=1)
        else:
            Ix = F.conv2d(x, self.kernel_x, padding=1)
            Iy = F.conv2d(x, self.kernel_y, padding=1)
        return Ix, Iy


def scharr_gradients(x):
    """
    Functional API for Scharr gradients.
    Args:
        x: [B, C, H, W]
    Returns:
        Ix, Iy: [B, C, H, W]
    """
    op = ScharrOperator()
    if x.is_cuda:
        op = op.to(x.device)
    return op(x)
