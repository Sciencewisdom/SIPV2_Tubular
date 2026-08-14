"""
Structure tensor computation with Gaussian smoothing.
Used for anchoring diffusion directions to image structure.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sobel import sobel_gradients


class StructureTensor(nn.Module):
    """
    Compute structure tensor from image gradients with Gaussian smoothing.

    J = G_sigma * [Ix^2, Ix*Iy; Ix*Iy, Iy^2]

    Returns eigenvalues and eigenvectors (directions).
    """

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        # Create Gaussian kernel for smoothing
        kernel_size = max(3, int(6 * sigma) | 1)  # odd number
        self.kernel_size = kernel_size
        kernel = self._gaussian_kernel_2d(kernel_size, sigma)
        self.register_buffer('gaussian_kernel', kernel)

    def _gaussian_kernel_2d(self, size, sigma):
        """Create 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.outer(g)
        return kernel_2d.unsqueeze(0).unsqueeze(0)

    def forward(self, image):
        """
        Args:
            image: [B, C, H, W] - typically green channel of fundus
        Returns:
            dict with:
                - lambda1: [B, C, H, W] larger eigenvalue
                - lambda2: [B, C, H, W] smaller eigenvalue
                - theta1: [B, C, H, W] direction of lambda1 (normal to vessel)
                - theta2: [B, C, H, W] direction of lambda2 (tangent to vessel)
                - v1x, v1y: [B, C, H, W] eigenvector components for lambda1
                - v2x, v2y: [B, C, H, W] eigenvector components for lambda2
        """
        B, C, H, W = image.shape

        # Compute gradients
        Ix, Iy = sobel_gradients(image)

        # Structure tensor components
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy

        # Gaussian smoothing
        padding = self.kernel_size // 2
        gaussian_kernel = self.gaussian_kernel
        if C > 1:
            # Expand kernel for multi-channel
            gaussian_kernel = gaussian_kernel.expand(C, 1, -1, -1)
            groups = C
        else:
            groups = 1

        Jxx = F.conv2d(Ixx, gaussian_kernel, padding=padding, groups=groups)
        Jxy = F.conv2d(Ixy, gaussian_kernel, padding=padding, groups=groups)
        Jyy = F.conv2d(Iyy, gaussian_kernel, padding=padding, groups=groups)

        # Eigenvalues: lambda = (trace +/- sqrt(trace^2 - 4*det)) / 2
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        discriminant = trace * trace - 4 * det
        discriminant = torch.clamp(discriminant, min=1e-8)
        sqrt_disc = torch.sqrt(discriminant)

        lambda1 = (trace + sqrt_disc) / 2.0  # larger
        lambda2 = (trace - sqrt_disc) / 2.0  # smaller

        # Eigenvector direction via direct angle formula (more stable)
        # angle = 0.5 * atan2(2*Jxy, Jyy - Jxx) gives the eigenvector direction
        # for the eigenvalue associated with the smaller eigenvalue (tangent direction)
        theta_eig = 0.5 * torch.atan2(2 * Jxy, Jyy - Jxx)

        # v2 (smaller eigenvalue direction = tangent to structure)
        v2x = torch.cos(theta_eig)
        v2y = torch.sin(theta_eig)

        # v1 (larger eigenvalue direction = normal), perpendicular to v2
        v1x = -v2y
        v1y = v2x

        # Angles
        theta2 = theta_eig
        theta1 = theta_eig + torch.where(theta_eig < 0,
                                         torch.tensor(np.pi, device=theta_eig.device),
                                         torch.tensor(-np.pi, device=theta_eig.device))

        return {
            'lambda1': lambda1,
            'lambda2': lambda2,
            'theta1': theta1,
            'theta2': theta2,
            'v1x': v1x, 'v1y': v1y,
            'v2x': v2x, 'v2y': v2y,
            'Jxx': Jxx, 'Jxy': Jxy, 'Jyy': Jyy,
        }


def compute_structure_tensor(image, sigma=1.0):
    """Functional API."""
    module = StructureTensor(sigma=sigma)
    if image.is_cuda:
        module = module.to(image.device)
    return module(image)
