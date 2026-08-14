"""
SIP-v2 full block with prototype competition and low-rank global field.

Stage-wise configuration:
  Stage 0: reaction + diffusion only
  Stage 1: reaction + diffusion + prototype
  Stage 2: reaction + diffusion + prototype + global
  Stage 3+: reaction + diffusion + global
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


class PrototypeCompetition(nn.Module):
    """Region prototype competition branch."""
    def __init__(self, channels, num_proto=8):
        super().__init__()
        self.channels = channels
        self.num_proto = num_proto
        self.attn = nn.Conv2d(channels, num_proto, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W
        # Attention: [B, M, H, W]
        A = F.softmax(self.attn(x), dim=1)  # [B, M, H, W]
        A_flat = A.view(B, self.num_proto, N)  # [B, M, N]
        x_flat = x.view(B, C, N)  # [B, C, N]
        # Prototypes: [B, M, C]
        numerator = torch.bmm(A_flat, x_flat.transpose(1, 2))  # [B, M, C]
        denominator = A_flat.sum(dim=2, keepdim=True) + 1e-6  # [B, M, 1]
        p = numerator / denominator  # [B, M, C]
        # Reconstruct: [B, N, C]
        out = torch.bmm(A_flat.transpose(1, 2), p)  # [B, N, C]
        out = out.transpose(1, 2).view(B, C, H, W)
        return out - x  # residual form


class LowRankGlobalField(nn.Module):
    """Low-rank global field branch."""
    def __init__(self, channels, rank=8):
        super().__init__()
        self.channels = channels
        self.rank = rank
        self.Z_proj = nn.Conv2d(channels, rank, kernel_size=1)
        self.W_g = nn.Conv2d(channels, channels, kernel_size=1)
        self.W_r = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W
        # Z = softmax(W_z * x): [B, r, H, W]
        Z = F.softmax(self.Z_proj(x), dim=1)  # [B, r, H, W]
        Z_flat = Z.view(B, self.rank, N)  # [B, r, N]
        x_flat = x.view(B, C, N)  # [B, C, N]
        # B = (Z^T * X) / (Z^T * 1): [B, r, C]
        B_mat = torch.bmm(Z_flat, x_flat.transpose(1, 2))  # [B, r, C]
        denom = Z_flat.sum(dim=2, keepdim=True) + 1e-6  # [B, r, 1]
        B_mat = B_mat / denom
        # G(X) = Z * B * W_g - X * W_r
        # First: Z * B -> [B, N, C]
        zb = torch.bmm(Z_flat.transpose(1, 2), B_mat)  # [B, N, C]
        zb = zb.transpose(1, 2).view(B, C, H, W)
        out = self.W_g(zb) - self.W_r(x)
        return out


class SIPV2FullBlock(nn.Module):
    """
    Full SIP-v2 block with all branches.

    Y = X + Phi(X) + beta_d * Diff(X, T) + beta_p * Proto(X) + beta_g * Global(X)
    """
    def __init__(
        self,
        channels,
        stage_id=0,
        rho=0.3,
        beta_d_init=-3.0,
        beta_p_init=-3.5,
        beta_g_init=-3.5,
        lambda_max=1.0,
        lambda_min=1e-4,
        directions=8,
        tensor_sigma=1.0,
        num_proto=8,
        rank=8,
        use_proto=True,
        use_global=True,
    ):
        super().__init__()
        self.channels = channels
        self.stage_id = stage_id
        self.rho = rho
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.directions = directions

        # Stage-wise branch enablement
        self.use_diffusion = True
        self.use_proto = use_proto and (stage_id >= 1)  # Stage 1+
        self.use_global = use_global and (stage_id >= 2)  # Stage 2+
        # Stage 3+: no proto
        if stage_id >= 3:
            self.use_proto = False

        # --- Reaction branch ---
        hidden = channels * 2
        self.react_norm = nn.GroupNorm(min(8, channels), channels)
        self.react_pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.react_act = nn.GELU()
        self.react_dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.react_norm_dw = nn.GroupNorm(min(8, hidden), hidden)
        self.react_pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

        # --- Diffusion branch ---
        self.st = compute_structure_tensor
        self.tensor_sigma = tensor_sigma
        self.lambda_norm = nn.GroupNorm(min(8, channels), channels)
        self.lambda_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, 2, kernel_size=1),
        )
        self.beta_d_logit = nn.Parameter(torch.tensor(beta_d_init, dtype=torch.float32))

        # --- Prototype branch ---
        if self.use_proto:
            self.proto = PrototypeCompetition(channels, num_proto=num_proto)
            self.beta_p_logit = nn.Parameter(torch.tensor(beta_p_init, dtype=torch.float32))

        # --- Global field branch ---
        if self.use_global:
            self.global_field = LowRankGlobalField(channels, rank=rank)
            self.beta_g_logit = nn.Parameter(torch.tensor(beta_g_init, dtype=torch.float32))

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
        if image.shape[1] == 3:
            green = image[:, 1:2]
        else:
            green = image
        st = self.st(green, sigma=self.tensor_sigma)
        u = self.lambda_norm(x)
        z = self.lambda_head(u)
        lambda_par = self.lambda_max * torch.sigmoid(z[:, 0:1])
        lambda_perp = self.lambda_min + (lambda_par - self.lambda_min) * torch.sigmoid(-z[:, 1:2])
        T = build_diffusion_tensor_from_structure(st, lambda_par, lambda_perp)
        diff = directional_diffusion(x, T, directions=self.directions)
        diff, scale = relative_norm_clip(diff, x, rho=self.rho)
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
        rea = self._reaction(x)
        y = x + rea
        aux = None

        if image is not None and self.use_diffusion:
            diff, aux = self._compute_diffusion(x, image)
            beta_d = torch.sigmoid(self.beta_d_logit)
            y = y + beta_d * diff

        if self.use_proto:
            proto_out = self.proto(x)
            beta_p = torch.sigmoid(self.beta_p_logit)
            y = y + beta_p * proto_out
            if aux is not None:
                aux['beta_p'] = beta_p.detach()

        if self.use_global:
            global_out = self.global_field(x)
            beta_g = torch.sigmoid(self.beta_g_logit)
            y = y + beta_g * global_out
            if aux is not None:
                aux['beta_g'] = beta_g.detach()

        if aux is not None:
            aux['beta_d'] = torch.sigmoid(self.beta_d_logit).detach()

        return y, aux


class SIPV2FullBlockWrapper(nn.Module):
    """Wrapper for nn.Sequential compatibility."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = SIPV2FullBlock(*args, **kwargs)
        self.return_aux = False

    def forward(self, x, image=None):
        y, aux = self.block(x, image)
        if self.return_aux:
            self._last_aux = aux
        return y
