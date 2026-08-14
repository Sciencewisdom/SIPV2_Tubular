"""
Standard convolution blocks for baseline and depthwise separable variants.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard conv block: Conv -> Norm -> Act -> Conv -> Norm -> Act
    Residual connection.
    """
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class DWBlock(nn.Module):
    """
    Depthwise separable conv block.
    Equivalent complexity depthwise separable conv as SIP diffusion alternative.
    """
    def __init__(self, channels, expand_ratio=2, **kwargs):
        super().__init__()
        hidden = int(channels * expand_ratio)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.norm_dw = nn.GroupNorm(min(8, hidden), hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.act(self.pw1(self.norm1(x)))
        out = self.act(self.norm_dw(self.dw(out)))
        out = self.pw2(out)
        return out + residual
