"""
DCNv2 block for the B2 controlled comparison (audit item B2).

Same macro-structure as DWBlock (norm -> 1x1 expand -> spatial conv -> norm ->
1x1 project -> residual), but the spatial mixing layer is a deformable conv
whose sampling offsets are *freely learned* from the features, not anchored to
the image structure tensor. If clDice harms DCN the way it harms the isotropic
DW-CNN, while it repairs SIP-v2, the decisive variable is structure-anchored
propagation direction rather than "having directional sampling" per se.

Parameter-matched to E1 (DWBlock, 1.42M): expand_ratio=1, deformable conv with
4 channel groups, offsets/masks predicted through a 16-channel bottleneck.
"""
import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DCNBlock(nn.Module):
    """
    Lightweight DCNv2 block, parameter-matched to DWBlock (E1).

    norm -> 1x1 expand -> DCNv2(grouped, bottleneck offsets) -> norm -> 1x1
    project -> residual.
    """

    def __init__(self, channels, expand_ratio=1, deform_groups=8, offset_bottleneck=8, **kwargs):
        super().__init__()
        hidden = int(channels * expand_ratio)
        self.groups = min(deform_groups, hidden)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        # bottleneck offset/mask predictor: 1x1 reduce -> 3x3 predict
        self.offset_reduce = nn.Conv2d(hidden, offset_bottleneck, kernel_size=1)
        self.offset_mask = nn.Conv2d(offset_bottleneck, self.groups * 3 * 9,
                                     kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.dcn = DeformConv2d(hidden, hidden, kernel_size=3, padding=1,
                                groups=self.groups)
        self.norm_dw = nn.GroupNorm(min(8, hidden), hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.act(self.pw1(self.norm1(x)))
        om = self.offset_mask(self.act(self.offset_reduce(out)))
        offset = om[:, : self.groups * 18]
        mask = torch.sigmoid(om[:, self.groups * 18:])
        out = self.dcn(out, offset, mask)
        out = self.act(self.norm_dw(out))
        out = self.pw2(out)
        return out + residual
