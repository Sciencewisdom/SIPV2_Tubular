"""Regression test: UNet must forward `image` to ALL sipv2-family blocks.

Root cause of the 2026-08-15 audit finding (commits 56c5d56 / ca8c3b2):
UNet.forward gated image passing on block_type == 'sipv2' only, so
sipv2_full (E5) and sipv2_road (R1/R2/ATW) ran with their diffusion
branch silently disabled (block.forward treats image=None as
"skip diffusion"). This test fails if the diffusion branch of any
sipv2-family model stops responding to the input image.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sipv2.models.model_factory import build_model

IMAGE_AWARE_BLOCKS = ['sipv2', 'sipv2_full', 'sipv2_road']


def _build(block_type, **kw):
    torch.manual_seed(0)
    return build_model(
        block_type=block_type, in_channels=3, num_classes=1,
        channels=[32, 64, 128, 256], blocks_per_stage=[2, 2, 2, 2],
        decoder_blocks=1, **kw,
    ).eval()


def test_image_reaches_diffusion_branch():
    """Output must differ when the input image content changes (fixed features
    is impossible end-to-end, so instead we vary tensor_sigma: a diffusion-only
    parameter. If the diffusion branch runs, sigma changes the output)."""
    x = torch.randn(1, 3, 128, 128)
    for bt in IMAGE_AWARE_BLOCKS:
        m1 = _build(bt, tensor_sigma=1.5)
        m2 = _build(bt, tensor_sigma=3.0)
        with torch.no_grad():
            d = (m1(x, image=x) - m2(x, image=x)).abs().max().item()
        assert d > 1e-6, (
            f"{bt}: output unresponsive to tensor_sigma with image passed — "
            f"diffusion branch is not receiving the image (regression of the "
            f"UNet image-forwarding bug)"
        )


def test_road_ablation_switches_are_live():
    """grad_op / stencil / use_confidence_gate must each change the forward."""
    x = torch.randn(1, 3, 128, 128)
    base = _build('sipv2_road')
    sobel = _build('sipv2_road', grad_op='sobel')
    stencil3 = _build('sipv2_road', stencil=3)
    nogate = _build('sipv2_road', use_confidence_gate=False)
    with torch.no_grad():
        y = base(x, image=x)
        for name, m in [('grad_op=sobel', sobel), ('stencil=3', stencil3),
                        ('gate off', nogate)]:
            d = (y - m(x, image=x)).abs().max().item()
            assert d > 1e-6, f"sipv2_road ablation switch has no effect: {name}"


def test_non_sipv2_blocks_ignore_image():
    """conv/dw must NOT change with image (guard against wiring image into
    blocks that were never designed for it)."""
    x = torch.randn(1, 3, 128, 128)
    for bt in ['conv', 'dw']:
        m = _build(bt)
        with torch.no_grad():
            d = (m(x) - m(x, image=x)).abs().max().item()
        assert d == 0.0, f"{bt}: output changed when image passed"
