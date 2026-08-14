"""
Model factory: create U-Net with different block types.
"""
from .unet import UNet
from .blocks_conv import ConvBlock, DWBlock
from .blocks_diffusion import IsoDiffusionBlock
from .blocks_old_sip import OldSIPBlock
from .blocks_sipv2 import SIPV2BlockWrapper
from .blocks_sipv2_full import SIPV2FullBlockWrapper
from .blocks_sipv2_road import SIPV2RoadBlockWrapper


BLOCK_REGISTRY = {
    'conv': ConvBlock,
    'dw': DWBlock,
    'iso': IsoDiffusionBlock,
    'old_sip': OldSIPBlock,
    'sipv2': SIPV2BlockWrapper,
    'sipv2_full': SIPV2FullBlockWrapper,
    'sipv2_road': SIPV2RoadBlockWrapper,
}


def make_block(block_type, channels, **kwargs):
    """Create a block instance."""
    if block_type not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block_type: {block_type}. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[block_type](channels, **kwargs)


def build_model(
    block_type='conv',
    in_channels=3,
    num_classes=1,
    channels=[32, 64, 128, 256],
    blocks_per_stage=[2, 2, 2, 2],
    decoder_blocks=1,
    use_deep_supervision=False,
    **kwargs
):
    """
    Build a U-Net model with specified block type.
    """
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channels=channels,
        blocks_per_stage=blocks_per_stage,
        block_type=block_type,
        decoder_blocks=decoder_blocks,
        use_deep_supervision=use_deep_supervision,
        **kwargs
    )


def build_experiment_model(exp_name, **override_kwargs):
    """
    Build a model for a specific experiment.

    Experiments:
        E0: U-Net with ConvBlock
        E1: U-Net with DWBlock
        E2: U-Net with IsoDiffusionBlock
        E3: U-Net with OldSIPBlock
        E4: U-Net with SIPV2Block (minimal)
        E5: U-Net with SIPV2FullBlock (full)
    """
    config = {
        'E0': {'block_type': 'conv'},
        'E1': {'block_type': 'dw'},
        'E2': {'block_type': 'iso'},
        'E3': {'block_type': 'old_sip'},
        'E4': {'block_type': 'sipv2'},
        'E5': {'block_type': 'sipv2_full'},
    }

    if exp_name not in config:
        raise ValueError(f"Unknown experiment: {exp_name}")

    cfg = config[exp_name].copy()
    cfg.update(override_kwargs)
    return build_model(**cfg)
