"""
Lightweight 2D U-Net backbone for retinal vessel segmentation.
All block types share the same encoder-decoder architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    """Downsampling by strided conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Upsample(nn.Module):
    """Upsampling by transposed conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class UNet(nn.Module):
    """
    Lightweight 2D U-Net.

    Args:
        in_channels: input channels (3 for RGB)
        num_classes: output classes (1 for binary vessel)
        channels: list of channel counts per stage, e.g., [32, 64, 128, 256]
        blocks_per_stage: number of blocks per encoder stage
        block_type: 'conv', 'dw', 'iso', 'old_sip', 'sipv2'
        decoder_blocks: number of blocks per decoder stage (0 = just fuse)
        use_deep_supervision: whether to output multi-scale predictions
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        channels=[32, 64, 128, 256],
        blocks_per_stage=[2, 2, 2, 2],
        block_type='conv',
        decoder_blocks=1,
        use_deep_supervision=False,
        **block_kwargs
    ):
        super().__init__()
        self.num_stages = len(channels)
        self.channels = channels
        self.use_deep_supervision = use_deep_supervision
        self.block_type = block_type

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
        )

        # Import block factory here to avoid circular imports
        from .model_factory import make_block

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, (ch, n_blk) in enumerate(zip(channels, blocks_per_stage)):
            blocks = []
            for _ in range(n_blk):
                blocks.append(make_block(
                    block_type, ch,
                    stage_id=i,
                    **block_kwargs
                ))
            self.enc_blocks.append(nn.Sequential(*blocks))
            if i < self.num_stages - 1:
                self.downs.append(Downsample(ch, channels[i + 1]))
            else:
                self.downs.append(nn.Identity())

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.aux_heads = nn.ModuleList() if use_deep_supervision else None

        for i in range(self.num_stages - 1, 0, -1):
            self.ups.append(Upsample(channels[i], channels[i - 1]))
            fuse_ch = channels[i - 1] * 2
            self.dec_fuse.append(nn.Sequential(
                nn.Conv2d(fuse_ch, channels[i - 1], kernel_size=1),
                nn.GroupNorm(min(8, channels[i - 1]), channels[i - 1]),
                nn.GELU(),
            ))
            # Decoder blocks
            dec_blks = []
            for _ in range(decoder_blocks):
                dec_blks.append(make_block(
                    block_type, channels[i - 1],
                    stage_id=i - 1,
                    **block_kwargs
                ))
            self.dec_blocks.append(nn.Sequential(*dec_blks) if dec_blks else nn.Identity())

            if use_deep_supervision:
                self.aux_heads.append(nn.Conv2d(channels[i - 1], num_classes, kernel_size=1))

        # Main head
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x, image=None):
        """
        Args:
            x: [B, 3, H, W] RGB image
            image: optional, same as x, passed to blocks that need original image
        Returns:
            If use_deep_supervision: list of [B, 1, H, W] at multiple scales
            Else: [B, 1, H, W]
        """
        x = self.stem(x)

        # Encoder
        skips = []
        for i in range(self.num_stages):
            # Pass image to blocks that need it
            if self.block_type in ('sipv2',) and image is not None:
                for block in self.enc_blocks[i]:
                    x = block(x, image)
            else:
                x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.downs[i](x)

        # Decoder
        seg_outputs = []
        for i in range(self.num_stages - 1, 0, -1):
            idx = self.num_stages - 1 - i
            x = self.ups[idx](x)
            skip = skips[i - 1]

            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = self.dec_fuse[idx](x)

            if self.block_type in ('sipv2',) and image is not None:
                for block in self.dec_blocks[idx]:
                    x = block(x, image)
            else:
                x = self.dec_blocks[idx](x)

            if self.use_deep_supervision:
                seg_outputs.append(self.aux_heads[idx](x))

        main_out = self.seg_head(x)

        if self.use_deep_supervision:
            seg_outputs.append(main_out)
            return seg_outputs[::-1]  # highest resolution first
        return main_out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
