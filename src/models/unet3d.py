"""
Symmetric 3D U-Net for volumetric medical image segmentation.

Architecture
------------
Encoder    : N resolution levels, each with 2× (Conv3d → BN → LeakyReLU) + MaxPool3d
Bottleneck : 2× (Conv3d → BN → LeakyReLU) at the coarsest resolution
Decoder    : N resolution levels, each with ConvTranspose3d (×2 upsample)
             + skip-connection concatenation + 2× (Conv3d → BN → LeakyReLU)
Output     : 1×1×1 Conv3d (raw logits; apply sigmoid externally)

Default feature sizes [32, 64, 128, 256] with bottleneck 512 give a good
balance between capacity and memory for 3D patches.

References
----------
Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation from
Sparse Annotation." MICCAI 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Two consecutive Conv3d → BatchNorm3d → LeakyReLU layers.

    This is the fundamental building block of the U-Net encoder and decoder.
    Padding=1 preserves spatial dimensions; bias=False because BN absorbs it.
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class UNet3D(nn.Module):
    """
    Symmetric 3D U-Net with configurable depth and feature map sizes.

    Parameters
    ----------
    in_channels  : number of input channels (3 for T2w + ADC + HBV)
    out_channels : number of output channels (1 for binary segmentation)
    features     : feature map sizes at each encoder level.
                   Length determines the number of pooling levels (depth).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.encoders.append(_conv_block(ch, f))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

        # ---- Bottleneck ----
        bottleneck_ch = ch * 2
        self.bottleneck = _conv_block(ch, bottleneck_ch)
        ch = bottleneck_ch

        # ---- Decoder ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for f in reversed(features):
            # Transposed convolution halves channels and doubles spatial dims
            self.upconvs.append(nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2))
            # After concatenation with skip: 2*f channels → f channels
            self.decoders.append(_conv_block(f * 2, f))
            ch = f

        # ---- Output head ----
        # 1×1×1 conv to desired number of classes; no activation (raw logits)
        self.output_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming initialisation for Conv layers; zero-init BN bias."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, D, H, W)

        Returns
        -------
        (B, out_channels, D, H, W) raw logits
        """
        # --- Encoder: collect skip connections ---
        skips: list[Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder: upsample + skip concatenation ---
        for upconv, decoder, skip in zip(
            self.upconvs, self.decoders, reversed(skips)
        ):
            x = upconv(x)

            # Pad upsampled tensor to match skip shape if spatial dims differ.
            # This handles cases where odd spatial dimensions cause off-by-one
            # mismatches after pooling + upsampling (e.g. D=5 → pool → 2 → up → 4).
            if x.shape[2:] != skip.shape[2:]:
                pad = []
                for s_dim, x_dim in zip(reversed(skip.shape[2:]), reversed(x.shape[2:])):
                    pad.extend([0, s_dim - x_dim])
                x = F.pad(x, pad)

            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.output_conv(x)
