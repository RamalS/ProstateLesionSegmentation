"""
Attention 3D U-Net for volumetric medical image segmentation.

Architecture
------------
Identical to UNet3D, with Attention Gates inserted on every skip connection
before concatenation in the decoder.

Each Attention Gate receives:
  g : upsampled decoder feature (gating signal), shape (B, f, D, H, W)
  x : encoder skip connection,                   shape (B, f, D, H, W)

and produces a spatially reweighted skip:
  alpha = sigmoid( psi( relu( W_g(g) + W_x(x) ) ) )   shape (B, 1, D, H, W)
  output = x * alpha

Deep Supervision
----------------
When ``deep_supervision=True`` the model attaches an auxiliary 1×1×1 output
head to every decoder level except the finest.  ``forward`` then returns a
``list[Tensor]`` ordered finest → coarsest:

    [logits_full, logits_D/2, logits_D/4, logits_D/8]

Use ``DeepSupervisionWrapper`` in ``src/losses.py`` to compute the weighted
multi-scale loss during training.  At inference, extract ``output[0]``.

When ``deep_supervision=False`` (default) ``forward`` returns a plain
``Tensor`` and the model is a drop-in replacement for the original.

References
----------
Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas."
MIDL 2018.  https://arxiv.org/abs/1804.03999

Çiçek et al. "3D U-Net: Learning Dense Volumetric Segmentation from
Sparse Annotation." MICCAI 2016.

Isensee et al. "nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation." Nature Methods 2021.
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


class AttentionGate3D(nn.Module):
    """
    3D Soft Attention Gate (Oktay et al. 2018).

    Computes a spatial attention map from a gating signal ``g`` and a skip
    connection ``x``, then returns the element-wise reweighted skip.

    Both ``g`` and ``x`` must have the same spatial dimensions
    (D, H, W).  In this U-Net the upsampled decoder feature and the encoder
    skip are already aligned, so no strided downsampling is needed.

    Parameters
    ----------
    f_g   : number of channels in the gating signal (decoder after upconv)
    f_l   : number of channels in the skip connection (encoder output)
    f_int : number of intermediate channels; typically ``f_l // 2``
    """

    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()

        # 1×1×1 projections — no bias because BN handles the offset
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, bias=True),
            nn.BatchNorm3d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, bias=True),
            nn.BatchNorm3d(f_int),
        )

        # Scalar attention coefficient per spatial location
        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for attention convolutions; zero BN bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """
        Compute attention-gated skip connection.

        Parameters
        ----------
        g : (B, f_g, D, H, W) — gating signal (upsampled decoder feature)
        x : (B, f_l, D, H, W) — skip connection from the encoder

        Returns
        -------
        (B, f_l, D, H, W) — spatially reweighted skip connection
        """
        g1 = self.W_g(g)                    # (B, f_int, D, H, W)
        x1 = self.W_x(x)                    # (B, f_int, D, H, W)
        alpha = self.psi(self.relu(g1 + x1))  # (B, 1, D, H, W)
        return x * alpha                    # broadcast over channel dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AttentionUNet3D(nn.Module):
    """
    3D U-Net with Attention Gates on all skip connections.

    An Attention Gate is applied to each encoder skip connection before it is
    concatenated with the upsampled decoder feature.  Everything else —
    encoder, bottleneck, decoder blocks, output head, and weight
    initialisation — is identical to ``UNet3D``.

    Parameters
    ----------
    in_channels      : number of input channels (3 for T2w + ADC + HBV)
    out_channels     : number of output channels (1 for binary segmentation)
    features         : feature map sizes at each encoder level.
                       Length determines the number of pooling levels (depth).
    deep_supervision : if True, attach auxiliary 1×1×1 output heads to all
                       decoder levels except the finest and return a
                       ``list[Tensor]`` ordered finest → coarsest from
                       ``forward``.  Set to False (default) for standard
                       single-output behaviour.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()

        self.deep_supervision = deep_supervision

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
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for f in reversed(features):
            # Transposed convolution: halves channels and doubles spatial dims
            self.upconvs.append(nn.ConvTranspose3d(ch, f, kernel_size=2, stride=2))
            # Attention gate: g and x both have f channels; f_int = f // 2
            self.attentions.append(AttentionGate3D(f_g=f, f_l=f, f_int=max(f // 2, 1)))
            # After attention-gated concatenation: 2*f channels → f channels
            self.decoders.append(_conv_block(f * 2, f))
            ch = f

        # ---- Output head ----
        # 1×1×1 conv to desired number of classes; no activation (raw logits)
        self.output_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

        # ---- Deep supervision auxiliary heads ----
        # One 1×1×1 head per decoder level except the finest.
        # reversed(features)[:-1] gives channel counts for the coarser levels,
        # e.g. [256, 128, 64] for default features=(32, 64, 128, 256).
        # Kaiming init in _init_weights covers these Conv3d layers automatically.
        if deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv3d(f, out_channels, kernel_size=1)
                for f in list(reversed(features))[:-1]
            ])

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

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, D, H, W)

        Returns
        -------
        deep_supervision=False (default)
            (B, out_channels, D, H, W) raw logits — plain Tensor.

        deep_supervision=True
            list[Tensor] ordered finest → coarsest:
            ``[logits_full, logits_D/2, logits_D/4, ...]``
            Pass to ``DeepSupervisionWrapper`` during training;
            at inference use ``output[0]``.
        """
        # --- Encoder: collect skip connections ---
        skips: list[Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder: upsample + attention-gated skip concatenation ---
        # Accumulate every decoder output; index 0 = coarsest, -1 = finest.
        decoder_outputs: list[Tensor] = []
        for upconv, attention, decoder, skip in zip(
            self.upconvs, self.attentions, self.decoders, reversed(skips)
        ):
            g = upconv(x)  # gating signal: upsampled to skip's resolution

            # Pad upsampled tensor to match skip shape if spatial dims differ.
            # Handles odd spatial dimensions (e.g. D=5 → pool → 2 → up → 4).
            if g.shape[2:] != skip.shape[2:]:
                pad = []
                for s_dim, g_dim in zip(reversed(skip.shape[2:]), reversed(g.shape[2:])):
                    pad.extend([0, s_dim - g_dim])
                g = F.pad(g, pad)

            attended_skip = attention(g=g, x=skip)
            x = torch.cat([attended_skip, g], dim=1)
            x = decoder(x)
            decoder_outputs.append(x)

        # decoder_outputs[-1] is the finest resolution (full patch size).
        main_logits = self.output_conv(decoder_outputs[-1])

        if self.deep_supervision:
            # Apply auxiliary heads to all but the finest decoder output.
            # decoder_outputs[:-1] is [coarsest, ..., second-finest]; reversing
            # gives [second-finest, ..., coarsest] so the final list is ordered
            # finest → coarsest: [main(D), aux(D/2), aux(D/4), aux(D/8)].
            aux_logits = [
                head(feat)
                for head, feat in zip(self.deep_heads, decoder_outputs[:-1])
            ]
            return [main_logits] + list(reversed(aux_logits))

        return main_logits
