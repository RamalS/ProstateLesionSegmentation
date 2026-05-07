"""
Slice-wise Fully Convolutional Transformer (FCT) for volumetric segmentation.

This module adapts the 2-D FCT idea (Tragakis et al., WACV 2023) to the
existing 3-D training pipeline in this repository:

- Input to ``forward`` remains volumetric: ``(B, C, D, H, W)``
- Each axial slice is processed by a 2-D FCT encoder-decoder
- Slices are folded back to a 3-D logit volume: ``(B, out_channels, D, H, W)``

The design preserves the existing model contract used by ``src/train.py``:

- ``deep_supervision=False`` -> return a plain ``Tensor``
- ``deep_supervision=True``  -> return ``list[Tensor]`` ordered finest ->
  coarsest, compatible with ``DeepSupervisionWrapper``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _depthwise_conv2d(
    channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """Depthwise 2-D convolution helper."""
    padding = dilation * (kernel_size // 2)
    return nn.Conv2d(
        channels,
        channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=False,
    )


def _safe_num_heads(channels: int, requested_heads: int) -> int:
    """
    Return the largest valid head count <= requested_heads that divides channels.
    """
    requested = max(1, min(requested_heads, channels))
    for h in range(requested, 0, -1):
        if channels % h == 0:
            return h
    return 1


class _LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for ``(B, C, H, W)`` tensors.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)   # BCHW -> BHWC
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)  # BHWC -> BCHW


class _DepthwiseSelfAttention2D(nn.Module):
    """
    MHSA with depthwise-convolutional Q/K/V projections.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = _safe_num_heads(channels, heads)
        self.head_dim = channels // self.heads
        self.dropout = dropout

        self.q_proj = _depthwise_conv2d(channels, kernel_size=3, stride=1)
        self.k_proj = _depthwise_conv2d(channels, kernel_size=3, stride=1)
        self.v_proj = _depthwise_conv2d(channels, kernel_size=3, stride=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def _to_heads(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, heads, N, head_dim)
        b, c, h, w = x.shape
        x = x.view(b, self.heads, self.head_dim, h * w)
        return x.transpose(2, 3)

    def _from_heads(self, x: Tensor, h: int, w: int) -> Tensor:
        # (B, heads, N, head_dim) -> (B, C, H, W)
        b = x.shape[0]
        x = x.transpose(2, 3).contiguous()
        return x.view(b, self.heads * self.head_dim, h, w)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q = self._to_heads(self.q_proj(x))
        k = self._to_heads(self.k_proj(x))
        v = self._to_heads(self.v_proj(x))

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = self._from_heads(attn, h=h, w=w)
        return self.out_proj(attn)


class _WideFocus2D(nn.Module):
    """
    Multi-branch dilated-convolution module used after attention.
    """

    def __init__(
        self,
        channels: int,
        dilations: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=d,
                dilation=d,
                bias=False,
            )
            for d in dilations
        ])
        self.agg = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        for branch in self.branches:
            out = out + F.gelu(branch(x))
        out = out / float(len(self.branches))
        out = self.agg(out)
        return self.drop(out)


class _FCTBlock2D(nn.Module):
    """
    Fully convolutional transformer block:
    conv stem -> convolutional attention -> wide-focus.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        patch_kernel_size: int = 7,
        patch_stride: int = 4,
        wide_focus_dilations: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if patch_kernel_size < 1 or patch_stride < 1:
            raise ValueError("patch_kernel_size and patch_stride must be >= 1.")

        self.pre_norm = _LayerNorm2d(channels)
        self.pre_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.pre_conv2 = _depthwise_conv2d(channels, kernel_size=3, stride=1)
        self.pre_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=patch_kernel_size,
                stride=patch_stride,
                padding=patch_kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            _LayerNorm2d(channels),
            nn.GELU(),
        )
        self.attn = _DepthwiseSelfAttention2D(channels, heads=heads, dropout=dropout)
        self.post_norm = _LayerNorm2d(channels)
        self.wide_focus = _WideFocus2D(
            channels=channels,
            dilations=wide_focus_dilations,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.pre_norm(x)
        x = F.gelu(self.pre_conv1(x))
        x = F.gelu(self.pre_conv2(x))
        x = self.pre_pool(x)

        in_hw = x.shape[2:]
        tokens = self.patch_embed(x)
        tokens = tokens + self.attn(tokens)
        if tokens.shape[2:] != in_hw:
            tokens = F.interpolate(tokens, size=in_hw, mode="bilinear", align_corners=False)

        x = x + tokens
        x = x + self.wide_focus(self.post_norm(x))
        return x + residual


class _FCTStage2D(nn.Module):
    """
    Stage wrapper that adjusts channels then applies one FCT block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        patch_kernel_size: int,
        patch_stride: int,
        wide_focus_dilations: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.block = _FCTBlock2D(
            channels=out_channels,
            heads=heads,
            patch_kernel_size=patch_kernel_size,
            patch_stride=patch_stride,
            wide_focus_dilations=wide_focus_dilations,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(self.proj(x))


class FCT(nn.Module):
    """
    Slice-wise FCT adapted to 3-D segmentation inputs.

    Parameters
    ----------
    in_channels      : number of input channels (active modalities)
    out_channels     : output channels (1 for binary segmentation)
    features         : channels per encoder stage
    deep_supervision : return multi-scale logits list when True
    heads            : optional per-stage head counts; defaults to ``f//16``
    bottleneck_channels : optional bottleneck width; default ``2 * features[-1]``
    patch_kernel_size   : depthwise patch-kernel size in attention path
    patch_strides       : per-stage patch strides; defaults to 4 for all stages
    wide_focus_dilations: dilation rates for wide-focus branches
    dropout             : dropout used in attention/wide-focus
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        deep_supervision: bool = False,
        heads: tuple[int, ...] | None = None,
        bottleneck_channels: int | None = None,
        patch_kernel_size: int = 7,
        patch_strides: tuple[int, ...] | None = None,
        wide_focus_dilations: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if len(features) < 2:
            raise ValueError("FCT requires at least 2 feature levels.")

        self.deep_supervision = deep_supervision
        self.features = features

        if heads is None:
            heads = tuple(max(1, f // 16) for f in features)
        if len(heads) != len(features):
            raise ValueError(
                "heads must have the same length as features: "
                f"got heads={len(heads)} features={len(features)}."
            )

        if patch_strides is None:
            patch_strides = tuple(4 for _ in features)
        if len(patch_strides) != len(features):
            raise ValueError(
                "patch_strides must have the same length as features: "
                f"got patch_strides={len(patch_strides)} features={len(features)}."
            )

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for f, h, s in zip(features, heads, patch_strides):
            self.encoders.append(
                _FCTStage2D(
                    in_channels=ch,
                    out_channels=f,
                    heads=h,
                    patch_kernel_size=patch_kernel_size,
                    patch_stride=s,
                    wide_focus_dilations=wide_focus_dilations,
                    dropout=dropout,
                )
            )
            self.pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
            ch = f

        bottleneck_ch = bottleneck_channels if bottleneck_channels is not None else ch * 2
        bottleneck_heads = max(1, bottleneck_ch // 16)
        self.bottleneck = _FCTStage2D(
            in_channels=ch,
            out_channels=bottleneck_ch,
            heads=bottleneck_heads,
            patch_kernel_size=patch_kernel_size,
            patch_stride=max(1, patch_strides[-1]),
            wide_focus_dilations=wide_focus_dilations,
            dropout=dropout,
        )
        ch = bottleneck_ch

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f, h, s in zip(reversed(features), reversed(heads), reversed(patch_strides)):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decoders.append(
                _FCTStage2D(
                    in_channels=f * 2,
                    out_channels=f,
                    heads=h,
                    patch_kernel_size=patch_kernel_size,
                    patch_stride=s,
                    wide_focus_dilations=wide_focus_dilations,
                    dropout=dropout,
                )
            )
            ch = f

        self.output_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

        if deep_supervision:
            self.deep_heads = nn.ModuleList([
                nn.Conv2d(f, out_channels, kernel_size=1)
                for f in list(reversed(features))[:-1]
            ])

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _vol_to_slices(x: Tensor) -> tuple[Tensor, int, int]:
        """
        Convert ``(B, C, D, H, W)`` volume to ``(B*D, C, H, W)`` slices.
        """
        b, c, d, h, w = x.shape
        x2d = x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        return x2d, b, d

    @staticmethod
    def _slices_to_vol(y: Tensor, b: int, d: int) -> Tensor:
        """
        Convert ``(B*D, C, H, W)`` slices back to ``(B, C, D, H, W)``.
        """
        bd, c, h, w = y.shape
        if bd != b * d:
            raise ValueError(f"Slice batch mismatch: got {bd}, expected {b*d}.")
        return y.view(b, d, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, x: Tensor) -> list[Tensor] | Tensor:
        x2d, b, d = self._vol_to_slices(x)

        skips: list[Tensor] = []
        for encoder, pool in zip(self.encoders, self.pools):
            x2d = encoder(x2d)
            skips.append(x2d)
            x2d = pool(x2d)

        x2d = self.bottleneck(x2d)

        decoder_outputs: list[Tensor] = []
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x2d = upconv(x2d)
            if x2d.shape[2:] != skip.shape[2:]:
                pad = []
                for s_dim, x_dim in zip(reversed(skip.shape[2:]), reversed(x2d.shape[2:])):
                    pad.extend([0, s_dim - x_dim])
                x2d = F.pad(x2d, pad)
            x2d = torch.cat([skip, x2d], dim=1)
            x2d = decoder(x2d)
            decoder_outputs.append(x2d)

        main_logits_2d = self.output_conv(decoder_outputs[-1])
        main_logits = self._slices_to_vol(main_logits_2d, b=b, d=d)

        if self.deep_supervision:
            aux_logits = [
                self._slices_to_vol(head(feat), b=b, d=d)
                for head, feat in zip(self.deep_heads, decoder_outputs[:-1])
            ]
            return [main_logits] + list(reversed(aux_logits))

        return main_logits
