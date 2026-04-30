from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

from src.models.attention_unet3d import AttentionUNet3D
from src.models.unet3d import UNet3D

# ---------------------------------------------------------------------------
# Deconver vendored package — add its parent to sys.path so the inner
# ``deconver`` package (src/models/deconver/deconver/) is importable.
# ---------------------------------------------------------------------------

_DECONVER_ROOT = Path(__file__).parent / "deconver"
if str(_DECONVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECONVER_ROOT))

try:
    from deconver.deconver import Deconver as _Deconver
    _DECONVER_AVAILABLE = True
except ImportError:
    _Deconver = None  # type: ignore[assignment,misc]
    _DECONVER_AVAILABLE = False

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps config ``model`` strings to their corresponding classes.
#: To add a new architecture, create a module in ``src/models/`` and register
#: it here.  No changes to training code are required.
_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "unet3d": UNet3D,
    "attention_unet3d": AttentionUNet3D,
}

if _DECONVER_AVAILABLE:
    _MODEL_REGISTRY["deconver"] = _Deconver  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate and return a model from the YAML config dict.

    The ``model`` key in ``cfg`` selects the architecture.  It defaults to
    ``"unet3d"`` when the key is absent so that existing configs without the
    key continue to work.

    The number of input channels is derived from the modality flags
    ``use_t2w``, ``use_adc``, and ``use_hbv`` (each defaults to ``True``).
    One channel is allocated per enabled modality in canonical order:
    ``[T2w, ADC, HBV]``.

    Parameters
    ----------
    cfg : dict
        Training configuration dict.  Relevant keys (common):

        ``model``            — architecture name (see ``_MODEL_REGISTRY``).
        ``input_channels``   — explicit input-channel override. When set, this
                               takes precedence over modality flags.
        ``use_t2w``          — include T2w channel (default ``True``).
        ``use_adc``          — include ADC channel (default ``True``).
        ``use_hbv``          — include HBV channel (default ``True``).
        ``out_channels``     — number of segmentation output channels (default 1).
        ``features``         — encoder feature sizes as a list (default [32,64,128,256]).
                               Used only for ``unet3d`` and ``attention_unet3d``.
        ``deep_supervision`` — enable auxiliary decoder heads (default ``False``).
                               For ``unet3d``/``attention_unet3d``: model returns
                               ``list[Tensor]`` (finest → coarsest); wrap the
                               criterion with ``DeepSupervisionWrapper``.
                               For ``deconver``: passed as ``num_deep_supr`` to
                               the built-in deep supervision mechanism.

        Additional keys for ``model: deconver``:

        ``spatial_dims``            — ``2`` or ``3`` (default ``3``).
        ``deconver_encoder_depth``  — blocks per encoder stage (default [1,1,1,1]).
        ``deconver_encoder_width``  — channels per encoder stage (default [64,128,256,512]).
        ``deconver_strides``        — stride per stage (default [1,2,2,2]).
        ``deconver_kernel_size``    — NDC kernel size (default [3,3,3] for 3D,
                                       [3,3] for 2D).
        ``deconver_groups``         — NDC groups; -1 = one group per channel (default -1).
        ``deconver_ndc_ratio``      — NDC channel expansion ratio (default 4).
        ``deconver_fp32_islands``   — run numerically sensitive NDC update math in FP32
                                       while keeping outer AMP policy (default False).
        ``deconver_fp32_scope``     — FP32 island scope: ``update_only`` or
                                       ``iterative_block`` (default ``update_only``).

    Returns
    -------
    nn.Module
        Not yet moved to a device.

    Raises
    ------
    ValueError
        If ``cfg["model"]`` is not present in ``_MODEL_REGISTRY``, or if
        all three modality flags are ``False`` (no input channels), or if
        ``model: deconver`` is requested but the package failed to import.
    """
    name = cfg.get("model", "unet3d").lower()
    spatial_dims = int(cfg.get("spatial_dims", 3))
    if spatial_dims not in (2, 3):
        raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

    if name == "deconver" and not _DECONVER_AVAILABLE:
        raise ValueError(
            "model='deconver' requested but the Deconver package could not be "
            "imported from src/models/deconver/. "
            "Check that the submodule is present and its dependencies are installed."
        )

    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available architectures: {available}"
        )

    if "input_channels" in cfg:
        in_channels = int(cfg["input_channels"])
    else:
        in_channels = sum([
            cfg.get("use_t2w", True),
            cfg.get("use_adc", True),
            cfg.get("use_hbv", True),
        ])
    if in_channels == 0:
        raise ValueError(
            "No input channels configured. Set input_channels > 0 or enable at least "
            "one of use_t2w/use_adc/use_hbv."
        )

    # -----------------------------------------------------------------------
    # Deconver has a different constructor signature; handle it separately.
    # -----------------------------------------------------------------------
    if name == "deconver":
        deep_supervision = cfg.get("deep_supervision", False)
        # num_deep_supr: False disables it; a positive int enables N auxiliary heads.
        num_deep_supr: bool | int = False
        if deep_supervision:
            encoder_depth = cfg.get("deconver_encoder_depth", [1, 1, 1, 1])
            # Number of auxiliary heads = number of decoder stages = len(encoder_depth) - 1
            num_deep_supr = max(1, len(encoder_depth) - 1)

        if spatial_dims == 2:
            norm_layer: type[nn.Module] = nn.InstanceNorm2d
            default_kernel_size = [3, 3]
        else:
            norm_layer = nn.InstanceNorm3d
            default_kernel_size = [3, 3, 3]

        return _Deconver(  # type: ignore[misc]
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", 1),
            spatial_dims=spatial_dims,
            encoder_depth=tuple(cfg.get("deconver_encoder_depth", [1, 1, 1, 1])),
            encoder_width=tuple(cfg.get("deconver_encoder_width", [64, 128, 256, 512])),
            strides=tuple(cfg.get("deconver_strides", [1, 2, 2, 2])),
            norm=norm_layer,
            kernel_size=tuple(cfg.get("deconver_kernel_size", default_kernel_size)),
            groups=cfg.get("deconver_groups", -1),
            ratio=cfg.get("deconver_ndc_ratio", 4),
            fp32_islands=cfg.get("deconver_fp32_islands", False),
            fp32_scope=cfg.get("deconver_fp32_scope", "update_only"),
            num_deep_supr=num_deep_supr,
        )

    # -----------------------------------------------------------------------
    # UNet3D / AttentionUNet3D share the same constructor signature.
    # -----------------------------------------------------------------------
    if spatial_dims != 3:
        raise ValueError(
            f"model='{name}' currently supports only spatial_dims=3, got {spatial_dims}. "
            "Use model='deconver' for 2D training."
        )

    cls = _MODEL_REGISTRY[name]
    return cls(
        in_channels=in_channels,
        out_channels=cfg.get("out_channels", 1),
        features=tuple(cfg.get("features", [32, 64, 128, 256])),
        deep_supervision=cfg.get("deep_supervision", False),
    )


__all__ = [
    "UNet3D",
    "AttentionUNet3D",
    "build_model",
]

if _DECONVER_AVAILABLE:
    __all__.append("Deconver")
