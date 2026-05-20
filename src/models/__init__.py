from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

from src.config import resolve_active_modalities
from src.models.attention_unet3d import AttentionUNet3D
from src.models.fct import FCT
from src.models.unet3d import UNet3D


logger = logging.getLogger(__name__)

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
    "fct": FCT,
}

if _DECONVER_AVAILABLE:
    Deconver = _Deconver  # type: ignore[misc,assignment]
    _MODEL_REGISTRY["deconver"] = _Deconver  # type: ignore[assignment]


if _DECONVER_AVAILABLE:
    class DeconverMultiTask(_Deconver):  # type: ignore[misc,valid-type]
        """
        Deconver segmentation model with an auxiliary lesion-presence head.

        The segmentation path is unchanged.  The classifier reads the deepest
        encoder feature and predicts whether the current patch/volume contains
        lesion voxels, so training can add a weak detection objective without
        making inference depend on a hard classification gate.
        """

        def __init__(
            self,
            *args,
            classification_dropout: float = 0.1,
            **kwargs,
        ) -> None:
            encoder_width = tuple(kwargs.get("encoder_width", [64, 128, 256, 512]))
            spatial_dims = int(kwargs.get("spatial_dims", 3))
            super().__init__(*args, **kwargs)

            pool_cls = getattr(nn, f"AdaptiveAvgPool{spatial_dims}d")
            self.classifier = nn.Sequential(
                pool_cls(1),
                nn.Flatten(1),
                nn.Dropout(float(classification_dropout)),
                nn.Linear(int(encoder_width[-1]), 1),
            )

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
            features = self.forward_features(x)

            if self.num_deep_supr:
                seg: torch.Tensor | list[torch.Tensor] = [
                    head(features[j]) for j, head in enumerate(self.heads)
                ]
            else:
                seg = self.head(features[0])

            cls = self.classifier(features[-1]).squeeze(1)
            return {"seg": seg, "cls": cls}

    _MODEL_REGISTRY["deconver_multitask"] = DeconverMultiTask  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> nn.Module:
    """
    Instantiate and return a model from the YAML config dict.

    The ``model`` key in ``cfg`` selects the architecture.  It defaults to
    ``"unet3d"`` when the key is absent so that existing configs without the
    key continue to work.

    The number of input channels is derived from the resolved active
    modalities (preferred config key: ``modalities``). One channel is
    allocated per active modality.

    Parameters
    ----------
    cfg : dict
        Training configuration dict.  Relevant keys (common):

        ``model``            — architecture name (see ``_MODEL_REGISTRY``).
        ``modalities``       — ordered active modalities, e.g.
                               ``["t2w", "adc", "hbv"]``.
                               Legacy ``use_t2w/use_adc/use_hbv`` keys are
                               still accepted for backward compatibility.
        ``out_channels``     — number of segmentation output channels (default 1).
        ``features``         — encoder feature sizes as a list (default [32,64,128,256]).
                               Used by ``unet3d``, ``attention_unet3d``, and ``fct``.
        ``deep_supervision`` — enable auxiliary decoder heads (default ``False``).
                               For ``unet3d``/``attention_unet3d``: model returns
                               ``list[Tensor]`` (finest → coarsest); wrap the
                               criterion with ``DeepSupervisionWrapper``.
                               For ``fct``: same list contract (finest → coarsest).
                               For ``deconver``: passed as ``num_deep_supr`` to
                               the built-in deep supervision mechanism.

        Additional keys for ``model: fct``:

        ``fct_heads``             — optional attention heads per stage
                                    (len must match ``features``).
        ``fct_bottleneck_channels`` — optional bottleneck channels.
        ``fct_patch_kernel_size`` — depthwise patch-kernel size (default 7).
        ``fct_patch_strides``     — per-stage patch stride for attention tokens
                                    (len must match ``features``; default all 4).
        ``fct_wide_focus_dilations`` — dilations for wide-focus branches
                                       (default [1, 2, 3]).
        ``fct_dropout``           — dropout in attention/wide-focus (default 0.0).

        Additional keys for ``model: deconver``:

        ``deconver_encoder_depth``  — blocks per encoder stage (default [1,1,1,1]).
        ``deconver_encoder_width``  — channels per encoder stage (default [64,128,256,512]).
        ``deconver_strides``        — stride per stage (default [1,2,2,2]).
        ``deconver_kernel_size``    — NDC kernel size (default [3,3,3]).
        ``deconver_groups``         — NDC groups; -1 = one group per channel (default -1).
        ``deconver_ndc_ratio``      — NDC channel expansion ratio (default 4).

    Returns
    -------
    nn.Module
        Not yet moved to a device.

    Raises
    ------
    ValueError
        If ``cfg["model"]`` is not present in ``_MODEL_REGISTRY``, or if
        no modalities are active (no input channels), or if
        ``model: deconver`` is requested but the package failed to import.
    """
    name = cfg.get("model", "unet3d").lower()

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

    in_channels = len(resolve_active_modalities(cfg, logger=logger))
    if in_channels == 0:
        raise ValueError(
            "No modalities enabled. Set config key 'modalities' to a non-empty "
            "ordered subset of ['t2w', 'adc', 'hbv']."
        )

    # -----------------------------------------------------------------------
    # Deconver has a different constructor signature; handle it separately.
    # -----------------------------------------------------------------------
    if name in {"deconver", "deconver_multitask"}:
        deep_supervision = cfg.get("deep_supervision", False)
        # num_deep_supr: False disables it; a positive int enables N auxiliary heads.
        num_deep_supr: bool | int = False
        if deep_supervision:
            encoder_depth = cfg.get("deconver_encoder_depth", [1, 1, 1, 1])
            # Number of auxiliary heads = number of decoder stages = len(encoder_depth) - 1
            num_deep_supr = max(1, len(encoder_depth) - 1)

        cls = _MODEL_REGISTRY[name]
        kwargs = {}
        if name == "deconver_multitask":
            kwargs["classification_dropout"] = cfg.get("classification_dropout", 0.1)

        return cls(  # type: ignore[misc]
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", 1),
            spatial_dims=3,
            encoder_depth=tuple(cfg.get("deconver_encoder_depth", [1, 1, 1, 1])),
            encoder_width=tuple(cfg.get("deconver_encoder_width", [64, 128, 256, 512])),
            strides=tuple(cfg.get("deconver_strides", [1, 2, 2, 2])),
            norm=nn.InstanceNorm3d,
            kernel_size=tuple(cfg.get("deconver_kernel_size", [3, 3, 3])),
            groups=cfg.get("deconver_groups", -1),
            ratio=cfg.get("deconver_ndc_ratio", 4),
            num_deep_supr=num_deep_supr,
            **kwargs,
        )

    if name == "fct":
        return FCT(
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", 1),
            features=tuple(cfg.get("features", [32, 64, 128, 256])),
            deep_supervision=cfg.get("deep_supervision", False),
            heads=tuple(cfg["fct_heads"]) if "fct_heads" in cfg else None,
            bottleneck_channels=cfg.get("fct_bottleneck_channels"),
            patch_kernel_size=int(cfg.get("fct_patch_kernel_size", 7)),
            patch_strides=tuple(cfg["fct_patch_strides"]) if "fct_patch_strides" in cfg else None,
            wide_focus_dilations=tuple(cfg.get("fct_wide_focus_dilations", [1, 2, 3])),
            dropout=float(cfg.get("fct_dropout", 0.0)),
        )

    # -----------------------------------------------------------------------
    # UNet3D / AttentionUNet3D share the same constructor signature.
    # -----------------------------------------------------------------------
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
    "FCT",
    "build_model",
]

if _DECONVER_AVAILABLE:
    __all__.append("Deconver")
    __all__.append("DeconverMultiTask")
