from __future__ import annotations

import inspect
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

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
_MONAI_MODEL_NAMES: tuple[str, str] = ("dynunet", "swinunetr")
_SUPPORTED_MODEL_NAMES: tuple[str, ...] = (
    "unet3d",
    "attention_unet3d",
    "fct",
    "deconver",
    "deconver_multitask",
    *_MONAI_MODEL_NAMES,
)


def _register_monai_models() -> None:
    """
    Lazily register MONAI architectures when MONAI is importable.

    MONAI is optional for local unit tests in this repo; loading these models
    lazily avoids turning every import of ``src.models`` into a hard MONAI
    dependency.
    """
    if "dynunet" in _MODEL_REGISTRY and "swinunetr" in _MODEL_REGISTRY:
        return

    try:
        from monai.networks.nets import DynUNet as _DynUNet, SwinUNETR as _SwinUNETR
    except ImportError:
        return

    globals()["DynUNet"] = _DynUNet
    globals()["SwinUNETR"] = _SwinUNETR
    _MODEL_REGISTRY.setdefault("dynunet", _DynUNet)
    _MODEL_REGISTRY.setdefault("swinunetr", _SwinUNETR)


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
# Helpers
# ---------------------------------------------------------------------------

def _coerce_positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config key '{key}' must be a positive integer, got boolean.")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{key}' must be a positive integer.") from exc
    if out <= 0:
        raise ValueError(f"Config key '{key}' must be > 0, got {out}.")
    return out


def _coerce_rate(value: Any, key: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{key}' must be a float in [0.0, 1.0].") from exc
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"Config key '{key}' must be in [0.0, 1.0], got {out}.")
    return out


def _coerce_sequence(value: Any, key: str, *, min_len: int = 1) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Config key '{key}' must be a list/tuple.")
    out = tuple(value)
    if len(out) < min_len:
        raise ValueError(
            f"Config key '{key}' must have at least {min_len} item(s), got {len(out)}."
        )
    return out


def _coerce_positive_int_sequence(
    value: Any,
    key: str,
    *,
    min_len: int = 1,
) -> tuple[int, ...]:
    seq = _coerce_sequence(value, key, min_len=min_len)
    return tuple(_coerce_positive_int(v, f"{key}[{i}]") for i, v in enumerate(seq))


def _coerce_spatial_block_sequence(
    value: Any,
    key: str,
    *,
    spatial_dims: int = 3,
    min_len: int = 1,
) -> tuple[int | tuple[int, ...], ...]:
    seq = _coerce_sequence(value, key, min_len=min_len)
    parsed: list[int | tuple[int, ...]] = []
    for i, raw in enumerate(seq):
        item_key = f"{key}[{i}]"
        if isinstance(raw, bool):
            raise ValueError(
                f"Config key '{item_key}' must be a positive integer or "
                f"{spatial_dims}-tuple of positive integers."
            )
        if isinstance(raw, int):
            parsed.append(_coerce_positive_int(raw, item_key))
            continue
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            nested = tuple(
                _coerce_positive_int(v, f"{item_key}[{j}]")
                for j, v in enumerate(raw)
            )
            if len(nested) != spatial_dims:
                raise ValueError(
                    f"Config key '{item_key}' must have exactly {spatial_dims} values, "
                    f"got {len(nested)}."
                )
            parsed.append(nested)
            continue
        raise ValueError(
            f"Config key '{item_key}' must be a positive integer or "
            f"{spatial_dims}-tuple of positive integers."
        )
    return tuple(parsed)


def _resolve_swinunetr_img_size(cfg: dict) -> tuple[int, int, int]:
    raw = cfg.get("swinunetr_img_size", cfg.get("patch_size", [16, 128, 128]))
    values = _coerce_positive_int_sequence(raw, "swinunetr_img_size", min_len=3)
    if len(values) != 3:
        raise ValueError(
            "Config key 'swinunetr_img_size' must contain exactly 3 values (D, H, W)."
        )
    return cast(tuple[int, int, int], values)


def _build_dynunet(
    cfg: dict,
    *,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    _register_monai_models()
    if "dynunet" not in _MODEL_REGISTRY:
        raise ValueError(
            "model='dynunet' requested but MONAI is not importable. "
            "Install the 'monai' package in this runtime."
        )

    cls = _MODEL_REGISTRY["dynunet"]
    signature = inspect.signature(cls)
    sig_params = signature.parameters

    strides = _coerce_spatial_block_sequence(
        cfg.get("dynunet_strides", [1, 2, 2, 2]),
        "dynunet_strides",
        spatial_dims=3,
        min_len=3,
    )
    kernel_size = _coerce_spatial_block_sequence(
        cfg.get("dynunet_kernel_size", [3] * len(strides)),
        "dynunet_kernel_size",
        spatial_dims=3,
        min_len=len(strides),
    )
    if len(kernel_size) != len(strides):
        raise ValueError(
            "Config keys 'dynunet_kernel_size' and 'dynunet_strides' must have the "
            f"same length, got {len(kernel_size)} and {len(strides)}."
        )

    upsample_default: tuple[int | tuple[int, ...], ...] = tuple(strides[1:])
    upsample_kernel_size = _coerce_spatial_block_sequence(
        cfg.get("dynunet_upsample_kernel_size", upsample_default),
        "dynunet_upsample_kernel_size",
        spatial_dims=3,
        min_len=len(strides) - 1,
    )
    if len(upsample_kernel_size) != len(strides) - 1:
        raise ValueError(
            "Config key 'dynunet_upsample_kernel_size' must have length "
            f"len(dynunet_strides)-1 ({len(strides) - 1}), got {len(upsample_kernel_size)}."
        )

    filters: tuple[int, ...] | None = None
    if "dynunet_filters" in cfg and cfg.get("dynunet_filters") is not None:
        filters = _coerce_positive_int_sequence(cfg["dynunet_filters"], "dynunet_filters")
        if len(filters) != len(strides):
            raise ValueError(
                "Config key 'dynunet_filters' must match len(dynunet_strides), "
                f"got {len(filters)} vs {len(strides)}."
            )

    norm_name = cfg.get("dynunet_norm_name", ("INSTANCE", {"affine": True}))
    act_name = cfg.get(
        "dynunet_act_name",
        ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    )
    dropout = cfg.get("dynunet_dropout", None)
    if dropout is not None:
        dropout = _coerce_rate(dropout, "dynunet_dropout")

    if cfg.get("deep_supervision", False):
        logger.info(
            "Config key 'deep_supervision=true' is ignored for model='dynunet' in this repo; "
            "forcing DynUNet deep_supervision=False."
        )

    if "deep_supervision" not in sig_params:
        raise ValueError(
            "Installed MONAI DynUNet signature does not expose 'deep_supervision'; "
            "cannot enforce deep_supervision=False for this training loop."
        )

    kwargs: dict[str, Any] = {
        "spatial_dims": 3,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "strides": strides,
        "upsample_kernel_size": upsample_kernel_size,
        "deep_supervision": False,
    }

    if "norm_name" in sig_params:
        kwargs["norm_name"] = norm_name
    elif "dynunet_norm_name" in cfg:
        raise ValueError(
            "Config key 'dynunet_norm_name' is set but this MONAI DynUNet "
            "version does not accept 'norm_name'."
        )

    if "act_name" in sig_params:
        kwargs["act_name"] = act_name
    elif "dynunet_act_name" in cfg:
        raise ValueError(
            "Config key 'dynunet_act_name' is set but this MONAI DynUNet "
            "version does not accept 'act_name'."
        )

    if filters is not None:
        if "filters" in sig_params:
            kwargs["filters"] = filters
        else:
            raise ValueError(
                "Config key 'dynunet_filters' is set but this MONAI DynUNet "
                "version does not accept 'filters'."
            )

    if dropout is not None:
        if "dropout" in sig_params:
            kwargs["dropout"] = dropout
        else:
            raise ValueError(
                "Config key 'dynunet_dropout' is set but this MONAI DynUNet "
                "version does not accept 'dropout'."
            )

    return cls(**kwargs)


def _build_swinunetr(
    cfg: dict,
    *,
    in_channels: int,
    out_channels: int,
) -> nn.Module:
    _register_monai_models()
    if "swinunetr" not in _MODEL_REGISTRY:
        raise ValueError(
            "model='swinunetr' requested but MONAI is not importable. "
            "Install the 'monai' package in this runtime."
        )

    cls = _MODEL_REGISTRY["swinunetr"]
    signature = inspect.signature(cls)
    sig_params = signature.parameters

    img_size = _resolve_swinunetr_img_size(cfg)
    feature_size = _coerce_positive_int(
        cfg.get("swinunetr_feature_size", 24),
        "swinunetr_feature_size",
    )
    depths = _coerce_positive_int_sequence(
        cfg.get("swinunetr_depths", [2, 2, 2, 2]),
        "swinunetr_depths",
    )
    num_heads = _coerce_positive_int_sequence(
        cfg.get("swinunetr_num_heads", [3, 6, 12, 24]),
        "swinunetr_num_heads",
    )
    if len(depths) != len(num_heads):
        raise ValueError(
            "Config keys 'swinunetr_depths' and 'swinunetr_num_heads' must have "
            f"the same length, got {len(depths)} and {len(num_heads)}."
        )

    norm_name = cfg.get("swinunetr_norm_name", "instance")
    drop_rate = _coerce_rate(cfg.get("swinunetr_drop_rate", 0.0), "swinunetr_drop_rate")
    attn_drop_rate = _coerce_rate(
        cfg.get("swinunetr_attn_drop_rate", 0.0),
        "swinunetr_attn_drop_rate",
    )
    dropout_path_rate = _coerce_rate(
        cfg.get("swinunetr_dropout_path_rate", 0.0),
        "swinunetr_dropout_path_rate",
    )
    use_checkpoint = bool(cfg.get("swinunetr_use_checkpoint", False))
    use_v2 = bool(cfg.get("swinunetr_use_v2", False))

    kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": out_channels,
    }

    if "img_size" in sig_params:
        kwargs["img_size"] = img_size
    elif "swinunetr_img_size" in cfg:
        logger.info(
            "Ignoring config key 'swinunetr_img_size': installed MONAI SwinUNETR "
            "signature does not accept 'img_size'."
        )

    if "feature_size" in sig_params:
        kwargs["feature_size"] = feature_size
    elif "swinunetr_feature_size" in cfg:
        raise ValueError(
            "Config key 'swinunetr_feature_size' is set but this MONAI SwinUNETR "
            "version does not accept 'feature_size'."
        )

    if "depths" in sig_params:
        kwargs["depths"] = depths
    elif "swinunetr_depths" in cfg:
        raise ValueError(
            "Config key 'swinunetr_depths' is set but this MONAI SwinUNETR "
            "version does not accept 'depths'."
        )

    if "num_heads" in sig_params:
        kwargs["num_heads"] = num_heads
    elif "swinunetr_num_heads" in cfg:
        raise ValueError(
            "Config key 'swinunetr_num_heads' is set but this MONAI SwinUNETR "
            "version does not accept 'num_heads'."
        )

    if "norm_name" in sig_params:
        kwargs["norm_name"] = norm_name
    elif "swinunetr_norm_name" in cfg:
        raise ValueError(
            "Config key 'swinunetr_norm_name' is set but this MONAI SwinUNETR "
            "version does not accept 'norm_name'."
        )

    if "drop_rate" in sig_params:
        kwargs["drop_rate"] = drop_rate
    elif "swinunetr_drop_rate" in cfg:
        raise ValueError(
            "Config key 'swinunetr_drop_rate' is set but this MONAI SwinUNETR "
            "version does not accept 'drop_rate'."
        )

    if "attn_drop_rate" in sig_params:
        kwargs["attn_drop_rate"] = attn_drop_rate
    elif "swinunetr_attn_drop_rate" in cfg:
        raise ValueError(
            "Config key 'swinunetr_attn_drop_rate' is set but this MONAI SwinUNETR "
            "version does not accept 'attn_drop_rate'."
        )

    if "dropout_path_rate" in sig_params:
        kwargs["dropout_path_rate"] = dropout_path_rate
    elif "swinunetr_dropout_path_rate" in cfg:
        raise ValueError(
            "Config key 'swinunetr_dropout_path_rate' is set but this MONAI SwinUNETR "
            "version does not accept 'dropout_path_rate'."
        )

    if "use_checkpoint" in sig_params:
        kwargs["use_checkpoint"] = use_checkpoint
    elif "swinunetr_use_checkpoint" in cfg:
        raise ValueError(
            "Config key 'swinunetr_use_checkpoint' is set but this MONAI SwinUNETR "
            "version does not accept 'use_checkpoint'."
        )

    if "spatial_dims" in sig_params:
        kwargs["spatial_dims"] = 3

    if "use_v2" in sig_params:
        kwargs["use_v2"] = use_v2
    elif use_v2:
        raise ValueError(
            "Config key 'swinunetr_use_v2=true' requested, but this MONAI SwinUNETR "
            "version does not support the 'use_v2' argument."
        )

    return cls(**kwargs)


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

        ``model``            — architecture name (see ``_SUPPORTED_MODEL_NAMES``).
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

        Additional keys for ``model: dynunet``:

        ``dynunet_kernel_size``          — per-stage kernel sizes (default all 3s).
        ``dynunet_strides``              — per-stage strides (default [1,2,2,2]).
        ``dynunet_upsample_kernel_size`` — per-upsample kernels
                                            (default ``dynunet_strides[1:]``).
        ``dynunet_filters``              — optional channels per stage
                                            (len must match ``dynunet_strides``).
        ``dynunet_norm_name``            — MONAI norm spec
                                            (default ``("INSTANCE", {"affine": True})``).
        ``dynunet_act_name``             — MONAI activation spec
                                            (default ``("leakyrelu", {...})``).
        ``dynunet_dropout``              — dropout ratio in [0,1] (default disabled).
        ``deep_supervision``             — ignored for DynUNet in this repo;
                                            always forced off.

        Additional keys for ``model: swinunetr``:

        ``swinunetr_img_size``           — legacy spatial size for MONAI builds that
                                            still require ``img_size``; ignored by the
                                            installed runtime when unsupported
                                            (defaults to ``patch_size``).
        ``swinunetr_feature_size``       — base embedding width (default 24).
        ``swinunetr_depths``             — transformer depth per stage
                                            (default [2,2,2,2]).
        ``swinunetr_num_heads``          — attention heads per stage
                                            (default [3,6,12,24]).
        ``swinunetr_norm_name``          — output norm block spec
                                            (default ``"instance"``).
        ``swinunetr_drop_rate``          — embedding dropout in [0,1] (default 0).
        ``swinunetr_attn_drop_rate``     — attention dropout in [0,1] (default 0).
        ``swinunetr_dropout_path_rate``  — stochastic-depth rate in [0,1] (default 0).
        ``swinunetr_use_checkpoint``     — enable gradient checkpointing (default false).
        ``swinunetr_use_v2``             — enable SwinUNETR v2 path when supported by
                                            installed MONAI.

    Returns
    -------
    nn.Module
        Not yet moved to a device.

    Raises
    ------
    ValueError
        If ``cfg["model"]`` is unknown, no modalities are active, or a selected
        optional architecture is unavailable in the current runtime.
    """
    name = cfg.get("model", "unet3d").lower()

    if name in {"deconver", "deconver_multitask"} and not _DECONVER_AVAILABLE:
        raise ValueError(
            f"model='{name}' requested but the Deconver package could not be "
            "imported from src/models/deconver/. "
            "Check that the submodule is present and its dependencies are installed."
        )

    _register_monai_models()
    if name in _MONAI_MODEL_NAMES and name not in _MODEL_REGISTRY:
        raise ValueError(
            f"model='{name}' requested but MONAI is not importable. "
            "Install the 'monai' package in this runtime."
        )

    if name not in _SUPPORTED_MODEL_NAMES:
        available = list(_SUPPORTED_MODEL_NAMES)
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

    if name == "dynunet":
        return _build_dynunet(
            cfg,
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", 1),
        )

    if name == "swinunetr":
        return _build_swinunetr(
            cfg,
            in_channels=in_channels,
            out_channels=cfg.get("out_channels", 1),
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


_register_monai_models()

__all__ = [
    "UNet3D",
    "AttentionUNet3D",
    "FCT",
    "build_model",
]

if _DECONVER_AVAILABLE:
    __all__.append("Deconver")
    __all__.append("DeconverMultiTask")
if "dynunet" in _MODEL_REGISTRY:
    __all__.append("DynUNet")
if "swinunetr" in _MODEL_REGISTRY:
    __all__.append("SwinUNETR")
