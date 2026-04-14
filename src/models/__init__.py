from __future__ import annotations

import torch.nn as nn

from src.models.attention_unet3d import AttentionUNet3D
from src.models.unet3d import UNet3D

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
        Training configuration dict.  Relevant keys:

        ``model``       — architecture name (see ``_MODEL_REGISTRY``).
        ``use_t2w``     — include T2w channel (default ``True``).
        ``use_adc``     — include ADC channel (default ``True``).
        ``use_hbv``     — include HBV channel (default ``True``).
        ``out_channels`` — number of segmentation output channels (default 1).
        ``features``    — encoder feature sizes as a list (default [32,64,128,256]).

    Returns
    -------
    nn.Module
        Uninitialised (weights set by the model's ``_init_weights``),
        not yet moved to a device.

    Raises
    ------
    ValueError
        If ``cfg["model"]`` is not present in ``_MODEL_REGISTRY``, or if
        all three modality flags are ``False`` (no input channels).
    """
    name = cfg.get("model", "unet3d").lower()
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY)
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available architectures: {available}"
        )

    in_channels = sum([
        cfg.get("use_t2w", True),
        cfg.get("use_adc", True),
        cfg.get("use_hbv", True),
    ])
    if in_channels == 0:
        raise ValueError(
            "No modalities enabled. At least one of use_t2w, use_adc, "
            "use_hbv must be true in the config."
        )

    cls = _MODEL_REGISTRY[name]
    return cls(
        in_channels=in_channels,
        out_channels=cfg.get("out_channels", 1),
        features=tuple(cfg.get("features", [32, 64, 128, 256])),
    )


__all__ = [
    "UNet3D",
    "AttentionUNet3D",
    "build_model",
]
