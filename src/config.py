from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import yaml


_CACHE_MODES: set[str] = {"none", "ram", "storage"}
SUPPORTED_MODALITIES: tuple[str, str, str] = ("t2w", "adc", "hbv")
_LEGACY_MODALITY_KEYS: tuple[str, str, str] = ("use_t2w", "use_adc", "use_hbv")


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_dataset_cache_dir() -> Path:
    """
    Return the default persistent dataset-cache directory.

    Preference order:
      1) ``/cache/dataset_cache`` when ``/cache`` exists and is writable
         (Docker Compose default mount in this repo).
      2) ``./cache/dataset_cache`` for local runs.
    """
    docker_cache_root = Path("/cache")
    if docker_cache_root.exists() and os.access(docker_cache_root, os.W_OK):
        return docker_cache_root / "dataset_cache"
    return Path("cache") / "dataset_cache"


def _coerce_bool(value: Any) -> bool:
    """
    Convert common YAML/string representations into a boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _warn(msg: str, logger: logging.Logger | None) -> None:
    if logger is not None:
        logger.warning(msg)
        return
    warnings.warn(msg, stacklevel=3)


def resolve_active_modalities(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[str, ...]:
    """
    Resolve active input modalities in channel order.

    Preferred config:
      - ``modalities``: ordered list subset of ``["t2w", "adc", "hbv"]``

    Backward-compatible legacy config:
      - ``use_t2w``, ``use_adc``, ``use_hbv`` boolean flags
      - when ``modalities`` is absent, legacy flags are read in canonical order
        and default to ``True`` when omitted

    Precedence:
      - when both ``modalities`` and legacy flags are present, ``modalities``
        wins and a warning is emitted.
    """
    modalities_raw = cfg.get("modalities", None)
    has_legacy = any(key in cfg for key in _LEGACY_MODALITY_KEYS)

    if modalities_raw is not None:
        if not isinstance(modalities_raw, (list, tuple)):
            raise ValueError(
                "Config key 'modalities' must be a list like ['t2w', 'adc', 'hbv']."
            )

        parsed: list[str] = []
        seen: set[str] = set()
        for raw in modalities_raw:
            if not isinstance(raw, str):
                raise ValueError(
                    "Config key 'modalities' must contain strings only "
                    "(allowed: t2w, adc, hbv)."
                )
            token = raw.strip().lower()
            if not token:
                raise ValueError(
                    "Config key 'modalities' contains an empty entry."
                )
            if token not in SUPPORTED_MODALITIES:
                supported = ", ".join(SUPPORTED_MODALITIES)
                raise ValueError(
                    f"Unknown modality '{raw}'. "
                    f"Supported modalities: {supported}."
                )
            if token in seen:
                raise ValueError(
                    f"Duplicate modality '{token}' in config key 'modalities'."
                )
            seen.add(token)
            parsed.append(token)

        if not parsed:
            raise ValueError(
                "Config key 'modalities' must not be empty. "
                "Choose at least one modality from: t2w, adc, hbv."
            )

        if has_legacy:
            _warn(
                "Both 'modalities' and deprecated legacy flags "
                "('use_t2w'/'use_adc'/'use_hbv') are set; "
                "ignoring legacy flags and using 'modalities'.",
                logger,
            )
        return tuple(parsed)

    active = tuple(
        key
        for key in SUPPORTED_MODALITIES
        if _coerce_bool(cfg.get(f"use_{key}", True))
    )
    if not active:
        raise ValueError(
            "No modalities enabled. Set config key 'modalities' to a non-empty "
            "ordered subset of ['t2w', 'adc', 'hbv']. "
            "Legacy flags 'use_t2w/use_adc/use_hbv' are still accepted but deprecated."
        )
    if has_legacy:
        _warn(
            "Config keys 'use_t2w'/'use_adc'/'use_hbv' are deprecated. "
            "Use 'modalities: [t2w, adc, hbv]' instead.",
            logger,
        )
    return active


def resolve_active_modality_pairs(
    cfg: Mapping[str, Any],
    suffix_by_modality: Mapping[str, str],
    logger: logging.Logger | None = None,
) -> list[tuple[str, str]]:
    """
    Resolve ordered ``(modality, suffix)`` pairs from ``cfg``.

    ``suffix_by_modality`` must include mappings for all supported modalities.
    """
    missing = [k for k in SUPPORTED_MODALITIES if k not in suffix_by_modality]
    if missing:
        raise ValueError(
            f"suffix_by_modality is missing required key(s): {missing}."
        )
    active = resolve_active_modalities(cfg, logger=logger)
    return [(key, suffix_by_modality[key]) for key in active]


def resolve_dataset_cache_config(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[str, float, Path | None]:
    """
    Resolve cache settings from config with legacy-key compatibility.

    Supported keys:
      - ``cache_mode``: ``none`` | ``ram`` | ``storage``
      - ``cache_rate``: float in [0, 1]
      - ``dataset_cache_dir``: optional directory for ``storage`` mode

    Legacy compatibility:
      - ``cache_dataset: true`` maps to ``cache_mode: ram``
      - ``cache_dataset: false`` maps to ``cache_mode: none``
      - when legacy mapping is used, a deprecation warning is emitted
    """
    cache_mode_raw = cfg.get("cache_mode")
    legacy_cache_dataset = cfg.get("cache_dataset", None)

    if cache_mode_raw is None:
        if legacy_cache_dataset is not None:
            cache_mode = "ram" if _coerce_bool(legacy_cache_dataset) else "none"
            _warn(
                "Config key 'cache_dataset' is deprecated. Use "
                "'cache_mode: ram|storage|none' instead.",
                logger,
            )
        else:
            cache_mode = "none"
    else:
        cache_mode = str(cache_mode_raw).strip().lower()
        if cache_mode not in _CACHE_MODES:
            supported = ", ".join(sorted(_CACHE_MODES))
            raise ValueError(
                f"Invalid cache_mode='{cache_mode_raw}'. "
                f"Expected one of: {supported}."
            )
        if legacy_cache_dataset is not None:
            _warn(
                "Both 'cache_mode' and deprecated 'cache_dataset' are set; "
                "ignoring 'cache_dataset'.",
                logger,
            )

    cache_rate = float(cfg.get("cache_rate", 1.0))
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(
            f"cache_rate must be in [0.0, 1.0], got {cache_rate}."
        )

    cache_dir_raw = cfg.get("dataset_cache_dir")
    cache_dir: Path | None = None
    if cache_mode == "storage":
        cache_dir = (
            Path(str(cache_dir_raw))
            if cache_dir_raw not in (None, "")
            else default_dataset_cache_dir()
        )
    elif cache_dir_raw not in (None, ""):
        _warn(
            "dataset_cache_dir is set but cache_mode is not 'storage'; "
            "the directory setting will be ignored.",
            logger,
        )

    return cache_mode, cache_rate, cache_dir


def resolve_roi_cache_config(
    cfg: Mapping[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[str, float, Path | None]:
    """
    Resolve ROI crop-cache settings from ``cfg["roi"]``.

    Supported keys under ``roi``:
      - ``cache_mode``: ``none`` | ``ram`` | ``storage``
      - ``cache_rate``: float in [0, 1]
      - ``cache_dir``: optional directory for ``storage`` mode
    """
    roi_cfg = cfg.get("roi", {}) or {}
    if not isinstance(roi_cfg, Mapping):
        raise ValueError("Config key 'roi' must be a mapping when provided.")

    cache_mode_raw = roi_cfg.get("cache_mode", "none")
    cache_mode = str(cache_mode_raw).strip().lower()
    if cache_mode not in _CACHE_MODES:
        supported = ", ".join(sorted(_CACHE_MODES))
        raise ValueError(
            f"Invalid roi.cache_mode='{cache_mode_raw}'. Expected one of: {supported}."
        )

    cache_rate = float(roi_cfg.get("cache_rate", 1.0))
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"roi.cache_rate must be in [0.0, 1.0], got {cache_rate}.")

    cache_dir_raw = roi_cfg.get("cache_dir")
    cache_dir: Path | None = None
    if cache_mode == "storage":
        cache_dir = (
            Path(str(cache_dir_raw))
            if cache_dir_raw not in (None, "")
            else default_dataset_cache_dir().parent / "roi_cache"
        )
    elif cache_dir_raw not in (None, ""):
        _warn(
            "roi.cache_dir is set but roi.cache_mode is not 'storage'; "
            "the directory setting will be ignored.",
            logger,
        )

    return cache_mode, cache_rate, cache_dir
