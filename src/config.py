from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import yaml


_CACHE_MODES: set[str] = {"none", "ram", "storage"}


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
