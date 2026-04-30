from __future__ import annotations

import logging
import math
from pathlib import Path
from datetime import datetime
import json
import re
import subprocess
import shutil
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CUDA runtime compatibility
# ---------------------------------------------------------------------------

def _parse_supported_sm_arches(arch_list: list[str]) -> list[tuple[int, int]]:
    """
    Parse `torch.cuda.get_arch_list()` entries into sorted `(major, minor)` SM tuples.
    """
    parsed: set[tuple[int, int]] = set()
    for arch in arch_list:
        m = re.match(r"^sm_(\d+)(?:a)?$", arch)
        if m is None:
            continue
        sm = int(m.group(1))
        parsed.add((sm // 10, sm % 10))
    return sorted(parsed)


def ensure_cuda_binary_compatibility(device: torch.device) -> None:
    """
    Raise a clear error when the current CUDA device is not supported by the
    installed PyTorch CUDA binaries.

    This catches unsupported-SM failures early (before training starts), instead
    of surfacing later as generic CUDA kernel launch/runtime errors.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(device_index)
    supported_sms = _parse_supported_sm_arches(torch.cuda.get_arch_list())

    if not supported_sms:
        return
    if (cc_major, cc_minor) in supported_sms:
        return

    # NVIDIA cubins are forward-compatible within the same major SM version:
    # code built for sm_Xy can run on sm_Xz where z >= y.
    same_major_supported = sorted(
        minor for major, minor in supported_sms if major == cc_major
    )
    if same_major_supported and any(minor <= cc_minor for minor in same_major_supported):
        closest_floor_minor = max(
            minor for minor in same_major_supported if minor <= cc_minor
        )
        gpu_name = torch.cuda.get_device_name(device_index)
        detected_sm = f"sm_{cc_major}{cc_minor}"
        floor_sm = f"sm_{cc_major}{closest_floor_minor}"
        supported_sm_str = ", ".join(f"sm_{m}{n}" for m, n in supported_sms)
        logger.warning(
            "GPU %s (%s) is not explicitly listed in torch CUDA arch list [%s], "
            "but binary compatibility is expected via %s. Proceeding.",
            gpu_name,
            detected_sm,
            supported_sm_str,
            floor_sm,
        )
        return

    gpu_name = torch.cuda.get_device_name(device_index)
    detected_sm = f"sm_{cc_major}{cc_minor}"
    supported_sm_str = ", ".join(f"sm_{m}{n}" for m, n in supported_sms)

    raise RuntimeError(
        "Installed PyTorch CUDA binaries are incompatible with the detected GPU: "
        f"'{gpu_name}' ({detected_sm}). Supported SM architectures in this build: "
        f"[{supported_sm_str}]. For TITAN V / Volta, use the cu126 stack: "
        "`docker compose -f compose.yml -f compose.volta.yml build trainer` and run "
        "commands with the same `-f` flags. To force CPU mode, set "
        "`CUDA_VISIBLE_DEVICES=\"\"`."
    )


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite_score(
    val_metrics: dict[str, float],
    w_sensitivity: float = 0.5,
    w_dice: float = 0.3,
    w_hd95: float = 0.2,
    hd95_scale: float = 10.0,
) -> float:
    """
    Compute a weighted composite validation score for best-checkpoint selection.

    score = w_sens * sensitivity + w_dice * dice + w_hd95 * (1 / (1 + hd95 / hd95_scale))

    Weights are normalised internally so they always sum to 1.0, making the
    score independent of whether the caller uses raw or pre-normalised values.

    The HD95 term is inverted (lower HD95 → higher score) via
    ``1 / (1 + hd95 / hd95_scale)``, which maps HD95 ∈ [0, ∞) to (0, 1].
    ``hd95_scale`` (default 10 mm) sets the midpoint: an HD95 equal to
    ``hd95_scale`` produces a term of 0.5.  Choose a value near the expected
    boundary between "acceptable" and "poor" HD95 for your dataset.

    NaN handling
    ------------
    - If ``sensitivity`` or ``dice`` is NaN (i.e. no positive cases were
      present in the validation set), the function returns ``float('nan')``.
      The caller should treat this as "no update" for best-checkpoint selection.
    - If only ``hd95`` is NaN (HD95 was not computed this epoch, or prediction
      / target was empty for every case), the HD95 term is dropped and its
      weight is redistributed proportionally between sensitivity and dice.

    Parameters
    ----------
    val_metrics   : dict with keys "sensitivity", "dice", "hd95"
    w_sensitivity : weight for sensitivity (TPR); default 0.5
    w_dice        : weight for Dice DSC; default 0.3
    w_hd95        : weight for inverted HD95; default 0.2
    hd95_scale    : HD95 value (mm) that maps to a term of 0.5; default 10.0

    Returns
    -------
    float in [0, 1], or float('nan') when primary metrics are undefined.
    """
    sens = val_metrics["sensitivity"]
    dice = val_metrics["dice"]
    hd95 = val_metrics["hd95"]

    # Primary metrics undefined → score is undefined
    if math.isnan(sens) or math.isnan(dice):
        return float("nan")

    if not math.isnan(hd95):
        hd95_term: float | None = 1.0 / (1.0 + hd95 / hd95_scale)
    else:
        hd95_term = None

    # Build weighted sum with normalised weights
    effective_w_hd95 = w_hd95 if hd95_term is not None else 0.0
    total_w = w_sensitivity + w_dice + effective_w_hd95

    score = (w_sensitivity * sens + w_dice * dice) / total_w
    if hd95_term is not None:
        score += (w_hd95 * hd95_term) / total_w

    return score


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base_dir: str, experiment_name: str) -> Path:
    run_name = f"{timestamp()}_{experiment_name}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    best_val_dice: float = 0.0,
    best_composite_score: float = 0.0,
    last_hd95: float = float("nan"),
) -> None:
    """
    Persist model, optimizer, and (optionally) scheduler/scaler state to *path*.

    Parameters
    ----------
    model                 : the ``torch.nn.Module`` being trained
    optimizer             : the optimizer whose state should be saved
    epoch                 : current epoch number (1-indexed), stored in the checkpoint
    path                  : destination file path (will be created or overwritten)
    scheduler             : learning-rate scheduler; its state is saved when provided
    scaler                : ``torch.amp.GradScaler``; its state is saved when provided
                            so FP16 loss-scaling factors survive across resume.
                            Pass ``None`` when using BF16 or no AMP.
    best_val_dice         : best validation Dice seen so far, stored for resuming
    best_composite_score  : best composite score seen so far, stored for resuming
    last_hd95             : most-recent finite HD95 value (mm), or NaN if HD95 has
                            not yet been computed; stored so composite score formula
                            remains consistent after a resume.
    """
    state: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
        "best_composite_score": best_composite_score,
        "last_hd95": last_hd95,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a checkpoint file and restore state into *model* (and optionally
    *optimizer* / *scheduler* / *scaler*).

    Parameters
    ----------
    path      : path to the ``.pt`` checkpoint file
    model     : model to restore weights into
    optimizer : if provided, optimizer state is restored from the checkpoint
    scheduler : if provided, scheduler state is restored from the checkpoint
    scaler    : if provided and a ``scaler_state_dict`` key exists in the
                checkpoint, the GradScaler loss-scale state is restored.
                Pass ``None`` when using BF16 or no AMP.
    device    : map_location for ``torch.load``; defaults to CPU

    Returns
    -------
    dict
        The raw checkpoint dictionary.  Callers can read ``ckpt["epoch"]``
        and ``ckpt.get("best_val_dice", 0.0)`` to resume training state.
    """
    map_location: torch.device | str = device if device is not None else "cpu"
    ckpt: dict = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", "?"))
    return ckpt


def rotate_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    """
    Delete old per-epoch checkpoints, keeping only the *keep_last_n* most recent.

    Only files whose names match the pattern ``epoch_NNNN.pt`` are considered.
    Named checkpoints such as ``best.pt`` are never touched.

    Parameters
    ----------
    checkpoint_dir : directory that contains the ``epoch_NNNN.pt`` files
    keep_last_n    : number of most-recent epoch checkpoints to retain;
                     if ≤ 0 all epoch checkpoints are kept (no-op)
    """
    if keep_last_n <= 0:
        return

    _EPOCH_RE = re.compile(r"^epoch_(\d{4})\.pt$")

    epoch_files: list[tuple[int, Path]] = []
    for f in checkpoint_dir.iterdir():
        m = _EPOCH_RE.match(f.name)
        if m:
            epoch_files.append((int(m.group(1)), f))

    # Sort ascending so the last keep_last_n entries are the most recent.
    epoch_files.sort(key=lambda x: x[0])

    to_delete = epoch_files[:-keep_last_n] if len(epoch_files) > keep_last_n else []
    for _, path in to_delete:
        path.unlink()
        logger.debug("Removed old checkpoint: %s", path.name)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_metadata(run_dir: Path, cfg: dict) -> None:
    metadata = {
        "experiment_name": cfg.get("experiment_name", "unknown"),
        "git_commit": get_git_commit(),
        "config": cfg,
    }

    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_config_copy(run_dir: Path, cfg: dict) -> None:
    config_copy_path = run_dir / "config.yaml"
    with open(config_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_latest_pointer(base_output_dir: str, run_dir: Path) -> None:
    latest_path = Path(base_output_dir) / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir, target_is_directory=True)


# ---------------------------------------------------------------------------
# Encoder transfer utilities (SSL pretraining -> supervised fine-tuning)
# ---------------------------------------------------------------------------

def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying module for torch.compile-wrapped models.
    """
    return getattr(model, "_orig_mod", model)


def _encoder_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    """
    Return state-dict key prefixes considered part of the encoder.
    """
    prefixes: list[str] = []

    # UNet3D / AttentionUNet3D
    if hasattr(model, "encoders") and hasattr(model, "bottleneck"):
        prefixes.extend(["encoders.", "bottleneck."])

    # Deconver
    if hasattr(model, "stem") and hasattr(model, "encoder"):
        prefixes.extend(["stem.", "encoder."])

    if not prefixes:
        raise ValueError(
            f"Unsupported model type for encoder transfer: {type(model).__name__}"
        )

    return tuple(prefixes)


def get_encoder_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Extract encoder-only tensors from ``model.state_dict()``.
    """
    base_model = _unwrap_model(model)
    prefixes = _encoder_prefixes(base_model)
    return {
        k: v
        for k, v in base_model.state_dict().items()
        if any(k.startswith(p) for p in prefixes)
    }


def load_pretrained_encoder(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device | None = None,
) -> dict[str, object]:
    """
    Load encoder weights from checkpoint into *model*.

    Accepted checkpoint layouts:
      1) ``{"encoder_state_dict": ...}``
      2) ``{"model_state_dict": ...}``
      3) raw state-dict mapping
    """
    base_model = _unwrap_model(model)
    map_location: torch.device | str = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        source_sd = ckpt["encoder_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        source_sd = ckpt["model_state_dict"]
    else:
        source_sd = ckpt

    if not isinstance(source_sd, dict):
        raise ValueError(
            f"Checkpoint at {path} does not contain a valid state_dict mapping."
        )

    prefixes = _encoder_prefixes(base_model)
    model_sd = base_model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []

    for k, v in source_sd.items():
        if k not in model_sd:
            continue
        if not any(k.startswith(p) for p in prefixes):
            continue
        if model_sd[k].shape != v.shape:
            shape_mismatch.append(k)
            continue
        matched[k] = v

    if not matched:
        raise ValueError(f"No encoder tensors from {path} matched current model.")

    updated_sd = model_sd.copy()
    updated_sd.update(matched)
    base_model.load_state_dict(updated_sd, strict=False)

    missing = [
        k
        for k in model_sd.keys()
        if any(k.startswith(p) for p in prefixes) and k not in matched
    ]

    return {
        "loaded": len(matched),
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> int:
    """
    Enable/disable gradients for encoder params.

    Returns the number of encoder parameters affected.
    """
    base_model = _unwrap_model(model)
    prefixes = _encoder_prefixes(base_model)

    n_params = 0
    for name, param in base_model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = trainable
            n_params += param.numel()

    # Keep normalization/dropout behavior consistent when frozen.
    if hasattr(base_model, "encoders"):
        base_model.encoders.train(trainable)
    if hasattr(base_model, "bottleneck"):
        base_model.bottleneck.train(trainable)
    if hasattr(base_model, "stem"):
        base_model.stem.train(trainable)
    if hasattr(base_model, "encoder"):
        base_model.encoder.train(trainable)

    return n_params
