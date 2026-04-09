from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
import json
import re
import subprocess
import shutil
import torch
import yaml

logger = logging.getLogger(__name__)


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
    best_val_dice: float = 0.0,
) -> None:
    """
    Persist model, optimizer, and (optionally) scheduler state to *path*.

    Parameters
    ----------
    model         : the ``torch.nn.Module`` being trained
    optimizer     : the optimizer whose state should be saved
    epoch         : current epoch number (1-indexed), stored in the checkpoint
    path          : destination file path (will be created or overwritten)
    scheduler     : learning-rate scheduler; its state is saved when provided
    best_val_dice : best validation Dice seen so far, stored for resuming
    """
    state: dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_dice": best_val_dice,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict:
    """
    Load a checkpoint file and restore state into *model* (and optionally
    *optimizer* / *scheduler*).

    Parameters
    ----------
    path      : path to the ``.pt`` checkpoint file
    model     : model to restore weights into
    optimizer : if provided, optimizer state is restored from the checkpoint
    scheduler : if provided, scheduler state is restored from the checkpoint
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
