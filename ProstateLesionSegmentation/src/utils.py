from pathlib import Path
from datetime import datetime
import json
import subprocess
import shutil
import torch
import yaml


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(base_dir: str, experiment_name: str) -> Path:
    run_name = f"{timestamp()}_{experiment_name}"
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


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
