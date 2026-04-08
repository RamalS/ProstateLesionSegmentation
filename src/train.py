"""
Training script for prostate lesion segmentation.

Usage (inside Docker):
    python -m src.train --config /workspace/configs/default.yaml

Pipeline
--------
1. Load config and set up output directories / TensorBoard.
2. Discover PI-CAI cases and split into train / validation sets.
3. Build PiCaiDataset with MONAI augmentation transforms.
4. Instantiate 3D U-Net, DiceBCELoss, AdamW + CosineAnnealingLR.
5. Train: random-patch forward pass → Dice+BCE loss → backward.
6. Validate: sliding-window inference over full volumes → Dice, IoU,
   Sensitivity, Specificity, HD95.
7. Save regular checkpoints + best-model checkpoint by validation Dice.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.dataset import PiCaiDataset, discover_cases, train_val_split
from src.losses import DiceBCELoss
from src.metrics import compute_all_metrics
from src.models import UNet3D
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    create_run_dir,
    ensure_dir,
    save_checkpoint,
    save_config_copy,
    save_latest_pointer,
    save_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, ...],
    sw_overlap: float,
    sw_batch_size: int,
) -> dict[str, float]:
    """
    Run sliding-window inference over full validation volumes and return
    averaged segmentation metrics.
    """
    model.eval()

    sums: dict[str, float] = {
        "dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "specificity": 0.0,
    }
    hd95_values: list[float] = []
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False, unit="vol"):
            images = batch["image"].to(device)   # (1, 3, D, H, W)
            labels = batch["label"].to(device)   # (1, 1, D, H, W)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
            )

            m = compute_all_metrics(logits, labels)

            for k in sums:
                sums[k] += m[k]
            if not math.isnan(m["hd95"]):
                hd95_values.append(m["hd95"])
            n += 1

    if n == 0:
        return {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0,
                "specificity": 0.0, "hd95": float("nan")}

    result = {k: v / n for k, v in sums.items()}
    result["hd95"] = float(sum(hd95_values) / len(hd95_values)) if hd95_values else float("nan")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train prostate lesion segmentation model"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Reproducibility ----
    seed = cfg.get("random_seed", 42)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )

    # ---- Output directories ----
    ensure_dir(cfg["base_output_dir"])
    run_dir = create_run_dir(cfg["base_output_dir"], cfg["experiment_name"])
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    save_metadata(run_dir, cfg)
    save_config_copy(run_dir, cfg)
    save_latest_pointer(cfg["base_output_dir"], run_dir)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # ---- Data ----
    patch_size: tuple[int, ...] = tuple(cfg.get("patch_size", [20, 128, 128]))
    target_spacing: tuple[float, ...] = tuple(cfg["target_spacing"])

    all_cases = discover_cases(
        images_dir=Path(cfg["images_dir"]),
        labels_dir=Path(cfg["labels_dir"]),
    )

    if not all_cases:
        raise RuntimeError(
            f"No cases found in {cfg['images_dir']}. "
            "Check that your data is mounted correctly (./data -> /data)."
        )

    train_cases, val_cases = train_val_split(
        all_cases,
        val_fraction=cfg.get("val_fraction", 0.2),
        seed=seed,
    )
    logger.info("Split: %d train | %d val", len(train_cases), len(val_cases))

    train_ds = PiCaiDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        target_spacing=target_spacing,
        transform=get_train_transforms(
            patch_size=patch_size,
            pos_fraction=cfg.get("pos_fraction", 0.75),
        ),
        cases=train_cases,
    )

    val_ds = PiCaiDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        target_spacing=target_spacing,
        transform=get_val_transforms(),
        cases=val_cases,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Validation uses batch_size=1: full volumes, sliding window handles memory
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ---- Model ----
    model = UNet3D(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 1),
        features=tuple(cfg.get("features", [32, 64, 128, 256])),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("UNet3D | trainable parameters: %s", f"{n_params:,}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    # Cosine annealing decays lr from initial down to lr * 1e-2 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=cfg["learning_rate"] * 1e-2,
    )

    # ---- Loss ----
    criterion = DiceBCELoss(
        dice_weight=cfg.get("dice_weight", 1.0),
        bce_weight=cfg.get("bce_weight", 1.0),
    )

    # ---- Training loop ----
    best_val_dice = 0.0
    sw_overlap = cfg.get("sw_overlap", 0.5)
    sw_batch_size = cfg.get("sw_batch_size", 4)
    epochs = cfg["epochs"]

    logger.info("Device: %s", device)
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("Run directory: %s", run_dir)
    logger.info("Loss: %s", criterion)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):

        # ---- Train ----
        model.train()
        epoch_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, unit="batch")
        for batch in batch_bar:
            images = batch["image"].to(device)   # (B, 3, D, H, W)
            labels = batch["label"].to(device)   # (B, 1, D, H, W)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        logger.info(
            "Epoch %d/%d | loss=%.4f | lr=%.2e",
            epoch, epochs, avg_loss, current_lr,
        )

        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, epoch,
            str(checkpoint_dir / f"epoch_{epoch:04d}.pt"),
        )

        # ---- Validate ----
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            patch_size=patch_size,
            sw_overlap=sw_overlap,
            sw_batch_size=sw_batch_size,
        )

        writer.add_scalar("val/dice",        val_metrics["dice"],        epoch)
        writer.add_scalar("val/iou",         val_metrics["iou"],         epoch)
        writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
        writer.add_scalar("val/specificity", val_metrics["specificity"], epoch)
        if not math.isnan(val_metrics["hd95"]):
            writer.add_scalar("val/hd95", val_metrics["hd95"], epoch)

        hd95_str = (
            f"{val_metrics['hd95']:.2f}"
            if not math.isnan(val_metrics["hd95"])
            else "n/a"
        )
        logger.info(
            "Epoch %d/%d | val_dice=%.4f | val_iou=%.4f"
            " | val_sens=%.4f | val_spec=%.4f | val_hd95=%s",
            epoch, epochs,
            val_metrics["dice"],
            val_metrics["iou"],
            val_metrics["sensitivity"],
            val_metrics["specificity"],
            hd95_str,
        )

        # Save best model checkpoint
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                model, optimizer, epoch,
                str(checkpoint_dir / "best.pt"),
            )
            logger.info(
                "New best model at epoch %d (val_dice=%.4f) → %s",
                epoch, best_val_dice, checkpoint_dir / "best.pt",
            )

    writer.close()
    logger.info("Training complete.")
    logger.info("Best validation Dice: %.4f", best_val_dice)
    logger.info("Artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()
