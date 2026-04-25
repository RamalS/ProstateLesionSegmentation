"""
Self-supervised masked reconstruction pretraining on unlabeled prostate MRI.

Input dataset:
  data/unlabeled_images/<case>_{t2,adc,dwi}.nii.gz

DWI is mapped to the HBV channel for 3-channel pretraining.
Checkpoint stores encoder_state_dict for transfer into src.train.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.dataset import PiCaiDataset, active_modality_pairs, discover_unlabeled_cases
from src.models import build_model
from src.utils import (
    create_run_dir,
    ensure_dir,
    get_encoder_state_dict,
    rotate_checkpoints,
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _pad_to_min_size(vol: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    """
    Ensure volume spatial size >= patch_size via zero-padding.

    Parameters
    ----------
    vol : Tensor
        Shape ``(1, C, D, H, W)``.
    patch_size : tuple[int, int, int]
        Spatial crop size ``(D, H, W)``.
    """
    _, _, d, h, w = vol.shape
    pd, ph, pw = patch_size
    pad_d = max(0, pd - d)
    pad_h = max(0, ph - h)
    pad_w = max(0, pw - w)
    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return vol
    return F.pad(vol, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)


def _sample_random_patches(
    vol: torch.Tensor,
    patch_size: tuple[int, int, int],
    n_patches: int,
) -> torch.Tensor:
    """
    Sample ``n_patches`` random patches from one volume.

    Parameters
    ----------
    vol : Tensor
        Shape ``(1, C, D, H, W)``.
    patch_size : tuple[int, int, int]
        Spatial crop size ``(D, H, W)``.
    n_patches : int
        Number of random crops to sample.

    Returns
    -------
    Tensor
        Shape ``(n_patches, C, D, H, W)``.
    """
    vol = _pad_to_min_size(vol, patch_size)
    _, _, d, h, w = vol.shape
    pd, ph, pw = patch_size

    max_z = max(0, d - pd)
    max_y = max(0, h - ph)
    max_x = max(0, w - pw)

    patches: list[torch.Tensor] = []
    for _ in range(n_patches):
        z = random.randint(0, max_z) if max_z > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        x = random.randint(0, max_x) if max_x > 0 else 0
        patches.append(vol[:, :, z:z + pd, y:y + ph, x:x + pw])

    return torch.cat(patches, dim=0)


def _augment_patches(x: torch.Tensor) -> torch.Tensor:
    """
    Lightweight geometric + intensity augmentation for SSL pretraining.
    """
    out = x
    if random.random() < 0.5:
        out = torch.flip(out, dims=[2])
    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])
    if random.random() < 0.5:
        out = torch.flip(out, dims=[4])

    b, c = out.shape[:2]
    scale = torch.empty((b, c, 1, 1, 1), device=out.device).uniform_(0.9, 1.1)
    shift = torch.empty((b, c, 1, 1, 1), device=out.device).uniform_(-0.1, 0.1)
    noise = torch.randn_like(out) * 0.05
    return out * scale + shift + noise


def _build_block_mask(
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    block_size: tuple[int, int, int],
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a random block mask for masked reconstruction.

    Returns a tensor of shape ``(B, 1, D, H, W)`` where 1 indicates masked voxels.
    """
    d, h, w = spatial_shape
    bd, bh, bw = block_size

    gd = math.ceil(d / bd)
    gh = math.ceil(h / bh)
    gw = math.ceil(w / bw)

    block_mask = (torch.rand((batch_size, gd, gh, gw), device=device) < mask_ratio).float()
    mask = block_mask.repeat_interleave(bd, dim=1)
    mask = mask.repeat_interleave(bh, dim=2)
    mask = mask.repeat_interleave(bw, dim=3)
    mask = mask[:, :d, :h, :w]
    return mask.unsqueeze(1)


def _save_ssl_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_loss: float,
) -> None:
    base_model = getattr(model, "_orig_mod", model)
    state = {
        "epoch": epoch,
        "model_state_dict": base_model.state_dict(),
        "encoder_state_dict": get_encoder_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-supervised encoder pretraining on unlabeled MRI"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Reproducibility
    seed = int(cfg.get("random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = bool(cfg.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )

    # Output directories
    base_output_dir = str(cfg.get("base_output_dir", "/outputs/pretrain_runs"))
    ensure_dir(base_output_dir)
    run_dir = create_run_dir(base_output_dir, str(cfg.get("experiment_name", "ssl_pretrain")))
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(run_dir, cfg)
    save_config_copy(run_dir, cfg)
    save_latest_pointer(base_output_dir, run_dir)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Data
    unlabeled_images_dir = Path(cfg["unlabeled_images_dir"])
    if not unlabeled_images_dir.exists():
        raise FileNotFoundError(f"unlabeled_images_dir not found: {unlabeled_images_dir}")

    active_keys = [k for k, _ in active_modality_pairs(cfg)]
    unlabeled_cases = discover_unlabeled_cases(
        images_dir=unlabeled_images_dir,
        active_keys=active_keys,
    )
    if not unlabeled_cases:
        raise RuntimeError(
            f"No unlabeled cases found in {unlabeled_images_dir}. "
            "Expected <case>_{t2,adc,dwi}.nii.gz files."
        )

    target_spacing: tuple[float, ...] = tuple(cfg.get("target_spacing", [3.0, 0.5, 0.5]))
    train_ds = PiCaiDataset(
        images_dir=unlabeled_images_dir,
        labels_dir=unlabeled_images_dir,  # labels unused for unlabeled cases
        target_spacing=target_spacing,
        transform=None,
        cases=unlabeled_cases,
        use_cache=bool(cfg.get("cache_dataset", False)),
        cache_rate=float(cfg.get("cache_rate", 1.0)),
        active_modalities=active_keys,
        dwi_hbv_preprocess=cfg.get("dwi_hbv_preprocess", {}),
    )

    num_workers = int(cfg.get("num_workers", 4))
    use_persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
    )

    # Reconstruction head predicts the full active modality stack.
    ssl_cfg = dict(cfg)
    ssl_cfg["out_channels"] = len(active_keys)
    ssl_cfg["deep_supervision"] = False
    model = build_model(ssl_cfg).to(device)

    if cfg.get("use_compile", False):
        logger.info("Compiling model with torch.compile ...")
        model = torch.compile(model)  # type: ignore[assignment]

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %s | trainable parameters: %s",
        ssl_cfg.get("model", "unknown"),
        f"{n_params:,}",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(cfg.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
        eta_min=float(cfg.get("learning_rate", 2e-4)) * 1e-2,
    )

    amp_dtype_str = str(cfg.get("amp_dtype", "bf16")).lower()
    _dtype_map: dict[str, torch.dtype] = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype: torch.dtype = _dtype_map.get(amp_dtype_str, torch.bfloat16)
    use_amp: bool = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    use_fp16: bool = use_amp and amp_dtype == torch.float16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # type: ignore[attr-defined]

    patch_size: tuple[int, int, int] = tuple(cfg.get("patch_size", [16, 128, 128]))
    mask_ratio: float = float(cfg.get("mask_ratio", 0.6))
    mask_block_size: tuple[int, int, int] = tuple(cfg.get("mask_block_size", [4, 16, 16]))
    num_patches_per_volume: int = max(1, int(cfg.get("num_patches_per_volume", 4)))
    full_l1_weight: float = float(cfg.get("full_l1_weight", 0.1))
    keep_last_n: int = int(cfg.get("keep_last_checkpoints", 3))

    logger.info("Device: %s | AMP (%s): %s", device, amp_dtype_str.upper(), use_amp)
    logger.info(
        "SSL setup: patch_size=%s | patches/volume=%d | mask_ratio=%.2f | mask_block=%s",
        patch_size,
        num_patches_per_volume,
        mask_ratio,
        mask_block_size,
    )
    logger.info("Run directory: %s", run_dir)

    best_loss = float("inf")

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        epoch_masked_l1 = 0.0
        epoch_full_l1 = 0.0

        batch_bar = tqdm(train_loader, desc=f"Pretrain {epoch}/{epochs}", leave=False, unit="vol")
        for batch in batch_bar:
            vol = batch["image"]  # (1, C, D, H, W), CPU

            patches = _sample_random_patches(
                vol=vol,
                patch_size=patch_size,
                n_patches=num_patches_per_volume,
            ).to(device, non_blocking=True)

            patches = _augment_patches(patches)

            mask = _build_block_mask(
                batch_size=patches.size(0),
                spatial_shape=(patches.shape[2], patches.shape[3], patches.shape[4]),
                block_size=mask_block_size,
                mask_ratio=mask_ratio,
                device=patches.device,
            )

            masked_input = patches * (1.0 - mask)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(masked_input)
                pred = pred[0] if isinstance(pred, list) else pred

                pred_f = pred.float()
                target_f = patches.float()
                mask_expanded = mask.expand_as(target_f)

                masked_l1 = (torch.abs(pred_f - target_f) * mask_expanded).sum() / mask_expanded.sum().clamp_min(1.0)
                full_l1 = F.l1_loss(pred_f, target_f)
                loss = masked_l1 + (full_l1_weight * full_l1)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite SSL loss at epoch {epoch}: {float(loss.item())}"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            epoch_masked_l1 += float(masked_l1.item())
            epoch_full_l1 += float(full_l1.item())

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                masked=f"{masked_l1.item():.4f}",
            )

            del pred, pred_f, target_f, masked_l1, full_l1, loss, patches, mask, masked_input

        scheduler.step()

        n_steps = max(1, len(train_loader))
        avg_loss = epoch_loss / n_steps
        avg_masked_l1 = epoch_masked_l1 / n_steps
        avg_full_l1 = epoch_full_l1 / n_steps
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/masked_l1", avg_masked_l1, epoch)
        writer.add_scalar("train/full_l1", avg_full_l1, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        logger.info(
            "Epoch %d/%d | loss=%.4f | masked_l1=%.4f | full_l1=%.4f | lr=%.2e",
            epoch,
            epochs,
            avg_loss,
            avg_masked_l1,
            avg_full_l1,
            current_lr,
        )

        _save_ssl_checkpoint(
            path=checkpoint_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_loss=best_loss,
        )
        rotate_checkpoints(checkpoint_dir, keep_last_n)

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_ssl_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_loss=best_loss,
            )
            logger.info("New best SSL checkpoint at epoch %d (loss=%.4f)", epoch, best_loss)

    writer.close()
    logger.info("SSL pretraining complete.")
    logger.info("Best loss: %.4f", best_loss)
    logger.info("Artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()
