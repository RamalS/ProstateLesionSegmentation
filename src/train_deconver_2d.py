"""
2D Deconver training script for JPG images and PNG labels.

Usage:
    python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import shutil
import warnings
from pathlib import Path

import monai.data.utils
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.dataset import default_split_manifest_path, resolve_train_val_split_from_manifest
from src.dataset_2d import ImageMask2DDataset, discover_cases_2d
from src.losses import DiceBCELoss, DeepSupervisionWrapper, TverskyBCELoss
from src.metrics import compute_all_metrics
from src.models import build_model
from src.notify import send_ntfy
from src.postprocess import postprocess_logits
from src.transforms_2d import get_train_transforms_2d, get_val_transforms_2d
from src.utils import (
    compute_composite_score,
    create_run_dir,
    ensure_cuda_binary_compatibility,
    ensure_dir,
    load_checkpoint,
    rotate_checkpoints,
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

warnings.filterwarnings(
    "ignore",
    message=".*unable to generate class balanced samples.*",
    category=UserWarning,
    module="monai",
)
logger = logging.getLogger(__name__)


def _fmt(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _log_cuda_memory(stage: str, device: torch.device) -> None:
    if device.type != "cuda":
        return

    alloc_gib = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserv_gib = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak_alloc_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserv_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    logger.info(
        "%s | CUDA alloc=%.2f GiB reserved=%.2f GiB peak_alloc=%.2f GiB peak_reserved=%.2f GiB",
        stage,
        alloc_gib,
        reserv_gib,
        peak_alloc_gib,
        peak_reserv_gib,
    )


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, ...],
    sw_overlap: float,
    sw_batch_size: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    compute_hd95: bool = True,
    threshold: float = 0.5,
    postprocess_enabled: bool = False,
    postprocess_min_component_volume_mm3: float = 30.0,
    postprocess_connectivity: int = 8,
    spacing_yx: tuple[float, float] = (0.5, 0.5),
) -> dict[str, float]:
    model.eval()

    pos_sums: dict[str, float] = {
        "dice": 0.0,
        "iou": 0.0,
        "sensitivity": 0.0,
        "precision": 0.0,
    }
    n_pos = 0
    n_all = 0
    hd95_values: list[float] = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    def _predictor(x):
        out = model(x)
        return out[0] if isinstance(out, list) else out

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False, unit="img"):
            images = batch["image"].to(device, non_blocking=True)  # (1, C, H, W)
            labels = batch["label"].to(device, non_blocking=True)  # (1, 1, H, W)

            with torch.autocast(
                device_type=autocast_device,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=_predictor,
                    overlap=sw_overlap,
                )

            logits = logits.float()
            metric_logits, _ = postprocess_logits(
                logits=logits,
                threshold=threshold,
                enabled=postprocess_enabled,
                spacing_zyx=spacing_yx,
                min_component_volume_mm3=postprocess_min_component_volume_mm3,
                connectivity=postprocess_connectivity,
            )
            m = compute_all_metrics(
                metric_logits,
                labels,
                threshold=threshold,
                compute_hd95=compute_hd95,
            )

            n_all += 1
            if not math.isnan(m["dice"]):
                for k in pos_sums:
                    pos_sums[k] += m[k]
                n_pos += 1
            if not math.isnan(m["hd95"]):
                hd95_values.append(m["hd95"])

            del metric_logits, logits, images, labels, m

    if n_all == 0:
        return {
            "dice": float("nan"),
            "iou": float("nan"),
            "sensitivity": float("nan"),
            "precision": float("nan"),
            "hd95": float("nan"),
            "n_pos": 0,
            "n_all": 0,
        }

    result: dict[str, float] = {
        "hd95": float(sum(hd95_values) / len(hd95_values)) if hd95_values else float("nan"),
        "n_pos": float(n_pos),
        "n_all": float(n_all),
    }
    if n_pos > 0:
        for k in pos_sums:
            result[k] = pos_sums[k] / n_pos
    else:
        result["dice"] = float("nan")
        result["iou"] = float("nan")
        result["sensitivity"] = float("nan")
        result["precision"] = float("nan")
    return result


def validate_with_oom_retry(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, ...],
    sw_overlap: float,
    sw_batch_size: int,
    min_sw_batch_size: int,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    compute_hd95: bool = True,
    threshold: float = 0.5,
    postprocess_enabled: bool = False,
    postprocess_min_component_volume_mm3: float = 30.0,
    postprocess_connectivity: int = 8,
    spacing_yx: tuple[float, float] = (0.5, 0.5),
) -> tuple[dict[str, float], int]:
    current_sw_batch_size = sw_batch_size

    while True:
        try:
            metrics = validate(
                model=model,
                loader=loader,
                device=device,
                patch_size=patch_size,
                sw_overlap=sw_overlap,
                sw_batch_size=current_sw_batch_size,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                compute_hd95=compute_hd95,
                threshold=threshold,
                postprocess_enabled=postprocess_enabled,
                postprocess_min_component_volume_mm3=postprocess_min_component_volume_mm3,
                postprocess_connectivity=postprocess_connectivity,
                spacing_yx=spacing_yx,
            )
            return metrics, current_sw_batch_size
        except RuntimeError as exc:
            if (
                device.type != "cuda"
                or not _is_cuda_oom(exc)
                or current_sw_batch_size <= min_sw_batch_size
            ):
                raise

            next_sw_batch_size = max(min_sw_batch_size, current_sw_batch_size // 2)
            if next_sw_batch_size == current_sw_batch_size:
                raise

            logger.warning(
                "CUDA OOM during validation with sw_batch_size=%d; retrying with sw_batch_size=%d.",
                current_sw_batch_size,
                next_sw_batch_size,
            )
            gc.collect()
            torch.cuda.empty_cache()
            current_sw_batch_size = next_sw_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2D Deconver segmentation model on JPG/PNG data",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume from",
    )
    parser.add_argument(
        "--new-split-manifest",
        action="store_true",
        help="Regenerate train/val split manifest before this run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    cfg_model = str(cfg.get("model", "deconver")).lower()
    if cfg_model != "deconver":
        raise ValueError(
            f"train_deconver_2d requires model='deconver', got {cfg_model!r}"
        )
    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(
            f"train_deconver_2d requires spatial_dims=2, got {spatial_dims}"
        )

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
    if cfg.get("device", "cuda") == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested in config but unavailable; falling back to CPU.")
    ensure_cuda_binary_compatibility(device)

    ensure_dir(str(cfg["base_output_dir"]))
    run_dir = create_run_dir(str(cfg["base_output_dir"]), str(cfg["experiment_name"]))
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    save_metadata(run_dir, cfg)
    save_config_copy(run_dir, cfg)
    save_latest_pointer(str(cfg["base_output_dir"]), run_dir)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    patch_size = tuple(int(v) for v in cfg.get("patch_size", [256, 256]))
    if len(patch_size) != 2:
        raise ValueError(f"patch_size must be [H, W], got {patch_size}")

    spacing_yx = tuple(float(v) for v in cfg.get("target_spacing", [0.5, 0.5]))
    if len(spacing_yx) != 2:
        raise ValueError(
            f"target_spacing must contain 2 values [y, x], got {spacing_yx}"
        )

    input_channels = int(cfg.get("input_channels", 1))
    image_ext = str(cfg.get("image_ext", ".jpg"))
    label_ext = str(cfg.get("label_ext", ".png"))
    recursive_discovery = bool(cfg.get("recursive_discovery", False))

    all_cases = discover_cases_2d(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        image_ext=image_ext,
        label_ext=label_ext,
        recursive=recursive_discovery,
        strict=True,
    )
    if not all_cases:
        raise RuntimeError(
            f"No JPG/PNG pairs found in {cfg['images_dir']} / {cfg['labels_dir']}."
        )

    split_manifest_raw = cfg.get("split_manifest_path", "")
    split_manifest_cfg = str(split_manifest_raw).strip() if split_manifest_raw is not None else ""
    split_manifest_path = (
        Path(split_manifest_cfg)
        if split_manifest_cfg
        else default_split_manifest_path(cfg["base_output_dir"])
    )
    lesion_flag_cache_path = split_manifest_path.parent / "lesion_flags.json"

    manifest_train_cases, manifest_val_cases, _, manifest_created = (
        resolve_train_val_split_from_manifest(
            cases=all_cases,
            val_fraction=float(cfg.get("val_fraction", 0.2)),
            seed=seed,
            manifest_path=split_manifest_path,
            new_split_manifest=args.new_split_manifest,
            cache_path=lesion_flag_cache_path,
        )
    )

    split_manifest_copy_path = run_dir / "train_val_split_manifest.json"
    shutil.copy2(split_manifest_path, split_manifest_copy_path)
    logger.info(
        "Split manifest: %s (%s) | copied to %s",
        split_manifest_path,
        "created" if manifest_created else "reused",
        split_manifest_copy_path,
    )

    train_cases = manifest_train_cases
    val_cases = manifest_val_cases
    logger.info("Split: %d train | %d val", len(train_cases), len(val_cases))

    train_ds = ImageMask2DDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        transform=get_train_transforms_2d(
            patch_size=patch_size,  # type: ignore[arg-type]
            pos_fraction=float(cfg.get("pos_fraction", 0.75)),
            num_samples=int(cfg.get("num_samples", 1)),
        ),
        cases=train_cases,
        use_cache=bool(cfg.get("cache_dataset", False)),
        cache_rate=float(cfg.get("cache_rate", 1.0)),
        input_channels=input_channels,
        image_ext=image_ext,
        label_ext=label_ext,
        recursive=recursive_discovery,
    )

    val_ds = ImageMask2DDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        transform=get_val_transforms_2d(),
        cases=val_cases,
        use_cache=bool(cfg.get("cache_dataset", False)),
        cache_rate=float(cfg.get("cache_rate", 1.0)),
        input_channels=input_channels,
        image_ext=image_ext,
        label_ext=label_ext,
        recursive=recursive_discovery,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)

    n_pos = sum(1 for c in train_cases if bool(c.get("has_lesion", False)))
    n_neg = len(train_cases) - n_pos
    num_workers = int(cfg["num_workers"])
    use_persistent = num_workers > 0

    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "All training cases are %s — WeightedRandomSampler disabled, falling back to shuffle=True.",
            "positive" if n_neg == 0 else "negative",
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            collate_fn=monai.data.utils.list_data_collate,
        )
    else:
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        sample_weights = [w_pos if c.get("has_lesion", False) else w_neg for c in train_cases]
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_cases),
            replacement=True,
            generator=sampler_generator,
        )
        logger.info(
            "WeightedRandomSampler: %d pos (w=%.4f) / %d neg (w=%.4f)",
            n_pos,
            w_pos,
            n_neg,
            w_neg,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["batch_size"]),
            sampler=weighted_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            collate_fn=monai.data.utils.list_data_collate,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )

    model = build_model(cfg).to(device)
    model_name = str(cfg.get("model", "deconver"))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s | trainable parameters: %s", model_name, f"{n_params:,}")

    compiled_model: torch.nn.Module = model
    if cfg.get("use_compile", False):
        cc_major, cc_minor = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        if device.type == "cuda" and (cc_major, cc_minor) < (7, 5):
            logger.warning(
                "Disabling torch.compile: GPU is sm_%d%d (requires sm_75+ for stable Triton support).",
                cc_major,
                cc_minor,
            )
        else:
            logger.info("Compiling model with torch.compile …")
            compiled_model = torch.compile(model)  # type: ignore[assignment]
    model = compiled_model

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
    cosine_t_max = max(1, int(cfg["epochs"]) - warmup_epochs)
    if warmup_epochs > 0:
        _warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        _cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=float(cfg["learning_rate"]) * 1e-2,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[_warmup_sched, _cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["epochs"]),
            eta_min=float(cfg["learning_rate"]) * 1e-2,
        )

    loss_fn = str(cfg.get("loss_fn", "dice_bce")).lower()
    criterion: torch.nn.Module
    if loss_fn == "tversky_bce":
        criterion = TverskyBCELoss(
            tversky_weight=float(cfg.get("dice_weight", 1.0)),
            bce_weight=float(cfg.get("bce_weight", 1.0)),
            alpha=float(cfg.get("tversky_alpha", 0.3)),
            beta=float(cfg.get("tversky_beta", 0.7)),
            pos_weight=float(cfg.get("bce_pos_weight", 1.0)),
        )
    else:
        criterion = DiceBCELoss(
            dice_weight=float(cfg.get("dice_weight", 1.0)),
            bce_weight=float(cfg.get("bce_weight", 1.0)),
            pos_weight=float(cfg.get("bce_pos_weight", 1.0)),
        )

    if cfg.get("deep_supervision", False):
        encoder_depth = cfg.get("deconver_encoder_depth", [1, 1, 1, 1])
        # Deconver returns num_deep_supr outputs, where
        # num_deep_supr = len(encoder_depth) - 1 in build_model.
        num_ds_levels = max(1, len(encoder_depth) - 1)
        criterion = DeepSupervisionWrapper(criterion, num_levels=num_ds_levels)
        logger.info(
            "Deep supervision enabled: %d levels, weights %s",
            num_ds_levels,
            [f"{w:.3f}" for w in criterion.weights],
        )

    best_val_dice = 0.0
    best_composite_score = 0.0
    last_hd95 = float("nan")
    start_epoch = 1

    sw_overlap = float(cfg.get("sw_overlap", 0.5))
    sw_batch_size = int(cfg.get("sw_batch_size", 8))
    val_min_sw_batch_size = max(1, int(cfg.get("val_min_sw_batch_size", 1)))
    val_compute_hd95_every = max(0, int(cfg.get("val_compute_hd95_every", 0)))
    epochs = int(cfg["epochs"])
    keep_last_n = int(cfg.get("keep_last_checkpoints", 3))
    val_every = max(1, int(cfg.get("val_every", 1)))
    val_start_epoch = max(1, int(cfg.get("val_start_epoch", 1)))
    pred_threshold = float(cfg.get("pred_threshold", 0.5))
    postprocess_enabled = bool(cfg.get("postprocess_enabled", False))
    postprocess_min_component_volume_mm3 = float(
        cfg.get("postprocess_min_component_volume_mm3", 30.0)
    )
    postprocess_connectivity = int(cfg.get("postprocess_connectivity", 8))

    if not (0.0 <= pred_threshold <= 1.0):
        raise ValueError(f"pred_threshold must be in [0, 1], got {pred_threshold}")
    if postprocess_min_component_volume_mm3 < 0.0:
        raise ValueError(
            "postprocess_min_component_volume_mm3 must be >= 0.0, "
            f"got {postprocess_min_component_volume_mm3}"
        )
    if postprocess_connectivity not in (4, 8):
        raise ValueError(
            "postprocess_connectivity must be one of {4, 8} for 2D, "
            f"got {postprocess_connectivity}"
        )

    amp_dtype_str = str(cfg.get("amp_dtype", "bf16"))
    dtype_map: dict[str, torch.dtype] = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype = dtype_map.get(amp_dtype_str, torch.bfloat16)
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
    bf16_supported = bool(bf16_supported_fn()) if callable(bf16_supported_fn) else False
    if use_amp and amp_dtype == torch.bfloat16 and not bf16_supported:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(
            "amp_dtype=bf16 requested, but detected device does not support BF16 autocast: "
            f"{gpu_name}. Set `amp_dtype: fp16` for older GPUs."
        )

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)  # type: ignore[attr-defined]
        if use_amp
        else torch.amp.autocast(device_type="cpu", enabled=False)  # type: ignore[attr-defined]
    )
    use_fp16 = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # type: ignore[attr-defined]

    w_sensitivity = float(cfg.get("best_ckpt_w_sensitivity", 0.5))
    w_dice = float(cfg.get("best_ckpt_w_dice", 0.3))
    w_hd95 = float(cfg.get("best_ckpt_w_hd95", 0.2))
    hd95_scale = float(cfg.get("best_ckpt_hd95_scale", 10.0))

    es_patience = int(cfg.get("early_stopping_patience", 20))
    es_min_delta = float(cfg.get("early_stopping_min_delta", 0.001))
    es_counter = 0
    es_enabled = es_patience > 0

    resume_path = args.resume or cfg.get("resume_checkpoint")
    if resume_path:
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_dice = float(ckpt.get("best_val_dice", 0.0))
        best_composite_score = float(ckpt.get("best_composite_score", 0.0))
        ckpt_hd95 = ckpt.get("last_hd95", float("nan"))
        last_hd95 = float(ckpt_hd95) if ckpt_hd95 is not None else float("nan")
        logger.info(
            "Resuming from epoch %d (best_composite_score=%.4f, best_val_dice=%.4f, "
            "last_hd95=%s) → starting at epoch %d",
            ckpt["epoch"],
            best_composite_score,
            best_val_dice,
            "nan" if math.isnan(last_hd95) else f"{last_hd95:.2f}px",
            start_epoch,
        )

    logger.info("Device: %s", device)
    logger.info("AMP (%s): %s", amp_dtype_str.upper(), use_amp)
    logger.info("torch.compile: %s", cfg.get("use_compile", False))
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("Run directory: %s", run_dir)
    logger.info("Loss: %s", criterion)
    logger.info(
        "Prediction: threshold=%.3f | postprocess=%s | min_component_volume=%.1f mm^3 | connectivity=%d",
        pred_threshold,
        "on" if postprocess_enabled else "off",
        postprocess_min_component_volume_mm3,
        postprocess_connectivity,
    )

    send_ntfy(
        cfg,
        title=f"Training started: {cfg['experiment_name']}",
        message=(
            f"Epochs: {epochs}\n"
            f"Train cases: {len(train_cases)} | Val cases: {len(val_cases)}\n"
            f"Device: {device} | AMP: {amp_dtype_str.upper() if use_amp else 'off'}\n"
            f"Run dir: {run_dir}"
        ),
        tags=["rocket"],
        priority="default",
    )

    max_nan_batches = 10
    nan_batch_count = 0

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):
        model.train()
        epoch_loss = 0.0

        batch_bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, unit="batch")
        for step, batch in enumerate(batch_bar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                logits = model(images)
                if isinstance(logits, list):
                    logits = [l.clamp(-10.0, 10.0) for l in logits]
                else:
                    logits = logits.clamp(-10.0, 10.0)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                loss_value = float(loss.detach().cpu().item())
                nan_batch_count += 1
                logger.warning(
                    "Non-finite loss at epoch %d, batch %d: %s — skipping (consecutive NaN skips: %d/%d)",
                    epoch,
                    step,
                    loss_value,
                    nan_batch_count,
                    max_nan_batches,
                )
                optimizer.zero_grad(set_to_none=True)
                if nan_batch_count >= max_nan_batches:
                    raise FloatingPointError(
                        f"Non-finite loss for {max_nan_batches} consecutive batches "
                        f"(epoch={epoch}, batch={step}): {loss_value}"
                    )
                continue
            nan_batch_count = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            del logits, images, labels, loss

        scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        logger.info("Epoch %d/%d | loss=%.4f | lr=%.2e", epoch, epochs, avg_loss, current_lr)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            str(checkpoint_dir / f"epoch_{epoch:04d}.pt"),
            scheduler=scheduler,
            scaler=scaler,
            best_val_dice=best_val_dice,
            best_composite_score=best_composite_score,
            last_hd95=last_hd95,
        )
        rotate_checkpoints(checkpoint_dir, keep_last_n)

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after train", device)

        run_val = (epoch >= val_start_epoch) and ((epoch % val_every == 0) or (epoch == epochs))
        if not run_val:
            logger.info(
                "Epoch %d/%d | validation skipped (val_every=%d)",
                epoch,
                epochs,
                val_every,
            )
            continue

        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        compute_hd95_now = (
            val_compute_hd95_every > 0
            and ((epoch % val_compute_hd95_every == 0) or (epoch == epochs))
        )
        val_metrics, used_sw_batch_size = validate_with_oom_retry(
            model=model,
            loader=val_loader,
            device=device,
            patch_size=patch_size,
            sw_overlap=sw_overlap,
            sw_batch_size=sw_batch_size,
            min_sw_batch_size=val_min_sw_batch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            compute_hd95=compute_hd95_now,
            threshold=pred_threshold,
            postprocess_enabled=postprocess_enabled,
            postprocess_min_component_volume_mm3=postprocess_min_component_volume_mm3,
            postprocess_connectivity=postprocess_connectivity,
            spacing_yx=spacing_yx,  # type: ignore[arg-type]
        )
        if used_sw_batch_size != sw_batch_size:
            logger.info(
                "Validation used reduced sw_batch_size=%d (configured=%d).",
                used_sw_batch_size,
                sw_batch_size,
            )

        writer.add_scalar("val/dice", val_metrics["dice"], epoch)
        writer.add_scalar("val/iou", val_metrics["iou"], epoch)
        writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
        writer.add_scalar("val/precision", val_metrics["precision"], epoch)
        if not math.isnan(val_metrics["hd95"]):
            writer.add_scalar("val/hd95", val_metrics["hd95"], epoch)

        n_pos_val = int(val_metrics["n_pos"])
        n_all_val = int(val_metrics["n_all"])
        logger.info(
            "Epoch %d/%d | val_dice=%s | val_iou=%s | val_sens=%s | val_prec=%s | val_hd95=%s | pos_cases=%d/%d",
            epoch,
            epochs,
            _fmt(val_metrics["dice"]),
            _fmt(val_metrics["iou"]),
            _fmt(val_metrics["sensitivity"]),
            _fmt(val_metrics["precision"]),
            _fmt(val_metrics["hd95"]),
            n_pos_val,
            n_all_val,
        )

        if compute_hd95_now and not math.isnan(val_metrics["hd95"]):
            last_hd95 = val_metrics["hd95"]
        score_metrics = dict(val_metrics)
        score_metrics["hd95"] = last_hd95
        composite_score = compute_composite_score(
            score_metrics,
            w_sensitivity=w_sensitivity,
            w_dice=w_dice,
            w_hd95=w_hd95,
            hd95_scale=hd95_scale,
        )

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after val", device)

        if not math.isnan(composite_score):
            writer.add_scalar("val/composite_score", composite_score, epoch)
            logger.info(
                "Epoch %d/%d | composite_score=%.4f (best=%.4f)",
                epoch,
                epochs,
                composite_score,
                best_composite_score,
            )

        if not math.isnan(composite_score) and composite_score > best_composite_score + es_min_delta:
            best_composite_score = composite_score
            if not math.isnan(val_metrics["dice"]):
                best_val_dice = val_metrics["dice"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                str(checkpoint_dir / "best.pt"),
                scheduler=scheduler,
                scaler=scaler,
                best_val_dice=best_val_dice,
                best_composite_score=best_composite_score,
                last_hd95=last_hd95,
            )
            logger.info(
                "New best model at epoch %d (composite_score=%.4f, val_dice=%s) → %s",
                epoch,
                best_composite_score,
                _fmt(val_metrics["dice"]),
                checkpoint_dir / "best.pt",
            )
            if cfg.get("ntfy_notify_best_model", True):
                send_ntfy(
                    cfg,
                    title=f"New best model: {cfg['experiment_name']}",
                    message=(
                        f"Epoch {epoch}/{epochs}\n"
                        f"Composite: {best_composite_score:.4f}\n"
                        f"Dice: {_fmt(val_metrics['dice'])} | "
                        f"Sensitivity: {_fmt(val_metrics['sensitivity'])} | "
                        f"Precision: {_fmt(val_metrics['precision'])}"
                    ),
                    tags=["trophy"],
                    priority="default",
                )
            es_counter = 0
        else:
            if es_enabled and not math.isnan(composite_score):
                es_counter += 1
                logger.info("Early stopping counter: %d / %d", es_counter, es_patience)

        if es_enabled and es_counter >= es_patience:
            logger.info(
                "Early stopping triggered at epoch %d — no improvement in composite score "
                "for %d consecutive epochs (min_delta=%.4f).",
                epoch,
                es_patience,
                es_min_delta,
            )
            send_ntfy(
                cfg,
                title=f"Training stopped early: {cfg['experiment_name']}",
                message=(
                    f"Early stopping at epoch {epoch}/{epochs}\n"
                    f"No improvement for {es_patience} consecutive epochs "
                    f"(min_delta={es_min_delta})\n"
                    f"Best composite: {best_composite_score:.4f} | "
                    f"Best dice: {best_val_dice:.4f}"
                ),
                tags=["warning"],
                priority="high",
            )
            break

    writer.close()
    logger.info("Training complete.")
    logger.info("Best composite score: %.4f", best_composite_score)
    logger.info("Best validation Dice: %.4f", best_val_dice)
    logger.info("Artifacts saved to: %s", run_dir)

    send_ntfy(
        cfg,
        title=f"Training complete: {cfg['experiment_name']}",
        message=(
            f"Best composite score: {best_composite_score:.4f}\n"
            f"Best val dice: {best_val_dice:.4f}\n"
            f"Artifacts: {run_dir}"
        ),
        tags=["white_check_mark"],
        priority="default",
    )


if __name__ == "__main__":
    _pre_parser = argparse.ArgumentParser(add_help=False)
    _pre_parser.add_argument("--config", type=str, default=None)
    _known, _ = _pre_parser.parse_known_args()

    _ntfy_cfg: dict = {}
    if _known.config:
        try:
            _ntfy_cfg = load_config(_known.config)
        except Exception:
            pass

    try:
        main()
    except Exception as exc:
        send_ntfy(
            _ntfy_cfg,
            title=f"Training FAILED: {_ntfy_cfg.get('experiment_name', 'unknown')}",
            message=f"{type(exc).__name__}: {exc}",
            tags=["x", "rotating_light"],
            priority="urgent",
        )
        raise
