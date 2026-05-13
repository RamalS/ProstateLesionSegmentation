"""
Training script for prostate lesion segmentation.

Usage (inside Docker):
    python -m src.train --config /workspace/configs/default.yaml

Pipeline
--------
1. Load config and set up output directories / TensorBoard.
2. Discover PI-CAI cases and split into train / validation sets.
3. Build PiCaiDataset with MONAI augmentation transforms.
4. Instantiate model via build_model(cfg), DiceBCELoss, AdamW + CosineAnnealingLR.
5. Train: random-patch forward pass → Dice+BCE loss → backward.
6. Validate: sliding-window inference over full volumes → Dice, IoU,
   Sensitivity, Specificity, HD95.
7. Save regular checkpoints + best-model checkpoint by validation Dice.
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

from src.config import load_config, resolve_dataset_cache_config
from src.dataset import (
    PiCaiDataset,
    annotate_cases_with_lesion_flags,
    active_modality_pairs,
    default_split_manifest_path,
    discover_cases,
    discover_prostate158_cases,
    resolve_train_val_split_from_manifest,
)
from src.losses import DiceBCELoss, DeepSupervisionWrapper, TverskyBCELoss
from src.metrics import compute_all_metrics
from src.models import build_model
from src.notify import send_ntfy
from src.postprocess import postprocess_logits
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    compute_composite_score,
    create_run_dir,
    ensure_cuda_binary_compatibility,
    ensure_dir,
    load_checkpoint,
    load_model_weights_for_current_config,
    load_pretrained_encoder,
    resolve_checkpoint_init_paths,
    rotate_checkpoints,
    save_checkpoint,
    save_config_copy,
    set_encoder_trainable,
    save_latest_pointer,
    save_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress MONAI class-balanced sampler warnings that fire on negative-only patches.
warnings.filterwarnings(
    "ignore",
    message=".*unable to generate class balanced samples.*",
    category=UserWarning,
    module="monai",
)
logger = logging.getLogger(__name__)


def _fmt(v: float) -> str:
    """Format a float metric for logging; returns 'n/a' when the value is NaN."""
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _seg_output(outputs):
    """
    Extract segmentation logits from a model output.

    Single-task models return Tensor/list directly. Multi-task models return
    {"seg": ..., "cls": ...}; keeping this helper centralized avoids branching
    throughout training and validation.
    """
    if isinstance(outputs, dict):
        return outputs["seg"]
    return outputs


def _cls_output(outputs) -> torch.Tensor | None:
    """Return auxiliary classification logits when the model provides them."""
    if isinstance(outputs, dict):
        cls = outputs.get("cls")
        if isinstance(cls, torch.Tensor):
            return cls
    return None


def _clamp_logits(outputs, min_value: float = -10.0, max_value: float = 10.0):
    """Clamp raw logits while preserving Tensor/list/dict output structure."""
    if isinstance(outputs, torch.Tensor):
        return outputs.clamp(min_value, max_value)
    if isinstance(outputs, list):
        return [item.clamp(min_value, max_value) for item in outputs]
    if isinstance(outputs, dict):
        clamped = dict(outputs)
        clamped["seg"] = _clamp_logits(clamped["seg"], min_value, max_value)
        if isinstance(clamped.get("cls"), torch.Tensor):
            clamped["cls"] = clamped["cls"].clamp(min_value, max_value)
        return clamped
    return outputs


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _seed_worker(worker_id: int) -> None:
    """
    Seed Python and NumPy RNGs inside each DataLoader worker process.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _is_cuda_oom(exc: RuntimeError) -> bool:
    """
    Return True if *exc* represents a CUDA out-of-memory error.
    """
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _log_cuda_memory(stage: str, device: torch.device) -> None:
    """
    Log current and peak CUDA memory usage in GiB.
    """
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
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    compute_hd95: bool = True,
    threshold: float = 0.5,
    postprocess_enabled: bool = False,
    postprocess_min_component_volume_mm3: float = 30.0,
    postprocess_connectivity: int = 26,
    spacing_zyx: tuple[float, float, float] = (3.0, 0.5, 0.5),
) -> dict[str, float]:
    """
    Run sliding-window inference over full validation volumes and return
    averaged segmentation metrics.

    Dice, IoU, and sensitivity are averaged only over positive cases (volumes
    whose ground-truth label contains at least one lesion voxel).  Specificity
    is averaged over all cases.  HD95 is averaged over cases where both
    prediction and target are non-empty.

    Parameters
    ----------
    use_amp      : enable autocast around sliding-window inference.
    amp_dtype    : dtype to use for autocast — ``torch.float16`` for Volta/Turing,
                   ``torch.bfloat16`` for Ampere+/Blackwell.
    compute_hd95 : if False, skip the expensive HD95 calculation and log
                   float('nan') for "hd95".  Useful during the training loop
                   where HD95 is not needed every epoch.
    threshold    : sigmoid threshold for logits -> binary prediction mask.
    postprocess_enabled : if True, remove tiny connected components from
                         predicted binary masks before metrics.
    postprocess_min_component_volume_mm3 : minimum component size in mm^3.
    postprocess_connectivity : 3-D connectivity for components (6/18/26).
    spacing_zyx : voxel spacing (z, y, x) in mm used for mm^3 -> voxel conversion.
    """
    model.eval()

    # Positive-case accumulators (dice / iou / sensitivity / precision)
    pos_sums: dict[str, float] = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "precision": 0.0}
    n_pos = 0   # volumes with ≥1 lesion voxel in ground truth
    n_all = 0   # total volumes processed

    hd95_values: list[float] = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    # When deep supervision is active the model returns list[Tensor].
    # sliding_window_inference requires a callable that returns a single Tensor,
    # so we wrap the model to extract the finest-resolution output (index 0).
    # This is a no-op for standard (non-DS) models that already return a Tensor.
    def _predictor(x):
        out = _seg_output(model(x))
        return out[0] if isinstance(out, list) else out

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False, unit="vol"):
            images = batch["image"].to(device, non_blocking=True)   # (1, 3, D, H, W)
            labels = batch["label"].to(device, non_blocking=True)   # (1, 1, D, H, W)

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

            # Cast back to float32 for metrics (avoids BF16 precision loss in distance
            # transforms and other numpy-backed metric operations).
            logits = logits.float()
            metric_logits, _ = postprocess_logits(
                logits=logits,
                threshold=threshold,
                enabled=postprocess_enabled,
                spacing_zyx=spacing_zyx,
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

            # Dice / IoU / sensitivity / precision: only for positive cases
            # compute_all_metrics returns nan when the target is empty
            if not math.isnan(m["dice"]):
                for k in pos_sums:
                    pos_sums[k] += m[k]
                n_pos += 1

            if not math.isnan(m["hd95"]):
                hd95_values.append(m["hd95"])

            del metric_logits, logits, images, labels, m

    if n_all == 0:
        return {"dice": float("nan"), "iou": float("nan"),
                "sensitivity": float("nan"), "precision": float("nan"),
                "hd95": float("nan"), "n_pos": 0, "n_all": 0}

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
    postprocess_connectivity: int = 26,
    spacing_zyx: tuple[float, float, float] = (3.0, 0.5, 0.5),
) -> tuple[dict[str, float], int]:
    """
    Run validation and retry with smaller sw_batch_size on CUDA OOM.

    Returns
    -------
    (metrics, used_sw_batch_size)
    """
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
                spacing_zyx=spacing_zyx,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train prostate lesion segmentation model"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume training from "
             "(overrides resume_checkpoint in config)",
    )
    parser.add_argument(
        "--current-config", action="store_true",
        help="When used with --resume (or resume_checkpoint in config), "
             "initialize model weights from that checkpoint but keep optimizer/"
             "scheduler/scaler state from the current config (fresh run at epoch 1).",
    )
    parser.add_argument(
        "--learnability", type=int, nargs="?", const=10, default=None, metavar="N",
        help="Learnability test: randomly sample N cases (default 10) and use them "
             "for both training and validation (no split). "
             "Useful to verify the model can overfit a small subset.",
    )
    parser.add_argument(
        "--new-split-manifest",
        action="store_true",
        help="Regenerate train/val split manifest before this run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    resume_path, use_current_config = resolve_checkpoint_init_paths(
        resume_cli=args.resume,
        resume_cfg=cfg.get("resume_checkpoint"),
        use_current_config=bool(args.current_config),
    )
    if use_current_config and resume_path is not None:
        cfg["current_config_checkpoint"] = resume_path

    cache_mode, cache_rate, cache_dir = resolve_dataset_cache_config(cfg, logger=logger)

    # ---- Reproducibility ----
    seed = cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic: bool = bool(cfg.get("deterministic", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )
    if cfg.get("device", "cuda") == "cuda" and device.type != "cuda":
        logger.warning("CUDA requested in config but unavailable; falling back to CPU.")
    ensure_cuda_binary_compatibility(device)

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
    logger.info(
        "Dataset cache: mode=%s, rate=%.2f, dir=%s",
        cache_mode,
        cache_rate,
        cache_dir if cache_dir is not None else "-",
    )

    # Resolve active modalities once; used for case discovery and dataset init.
    _active_keys = [k for k, _ in active_modality_pairs(cfg)]

    dataset_type = str(cfg.get("dataset_type", "picai")).strip().lower()
    if dataset_type == "prostate158":
        prostate158_train_dir = Path(
            cfg.get(
                "prostate158_train_dir",
                cfg.get("images_dir", "/data/prostate158_train"),
            )
        )
        label_target = str(cfg.get("prostate158_label_target", "tumor"))
        label_reader = cfg.get("prostate158_label_reader", 1)
        label_modality = cfg.get("prostate158_label_modality")

        train_cases = discover_prostate158_cases(
            root_dir=prostate158_train_dir,
            split=str(cfg.get("prostate158_train_split", "train")),
            active_keys=_active_keys,
            label_target=label_target,
            label_reader=label_reader,
            label_modality=label_modality,
        )
        val_cases = discover_prostate158_cases(
            root_dir=prostate158_train_dir,
            split=str(cfg.get("prostate158_val_split", "valid")),
            active_keys=_active_keys,
            label_target=label_target,
            label_reader=label_reader,
            label_modality=label_modality,
        )
        all_cases = train_cases + val_cases
        lesion_flag_cache_path = (
            Path(cfg["base_output_dir"]).parent / "splits" / "prostate158_lesion_flags.json"
        )
        annotate_cases_with_lesion_flags(all_cases, cache_path=lesion_flag_cache_path)

        if args.new_split_manifest:
            logger.warning(
                "--new-split-manifest is ignored for dataset_type='prostate158'; "
                "using upstream train.csv and valid.csv."
            )
    elif dataset_type == "picai":
        all_cases = discover_cases(
            images_dir=Path(cfg["images_dir"]),
            labels_dir=Path(cfg["labels_dir"]),
            active_keys=_active_keys,
        )

        if not all_cases:
            raise RuntimeError(
                f"No cases found in {cfg['images_dir']}. "
                "Check that your data is mounted correctly (./data -> /data)."
            )

        split_manifest_raw = cfg.get("split_manifest_path", "")
        split_manifest_cfg = (
            str(split_manifest_raw).strip()
            if split_manifest_raw is not None
            else ""
        )
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
    else:
        raise ValueError(
            f"Unsupported dataset_type='{dataset_type}'. Expected 'picai' or 'prostate158'."
        )

    if not all_cases:
        raise RuntimeError(f"No cases found for dataset_type='{dataset_type}'.")
    if not train_cases or not val_cases:
        raise RuntimeError(
            f"Invalid split for dataset_type='{dataset_type}': "
            f"{len(train_cases)} train / {len(val_cases)} val cases."
        )

    if args.learnability is not None:
        n = max(1, args.learnability)
        n = min(n, len(all_cases))
        rng = random.Random(seed)
        subset = rng.sample(all_cases, n)
        train_cases = subset
        val_cases = subset
        logger.info(
            "[LEARNABILITY MODE] Using %d randomly sampled cases for both train and val (no split).",
            n,
        )
    else:
        logger.info("Split: %d train | %d val", len(train_cases), len(val_cases))

    train_ds = PiCaiDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        target_spacing=target_spacing,
        transform=get_train_transforms(
            patch_size=patch_size,
            pos_fraction=cfg.get("pos_fraction", 0.75),
            num_samples=cfg.get("num_samples", 1),
        ),
        cases=train_cases,
        cache_mode=cache_mode,
        cache_rate=cache_rate,
        cache_dir=cache_dir,
        active_modalities=_active_keys,
        dwi_hbv_preprocess=cfg.get("dwi_hbv_preprocess", {}),
    )

    val_ds = PiCaiDataset(
        images_dir=cfg["images_dir"],
        labels_dir=cfg["labels_dir"],
        target_spacing=target_spacing,
        transform=get_val_transforms(),
        cases=val_cases,
        cache_mode=cache_mode,
        cache_rate=cache_rate,
        cache_dir=cache_dir,
        active_modalities=_active_keys,
        dwi_hbv_preprocess=cfg.get("dwi_hbv_preprocess", {}),
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)

    # ---- Weighted sampler: over-sample positive cases ----
    # Each positive case gets weight 1/n_pos; each negative gets 1/n_neg.
    # This gives each batch a roughly balanced mix without duplicating data.
    n_pos = sum(1 for c in train_cases if c.get("has_lesion", False))
    n_neg = len(train_cases) - n_pos
    if cache_mode == "ram" and cfg["num_workers"] == 0:
        logger.warning(
            "cache_mode='ram' with num_workers=0 keeps cache in the main process only; "
            "this is valid but may reduce throughput."
        )

    num_workers: int = cfg["num_workers"]
    use_persistent: bool = num_workers > 0
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "All training cases are %s — WeightedRandomSampler disabled, "
            "falling back to shuffle=True.",
            "positive" if n_neg == 0 else "negative",
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
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
        sample_weights = [
            w_pos if c.get("has_lesion", False) else w_neg
            for c in train_cases
        ]
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_cases),
            replacement=True,
            generator=sampler_generator,
        )
        logger.info(
            "WeightedRandomSampler: %d pos (w=%.4f) / %d neg (w=%.4f)",
            n_pos, w_pos, n_neg, w_neg,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=weighted_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            collate_fn=monai.data.utils.list_data_collate,
        )

    # Validation uses batch_size=1: full volumes, sliding window handles memory
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

    # ---- Model ----
    model = build_model(cfg).to(device)

    pretrained_encoder_checkpoint: str = str(
        cfg.get("pretrained_encoder_checkpoint", "")
    ).strip()
    if use_current_config and pretrained_encoder_checkpoint:
        logger.info(
            "Skipping pretrained_encoder_checkpoint (%s) because --current-config "
            "was provided.",
            pretrained_encoder_checkpoint,
        )
    elif pretrained_encoder_checkpoint:
        enc_stats = load_pretrained_encoder(
            model=model,
            path=pretrained_encoder_checkpoint,
            device=device,
        )
        logger.info(
            "Loaded pretrained encoder from %s | tensors=%d | missing=%d | shape_mismatch=%d",
            pretrained_encoder_checkpoint,
            enc_stats["loaded"],
            len(enc_stats["missing"]),
            len(enc_stats["shape_mismatch"]),
        )
    if use_current_config and resume_path is not None:
        init_stats = load_model_weights_for_current_config(
            model=model,
            path=resume_path,
            device=device,
            strict_shape=True,
        )
        logger.info(
            "Initialized model from current-config checkpoint %s | tensors=%d "
            "| missing=%d | unexpected=%d",
            resume_path,
            init_stats["loaded"],
            len(init_stats["missing"]),
            len(init_stats["unexpected"]),
        )

    model_name = cfg.get("model", "unet3d")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s | trainable parameters: %s", model_name, f"{n_params:,}")

    # Optional: torch.compile (triton-based kernel fusion).  Disabled by default
    # until the user has verified convergence; enable via use_compile: true in
    # the config.  Safe here because patch_size is fixed, so no recompilation.
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

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    # Optional linear warm-up phase followed by cosine annealing.
    # warmup_epochs=0 (default) falls back to pure cosine — backward compatible.
    # During warm-up the LR rises linearly from 10 % → 100 % of learning_rate
    # over the first warmup_epochs epochs.  This lets BatchNorm statistics
    # stabilise before the full learning rate is applied, reducing the risk of
    # early collapse into the all-background local minimum.
    warmup_epochs: int = max(0, int(cfg.get("warmup_epochs", 0)))
    cosine_t_max: int = max(1, cfg["epochs"] - warmup_epochs)

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
            eta_min=cfg["learning_rate"] * 1e-2,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[_warmup_sched, _cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["epochs"],
            eta_min=cfg["learning_rate"] * 1e-2,
        )

    # ---- Loss ----
    _loss_fn: str = cfg.get("loss_fn", "dice_bce").lower()
    criterion: torch.nn.Module
    if _loss_fn == "tversky_bce":
        criterion = TverskyBCELoss(
            tversky_weight=cfg.get("dice_weight", 1.0),
            bce_weight=cfg.get("bce_weight", 1.0),
            alpha=cfg.get("tversky_alpha", 0.3),
            beta=cfg.get("tversky_beta", 0.7),
            pos_weight=cfg.get("bce_pos_weight", 1.0),
        )
    else:
        criterion = DiceBCELoss(
            dice_weight=cfg.get("dice_weight", 1.0),
            bce_weight=cfg.get("bce_weight", 1.0),
            pos_weight=cfg.get("bce_pos_weight", 1.0),
        )

    # Wrap with DeepSupervisionWrapper when deep supervision is enabled.
    # The wrapper accepts both list[Tensor] (DS mode) and plain Tensor (standard
    # mode) outputs, so the training loop below is identical in either case.
    if cfg.get("deep_supervision", False):
        _num_ds_levels = len(cfg.get("features", [32, 64, 128, 256]))
        criterion = DeepSupervisionWrapper(criterion, num_levels=_num_ds_levels)
        logger.info(
            "Deep supervision enabled: %d levels, weights %s",
            _num_ds_levels,
            [f"{w:.3f}" for w in criterion.weights],
        )

    classification_loss_weight = float(cfg.get("classification_loss_weight", 0.0))
    classification_pos_weight = float(cfg.get("classification_pos_weight", 1.0))
    cls_criterion: torch.nn.Module | None = None
    if classification_loss_weight > 0.0:
        cls_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([classification_pos_weight], device=device)
        )
        logger.info(
            "Auxiliary classification loss enabled: weight=%.3f, pos_weight=%.3f",
            classification_loss_weight,
            classification_pos_weight,
        )

    # Optional: freeze encoder for a few warm-up epochs after loading
    # pretrained encoder weights.
    freeze_encoder_epochs: int = max(0, int(cfg.get("freeze_encoder_epochs", 0)))
    encoder_is_frozen: bool = False
    if freeze_encoder_epochs > 0:
        n_frozen = set_encoder_trainable(model, trainable=False)
        encoder_is_frozen = True
        logger.info(
            "Encoder frozen for first %d epoch(s) (%s parameters).",
            freeze_encoder_epochs,
            f"{n_frozen:,}",
        )

    # ---- Training loop ----
    best_val_dice = 0.0
    best_composite_score = 0.0
    last_hd95: float = float("nan")   # most-recent finite HD95; NaN until first computation
    start_epoch = 1
    sw_overlap = cfg.get("sw_overlap", 0.5)
    sw_batch_size = cfg.get("sw_batch_size", 4)
    val_min_sw_batch_size: int = max(1, int(cfg.get("val_min_sw_batch_size", 1)))
    val_compute_hd95_every: int = max(0, int(cfg.get("val_compute_hd95_every", 0)))
    epochs = cfg["epochs"]
    keep_last_n: int = cfg.get("keep_last_checkpoints", 3)
    val_every: int = max(1, cfg.get("val_every", 1))
    val_start_epoch: int = max(1, int(cfg.get("val_start_epoch", 1)))
    pred_threshold: float = float(cfg.get("pred_threshold", 0.5))
    postprocess_enabled: bool = bool(cfg.get("postprocess_enabled", False))
    postprocess_min_component_volume_mm3: float = float(
        cfg.get("postprocess_min_component_volume_mm3", 30.0)
    )
    postprocess_connectivity: int = int(cfg.get("postprocess_connectivity", 26))

    if not (0.0 <= pred_threshold <= 1.0):
        raise ValueError(f"pred_threshold must be in [0, 1], got {pred_threshold}")
    if postprocess_min_component_volume_mm3 < 0.0:
        raise ValueError(
            "postprocess_min_component_volume_mm3 must be >= 0.0, "
            f"got {postprocess_min_component_volume_mm3}"
        )
    if postprocess_connectivity not in (6, 18, 26):
        raise ValueError(
            "postprocess_connectivity must be one of {6, 18, 26}, "
            f"got {postprocess_connectivity}"
        )

    # AMP dtype: "fp16" for Volta/Turing (TITAN V, V100), "bf16" for Ampere+/Blackwell.
    # FP16 requires GradScaler (limited exponent range); BF16 does not.
    amp_dtype_str: str = cfg.get("amp_dtype", "bf16")
    _dtype_map: dict[str, torch.dtype] = {"fp16": torch.float16, "bf16": torch.bfloat16}
    amp_dtype: torch.dtype = _dtype_map.get(amp_dtype_str, torch.bfloat16)
    use_amp: bool = cfg.get("use_amp", True) and device.type == "cuda"
    bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
    bf16_supported = bool(bf16_supported_fn()) if callable(bf16_supported_fn) else False
    if use_amp and amp_dtype == torch.bfloat16 and not bf16_supported:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        raise RuntimeError(
            "amp_dtype=bf16 requested, but detected device does not support BF16 autocast: "
            f"{gpu_name}. Set `amp_dtype: fp16` for Volta/Turing GPUs."
        )
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)  # type: ignore[attr-defined]
        if use_amp
        else torch.amp.autocast(device_type="cpu", enabled=False)  # type: ignore[attr-defined]
    )
    # GradScaler: required for FP16 (prevents underflow), must be disabled for BF16/FP32.
    use_fp16: bool = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # type: ignore[attr-defined]

    # ---- Composite score weights ----
    w_sensitivity: float = cfg.get("best_ckpt_w_sensitivity", 0.5)
    w_dice: float = cfg.get("best_ckpt_w_dice", 0.3)
    w_hd95: float = cfg.get("best_ckpt_w_hd95", 0.2)
    hd95_scale: float = float(cfg.get("best_ckpt_hd95_scale", 10.0))
    if val_compute_hd95_every == 0 and w_hd95 > 0.0:
        logger.warning(
            "best_ckpt_w_hd95=%.2f but val_compute_hd95_every=0; "
            "HD95 will never be computed, so the HD95 weight will always be "
            "redistributed to sensitivity and dice.",
            w_hd95,
        )

    # ---- Early stopping ----
    es_patience: int = cfg.get("early_stopping_patience", 20)
    es_min_delta: float = cfg.get("early_stopping_min_delta", 0.001)
    es_counter: int = 0
    es_enabled: bool = es_patience > 0

    # ---- Resume from checkpoint (full state restore) ----
    if resume_path and not use_current_config:
        ckpt = load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = ckpt["epoch"] + 1
        best_val_dice = float(ckpt.get("best_val_dice", 0.0))
        best_composite_score = float(ckpt.get("best_composite_score", 0.0))
        _ckpt_hd95 = ckpt.get("last_hd95", float("nan"))
        last_hd95 = float(_ckpt_hd95) if _ckpt_hd95 is not None else float("nan")
        logger.info(
            "Resuming from epoch %d (best_composite_score=%.4f, best_val_dice=%.4f,"
            " last_hd95=%s) → starting at epoch %d",
            ckpt["epoch"], best_composite_score, best_val_dice,
            "nan" if math.isnan(last_hd95) else f"{last_hd95:.2f}mm",
            start_epoch,
        )
    elif resume_path and use_current_config:
        logger.info(
            "Current-config init active from %s: starting at epoch 1 with "
            "fresh optimizer/scheduler/scaler state from current config.",
            resume_path,
        )

    logger.info("Device: %s", device)
    logger.info("AMP (%s): %s", amp_dtype_str.upper(), use_amp)
    logger.info("torch.compile: %s", cfg.get("use_compile", False))
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("Run directory: %s", run_dir)
    logger.info("Loss: %s", criterion)
    logger.info(
        "Scheduler: %s warmup → CosineAnnealing (T_max=%d, eta_min=%.2e)",
        f"{warmup_epochs}-epoch linear" if warmup_epochs > 0 else "no",
        cosine_t_max,
        cfg["learning_rate"] * 1e-2,
    )
    logger.info(
        "Best checkpoint metric: composite score "
        "(w_sensitivity=%.2f, w_dice=%.2f, w_hd95=%.2f, hd95_scale=%.1fmm)",
        w_sensitivity, w_dice, w_hd95, hd95_scale,
    )
    logger.info(
        "Prediction: threshold=%.3f | postprocess=%s | min_component_volume=%.1f mm^3 | connectivity=%d",
        pred_threshold,
        "on" if postprocess_enabled else "off",
        postprocess_min_component_volume_mm3,
        postprocess_connectivity,
    )
    if es_enabled:
        logger.info(
            "Early stopping: patience=%d, min_delta=%.4f",
            es_patience, es_min_delta,
        )
        if val_every > 1:
            logger.warning(
                "val_every=%d — early stopping patience counts validation epochs, "
                "so no improvement for %d epochs = %d training epochs.",
                val_every, es_patience, es_patience * val_every,
            )
    else:
        logger.info("Early stopping: disabled")
    if val_start_epoch > 1:
        logger.info(
            "Validation deferred until epoch %d (val_start_epoch)",
            val_start_epoch,
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

    # Maximum consecutive NaN batches tolerated before aborting training.
    _MAX_NAN_BATCHES: int = 10
    nan_batch_count: int = 0

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):

        # ---- Train ----
        model.train()
        if freeze_encoder_epochs > 0:
            if epoch <= freeze_encoder_epochs:
                set_encoder_trainable(model, trainable=False)
                writer.add_scalar("train/encoder_frozen", 1.0, epoch)
            else:
                if encoder_is_frozen:
                    n_unfrozen = set_encoder_trainable(model, trainable=True)
                    encoder_is_frozen = False
                    logger.info(
                        "Encoder unfrozen at epoch %d (%s parameters).",
                        epoch,
                        f"{n_unfrozen:,}",
                    )
                writer.add_scalar("train/encoder_frozen", 0.0, epoch)
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_cls_loss = 0.0
        cls_batches = 0

        batch_bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, unit="batch")
        for step, batch in enumerate(batch_bar, start=1):
            images = batch["image"].to(device, non_blocking=True)   # (B, 3, D, H, W)
            labels = batch["label"].to(device, non_blocking=True)   # (B, 1, D, H, W)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(images)
                # Clamp to ±10 to prevent FP16 sigmoid overflow (sigmoid(±10) ≈
                # 0.99995 / 0.00005 — sufficient confidence range for segmentation).
                # Under BF16 or FP32 this is a no-op in practice.
                outputs = _clamp_logits(outputs)
                seg_logits = _seg_output(outputs)
                seg_loss = criterion(seg_logits, labels)
                cls_loss = None
                cls_logits = _cls_output(outputs)
                if cls_criterion is not None and cls_logits is not None:
                    cls_targets = (labels.flatten(1).amax(dim=1) > 0).float()
                    cls_loss = cls_criterion(cls_logits.float(), cls_targets)
                    loss = seg_loss + classification_loss_weight * cls_loss
                else:
                    loss = seg_loss

            if not torch.isfinite(loss):
                loss_value = float(loss.detach().cpu().item())
                nan_batch_count += 1
                logger.warning(
                    "Non-finite loss at epoch %d, batch %d: %s — skipping"
                    " (consecutive NaN skips: %d/%d)",
                    epoch,
                    step,
                    loss_value,
                    nan_batch_count,
                    _MAX_NAN_BATCHES,
                )
                optimizer.zero_grad(set_to_none=True)
                if nan_batch_count >= _MAX_NAN_BATCHES:
                    raise FloatingPointError(
                        f"Non-finite loss for {_MAX_NAN_BATCHES} consecutive"
                        f" batches (epoch={epoch}, batch={step}): {loss_value}"
                    )
                continue
            nan_batch_count = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_item = loss.item()
            seg_loss_item = seg_loss.item()
            epoch_loss += loss_item
            epoch_seg_loss += seg_loss_item
            postfix = {"loss": f"{loss_item:.4f}"}
            if cls_loss is not None:
                cls_loss_item = cls_loss.item()
                epoch_cls_loss += cls_loss_item
                cls_batches += 1
                postfix["cls"] = f"{cls_loss_item:.4f}"
            batch_bar.set_postfix(**postfix)

            del outputs, seg_logits, images, labels, loss, seg_loss, cls_loss, cls_logits

        scheduler.step()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_seg_loss = epoch_seg_loss / max(len(train_loader), 1)
        avg_cls_loss = epoch_cls_loss / max(cls_batches, 1) if cls_batches > 0 else float("nan")
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/seg_loss", avg_seg_loss, epoch)
        if not math.isnan(avg_cls_loss):
            writer.add_scalar("train/cls_loss", avg_cls_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        if not math.isnan(avg_cls_loss):
            logger.info(
                "Epoch %d/%d | loss=%.4f | seg_loss=%.4f | cls_loss=%.4f | lr=%.2e",
                epoch, epochs, avg_loss, avg_seg_loss, avg_cls_loss, current_lr,
            )
        else:
            logger.info(
                "Epoch %d/%d | loss=%.4f | lr=%.2e",
                epoch, epochs, avg_loss, current_lr,
            )
        save_checkpoint(
            model, optimizer, epoch,
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

        # ---- Validate ----
        run_val = (epoch >= val_start_epoch) and ((epoch % val_every == 0) or (epoch == epochs))
        if run_val:
            # Free leftover train grads/cache before validation.
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
                spacing_zyx=target_spacing,
            )

            if used_sw_batch_size != sw_batch_size:
                logger.info(
                    "Validation used reduced sw_batch_size=%d (configured=%d).",
                    used_sw_batch_size,
                    sw_batch_size,
                )

            writer.add_scalar("val/dice",        val_metrics["dice"],        epoch)
            writer.add_scalar("val/iou",         val_metrics["iou"],         epoch)
            writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
            writer.add_scalar("val/precision",   val_metrics["precision"],   epoch)
            if not math.isnan(val_metrics["hd95"]):
                writer.add_scalar("val/hd95", val_metrics["hd95"], epoch)

            n_pos_val = int(val_metrics["n_pos"])
            n_all_val = int(val_metrics["n_all"])

            hd95_str = _fmt(val_metrics["hd95"])
            logger.info(
                "Epoch %d/%d | val_dice=%s | val_iou=%s"
                " | val_sens=%s | val_prec=%s | val_hd95=%s"
                " | pos_cases=%d/%d",
                epoch, epochs,
                _fmt(val_metrics["dice"]),
                _fmt(val_metrics["iou"]),
                _fmt(val_metrics["sensitivity"]),
                _fmt(val_metrics["precision"]),
                hd95_str,
                 n_pos_val, n_all_val,
            )

            # ---- Composite score: best.pt selection + early stopping ----
            # Cache the most-recent finite HD95 so the score formula is
            # consistent across epochs regardless of compute_hd95_now.
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
                    epoch, epochs, composite_score, best_composite_score,
                )

            # Save best model checkpoint when composite score improves
            if not math.isnan(composite_score) and composite_score > best_composite_score + es_min_delta:
                best_composite_score = composite_score
                # Also track best dice for logging convenience
                if not math.isnan(val_metrics["dice"]):
                    best_val_dice = val_metrics["dice"]
                save_checkpoint(
                    model, optimizer, epoch,
                    str(checkpoint_dir / "best.pt"),
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_dice=best_val_dice,
                    best_composite_score=best_composite_score,
                    last_hd95=last_hd95,
                )
                logger.info(
                    "New best model at epoch %d (composite_score=%.4f, val_dice=%s) → %s",
                    epoch, best_composite_score, _fmt(val_metrics["dice"]),
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
                    logger.info(
                        "Early stopping counter: %d / %d",
                        es_counter, es_patience,
                    )

            if es_enabled and es_counter >= es_patience:
                logger.info(
                    "Early stopping triggered at epoch %d — no improvement in composite score "
                    "for %d consecutive epochs (min_delta=%.4f).",
                    epoch, es_patience, es_min_delta,
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
        else:
            logger.info(
                "Epoch %d/%d | validation skipped (val_every=%d)",
                epoch, epochs, val_every,
            )

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
    # Pre-load config for the failure notification.  parse_known_args lets us
    # extract --config without duplicating the full argument parser.
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
    except Exception as _exc:
        send_ntfy(
            _ntfy_cfg,
            title=f"Training FAILED: {_ntfy_cfg.get('experiment_name', 'unknown')}",
            message=f"{type(_exc).__name__}: {_exc}",
            tags=["x", "rotating_light"],
            priority="urgent",
        )
        raise
