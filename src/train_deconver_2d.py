"""
2D Deconver training for Gleason consensus labels (4-class segmentation).

Usage:
    python -m src.train_deconver_2d --config configs/deconver_2d_local.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import multiprocessing as mp
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.gleason_consensus_dataset import GleasonConsensusDataset
from src.models import build_model
from src.notify import send_ntfy
from src.utils import (
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


def _resolve_dataloader_context(
    cfg: dict,
    num_workers: int,
) -> tuple[int, mp.context.BaseContext | None, str]:
    """
    Resolve DataLoader multiprocessing context robustly.

    Python 3.14 changed POSIX default start method to ``forkserver`` which
    requires opening a listener socket. In restricted environments this can
    raise PermissionError during worker startup. We default to ``fork`` here
    and gracefully fall back to single-process loading if unavailable.
    """
    raw = str(cfg.get("dataloader_start_method", "fork")).strip().lower()
    method = raw or "fork"
    if num_workers <= 0:
        return 0, None, method

    if method in {"default", "auto", "none"}:
        return num_workers, None, method

    try:
        ctx = mp.get_context(method)
        return num_workers, ctx, method
    except Exception as exc:
        logger.warning(
            "DataLoader start method '%s' unavailable (%s). Falling back to num_workers=0.",
            method,
            exc,
        )
        return 0, None, method


def _safe_read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _infer_qc_flags(
    qc: dict,
    fail_keys: tuple[str, ...],
    suspicious_keys: tuple[str, ...],
) -> tuple[bool, bool]:
    fail = any(bool(qc.get(k, False)) for k in fail_keys)
    suspicious = any(bool(qc.get(k, False)) for k in suspicious_keys)
    return fail, suspicious


def _infer_qc_weight(
    qc: dict,
    base_downweight: float,
    per_rater_penalty: float,
    min_weight: float,
) -> float:
    weight = 1.0
    excluded = qc.get("excluded_raters", [])
    downweighted = qc.get("downweighted_raters", [])

    n_excluded = len(excluded) if isinstance(excluded, list) else 0
    n_downweighted = len(downweighted) if isinstance(downweighted, list) else 0

    if bool(qc.get("suspicious", False)):
        weight *= base_downweight

    if n_excluded > 0 or n_downweighted > 0:
        penalty_steps = n_excluded + n_downweighted
        weight *= max(0.0, 1.0 - (per_rater_penalty * penalty_steps))

    return float(max(min_weight, min(1.0, weight)))


def _case_flags_from_hard_mask(mask_path: Path) -> tuple[bool, bool]:
    with Image.open(mask_path) as img:
        arr = np.asarray(img, dtype=np.uint8)
    has_cancer = bool((arr > 0).any())
    has_grade5 = bool((arr == 3).any())
    return has_cancer, has_grade5


def _build_sample_metadata(dataset: GleasonConsensusDataset, cfg: dict) -> list[dict]:
    supervised_subdirs = tuple(
        str(s) for s in cfg.get("supervised_image_subdirs", ["Trains_imgs"])
    )
    qc_policy = str(cfg.get("qc_policy", "warn_downweight")).strip().lower()
    fail_keys = tuple(
        str(s) for s in cfg.get("qc_fail_keys", ["hard_fail", "failed", "qc_failed"])
    )
    suspicious_keys = tuple(
        str(s) for s in cfg.get("qc_suspicious_keys", ["suspicious", "needs_review"])
    )
    qc_downweight = float(cfg.get("qc_downweight_factor", 0.7))
    qc_per_rater_penalty = float(cfg.get("qc_per_rater_penalty", 0.1))
    qc_min_weight = float(cfg.get("qc_min_weight", 0.2))

    rows: list[dict] = []
    n_out_of_scope = 0
    n_hard_fail = 0
    n_suspicious = 0

    for i, item in enumerate(dataset.items):
        image_subdir = str(item.get("image_subdir", ""))
        if image_subdir not in supervised_subdirs:
            n_out_of_scope += 1
            continue

        has_cancer, has_grade5 = _case_flags_from_hard_mask(Path(item["hard_path"]))
        qc = _safe_read_json(Path(item["qc_path"]))
        qc_fail, qc_suspicious = _infer_qc_flags(qc, fail_keys=fail_keys, suspicious_keys=suspicious_keys)

        if qc_fail:
            n_hard_fail += 1
        if qc_suspicious:
            n_suspicious += 1

        qc_weight = 1.0
        if qc_policy in {"warn_downweight", "strict_skip"}:
            qc_weight = _infer_qc_weight(
                qc,
                base_downweight=qc_downweight,
                per_rater_penalty=qc_per_rater_penalty,
                min_weight=qc_min_weight,
            )

        if qc_policy == "strict_skip" and qc_fail:
            continue

        rows.append(
            {
                "dataset_index": i,
                "image_id": str(item["image_id"]),
                "image_subdir": image_subdir,
                "has_cancer": has_cancer,
                "has_grade5": has_grade5,
                "qc_fail": qc_fail,
                "qc_suspicious": qc_suspicious,
                "qc_weight": qc_weight,
            }
        )

    logger.info(
        "Consensus supervised pool: %d included | out_of_scope=%d | qc_hard_fail=%d | qc_suspicious=%d | policy=%s",
        len(rows),
        n_out_of_scope,
        n_hard_fail,
        n_suspicious,
        qc_policy,
    )

    if not rows:
        raise RuntimeError("No supervised consensus samples available after QC/scope filtering.")

    return rows


def _stratify_key(row: dict) -> str:
    return f"cancer={int(bool(row['has_cancer']))}|grade5={int(bool(row['has_grade5']))}"


def _split_two_way_stratified(
    rows: list[dict],
    right_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < right_fraction < 1.0:
        raise ValueError(f"right_fraction must be in (0, 1), got {right_fraction}")

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(_stratify_key(r), []).append(r)

    rng = np.random.default_rng(seed)
    left: list[dict] = []
    right: list[dict] = []

    for key in sorted(grouped):
        grp = grouped[key]
        rng.shuffle(grp)
        n = len(grp)
        if n == 1:
            left.extend(grp)
            continue

        n_right = int(round(n * right_fraction))
        n_right = max(1, min(n - 1, n_right))

        right.extend(grp[:n_right])
        left.extend(grp[n_right:])

    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def _build_split_rows(
    rows: list[dict],
    split_mode: str,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    mode = split_mode.strip().lower()
    if mode not in {"iter_80_20", "final_80_10_10"}:
        raise ValueError(
            "split_mode must be one of {'iter_80_20', 'final_80_10_10'}, "
            f"got {split_mode!r}"
        )

    if mode == "iter_80_20":
        train_rows, val_rows = _split_two_way_stratified(rows, right_fraction=0.2, seed=seed)
        test_rows: list[dict] = []
    else:
        train_rows, holdout_rows = _split_two_way_stratified(rows, right_fraction=0.2, seed=seed)
        val_rows, test_rows = _split_two_way_stratified(holdout_rows, right_fraction=0.5, seed=seed + 1)

    if not train_rows or not val_rows:
        raise RuntimeError(
            f"Split produced empty subset(s): train={len(train_rows)} val={len(val_rows)} "
            f"test={len(test_rows)}"
        )

    train_ids = {r["image_id"] for r in train_rows}
    val_ids = {r["image_id"] for r in val_rows}
    test_ids = {r["image_id"] for r in test_rows}

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Split overlap detected across train/val/test.")

    return train_rows, val_rows, test_rows


def _write_split_manifest(
    path: Path,
    split_mode: str,
    seed: int,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    def _summ(rows: list[dict]) -> dict[str, int]:
        return {
            "n": len(rows),
            "n_cancer": int(sum(1 for r in rows if r["has_cancer"])),
            "n_grade5": int(sum(1 for r in rows if r["has_grade5"])),
            "n_qc_suspicious": int(sum(1 for r in rows if r["qc_suspicious"])),
            "n_qc_fail": int(sum(1 for r in rows if r["qc_fail"])),
        }

    manifest = {
        "version": 1,
        "split_mode": split_mode,
        "seed": int(seed),
        "summary": {
            "train": _summ(train_rows),
            "val": _summ(val_rows),
            "test": _summ(test_rows),
        },
        "train_image_ids": [str(r["image_id"]) for r in train_rows],
        "val_image_ids": [str(r["image_id"]) for r in val_rows],
        "test_image_ids": [str(r["image_id"]) for r in test_rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _resize_targets_for_logits(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial = logits.shape[2:]
    if hard_mask.shape[1:] == spatial:
        return hard_mask, soft_probs, ignore_mask

    hard_rs = F.interpolate(
        hard_mask.unsqueeze(1).float(),
        size=spatial,
        mode="nearest",
    ).squeeze(1).long()
    ignore_rs = F.interpolate(
        ignore_mask.unsqueeze(1).float(),
        size=spatial,
        mode="nearest",
    ).squeeze(1).to(ignore_mask.dtype)

    soft_rs = F.interpolate(
        soft_probs.float(),
        size=spatial,
        mode="bilinear",
        align_corners=False,
    )
    soft_rs = torch.nan_to_num(soft_rs, nan=0.0, posinf=1.0, neginf=0.0)
    soft_rs = torch.clamp(soft_rs, min=0.0)
    soft_sum = torch.clamp(soft_rs.sum(dim=1, keepdim=True), min=1e-8)
    soft_rs = soft_rs / soft_sum
    return hard_rs, soft_rs, ignore_rs


def _make_valid_mask(
    ignore_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
) -> torch.Tensor:
    valid = ignore_mask == 0
    if use_confidence_mask:
        conf = soft_probs.max(dim=1).values
        valid = valid & (conf >= confidence_threshold)
    return valid


def _soft_loss_map(
    logits: torch.Tensor,
    soft_probs: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    log_p = F.log_softmax(logits.float(), dim=1)
    if loss_type == "kl":
        return F.kl_div(log_p, soft_probs.float(), reduction="none").sum(dim=1)
    # cross-entropy with soft targets
    return -(soft_probs.float() * log_p).sum(dim=1)


def _hard_dice_per_class(
    probs: torch.Tensor,
    hard_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    target = F.one_hot(hard_mask.long().clamp(0, num_classes - 1), num_classes=num_classes)
    target = target.permute(0, 3, 1, 2).float()
    valid = valid_mask.unsqueeze(1).float()

    p = probs * valid
    t = target * valid

    intersection = (p * t).sum(dim=(0, 2, 3))
    denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return dice


def _single_scale_loss(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    hard_rs, soft_rs, ignore_rs = _resize_targets_for_logits(
        logits=logits,
        hard_mask=hard_mask,
        soft_probs=soft_probs,
        ignore_mask=ignore_mask,
    )

    valid_mask = _make_valid_mask(
        ignore_mask=ignore_rs,
        soft_probs=soft_rs,
        use_confidence_mask=use_confidence_mask,
        confidence_threshold=confidence_threshold,
    )

    soft_map = _soft_loss_map(logits, soft_rs, loss_type=soft_loss_type)
    expected_cls_weight = (soft_rs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
    soft_map = soft_map * expected_cls_weight

    pixel_weight = sample_weights.view(-1, 1, 1)
    valid_float = valid_mask.float()
    soft_num = (soft_map * valid_float * pixel_weight).sum()
    soft_den = (valid_float * pixel_weight).sum().clamp_min(1e-8)
    soft_loss = soft_num / soft_den

    probs = F.softmax(logits.float(), dim=1)
    dice_c = _hard_dice_per_class(
        probs=probs,
        hard_mask=hard_rs,
        valid_mask=valid_mask,
        num_classes=logits.shape[1],
    )

    if include_background_in_dice:
        dice_used = dice_c
    else:
        dice_used = dice_c[1:]

    hard_dice_loss = 1.0 - dice_used.mean()
    total = (lambda_soft * soft_loss) + (lambda_dice * hard_dice_loss)

    with torch.no_grad():
        stats = {
            "soft_loss": float(soft_loss.detach().cpu().item()),
            "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
            "valid_fraction": float(valid_mask.float().mean().detach().cpu().item()),
        }
    return total, stats


def _consensus_loss(
    outputs: list[torch.Tensor] | torch.Tensor,
    hard_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    ignore_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    class_weights: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(outputs, torch.Tensor):
        return _single_scale_loss(
            logits=outputs,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
        )

    raw = [1.0 / (2 ** i) for i in range(len(outputs))]
    total_w = sum(raw)
    weights = [w / total_w for w in raw]

    total_loss = torch.zeros((), device=outputs[0].device, dtype=torch.float32)
    soft_acc = 0.0
    dice_acc = 0.0
    valid_acc = 0.0

    for out, w in zip(outputs, weights):
        l, stats = _single_scale_loss(
            logits=out,
            hard_mask=hard_mask,
            soft_probs=soft_probs,
            ignore_mask=ignore_mask,
            sample_weights=sample_weights,
            class_weights=class_weights,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
        )
        total_loss = total_loss + (w * l)
        soft_acc += w * stats["soft_loss"]
        dice_acc += w * stats["hard_dice_loss"]
        valid_acc += w * stats["valid_fraction"]

    return total_loss, {
        "soft_loss": soft_acc,
        "hard_dice_loss": dice_acc,
        "valid_fraction": valid_acc,
    }


def _compute_val_metrics(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    include_background_in_dice: bool,
) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    valid = ignore_mask == 0

    num_classes = logits.shape[1]
    dice_values: list[float] = []
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (hard_mask == c) & valid
        denom = p.sum().item() + t.sum().item()
        if denom == 0:
            dice_values.append(float("nan"))
        else:
            inter = (p & t).sum().item()
            dice_values.append((2.0 * inter + 1e-5) / (denom + 1e-5))

    start = 0 if include_background_in_dice else 1
    used = [d for d in dice_values[start:] if not math.isnan(d)]
    macro_dice = float(sum(used) / len(used)) if used else float("nan")

    grade5_dice = dice_values[3] if len(dice_values) > 3 else float("nan")

    p_pos = (pred > 0) & valid
    t_pos = (hard_mask > 0) & valid
    tp = float((p_pos & t_pos).sum().item())
    fn = float((~p_pos & t_pos).sum().item())
    fp = float((p_pos & ~t_pos).sum().item())

    sens = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else float("nan")
    prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else float("nan")

    return {
        "macro_dice": macro_dice,
        "grade5_dice": grade5_dice,
        "sensitivity": sens,
        "precision": prec,
    }


def _pad_to_hw(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    value: float | int,
) -> torch.Tensor:
    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return x
    # F.pad uses (left, right, top, bottom) for 2D trailing dims.
    return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=value)


def _collate_consensus_batch(batch: list[dict]) -> dict:
    if not batch:
        raise RuntimeError("Empty batch in collate function.")

    max_h = max(int(s["image"].shape[-2]) for s in batch)
    max_w = max(int(s["image"].shape[-1]) for s in batch)

    images: list[torch.Tensor] = []
    soft_probs: list[torch.Tensor] = []
    hard_masks: list[torch.Tensor] = []
    ignore_masks: list[torch.Tensor] = []
    tissue_masks: list[torch.Tensor] = []
    image_ids: list[str] = []

    for s in batch:
        images.append(_pad_to_hw(s["image"], max_h, max_w, value=0.0))
        soft_probs.append(_pad_to_hw(s["soft_probs"], max_h, max_w, value=0.0))
        hard_masks.append(_pad_to_hw(s["hard_mask"], max_h, max_w, value=0))
        # Padded pixels must be ignored by loss/metrics.
        ignore_masks.append(_pad_to_hw(s["ignore_mask"], max_h, max_w, value=1))
        tissue_masks.append(_pad_to_hw(s["tissue_mask"], max_h, max_w, value=0))
        image_ids.append(str(s["image_id"]))

    return {
        "image": torch.stack(images, dim=0),
        "soft_probs": torch.stack(soft_probs, dim=0),
        "hard_mask": torch.stack(hard_masks, dim=0),
        "ignore_mask": torch.stack(ignore_masks, dim=0),
        "tissue_mask": torch.stack(tissue_masks, dim=0),
        "image_id": image_ids,
    }


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    image_weight_map: dict[str, float],
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()

    loss_sum = 0.0
    n_batches = 0

    sums = {
        "macro_dice": 0.0,
        "grade5_dice": 0.0,
        "sensitivity": 0.0,
        "precision": 0.0,
    }
    counts = {k: 0 for k in sums}

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Val", leave=False, unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            soft_probs = batch["soft_probs"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            image_ids = [str(x) for x in batch["image_id"]]
            sample_w = torch.tensor(
                [float(image_weight_map.get(i, 1.0)) for i in image_ids],
                device=device,
                dtype=torch.float32,
            )

            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=use_amp):
                out = model(images)
                loss, _ = _consensus_loss(
                    outputs=out,
                    hard_mask=hard_mask,
                    soft_probs=soft_probs,
                    ignore_mask=ignore_mask,
                    sample_weights=sample_w,
                    class_weights=class_weights,
                    use_confidence_mask=use_confidence_mask,
                    confidence_threshold=confidence_threshold,
                    soft_loss_type=soft_loss_type,
                    lambda_soft=lambda_soft,
                    lambda_dice=lambda_dice,
                    include_background_in_dice=include_background_in_dice,
                )

            logits = out[0] if isinstance(out, list) else out
            hard_rs, _, ignore_rs = _resize_targets_for_logits(
                logits=logits,
                hard_mask=hard_mask,
                soft_probs=soft_probs,
                ignore_mask=ignore_mask,
            )
            m = _compute_val_metrics(
                logits=logits.float(),
                hard_mask=hard_rs,
                ignore_mask=ignore_rs,
                include_background_in_dice=include_background_in_dice,
            )

            loss_sum += float(loss.detach().cpu().item())
            n_batches += 1
            for k in sums:
                if not math.isnan(m[k]):
                    sums[k] += m[k]
                    counts[k] += 1

    out_metrics = {
        "val_loss": (loss_sum / max(1, n_batches)),
    }
    for k in sums:
        out_metrics[k] = (sums[k] / counts[k]) if counts[k] > 0 else float("nan")
    return out_metrics


def validate_with_oom_retry(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    image_weight_map: dict[str, float],
    use_confidence_mask: bool,
    confidence_threshold: float,
    soft_loss_type: str,
    lambda_soft: float,
    lambda_dice: float,
    include_background_in_dice: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    try:
        return validate(
            model=model,
            loader=loader,
            device=device,
            class_weights=class_weights,
            image_weight_map=image_weight_map,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
    except RuntimeError as exc:
        if device.type == "cuda" and _is_cuda_oom(exc):
            logger.warning("CUDA OOM during validation. Clearing cache and retrying once.")
            gc.collect()
            torch.cuda.empty_cache()
            return validate(
                model=model,
                loader=loader,
                device=device,
                class_weights=class_weights,
                image_weight_map=image_weight_map,
                use_confidence_mask=use_confidence_mask,
                confidence_threshold=confidence_threshold,
                soft_loss_type=soft_loss_type,
                lambda_soft=lambda_soft,
                lambda_dice=lambda_dice,
                include_background_in_dice=include_background_in_dice,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2D Deconver model on Gleason consensus labels.",
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
        help="Regenerate split manifest before this run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    cfg_model = str(cfg.get("model", "deconver")).lower()
    if cfg_model != "deconver":
        raise ValueError(f"train_deconver_2d requires model='deconver', got {cfg_model!r}")
    spatial_dims = int(cfg.get("spatial_dims", 2))
    if spatial_dims != 2:
        raise ValueError(f"train_deconver_2d requires spatial_dims=2, got {spatial_dims}")

    # Requested class mapping: 0=benign, 1=G3, 2=G4, 3=G5.
    out_channels = int(cfg.get("out_channels", 4))
    if out_channels != 4:
        raise ValueError(
            f"This consensus trainer requires out_channels=4, got {out_channels}."
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

    max_long_side = int(cfg.get("max_long_side", 0))
    deconver_strides = tuple(int(x) for x in cfg.get("deconver_strides", [1, 2, 2, 2]))
    resize_divisor = int(math.prod([s for s in deconver_strides if s > 1])) or 1

    dataset = GleasonConsensusDataset(
        data_root=cfg["data_root"],
        consensus_root=cfg["consensus_root"],
        image_subdirs=tuple(str(x) for x in cfg.get("image_subdirs", ["Trains_imgs", "Test_imgs"])),
        transform=None,
        renormalize_probs=bool(cfg.get("renormalize_probs", True)),
        enforce_background_ignore=bool(cfg.get("enforce_background_ignore", True)),
        otsu_close_radius=int(cfg.get("otsu_close_radius", 3)),
        otsu_min_object_size=int(cfg.get("otsu_min_object_size", 4096)),
        otsu_min_hole_size=int(cfg.get("otsu_min_hole_size", 4096)),
        probs_eps=float(cfg.get("probs_eps", 1e-8)),
        load_qc_report=False,
        max_long_side=max_long_side or None,
        resize_divisor=resize_divisor,
    )
    if max_long_side > 0:
        logger.info(
            "Consensus loader resize enabled: max_long_side=%d px (divisor=%d from strides=%s)",
            max_long_side,
            resize_divisor,
            list(deconver_strides),
        )

    all_rows = _build_sample_metadata(dataset=dataset, cfg=cfg)

    split_mode = str(cfg.get("split_mode", "iter_80_20"))
    split_manifest_cfg = str(cfg.get("split_manifest_path", "")).strip()
    split_manifest_path = (
        Path(split_manifest_cfg)
        if split_manifest_cfg
        else Path(cfg["base_output_dir"]).parent / "splits" / "gleason_consensus_split.json"
    )

    if args.new_split_manifest or not split_manifest_path.exists():
        train_rows, val_rows, test_rows = _build_split_rows(
            rows=all_rows,
            split_mode=split_mode,
            seed=seed,
        )
        _write_split_manifest(
            path=split_manifest_path,
            split_mode=split_mode,
            seed=seed,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
        )
        logger.info("Wrote split manifest to %s", split_manifest_path)
    else:
        manifest = _safe_read_json(split_manifest_path)
        train_ids = set(str(x) for x in manifest.get("train_image_ids", []))
        val_ids = set(str(x) for x in manifest.get("val_image_ids", []))
        test_ids = set(str(x) for x in manifest.get("test_image_ids", []))

        def _pick(ids: set[str]) -> list[dict]:
            return [r for r in all_rows if str(r["image_id"]) in ids]

        train_rows = _pick(train_ids)
        val_rows = _pick(val_ids)
        test_rows = _pick(test_ids)

        if not train_rows or not val_rows:
            raise RuntimeError(
                "Existing split manifest does not match discovered samples. "
                "Pass --new-split-manifest to regenerate."
            )
        logger.info("Using existing split manifest at %s", split_manifest_path)

    split_manifest_copy_path = run_dir / "train_val_split_manifest.json"
    shutil.copy2(split_manifest_path, split_manifest_copy_path)

    train_indices = [int(r["dataset_index"]) for r in train_rows]
    val_indices = [int(r["dataset_index"]) for r in val_rows]
    test_indices = [int(r["dataset_index"]) for r in test_rows]

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices) if test_indices else None

    image_weight_map = {
        str(r["image_id"]): float(r.get("qc_weight", 1.0)) for r in all_rows
    }

    num_workers_cfg = int(cfg.get("num_workers", 0))
    num_workers, mp_context, start_method = _resolve_dataloader_context(
        cfg=cfg,
        num_workers=num_workers_cfg,
    )
    use_persistent = num_workers > 0
    if num_workers != num_workers_cfg:
        logger.warning(
            "Reduced num_workers from %d to %d due to DataLoader context constraints.",
            num_workers_cfg,
            num_workers,
        )
    logger.info(
        "DataLoader workers=%d (requested=%d), start_method=%s",
        num_workers,
        num_workers_cfg,
        start_method,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
        multiprocessing_context=mp_context if use_persistent else None,
        collate_fn=_collate_consensus_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("val_batch_size", 4)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        prefetch_factor=2 if use_persistent else None,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
        multiprocessing_context=mp_context if use_persistent else None,
        collate_fn=_collate_consensus_batch,
    )
    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg.get("val_batch_size", 4)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=use_persistent,
            prefetch_factor=2 if use_persistent else None,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            multiprocessing_context=mp_context if use_persistent else None,
            collate_fn=_collate_consensus_batch,
        )

    logger.info(
        "Split mode=%s | train=%d | val=%d | test=%d",
        split_mode,
        len(train_ds),
        len(val_ds),
        len(test_ds) if test_ds is not None else 0,
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("deconver_2d_consensus | trainable parameters: %s", f"{n_params:,}")

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
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
    epochs = int(cfg.get("epochs", 100))
    cosine_t_max = max(1, epochs - warmup_epochs)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_t_max,
            eta_min=float(cfg.get("learning_rate", 2e-4)) * 1e-2,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(cfg.get("learning_rate", 2e-4)) * 1e-2,
        )

    class_weights_cfg = cfg.get("class_weights", None)
    if class_weights_cfg is not None:
        if not isinstance(class_weights_cfg, list) or len(class_weights_cfg) != 4:
            raise ValueError("class_weights must be a list of 4 floats [w0,w1,w2,w3].")
        class_weights = torch.tensor([float(x) for x in class_weights_cfg], dtype=torch.float32, device=device)
    else:
        grade5_boost = float(cfg.get("grade5_boost", 1.0))
        class_weights = torch.tensor([1.0, 1.0, 1.0, grade5_boost], dtype=torch.float32, device=device)

    lambda_soft = float(cfg.get("lambda_soft", 1.0))
    lambda_dice = float(cfg.get("lambda_dice", 1.0))
    soft_loss_type = str(cfg.get("soft_label_loss", "ce")).strip().lower()
    if soft_loss_type not in {"ce", "kl"}:
        raise ValueError(f"soft_label_loss must be 'ce' or 'kl', got {soft_loss_type!r}")

    use_confidence_mask = bool(cfg.get("use_confidence_mask", False))
    confidence_threshold = float(cfg.get("confidence_threshold", 0.6))
    include_background_in_dice = bool(cfg.get("include_background_in_dice", False))

    amp_dtype_str = str(cfg.get("amp_dtype", "bf16")).lower()
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

    best_metric = float("-inf")
    best_val_macro_dice = float("nan")
    start_epoch = 1

    es_patience = int(cfg.get("early_stopping_patience", 30))
    es_min_delta = float(cfg.get("early_stopping_min_delta", 0.0005))
    es_enabled = es_patience > 0
    es_counter = 0

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
        best_metric = float(ckpt.get("best_composite_score", float("-inf")))
        best_val_d = ckpt.get("best_val_dice", float("nan"))
        best_val_macro_dice = float(best_val_d) if best_val_d is not None else float("nan")
        logger.info(
            "Resuming from epoch %d (best_metric=%s, best_val_macro_dice=%s) -> epoch %d",
            ckpt["epoch"],
            _fmt(best_metric),
            _fmt(best_val_macro_dice),
            start_epoch,
        )

    logger.info("Device: %s", device)
    logger.info("AMP (%s): %s", amp_dtype_str.upper(), use_amp)
    if str(cfg.get("model", "")).strip().lower() == "deconver":
        fp32_islands = bool(cfg.get("deconver_fp32_islands", False))
        fp32_scope = str(cfg.get("deconver_fp32_scope", "update_only")).strip().lower()
        logger.info(
            "Deconver FP32 islands: %s (scope=%s)",
            fp32_islands,
            fp32_scope,
        )
    logger.info("torch.compile: %s", cfg.get("use_compile", False))
    logger.info("Experiment: %s", cfg["experiment_name"])
    logger.info("Run directory: %s", run_dir)
    logger.info(
        "Loss setup: soft=%s (lambda=%.3f), hard_dice(lambda=%.3f), confidence_mask=%s(th=%.2f)",
        soft_loss_type,
        lambda_soft,
        lambda_dice,
        use_confidence_mask,
        confidence_threshold,
    )
    logger.info("Class weights: %s", [float(x) for x in class_weights.detach().cpu().tolist()])

    send_ntfy(
        cfg,
        title=f"Training started: {cfg['experiment_name']}",
        message=(
            f"Epochs: {epochs}\n"
            f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds) if test_ds is not None else 0}\n"
            f"Split mode: {split_mode}\n"
            f"Device: {device} | AMP: {amp_dtype_str.upper() if use_amp else 'off'}\n"
            f"Run dir: {run_dir}"
        ),
        tags=["rocket"],
        priority="default",
    )

    max_nan_batches = 10
    nan_batch_count = 0

    val_every = max(1, int(cfg.get("val_every", 1)))
    val_start_epoch = max(1, int(cfg.get("val_start_epoch", 1)))
    keep_last_n = int(cfg.get("keep_last_checkpoints", 3))

    w_macro = float(cfg.get("best_ckpt_w_macro_dice", 0.6))
    w_sens = float(cfg.get("best_ckpt_w_sensitivity", 0.4))

    for epoch in tqdm(range(start_epoch, epochs + 1), desc="Epochs", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        epoch_soft = 0.0
        epoch_hard = 0.0
        epoch_valid_frac = 0.0

        batch_bar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False, unit="batch")
        for step, batch in enumerate(batch_bar, start=1):
            images = batch["image"].to(device, non_blocking=True)
            soft_probs = batch["soft_probs"].to(device, non_blocking=True)
            hard_mask = batch["hard_mask"].to(device, non_blocking=True)
            ignore_mask = batch["ignore_mask"].to(device, non_blocking=True)
            image_ids = [str(x) for x in batch["image_id"]]
            sample_w = torch.tensor(
                [float(image_weight_map.get(i, 1.0)) for i in image_ids],
                device=device,
                dtype=torch.float32,
            )

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                out = model(images)
                if isinstance(out, list):
                    out = [o.clamp(-15.0, 15.0) for o in out]
                else:
                    out = out.clamp(-15.0, 15.0)

                loss, stats = _consensus_loss(
                    outputs=out,
                    hard_mask=hard_mask,
                    soft_probs=soft_probs,
                    ignore_mask=ignore_mask,
                    sample_weights=sample_w,
                    class_weights=class_weights,
                    use_confidence_mask=use_confidence_mask,
                    confidence_threshold=confidence_threshold,
                    soft_loss_type=soft_loss_type,
                    lambda_soft=lambda_soft,
                    lambda_dice=lambda_dice,
                    include_background_in_dice=include_background_in_dice,
                )

            if not torch.isfinite(loss):
                loss_value = float(loss.detach().cpu().item())
                nan_batch_count += 1
                logger.warning(
                    "Non-finite loss at epoch %d, batch %d: %s (consecutive=%d/%d).",
                    epoch,
                    step,
                    loss_value,
                    nan_batch_count,
                    max_nan_batches,
                )
                optimizer.zero_grad(set_to_none=True)
                del out, images, soft_probs, hard_mask, ignore_mask, sample_w, loss
                if nan_batch_count >= max_nan_batches:
                    raise FloatingPointError(
                        f"Non-finite loss for {max_nan_batches} consecutive batches "
                        f"(epoch={epoch}, batch={step})."
                    )
                continue
            nan_batch_count = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_item = float(loss.detach().cpu().item())
            epoch_loss += loss_item
            epoch_soft += float(stats["soft_loss"])
            epoch_hard += float(stats["hard_dice_loss"])
            epoch_valid_frac += float(stats["valid_fraction"])
            batch_bar.set_postfix(
                loss=f"{loss_item:.4f}",
                soft=f"{stats['soft_loss']:.4f}",
                dice=f"{stats['hard_dice_loss']:.4f}",
            )

            del out, images, soft_probs, hard_mask, ignore_mask, sample_w, loss

        scheduler.step()

        nb = max(1, len(train_loader))
        avg_loss = epoch_loss / nb
        avg_soft = epoch_soft / nb
        avg_hard = epoch_hard / nb
        avg_valid = epoch_valid_frac / nb
        current_lr = scheduler.get_last_lr()[0]

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/soft_loss", avg_soft, epoch)
        writer.add_scalar("train/hard_dice_loss", avg_hard, epoch)
        writer.add_scalar("train/valid_pixel_fraction", avg_valid, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        logger.info(
            "Epoch %d/%d | loss=%.4f soft=%.4f hard_dice=%.4f valid_px=%.3f lr=%.2e",
            epoch,
            epochs,
            avg_loss,
            avg_soft,
            avg_hard,
            avg_valid,
            current_lr,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            str(checkpoint_dir / f"epoch_{epoch:04d}.pt"),
            scheduler=scheduler,
            scaler=scaler,
            best_val_dice=best_val_macro_dice,
            best_composite_score=best_metric,
            last_hd95=float("nan"),
        )
        rotate_checkpoints(checkpoint_dir, keep_last_n)

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after train", device)

        run_val = (epoch >= val_start_epoch) and ((epoch % val_every == 0) or (epoch == epochs))
        if not run_val:
            logger.info("Epoch %d/%d | validation skipped (val_every=%d)", epoch, epochs, val_every)
            continue

        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        val_metrics = validate_with_oom_retry(
            model=model,
            loader=val_loader,
            device=device,
            class_weights=class_weights,
            image_weight_map=image_weight_map,
            use_confidence_mask=use_confidence_mask,
            confidence_threshold=confidence_threshold,
            soft_loss_type=soft_loss_type,
            lambda_soft=lambda_soft,
            lambda_dice=lambda_dice,
            include_background_in_dice=include_background_in_dice,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        writer.add_scalar("val/macro_dice", val_metrics["macro_dice"], epoch)
        writer.add_scalar("val/grade5_dice", val_metrics["grade5_dice"], epoch)
        writer.add_scalar("val/sensitivity", val_metrics["sensitivity"], epoch)
        writer.add_scalar("val/precision", val_metrics["precision"], epoch)

        logger.info(
            "Epoch %d/%d | val_loss=%s | val_macro_dice=%s | val_grade5_dice=%s | "
            "val_sens=%s | val_prec=%s",
            epoch,
            epochs,
            _fmt(val_metrics["val_loss"]),
            _fmt(val_metrics["macro_dice"]),
            _fmt(val_metrics["grade5_dice"]),
            _fmt(val_metrics["sensitivity"]),
            _fmt(val_metrics["precision"]),
        )

        if device.type == "cuda":
            _log_cuda_memory(f"Epoch {epoch} after val", device)

        macro = val_metrics["macro_dice"]
        sens = val_metrics["sensitivity"]
        if math.isnan(macro) and math.isnan(sens):
            composite = float("nan")
        else:
            macro0 = 0.0 if math.isnan(macro) else macro
            sens0 = 0.0 if math.isnan(sens) else sens
            composite = (w_macro * macro0) + (w_sens * sens0)

        if not math.isnan(composite):
            writer.add_scalar("val/composite_score", composite, epoch)
            logger.info(
                "Epoch %d/%d | composite_score=%.4f (best=%s)",
                epoch,
                epochs,
                composite,
                _fmt(best_metric),
            )

        if not math.isnan(composite) and composite > best_metric + es_min_delta:
            best_metric = composite
            best_val_macro_dice = macro
            save_checkpoint(
                model,
                optimizer,
                epoch,
                str(checkpoint_dir / "best.pt"),
                scheduler=scheduler,
                scaler=scaler,
                best_val_dice=best_val_macro_dice,
                best_composite_score=best_metric,
                last_hd95=float("nan"),
            )
            logger.info(
                "New best model at epoch %d (composite=%.4f, val_macro_dice=%s) -> %s",
                epoch,
                best_metric,
                _fmt(best_val_macro_dice),
                checkpoint_dir / "best.pt",
            )
            if cfg.get("ntfy_notify_best_model", True):
                send_ntfy(
                    cfg,
                    title=f"New best model: {cfg['experiment_name']}",
                    message=(
                        f"Epoch {epoch}/{epochs}\n"
                        f"Composite: {best_metric:.4f}\n"
                        f"Macro Dice: {_fmt(val_metrics['macro_dice'])}\n"
                        f"Grade5 Dice: {_fmt(val_metrics['grade5_dice'])}\n"
                        f"Sensitivity: {_fmt(val_metrics['sensitivity'])}\n"
                        f"Precision: {_fmt(val_metrics['precision'])}"
                    ),
                    tags=["trophy"],
                    priority="default",
                )
            es_counter = 0
        else:
            if es_enabled and not math.isnan(composite):
                es_counter += 1
                logger.info("Early stopping counter: %d / %d", es_counter, es_patience)

        if es_enabled and es_counter >= es_patience:
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d, min_delta=%.4f).",
                epoch,
                es_patience,
                es_min_delta,
            )
            send_ntfy(
                cfg,
                title=f"Training stopped early: {cfg['experiment_name']}",
                message=(
                    f"Early stopping at epoch {epoch}/{epochs}\n"
                    f"No improvement for {es_patience} validation checks\n"
                    f"Best composite: {best_metric:.4f}\n"
                    f"Best val macro dice: {_fmt(best_val_macro_dice)}"
                ),
                tags=["warning"],
                priority="high",
            )
            break

    writer.close()
    logger.info("Training complete.")
    logger.info("Best composite score: %s", _fmt(best_metric))
    logger.info("Best validation macro Dice: %s", _fmt(best_val_macro_dice))
    logger.info("Artifacts saved to: %s", run_dir)

    if test_loader is not None and len(test_loader) > 0:
        logger.info(
            "Test split present (%d samples). Supervised metrics are valid only if this split is from labeled Trains_imgs.",
            len(test_loader.dataset),
        )

    send_ntfy(
        cfg,
        title=f"Training complete: {cfg['experiment_name']}",
        message=(
            f"Best composite score: {_fmt(best_metric)}\n"
            f"Best val macro dice: {_fmt(best_val_macro_dice)}\n"
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
