from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROI_DISABLED = "disabled"
ROI_GT_MASK = "gt_mask"
ROI_PREDICTED_MASK = "predicted_mask"
ROI_MODES = {ROI_DISABLED, ROI_GT_MASK, ROI_PREDICTED_MASK}
TASK_LESION_SEGMENTATION = "lesion_segmentation"
TASK_PROSTATE_LOCALIZATION = "prostate_localization"
TASKS = {TASK_LESION_SEGMENTATION, TASK_PROSTATE_LOCALIZATION}


@dataclass(frozen=True)
class ROISettings:
    mode: str = ROI_DISABLED
    target: str = "prostate"
    margin_mm: tuple[float, float, float] = (3.0, 6.0, 6.0)
    min_size_vox: tuple[int, int, int] = (16, 128, 128)
    fallback_to_full_volume: bool = True
    gt_dataset_types: tuple[str, ...] = ("prostate158",)
    localizer_run: str = ""
    localizer_threshold: float = 0.5
    localizer_keep_largest_component: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != ROI_DISABLED


@dataclass(frozen=True)
class CropBounds:
    start_zyx: tuple[int, int, int]
    end_zyx: tuple[int, int, int]
    full_shape_zyx: tuple[int, int, int]
    used_fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_zyx": [int(v) for v in self.start_zyx],
            "end_zyx": [int(v) for v in self.end_zyx],
            "full_shape_zyx": [int(v) for v in self.full_shape_zyx],
            "used_fallback": bool(self.used_fallback),
        }


def resolve_task(cfg: Mapping[str, Any]) -> str:
    task = str(cfg.get("task", TASK_LESION_SEGMENTATION)).strip().lower()
    if task not in TASKS:
        supported = ", ".join(sorted(TASKS))
        raise ValueError(f"Unsupported task='{task}'. Expected one of: {supported}.")
    return task


def resolve_roi_settings(cfg: Mapping[str, Any]) -> ROISettings:
    roi_cfg = cfg.get("roi", {}) or {}
    if not isinstance(roi_cfg, Mapping):
        raise ValueError("Config key 'roi' must be a mapping when provided.")

    mode = str(roi_cfg.get("mode", ROI_DISABLED)).strip().lower()
    if mode not in ROI_MODES:
        supported = ", ".join(sorted(ROI_MODES))
        raise ValueError(f"Unsupported roi.mode='{mode}'. Expected one of: {supported}.")

    target = str(roi_cfg.get("target", "prostate")).strip().lower()
    if target != "prostate":
        raise ValueError(f"Unsupported roi.target='{target}'. Expected 'prostate'.")

    margin_mm = _coerce_triplet(roi_cfg.get("margin_mm", (3.0, 6.0, 6.0)), float, "roi.margin_mm")
    min_size_vox = _coerce_triplet(roi_cfg.get("min_size_vox", (16, 128, 128)), int, "roi.min_size_vox")
    gt_dataset_types = tuple(
        str(v).strip().lower()
        for v in roi_cfg.get("gt_dataset_types", ["prostate158"])
    )

    settings = ROISettings(
        mode=mode,
        target=target,
        margin_mm=margin_mm,
        min_size_vox=min_size_vox,
        fallback_to_full_volume=bool(roi_cfg.get("fallback_to_full_volume", True)),
        gt_dataset_types=gt_dataset_types,
        localizer_run=str(roi_cfg.get("localizer_run", "")).strip(),
        localizer_threshold=float(roi_cfg.get("localizer_threshold", 0.5)),
        localizer_keep_largest_component=bool(
            roi_cfg.get("localizer_keep_largest_component", True)
        ),
    )

    if not (0.0 <= settings.localizer_threshold <= 1.0):
        raise ValueError(
            f"roi.localizer_threshold must be in [0, 1], got {settings.localizer_threshold}."
        )
    if any(v <= 0 for v in settings.min_size_vox):
        raise ValueError(f"roi.min_size_vox must contain positive integers, got {settings.min_size_vox}.")
    if any(v < 0 for v in settings.margin_mm):
        raise ValueError(f"roi.margin_mm must contain non-negative values, got {settings.margin_mm}.")

    return settings


def validate_task_and_roi_config(
    cfg: Mapping[str, Any],
    dataset_type: str,
) -> tuple[str, ROISettings]:
    task = resolve_task(cfg)
    roi = resolve_roi_settings(cfg)
    dataset_type_norm = str(dataset_type).strip().lower()

    if task == TASK_PROSTATE_LOCALIZATION:
        if dataset_type_norm != "prostate158":
            raise ValueError(
                "task='prostate_localization' currently requires dataset_type='prostate158'."
            )
        prostate_label_col = str(cfg.get("prostate158_prostate_label_col", "")).strip()
        if not prostate_label_col:
            raise ValueError(
                "task='prostate_localization' requires 'prostate158_prostate_label_col'."
            )
        if roi.enabled:
            raise ValueError("ROI cropping is only supported for task='lesion_segmentation'.")

    if roi.mode == ROI_GT_MASK:
        if dataset_type_norm not in set(roi.gt_dataset_types):
            supported = ", ".join(roi.gt_dataset_types)
            raise ValueError(
                f"roi.mode='gt_mask' requires dataset_type in {{{supported}}}, got '{dataset_type_norm}'."
            )
        prostate_label_col = str(cfg.get("prostate158_prostate_label_col", "")).strip()
        if not prostate_label_col:
            raise ValueError(
                "roi.mode='gt_mask' requires 'prostate158_prostate_label_col'."
            )
    elif roi.mode == ROI_PREDICTED_MASK:
        if not roi.localizer_run:
            raise ValueError("roi.mode='predicted_mask' requires roi.localizer_run.")

    return task, roi


def binarize_mask(mask: np.ndarray | torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected ROI mask with shape (D,H,W), got {arr.shape}.")
    return (arr >= threshold).astype(np.uint8, copy=False)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return mask.astype(np.uint8, copy=False)

    mask_bool = mask > 0
    visited = np.zeros(mask_bool.shape, dtype=bool)
    best_component: list[tuple[int, int, int]] = []

    neighbors = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for start in coords:
        z, y, x = (int(start[0]), int(start[1]), int(start[2]))
        if visited[z, y, x]:
            continue
        stack = [(z, y, x)]
        component: list[tuple[int, int, int]] = []
        visited[z, y, x] = True

        while stack:
            cz, cy, cx = stack.pop()
            component.append((cz, cy, cx))
            for dz, dy, dx in neighbors:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if (
                    0 <= nz < mask_bool.shape[0]
                    and 0 <= ny < mask_bool.shape[1]
                    and 0 <= nx < mask_bool.shape[2]
                    and mask_bool[nz, ny, nx]
                    and not visited[nz, ny, nx]
                ):
                    visited[nz, ny, nx] = True
                    stack.append((nz, ny, nx))

        if len(component) > len(best_component):
            best_component = component

    out = np.zeros(mask_bool.shape, dtype=np.uint8)
    for z, y, x in best_component:
        out[z, y, x] = 1
    return out


def compute_crop_bounds(
    mask: np.ndarray,
    spacing_zyx: Sequence[float],
    margin_mm: Sequence[float],
    min_size_vox: Sequence[int],
    fallback_to_full_volume: bool,
) -> CropBounds:
    mask = binarize_mask(mask, threshold=0.5)
    full_shape = tuple(int(v) for v in mask.shape)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        if fallback_to_full_volume:
            return CropBounds(
                start_zyx=(0, 0, 0),
                end_zyx=full_shape,
                full_shape_zyx=full_shape,
                used_fallback=True,
            )
        raise ValueError("ROI mask is empty.")

    mins = coords.min(axis=0).astype(int)
    maxs = coords.max(axis=0).astype(int) + 1
    spacing = np.asarray([float(v) for v in spacing_zyx], dtype=np.float32)
    margins = np.asarray([float(v) for v in margin_mm], dtype=np.float32)
    min_sizes = np.asarray([int(v) for v in min_size_vox], dtype=np.int32)
    full = np.asarray(full_shape, dtype=np.int32)

    margin_vox = np.ceil(margins / np.maximum(spacing, 1e-8)).astype(np.int32)
    start = mins - margin_vox
    end = maxs + margin_vox

    start = np.maximum(start, 0)
    end = np.minimum(end, full)

    size = end - start
    deficit = np.maximum(min_sizes - size, 0)
    start = start - deficit // 2
    end = end + deficit - deficit // 2

    if np.any(start < 0):
        end = np.minimum(end - start, full)
        start = np.maximum(start, 0)
    if np.any(end > full):
        shift = end - full
        start = np.maximum(start - shift, 0)
        end = np.minimum(end, full)

    start = np.maximum(start, 0)
    end = np.minimum(end, full)

    return CropBounds(
        start_zyx=tuple(int(v) for v in start.tolist()),
        end_zyx=tuple(int(v) for v in end.tolist()),
        full_shape_zyx=full_shape,
        used_fallback=False,
    )


def crop_tensor(
    tensor: torch.Tensor,
    bounds: Mapping[str, Any] | CropBounds,
) -> torch.Tensor:
    crop = bounds if isinstance(bounds, CropBounds) else crop_bounds_from_dict(bounds)
    slices = (
        slice(None),
        slice(crop.start_zyx[0], crop.end_zyx[0]),
        slice(crop.start_zyx[1], crop.end_zyx[1]),
        slice(crop.start_zyx[2], crop.end_zyx[2]),
    )
    return tensor[slices]


def restore_from_roi(
    roi_tensor: torch.Tensor,
    bounds: Mapping[str, Any] | CropBounds,
) -> torch.Tensor:
    crop = bounds if isinstance(bounds, CropBounds) else crop_bounds_from_dict(bounds)
    if roi_tensor.ndim != 5:
        raise ValueError(
            f"Expected ROI tensor with shape (B,C,D,H,W), got {tuple(roi_tensor.shape)}."
        )
    full_shape = crop.full_shape_zyx
    out = roi_tensor.new_zeros(
        (roi_tensor.shape[0], roi_tensor.shape[1], full_shape[0], full_shape[1], full_shape[2])
    )
    out[
        :,
        :,
        crop.start_zyx[0]:crop.end_zyx[0],
        crop.start_zyx[1]:crop.end_zyx[1],
        crop.start_zyx[2]:crop.end_zyx[2],
    ] = roi_tensor
    return out


def crop_bounds_from_dict(payload: Mapping[str, Any]) -> CropBounds:
    return CropBounds(
        start_zyx=tuple(int(v) for v in payload["start_zyx"]),
        end_zyx=tuple(int(v) for v in payload["end_zyx"]),
        full_shape_zyx=tuple(int(v) for v in payload["full_shape_zyx"]),
        used_fallback=bool(payload.get("used_fallback", False)),
    )


def resolve_localizer_checkpoint(localizer_run: str | Path) -> tuple[Path, Path]:
    path = Path(localizer_run).expanduser().resolve()
    if path.is_dir():
        cfg_path = path / "config.yaml"
        ckpt_path = path / "checkpoints" / "best.pt"
    else:
        ckpt_path = path
        if ckpt_path.parent.name == "checkpoints":
            cfg_path = ckpt_path.parent.parent / "config.yaml"
        else:
            raise ValueError(
                "roi.localizer_run must be a run directory or a checkpoint inside <run>/checkpoints/."
            )
    if not cfg_path.exists():
        raise FileNotFoundError(f"Localizer config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Localizer checkpoint not found: {ckpt_path}")
    return cfg_path, ckpt_path


def _coerce_triplet(
    value: Any,
    cast: type[float] | type[int],
    key: str,
) -> tuple[Any, Any, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must be a 3-item list/tuple, got {value!r}.")
    return tuple(cast(v) for v in value)
