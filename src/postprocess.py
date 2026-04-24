from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

_ndi = None

try:
    import cc3d

    _HAS_CC3D = True
except ImportError:
    cc3d = None  # type: ignore[assignment]
    _HAS_CC3D = False
    try:
        from scipy import ndimage as _ndi
    except ImportError:
        _ndi = None


def logits_to_binary_mask(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert raw logits to binary mask {0,1} with sigmoid+threshold."""
    return (torch.sigmoid(logits) >= threshold).to(torch.uint8)


def binary_mask_to_pseudo_logits(
    binary_mask: Tensor,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Convert binary mask {0,1} to large-magnitude logits (-100,+100).
    This keeps compute_all_metrics() behavior stable.
    """
    mask = binary_mask.to(dtype=dtype)
    return (mask * 200.0) - 100.0


def _min_component_voxels(
    min_component_volume_mm3: float,
    spacing_zyx: tuple[float, float, float],
) -> int:
    if min_component_volume_mm3 <= 0.0:
        return 1

    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_zyx, dtype=np.float64)))
    if voxel_volume_mm3 <= 0.0:
        raise ValueError(f"spacing_zyx must contain positive values, got {spacing_zyx}")
    return max(1, int(math.ceil(min_component_volume_mm3 / voxel_volume_mm3)))


def _label_connected_components(mask: np.ndarray, connectivity: int) -> np.ndarray:
    if _HAS_CC3D:
        return cc3d.connected_components(  # type: ignore[union-attr]
            mask.astype(np.uint8),
            connectivity=connectivity,
        )

    if _ndi is None:
        raise RuntimeError(
            "Connected-component backend unavailable: install connected-components-3d or scipy."
        )

    connectivity_map = {6: 1, 18: 2, 26: 3}
    structure = _ndi.generate_binary_structure(
        rank=3,
        connectivity=connectivity_map[connectivity],
    )
    labels, _ = _ndi.label(mask.astype(bool), structure=structure)
    return labels.astype(np.int32, copy=False)


def _remove_small_components_single(
    mask_zyx: np.ndarray,
    min_component_voxels: int,
    connectivity: int,
) -> np.ndarray:
    if min_component_voxels <= 1 or not np.any(mask_zyx):
        return mask_zyx.astype(np.uint8, copy=False)

    labels = _label_connected_components(mask_zyx, connectivity=connectivity)
    counts = np.bincount(labels.ravel())

    if counts.size <= 1:
        return np.zeros_like(mask_zyx, dtype=np.uint8)

    keep = np.flatnonzero(counts >= min_component_voxels)
    keep = keep[keep != 0]  # drop background label
    if keep.size == 0:
        return np.zeros_like(mask_zyx, dtype=np.uint8)

    return np.isin(labels, keep).astype(np.uint8)


def apply_outlier_postprocessing(
    binary_mask: Tensor,
    spacing_zyx: tuple[float, float, float],
    min_component_volume_mm3: float = 30.0,
    connectivity: int = 26,
) -> Tensor:
    """
    Remove small connected components from binary prediction mask.
    Expects shape (B,1,D,H,W). Returns uint8 mask on original device.
    """
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be one of {{6, 18, 26}}, got {connectivity}")

    if binary_mask.ndim != 5 or binary_mask.size(1) != 1:
        raise ValueError(
            "binary_mask must have shape (B, 1, D, H, W), "
            f"got {tuple(binary_mask.shape)}"
        )

    if min_component_volume_mm3 <= 0.0:
        return binary_mask.to(dtype=torch.uint8)

    min_component_voxels = _min_component_voxels(
        min_component_volume_mm3=min_component_volume_mm3,
        spacing_zyx=spacing_zyx,
    )

    mask_np = binary_mask.detach().to(device="cpu", dtype=torch.uint8).numpy()
    filtered_np = np.zeros_like(mask_np, dtype=np.uint8)

    for b in range(mask_np.shape[0]):
        filtered_np[b, 0] = _remove_small_components_single(
            mask_zyx=(mask_np[b, 0] > 0),
            min_component_voxels=min_component_voxels,
            connectivity=connectivity,
        )

    return torch.from_numpy(filtered_np).to(
        device=binary_mask.device,
        dtype=torch.uint8,
    )


def postprocess_logits(
    logits: Tensor,
    threshold: float = 0.5,
    enabled: bool = False,
    spacing_zyx: tuple[float, float, float] = (3.0, 0.5, 0.5),
    min_component_volume_mm3: float = 30.0,
    connectivity: int = 26,
) -> tuple[Tensor, Tensor]:
    """
    logits -> binary mask (threshold) -> optional small-component removal
    -> pseudo-logits for metric pipeline.
    Returns (metric_logits, pred_binary_mask).
    """
    pred_bin = logits_to_binary_mask(logits, threshold=threshold)
    if enabled:
        pred_bin = apply_outlier_postprocessing(
            binary_mask=pred_bin,
            spacing_zyx=spacing_zyx,
            min_component_volume_mm3=min_component_volume_mm3,
            connectivity=connectivity,
        )

    pred_logits = binary_mask_to_pseudo_logits(
        binary_mask=pred_bin,
        dtype=logits.dtype,
    )
    return pred_logits, pred_bin


__all__ = [
    "apply_outlier_postprocessing",
    "binary_mask_to_pseudo_logits",
    "logits_to_binary_mask",
    "postprocess_logits",
]
