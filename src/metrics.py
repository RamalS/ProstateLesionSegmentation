"""
Segmentation evaluation metrics for binary prostate lesion masks.

All public functions accept raw logits (or probabilities ≥ 0) and binary
ground-truth masks; binarisation at *threshold* is applied internally.

Functions return Python floats averaged over the batch dimension.
Samples where both prediction and ground-truth are empty are excluded from
metrics that are undefined in that case (e.g. HD95).

Metrics implemented
-------------------
dice_coefficient   : Dice Similarity Coefficient (DSC)
iou_score          : Intersection over Union (Jaccard index)
sensitivity        : True Positive Rate (Recall)
specificity        : True Negative Rate
hausdorff_distance_95 : 95th-percentile Hausdorff Distance (voxels)
compute_all_metrics   : convenience wrapper returning all five metrics
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_AVAILABLE = True
except ImportError:
    _edt = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binarise(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Apply sigmoid then threshold to produce a binary {0, 1} mask."""
    return (torch.sigmoid(logits) >= threshold).float()


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def dice_coefficient(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Volumetric Dice Similarity Coefficient (DSC).

    DSC = 2|P ∩ T| / (|P| + |T|)

    Range: [0, 1]; higher is better.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    dsc = (2.0 * intersection + smooth) / (p.sum(dim=1) + t.sum(dim=1) + smooth)
    return dsc.mean().item()


def iou_score(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Intersection over Union (Jaccard index).

    IoU = |P ∩ T| / |P ∪ T|

    Range: [0, 1]; higher is better.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    union = (p + t - p * t).sum(dim=1)
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def sensitivity(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Sensitivity (Recall / True Positive Rate).

    Sensitivity = TP / (TP + FN)

    Range: [0, 1]; higher is better.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    tp = (p * t).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    sens = (tp + smooth) / (tp + fn + smooth)
    return sens.mean().item()


def specificity(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Specificity (True Negative Rate).

    Specificity = TN / (TN + FP)

    Range: [0, 1]; higher is better.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    tn = ((1.0 - p) * (1.0 - t)).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    spec = (tn + smooth) / (tn + fp + smooth)
    return spec.mean().item()


def hausdorff_distance_95(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> float:
    """
    95th-percentile Hausdorff Distance (HD95) in voxels.

    Measures the maximum surface distance between prediction and ground-truth,
    using the 95th percentile for robustness against single-voxel outliers.

    Returns float('nan') if:
    - scipy is not installed
    - all batch samples have empty predictions or targets

    Lower is better; 0.0 is a perfect match.
    """
    if not _SCIPY_AVAILABLE:
        return float("nan")

    # Local import: guaranteed to succeed since _SCIPY_AVAILABLE is True
    from scipy.ndimage import distance_transform_edt
    p_np = _binarise(preds, threshold).cpu().numpy().astype(bool)  # (B, 1, D, H, W)
    t_np = targets.cpu().numpy().astype(bool)

    hd95_values: list[float] = []

    for b in range(p_np.shape[0]):
        pred_vol = p_np[b, 0]
        tgt_vol = t_np[b, 0]

        # Skip samples where either mask is empty (metric undefined)
        if not pred_vol.any() or not tgt_vol.any():
            continue

        # Distance transforms: each voxel stores distance to nearest surface.
        # scipy stubs declare distance_transform_edt as returning ndarray|tuple|None;
        # with default args (return_distances=True, return_indices=False) it is
        # always an ndarray — the type: ignore comments suppress the false positive.
        dt_pred = distance_transform_edt(~pred_vol)
        dt_tgt = distance_transform_edt(~tgt_vol)

        d_pred_to_tgt = dt_tgt[pred_vol]    # type: ignore[index]
        d_tgt_to_pred = dt_pred[tgt_vol]    # type: ignore[index]

        hd95 = max(
            float(np.percentile(d_pred_to_tgt, 95)),
            float(np.percentile(d_tgt_to_pred, 95)),
        )
        hd95_values.append(hd95)

    return float(np.mean(hd95_values)) if hd95_values else float("nan")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def compute_all_metrics(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute all five segmentation metrics in a single call.

    Parameters
    ----------
    preds     : (B, 1, D, H, W) raw logits
    targets   : (B, 1, D, H, W) binary ground-truth mask {0.0, 1.0}
    threshold : binarisation threshold applied to sigmoid(preds)

    Returns
    -------
    dict with keys: "dice", "iou", "sensitivity", "specificity", "hd95"
    """
    return {
        "dice":        dice_coefficient(preds, targets, threshold),
        "iou":         iou_score(preds, targets, threshold),
        "sensitivity": sensitivity(preds, targets, threshold),
        "specificity": specificity(preds, targets, threshold),
        "hd95":        hausdorff_distance_95(preds, targets, threshold),
    }
