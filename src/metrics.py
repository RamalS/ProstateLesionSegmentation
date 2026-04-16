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
precision_score    : Positive Predictive Value (Precision)
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

    Samples where the ground-truth target is entirely empty (no positive
    voxels) are excluded from the average.  This prevents negative cases
    from inflating the metric via the smooth term: when both prediction and
    target are all-zero, the smooth numerator and denominator cancel to 1.0,
    which would misleadingly reward a model that never predicts any lesion.

    Returns float('nan') if all samples in the batch have empty targets.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    mask = t.sum(dim=1) > 0          # True for samples with ≥1 positive voxel
    if not mask.any():
        return float("nan")
    p, t = p[mask], t[mask]

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

    Samples with empty ground-truth targets are excluded from the average
    for the same reason as in ``dice_coefficient``.

    Returns float('nan') if all samples in the batch have empty targets.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    mask = t.sum(dim=1) > 0
    if not mask.any():
        return float("nan")
    p, t = p[mask], t[mask]

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

    Samples with empty ground-truth targets are excluded from the average.
    When ``t = 0`` everywhere, FN = 0 regardless of the prediction, so the
    smooth term would always return 1.0 — making sensitivity meaningless for
    those cases and masking a model that predicts nothing.

    Returns float('nan') if all samples in the batch have empty targets.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    mask = t.sum(dim=1) > 0
    if not mask.any():
        return float("nan")
    p, t = p[mask], t[mask]

    tp = (p * t).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    sens = (tp + smooth) / (tp + fn + smooth)
    return sens.mean().item()


def precision_score(
    preds: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    """
    Precision (Positive Predictive Value).

    Precision = TP / (TP + FP)

    Range: [0, 1]; higher is better.

    Samples with empty ground-truth targets are excluded from the average
    for consistency with sensitivity and dice.  For negative cases the
    model should predict nothing; any positive prediction is a false
    positive that inflates the denominator, making precision unreliable as
    a per-case metric on empty-target volumes.

    Returns float('nan') if all samples in the batch have empty targets.
    """
    p = _binarise(preds, threshold).view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)

    mask = t.sum(dim=1) > 0
    if not mask.any():
        return float("nan")
    p, t = p[mask], t[mask]

    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec.mean().item()


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
    compute_hd95: bool = True,
) -> dict[str, float]:
    """
    Compute all five segmentation metrics in a single call.

    Parameters
    ----------
    preds        : (B, 1, D, H, W) raw logits
    targets      : (B, 1, D, H, W) binary ground-truth mask {0.0, 1.0}
    threshold    : binarisation threshold applied to sigmoid(preds)
    compute_hd95 : if False, skip the expensive HD95 computation and return
                   float('nan') for the "hd95" key.  Useful during the
                   training loop where HD95 is not needed every epoch.

    Returns
    -------
    dict with keys: "dice", "iou", "sensitivity", "precision", "hd95"

    Notes
    -----
    - sigmoid + threshold is applied **once** and reused across all metrics.
    - "dice", "iou", "sensitivity", "precision": computed only over samples
      whose ground-truth target is non-empty (has at least one positive voxel).
      Returns float('nan') when all samples in the batch are empty-target.
    - "hd95": computed only over samples where both prediction and target
      are non-empty; returns float('nan') otherwise.
    """
    # Binarise once; pass the already-binary tensor to each metric.
    # Each individual metric function calls _binarise() internally, so we pass
    # logits directly — the binarised intermediate is computed once here and
    # the individual functions each do their own cheap view/mask operations.
    # To avoid modifying the individual metric APIs (which are also public),
    # we pass large-magnitude logits that are equivalent to the pre-binarised
    # result: positive voxels → +100, negative → -100.
    binary = _binarise(preds, threshold)
    # Convert binary mask back to "pseudo-logits" so individual metric
    # functions (which call _binarise internally) reproduce the same mask.
    pseudo_logits = (binary * 200.0) - 100.0  # 0→-100, 1→+100

    hd95_val = (
        hausdorff_distance_95(pseudo_logits, targets, threshold)
        if compute_hd95
        else float("nan")
    )

    return {
        "dice":        dice_coefficient(pseudo_logits, targets, threshold),
        "iou":         iou_score(pseudo_logits, targets, threshold),
        "sensitivity": sensitivity(pseudo_logits, targets, threshold),
        "precision":   precision_score(pseudo_logits, targets, threshold),
        "hd95":        hd95_val,
    }
