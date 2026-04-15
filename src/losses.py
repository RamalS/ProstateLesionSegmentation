"""
Loss functions for binary prostate lesion segmentation.

All functions accept raw logits (before sigmoid) and binary ground-truth
masks to remain numerically stable (sigmoid is applied internally).

Two combined losses are available:

DiceBCELoss (original)
    Soft Dice + BCE.  FP and FN are penalised equally by the Dice term.
    Good baseline; use when the class imbalance is modest.

TverskyBCELoss (recommended for lesion segmentation)
    Tversky + BCE.  Tversky generalises Dice with independent FP (alpha)
    and FN (beta) weights.  Setting alpha < beta (e.g. 0.3 / 0.7) makes
    the model prefer sensitivity over precision — missing a lesion is
    penalised 2.3× more than a false alarm.  This is the right trade-off
    for csPCa detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Soft Dice loss
# ---------------------------------------------------------------------------

def dice_loss(
    logits: Tensor,
    targets: Tensor,
    smooth: float = 1.0,
) -> Tensor:
    """
    Soft Dice loss for binary segmentation.

    DSC = 2|P ∩ T| / (|P| + |T|)
    loss = 1 - mean(DSC over batch)

    Sigmoid is applied internally so logits are expected (not probabilities).

    Parameters
    ----------
    logits  : (B, 1, D, H, W) raw model output
    targets : (B, 1, D, H, W) binary ground-truth mask {0.0, 1.0}
    smooth  : Laplace smoothing constant; prevents division by zero on
              empty predictions/targets (set to 1.0 following standard practice)

    Returns
    -------
    Scalar Dice loss in [0, 1].
    """
    probs = torch.sigmoid(logits)

    # Flatten spatial dimensions; keep batch dimension
    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * tgts_flat).sum(dim=1)
    cardinality = probs_flat.sum(dim=1) + tgts_flat.sum(dim=1)

    dsc = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dsc).mean()


# ---------------------------------------------------------------------------
# Combined Dice + BCE loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """
    Weighted combination of soft Dice loss and Binary Cross-Entropy loss.

        loss = dice_weight × Dice(logits, targets)
             + bce_weight  × BCE(logits, targets)

    Both terms operate on raw logits for numerical stability.

    Parameters
    ----------
    dice_weight : weight applied to the Dice term (default 1.0)
    bce_weight  : weight applied to the BCE term  (default 1.0)
    pos_weight  : scalar multiplier applied to the loss at positive (lesion)
                  voxels in the BCE term.  Compensates for the severe class
                  imbalance typical in prostate lesion segmentation, where
                  lesion voxels are ~1-2 % of the total volume.  Without this,
                  the BCE gradient from the majority background voxels can
                  overwhelm the lesion gradient and drive the model toward
                  predicting all-zeros.
                  Rule of thumb: set to (background voxels) / (lesion voxels).
                  Default is 1.0 (no re-weighting) for backward compatibility.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        # Register as a buffer so it moves to the correct device automatically
        # with .to(device) / .cuda(), and is not treated as a trainable parameter.
        # This avoids re-creating a new tensor on every forward pass.
        self.register_buffer("_pw", torch.tensor([pos_weight]))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        d_loss = dice_loss(logits, targets)
        # Cast _pw to match logits device and dtype (important for BF16 autocast and
        # cases where the criterion is not explicitly moved to the target device).
        pw: Tensor = self._pw.to(device=logits.device, dtype=logits.dtype)  # type: ignore[union-attr]
        b_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw
        )
        return self.dice_weight * d_loss + self.bce_weight * b_loss

    def __repr__(self) -> str:
        pw_val = float(self._pw.item()) if hasattr(self, "_pw") else "?"  # type: ignore[union-attr]
        return (
            f"{self.__class__.__name__}("
            f"dice_weight={self.dice_weight}, "
            f"bce_weight={self.bce_weight}, "
            f"pos_weight={pw_val})"
        )


# ---------------------------------------------------------------------------
# Tversky loss
# ---------------------------------------------------------------------------

def tversky_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> Tensor:
    """
    Soft Tversky loss for binary segmentation.

    Generalises soft Dice by independently weighting false positives (alpha)
    and false negatives (beta):

        TI  = TP / (TP + alpha·FP + beta·FN)
        loss = 1 - mean(TI over batch)

    Setting alpha = beta = 0.5 recovers standard Dice loss.

    For prostate lesion segmentation, use alpha < beta so that missing a
    lesion (FN) is penalised more heavily than predicting a false alarm (FP).
    The canonical choice is alpha=0.3, beta=0.7, which weights FN 2.3× more
    than FP.

    Sigmoid is applied internally so raw logits are expected.

    Parameters
    ----------
    logits  : (B, 1, D, H, W) raw model output
    targets : (B, 1, D, H, W) binary ground-truth mask {0.0, 1.0}
    alpha   : FP penalty weight; default 0.3
    beta    : FN penalty weight; default 0.7
    smooth  : Laplace smoothing constant; prevents division by zero on
              empty predictions/targets (set to 1.0 following standard practice)

    Returns
    -------
    Scalar Tversky loss in [0, 1].
    """
    probs = torch.sigmoid(logits)

    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1)

    tp = (probs_flat * tgts_flat).sum(dim=1)
    fp = (probs_flat * (1.0 - tgts_flat)).sum(dim=1)
    fn = ((1.0 - probs_flat) * tgts_flat).sum(dim=1)

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - tversky_index).mean()


# ---------------------------------------------------------------------------
# Combined Tversky + BCE loss
# ---------------------------------------------------------------------------

class TverskyBCELoss(nn.Module):
    """
    Weighted combination of soft Tversky loss and Binary Cross-Entropy loss.

        loss = tversky_weight × Tversky(logits, targets, alpha, beta)
             + bce_weight     × BCE(logits, targets, pos_weight)

    Recommended over DiceBCELoss for prostate lesion segmentation because
    the Tversky term directly penalises false negatives more than false
    positives (alpha < beta), matching the clinical cost asymmetry where
    missing a lesion is worse than a false alarm.

    Parameters
    ----------
    tversky_weight : weight applied to the Tversky term (default 1.0)
    bce_weight     : weight applied to the BCE term (default 1.0)
    alpha          : FP penalty in the Tversky index (default 0.3)
    beta           : FN penalty in the Tversky index (default 0.7)
    pos_weight     : scalar multiplier on BCE loss at positive voxels.
                     Compensates for voxel-level class imbalance (~50:1 for
                     PI-CAI at the default target spacing).  Rule of thumb:
                     set to (background voxels) / (lesion voxels).
    """

    def __init__(
        self,
        tversky_weight: float = 1.0,
        bce_weight: float = 1.0,
        alpha: float = 0.3,
        beta: float = 0.7,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.alpha = alpha
        self.beta = beta
        self.register_buffer("_pw", torch.tensor([pos_weight]))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        t_loss = tversky_loss(logits, targets, self.alpha, self.beta)
        pw: Tensor = self._pw.to(device=logits.device, dtype=logits.dtype)  # type: ignore[union-attr]
        b_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw
        )
        return self.tversky_weight * t_loss + self.bce_weight * b_loss

    def __repr__(self) -> str:
        pw_val = float(self._pw.item()) if hasattr(self, "_pw") else "?"  # type: ignore[union-attr]
        return (
            f"{self.__class__.__name__}("
            f"tversky_weight={self.tversky_weight}, "
            f"bce_weight={self.bce_weight}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"pos_weight={pw_val})"
        )
