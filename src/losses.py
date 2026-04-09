"""
Loss functions for binary prostate lesion segmentation.

All functions accept raw logits (before sigmoid) and binary ground-truth
masks to remain numerically stable (sigmoid is applied internally).

DiceBCELoss is the recommended loss:
  - Soft Dice loss directly optimises the overlap metric, handling class
    imbalance by normalising by prediction + target volume.
  - BCE provides dense per-voxel gradients, stabilising early training
    when the Dice numerator is near zero.
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
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        d_loss = dice_loss(logits, targets)
        # Build pos_weight tensor on the same device/dtype as the logits so
        # the loss function works on both CPU and CUDA without manual moves.
        pw = torch.tensor(
            [self.pos_weight], device=logits.device, dtype=logits.dtype
        )
        b_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw
        )
        return self.dice_weight * d_loss + self.bce_weight * b_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dice_weight={self.dice_weight}, "
            f"bce_weight={self.bce_weight}, "
            f"pos_weight={self.pos_weight})"
        )
