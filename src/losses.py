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

DeepSupervisionWrapper
    Wraps any base loss to support deep supervision.  Accepts model outputs
    as a list[Tensor] (finest → coarsest) and computes a geometrically
    weighted sum of per-scale losses.  Transparently delegates to the base
    criterion when given a plain Tensor (deep_supervision=False mode).
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
    loss = 1 - mean(DSC over positive samples in batch)

    Sigmoid is applied internally so logits are expected (not probabilities).

    Samples whose ground-truth target is entirely empty (no positive voxels)
    are excluded from the Dice computation.  For negative cases the Dice
    gradient pushes the model toward predicting all-zero — the opposite of
    what we want.  The BCE term (applied to all samples) is sufficient to
    learn from negative cases.  This aligns the training signal with the
    validation metrics, which also exclude empty-target samples.

    Returns ``torch.tensor(0.0)`` when all samples in the batch have empty
    targets (no Dice gradient is generated; only BCE contributes).

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
    # Cast to FP32 before any reduction: sigmoid output summed over 327 K
    # voxels (patch 20×128×128) overflows FP16 max (~65 k), producing inf
    # and locking the loss at 1.0 for the entire training run.
    probs = torch.sigmoid(logits).float()

    # Flatten spatial dimensions; keep batch dimension
    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1).float()

    # Exclude negative samples (empty ground-truth) from the Dice term.
    # For negative cases the gradient of the Dice numerator is zero while
    # the denominator penalises any positive prediction, actively driving
    # the model toward all-background.
    has_positive = tgts_flat.sum(dim=1) > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    probs_flat = probs_flat[has_positive]
    tgts_flat = tgts_flat[has_positive]

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
        loss = 1 - mean(TI over positive samples in batch)

    Setting alpha = beta = 0.5 recovers standard Dice loss.

    For prostate lesion segmentation, use alpha < beta so that missing a
    lesion (FN) is penalised more heavily than predicting a false alarm (FP).
    The canonical choice is alpha=0.3, beta=0.7, which weights FN 2.3× more
    than FP.

    Samples whose ground-truth target is entirely empty (no positive voxels)
    are excluded from the Tversky computation.  For negative cases the
    Tversky gradient penalises any positive prediction (FP), actively driving
    the model toward all-background — the opposite of what we want.  The BCE
    term (applied to all samples) is sufficient to learn from negative cases.
    This aligns the training signal with the validation metrics, which also
    exclude empty-target samples.

    Returns ``torch.tensor(0.0)`` when all samples in the batch have empty
    targets (no Tversky gradient is generated; only BCE contributes).

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
    # Cast to FP32 before any reduction: sigmoid output summed over 327 K
    # voxels (patch 20×128×128) overflows FP16 max (~65 k), producing inf
    # and locking the loss at 1.0 for the entire training run.
    probs = torch.sigmoid(logits).float()

    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1).float()

    # Exclude negative samples (empty ground-truth) from the Tversky term.
    # For negative cases the gradient of the Tversky numerator is zero while
    # the FP term in the denominator penalises any positive prediction,
    # actively driving the model toward all-background.
    has_positive = tgts_flat.sum(dim=1) > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    probs_flat = probs_flat[has_positive]
    tgts_flat = tgts_flat[has_positive]

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


# ---------------------------------------------------------------------------
# Deep supervision wrapper
# ---------------------------------------------------------------------------

class DeepSupervisionWrapper(nn.Module):
    """
    Wraps any base loss to support deep supervision.

    When a model is trained with ``deep_supervision=True`` its ``forward``
    returns a ``list[Tensor]`` ordered finest → coarsest (e.g.
    ``[logits_full, logits_D/2, logits_D/4, logits_D/8]``).  This wrapper
    accepts that list, downsamples the ground-truth label to each auxiliary
    scale with nearest-neighbour interpolation, computes the base loss at
    every scale independently, and returns a geometrically weighted sum.

    Weights are ``[1, 1/2, 1/4, …]`` (finest → coarsest), normalised so
    they sum to 1.  For the default 4-level model they become approximately
    ``[0.533, 0.267, 0.133, 0.067]``.

    When the model output is a plain ``Tensor`` (i.e. ``deep_supervision=False``
    at construction time), the wrapper delegates directly to the base criterion
    without modification.  This means the same ``criterion`` object works for
    both modes and the training loop requires no conditional logic.

    Parameters
    ----------
    base_criterion : nn.Module
        Underlying loss instance, e.g. ``TverskyBCELoss`` or ``DiceBCELoss``.
        Must accept ``(logits, targets)`` and return a scalar tensor.
    num_levels : int
        Total number of output scales (main + auxiliary).  Should equal
        ``len(cfg["features"])``.  Default 4.
    """

    def __init__(self, base_criterion: nn.Module, num_levels: int = 4) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        raw: list[float] = [1.0 / (2 ** i) for i in range(num_levels)]
        total = sum(raw)
        self.weights: list[float] = [w / total for w in raw]

    def forward(self, outputs: list[Tensor] | Tensor, targets: Tensor) -> Tensor:
        """
        Compute (weighted) deep-supervision loss.

        Parameters
        ----------
        outputs : list[Tensor] | Tensor
            Model output.  If a plain ``Tensor``, delegates to the base
            criterion unchanged.  If a ``list[Tensor]``, index 0 must be the
            finest-resolution logits (matching ``targets`` shape) and
            subsequent indices progressively coarser.
        targets : (B, 1, D, H, W)
            Binary ground-truth mask; downsampled per scale as needed.

        Returns
        -------
        Scalar loss tensor.
        """
        if isinstance(outputs, Tensor):
            return self.base_criterion(outputs, targets)

        total_loss: Tensor = torch.zeros(
            1, device=targets.device, dtype=targets.dtype
        ).squeeze()

        for logits, weight in zip(outputs, self.weights):
            if logits.shape[2:] != targets.shape[2:]:
                # Nearest-neighbour downsampling preserves binary label values.
                scaled_targets = F.interpolate(
                    targets.float(),
                    size=logits.shape[2:],
                    mode="nearest",
                )
            else:
                scaled_targets = targets

            total_loss = total_loss + weight * self.base_criterion(logits, scaled_targets)

        return total_loss

    def __repr__(self) -> str:
        weight_str = ", ".join(f"{w:.3f}" for w in self.weights)
        return (
            f"{self.__class__.__name__}("
            f"base_criterion={self.base_criterion}, "
            f"weights=[{weight_str}])"
        )
