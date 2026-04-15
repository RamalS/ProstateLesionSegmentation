# Fix Model Collapse to All-Background Predictions

## Problem

The 3D segmentation model collapses to predicting only background during training:
- `val_dice ~ 0.0`, `val_sens ~ 0.0`, `val_spec = 1.0`
- Briefly achieved `best_val_dice = 0.0898` early, then collapsed
- Early stopping triggered at epoch 92

## Root Causes

### 1. No gradient clipping (HIGH IMPACT)
**File:** `src/train.py:683-685`

The training loop goes `scaler.scale(loss).backward()` -> `scaler.step(optimizer)` with no gradient clipping. With `pos_weight=50` and `dice_weight=3.0`, gradient spikes can push the model into the all-background local minimum in a single step.

### 2. Tversky/Dice loss includes negative samples (HIGH IMPACT)
**File:** `src/losses.py:67,184`

Both `dice_loss()` and `tversky_loss()` compute per-sample loss and `.mean()` over ALL batch samples, including cases with empty ground-truth (no lesion). For negative samples, the Tversky/Dice gradient actively rewards predicting all-zero (any false positive increases loss). Meanwhile, validation metrics (`src/metrics.py:74`) correctly EXCLUDE negative samples. This train/eval mismatch means the model optimizes a fundamentally different objective than it's evaluated on.

### 3. FP16 overflow in dice_loss denominator (LATENT BUG)
**File:** `src/losses.py:64`

Summing sigmoid probabilities (~0.5) over 327,680 voxels yields ~163,840, exceeding FP16 max (65,504). Causes `inf` -> DSC=0 -> dice_loss stuck at 1.0. Currently masked because TverskyBCELoss accidentally promotes to FP32, but `dice_loss` is broken for any direct FP16 usage.

---

## Fix 1: Add gradient clipping in train.py

**Location:** `src/train.py`, lines 683-685

**Current code:**
```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Replace with:**
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

**Why:** `scaler.unscale_()` converts gradients back from scaled FP16 to their true FP32 values. `clip_grad_norm_` then caps the total gradient magnitude at 1.0, preventing any single batch from causing a catastrophically large parameter update. This is standard practice with AMP + GradScaler.

---

## Fix 2: Exclude negative samples from Tversky/Dice loss

**Location:** `src/losses.py`, functions `dice_loss()` and `tversky_loss()`

### dice_loss() - replace the full function body:

```python
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

    Samples whose ground-truth target is entirely empty (no positive voxels)
    are excluded from the Dice computation.  For negative cases the Dice
    gradient pushes the model toward predicting all-zero — the opposite of
    what we want.  The BCE term (applied to all samples) is sufficient to
    learn from negative cases.

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
    probs = torch.sigmoid(logits).float()   # FP32 for numerical safety

    # Flatten spatial dimensions; keep batch dimension
    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1).float()

    # Exclude negative samples (empty ground-truth) from the Dice term
    has_positive = tgts_flat.sum(dim=1) > 0
    if not has_positive.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    probs_flat = probs_flat[has_positive]
    tgts_flat = tgts_flat[has_positive]

    intersection = (probs_flat * tgts_flat).sum(dim=1)
    cardinality = probs_flat.sum(dim=1) + tgts_flat.sum(dim=1)

    dsc = (2.0 * intersection + smooth) / (cardinality + smooth)
    return (1.0 - dsc).mean()
```

### tversky_loss() - replace the full function body:

```python
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

        TI  = TP / (TP + alpha*FP + beta*FN)
        loss = 1 - mean(TI over batch)

    Setting alpha = beta = 0.5 recovers standard Dice loss.

    For prostate lesion segmentation, use alpha < beta so that missing a
    lesion (FN) is penalised more heavily than predicting a false alarm (FP).
    The canonical choice is alpha=0.3, beta=0.7, which weights FN 2.3x more
    than FP.

    Samples whose ground-truth target is entirely empty (no positive voxels)
    are excluded from the Tversky computation.  For negative cases the
    Tversky gradient pushes the model toward predicting all-zero — the
    opposite of what we want.  The BCE term (applied to all samples) is
    sufficient to learn from negative cases.

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
    probs = torch.sigmoid(logits).float()   # FP32 for numerical safety

    probs_flat = probs.view(probs.size(0), -1)
    tgts_flat = targets.view(targets.size(0), -1).float()

    # Exclude negative samples (empty ground-truth) from the Tversky term
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
```

**Why:** This aligns the training loss with the validation metrics. Negative samples only contribute via the BCE term (which correctly uses `pos_weight` to handle class imbalance). The Tversky/Dice term now only optimizes on samples that actually contain lesions -- exactly what validation measures.

---

## Fix 3: FP32 casting (included in Fix 2 above)

Both replacement functions above already include `.float()` casting:
- `probs = torch.sigmoid(logits).float()` -- ensures FP32 after sigmoid
- `tgts_flat = targets.view(...).float()` -- ensures targets are FP32 too

This prevents the FP16 overflow where `sum(sigmoid(logits))` over 327K voxels exceeds 65,504.

---

## Fix 4: Update smoke_test.py

Add test blocks for:
1. **Negative-sample exclusion:** Create a batch where some samples have empty targets, verify `dice_loss` and `tversky_loss` return values that only reflect the positive samples
2. **All-negative batch:** Verify both losses return `0.0` when all targets are empty
3. **FP16 safety:** Run `dice_loss` with FP16 inputs on large spatial dimensions, verify no inf/nan

---

## Verification

After all edits:
```bash
PYTHONPATH=. python scripts/smoke_test.py
```

Must exit with code 0.
