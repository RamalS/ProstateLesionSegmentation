"""
Minimal integration examples for 2D Gleason consensus training.

These examples are intentionally compact and focus on the loss wiring:
- ignore/confidence masking
- soft-label CE/KL
- hard Dice (or MONAI DiceLoss)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_mask(
    ignore_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    use_confidence_mask: bool,
    confidence_threshold: float,
) -> torch.Tensor:
    valid = ignore_mask == 0
    if use_confidence_mask:
        valid = valid & (soft_probs.max(dim=1).values >= confidence_threshold)
    return valid


def plain_pytorch_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    class_weights: torch.Tensor,
    lambda_soft: float = 1.0,
    lambda_dice: float = 1.0,
    soft_loss_type: str = "ce",
    use_confidence_mask: bool = True,
    confidence_threshold: float = 0.6,
) -> dict[str, float]:
    """
    Plain PyTorch training step using soft-label CE/KL + hard Dice.
    """
    model.train()
    image = batch["image"]
    soft_probs = batch["soft_probs"]
    hard_mask = batch["hard_mask"]
    ignore_mask = batch["ignore_mask"]

    logits = model(image)
    if isinstance(logits, list):
        logits = logits[0]

    valid = _valid_mask(ignore_mask, soft_probs, use_confidence_mask, confidence_threshold)
    valid_f = valid.float()

    log_p = F.log_softmax(logits.float(), dim=1)
    if soft_loss_type == "kl":
        soft_map = F.kl_div(log_p, soft_probs.float(), reduction="none").sum(dim=1)
    else:
        soft_map = -(soft_probs.float() * log_p).sum(dim=1)

    expected_w = (soft_probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
    soft_loss = (soft_map * expected_w * valid_f).sum() / valid_f.sum().clamp_min(1e-8)

    pred_probs = F.softmax(logits.float(), dim=1)
    target_1h = F.one_hot(hard_mask.long(), num_classes=4).permute(0, 3, 1, 2).float()
    valid_c = valid_f.unsqueeze(1)
    inter = (pred_probs * target_1h * valid_c).sum(dim=(0, 2, 3))
    denom = ((pred_probs + target_1h) * valid_c).sum(dim=(0, 2, 3))
    hard_dice_loss = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5))[1:].mean()

    loss = (lambda_soft * soft_loss) + (lambda_dice * hard_dice_loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu().item()),
        "soft_loss": float(soft_loss.detach().cpu().item()),
        "hard_dice_loss": float(hard_dice_loss.detach().cpu().item()),
    }


def monai_compatible_loss_example(
    logits: torch.Tensor,
    hard_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    soft_probs: torch.Tensor,
    class_weights: torch.Tensor,
    lambda_soft: float = 1.0,
    lambda_dice: float = 1.0,
) -> torch.Tensor:
    """
    MONAI-compatible loss wiring to reduce custom code.

    Uses MONAI DiceLoss for hard labels + soft-label CE for consensus probs.
    """
    from monai.losses import DiceLoss

    valid = (ignore_mask == 0)
    valid_f = valid.float()

    log_p = F.log_softmax(logits.float(), dim=1)
    soft_map = -(soft_probs.float() * log_p).sum(dim=1)
    expected_w = (soft_probs * class_weights.view(1, -1, 1, 1)).sum(dim=1)
    soft_loss = (soft_map * expected_w * valid_f).sum() / valid_f.sum().clamp_min(1e-8)

    target_1h = F.one_hot(hard_mask.long(), num_classes=4).permute(0, 3, 1, 2).float()
    pred_probs = F.softmax(logits.float(), dim=1)

    # Mask out ignored pixels before MONAI Dice to keep behavior aligned.
    valid_c = valid_f.unsqueeze(1)
    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=False)(
        pred_probs * valid_c,
        target_1h * valid_c,
    )

    return (lambda_soft * soft_loss) + (lambda_dice * dice_loss)


__all__ = [
    "plain_pytorch_train_step",
    "monai_compatible_loss_example",
]
