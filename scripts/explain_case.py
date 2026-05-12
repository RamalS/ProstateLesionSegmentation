"""


This script reuses the repository's run config, checkpoint loading,
single-case preprocessing, sliding-window prediction, postprocessing, and
metric code.  It then computes ROI-based attribution maps and writes PNG/JSON
artifacts under visualizations/xai/ by default.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve imports for both repo-root and scripts/ execution.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config  # noqa: E402
from dataset import active_modality_pairs  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from utils import load_checkpoint  # noqa: E402
from visualize_3d import (  # noqa: E402
    _case_id_from_t2w,
    _load_model_inputs,
    _resolve_checkpoint_path,
    _resolve_seg_path,
    _run_inference,
    resolve_run_dir,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_N_VIS_COLS = 12
_GT_COLOR = (0.0, 1.0, 0.0)
_PRED_COLOR = (1.0, 0.0, 0.0)
_OVL_COLOR = (1.0, 1.0, 0.0)


def _json_float(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


def _safe_filename_component(raw: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in raw)
    return token.strip("_") or "item"


def _resolve_device(device_arg: str | None, cfg: dict[str, Any]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    use_cuda = torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
    return torch.device("cuda" if use_cuda else "cpu")


def _first_output(output: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, (list, tuple)) else output


def _postprocess_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": float(cfg.get("pred_threshold", 0.5)),
        "enabled": bool(cfg.get("postprocess_enabled", False)),
        "spacing_zyx": tuple(float(v) for v in cfg.get("target_spacing", [3.0, 0.5, 0.5])),
        "min_component_volume_mm3": float(cfg.get("postprocess_min_component_volume_mm3", 30.0)),
        "connectivity": int(cfg.get("postprocess_connectivity", 26)),
    }


def _select_roi_center(
    pred_bin: torch.Tensor,
    label_t: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[int, int, int]:
    pred_np = pred_bin[0, 0].detach().cpu().numpy() > 0
    label_np = label_t[0].detach().cpu().numpy() > 0

    if pred_np.any():
        coords = np.argwhere(pred_np)
        return tuple(int(v) for v in np.round(coords.mean(axis=0)))  # type: ignore[return-value]
    if label_np.any():
        coords = np.argwhere(label_np)
        return tuple(int(v) for v in np.round(coords.mean(axis=0)))  # type: ignore[return-value]

    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    flat_idx = int(np.argmax(prob))
    return tuple(int(v) for v in np.unravel_index(flat_idx, prob.shape))  # type: ignore[return-value]


def _roi_slices(
    shape_zyx: tuple[int, int, int],
    center_zyx: tuple[int, int, int],
    roi_size_zyx: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for dim, center, requested in zip(shape_zyx, center_zyx, roi_size_zyx):
        length = min(int(requested), int(dim))
        start = max(0, min(int(center) - length // 2, int(dim) - length))
        slices.append(slice(start, start + length))
    return tuple(slices)  # type: ignore[return-value]


def _insert_roi(full_shape: tuple[int, int, int], roi: np.ndarray, roi_slices: tuple[slice, slice, slice]) -> np.ndarray:
    out = np.zeros(full_shape, dtype=np.float32)
    out[roi_slices] = roi.astype(np.float32, copy=False)
    return out


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _candidate_module_names(model: nn.Module) -> list[str]:
    candidates: list[str] = []
    blocked = ("output", "head", "classifier", "deep_heads", "seg")
    for name, module in model.named_modules():
        if not name:
            continue
        if any(part in name.lower() for part in blocked):
            continue
        if isinstance(module, (nn.Conv3d, nn.Sequential)):
            candidates.append(name)
    return candidates


def _default_target_layer_name(model: nn.Module, model_name: str) -> str:
    modules = dict(model.named_modules())
    for preferred in ("bottleneck", "encoder.blocks.3", "encoder.3", "encoder"):
        if preferred in modules:
            return preferred

    conv_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv3d)
        and name
        and not any(part in name.lower() for part in ("output", "head", "classifier", "seg"))
    ]
    if conv_names:
        return conv_names[-1]

    candidates = _candidate_module_names(model)
    if candidates:
        return candidates[-1]

    raise ValueError(f"No Grad-CAM target layer candidates found for model '{model_name}'.")


class _FeatureHook:
    def __init__(self, module: nn.Module) -> None:
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        self._handles = [
            module.register_forward_hook(self._forward_hook),
            module.register_full_backward_hook(self._backward_hook),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _forward_hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        out = _first_output(output) if isinstance(output, (list, tuple)) else output
        if torch.is_tensor(out) and out.ndim == 5:
            self.activation = out

    def _backward_hook(self, _module: nn.Module, _grad_input: tuple[Any, ...], grad_output: tuple[Any, ...]) -> None:
        grad = grad_output[0] if grad_output else None
        if torch.is_tensor(grad) and grad.ndim == 5:
            self.gradient = grad


def _target_scalar(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    mask = (probs.detach() >= threshold).float()
    if mask.sum() > 0:
        return (logits * mask).sum() / mask.sum().clamp_min(1.0)
    return logits.flatten()[torch.argmax(probs.detach().flatten())]


def _run_gradcam_roi(
    model: nn.Module,
    image_roi: torch.Tensor,
    device: torch.device,
    target_layer_name: str,
    threshold: float,
) -> np.ndarray:
    modules = dict(model.named_modules())
    if target_layer_name not in modules:
        candidates = "\n  - " + "\n  - ".join(_candidate_module_names(model)[-40:])
        raise ValueError(f"Unknown target layer '{target_layer_name}'. Candidate layers:{candidates}")

    hook = _FeatureHook(modules[target_layer_name])
    try:
        model.zero_grad(set_to_none=True)
        x = image_roi.unsqueeze(0).to(device)
        logits = _first_output(model(x))
        target = _target_scalar(logits, threshold=threshold)
        target.backward()

        if hook.activation is None or hook.gradient is None:
            raise RuntimeError(
                f"Target layer '{target_layer_name}' did not produce a 5D activation/gradient pair."
            )

        activations = hook.activation.detach()
        gradients = hook.gradient.detach()
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=tuple(int(v) for v in image_roi.shape[1:]),
            mode="trilinear",
            align_corners=False,
        )
        return _normalize01(cam[0, 0].detach().cpu().numpy())
    finally:
        hook.close()


def _run_saliency_roi(
    model: nn.Module,
    image_roi: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.zero_grad(set_to_none=True)
    x = image_roi.unsqueeze(0).to(device).detach()
    x.requires_grad_(True)
    logits = _first_output(model(x))
    target = _target_scalar(logits, threshold=threshold)
    target.backward()

    if x.grad is None:
        raise RuntimeError("Input saliency failed because input gradients were not produced.")

    per_channel = _normalize01(x.grad.detach().abs()[0].cpu().numpy().astype(np.float32))
    combined = _normalize01(per_channel.max(axis=0))
    return combined, per_channel


def _segmentation_overlay(gt: np.ndarray, pred: np.ndarray, alpha: float) -> np.ndarray:
    h, w = gt.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    gt_only = (gt > 0) & (pred == 0)
    pred_only = (pred > 0) & (gt == 0)
    both = (gt > 0) & (pred > 0)

    rgba[gt_only, :3] = _GT_COLOR
    rgba[pred_only, :3] = _PRED_COLOR
    rgba[both, :3] = _OVL_COLOR
    rgba[gt_only | pred_only | both, 3] = alpha
    return rgba


def _choose_slices(
    heatmap: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    n_cols: int,
) -> np.ndarray:
    activity = heatmap.sum(axis=(1, 2)) + pred.sum(axis=(1, 2)).astype(np.float32) + gt.sum(axis=(1, 2)).astype(np.float32)
    if activity.max() <= 0:
        return np.linspace(0, heatmap.shape[0] - 1, n_cols, dtype=int)

    center = int(np.argmax(activity))
    half = max(1, n_cols // 2)
    start = max(0, min(center - half, heatmap.shape[0] - n_cols))
    stop = min(heatmap.shape[0], start + n_cols)
    if stop - start < n_cols:
        start = max(0, stop - n_cols)
    return np.arange(start, stop, dtype=int)


def _save_heatmap_overlay(
    output_path: Path,
    title: str,
    t2w: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    heatmap: np.ndarray,
    n_cols: int = _N_VIS_COLS,
    view_mode: str = "panels",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError("matplotlib is required for XAI visualization.") from exc

    n_cols = min(n_cols, int(t2w.shape[0]))
    slice_indices = _choose_slices(heatmap, pred, gt, n_cols=n_cols)
    if view_mode == "overlay":
        _save_stacked_overlay(
            output_path=output_path,
            title=title,
            t2w=t2w,
            gt=gt,
            pred=pred,
            heatmap=heatmap,
            slice_indices=slice_indices,
            plt=plt,
            mpatches=mpatches,
        )
        return

    if view_mode != "panels":
        raise ValueError(f"view_mode must be 'panels' or 'overlay', got {view_mode!r}")

    _save_panel_overlay(
        output_path=output_path,
        title=title,
        t2w=t2w,
        gt=gt,
        pred=pred,
        heatmap=heatmap,
        slice_indices=slice_indices,
        plt=plt,
        mpatches=mpatches,
    )


def _draw_contours(ax: Any, gt_slice: np.ndarray, pred_slice: np.ndarray, *, linewidth: float = 0.85) -> None:
    if np.any(gt_slice):
        ax.contour(gt_slice > 0, levels=[0.5], colors=["#00ff66"], linewidths=linewidth)
    if np.any(pred_slice):
        ax.contour(pred_slice > 0, levels=[0.5], colors=["#ff3333"], linewidths=linewidth)


def _show_t2w(ax: Any, t2w_slice: np.ndarray) -> None:
    ax.imshow(t2w_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show_heatmap(ax: Any, heatmap_slice: np.ndarray, *, threshold: float, alpha: float) -> None:
    hm = np.ma.masked_where(heatmap_slice <= threshold, heatmap_slice)
    ax.imshow(hm, cmap="magma", vmin=0.0, vmax=1.0, alpha=alpha, interpolation="nearest")


def _save_stacked_overlay(
    output_path: Path,
    title: str,
    t2w: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    heatmap: np.ndarray,
    slice_indices: np.ndarray,
    plt: Any,
    mpatches: Any,
) -> None:
    """Old compact view: all layers in one panel per slice."""

    fig, axes = plt.subplots(
        1,
        len(slice_indices),
        figsize=(len(slice_indices) * 1.35, 1.95),
        squeeze=False,
        gridspec_kw={"wspace": 0.03},
    )
    fig.patch.set_facecolor("#101010")

    for ax, z in zip(axes[0], slice_indices):
        ax.set_facecolor("black")
        _show_t2w(ax, t2w[z])
        _show_heatmap(ax, heatmap[z], threshold=0.10, alpha=0.55)
        ax.imshow(_segmentation_overlay(gt[z], pred[z], alpha=0.38), interpolation="nearest")
        ax.set_title(str(int(z)), fontsize=6, color="#dddddd", pad=1.5)

    legend = [
        mpatches.Patch(facecolor=(*_GT_COLOR, 0.8), label="GT"),
        mpatches.Patch(facecolor=(*_PRED_COLOR, 0.8), label="Pred"),
        mpatches.Patch(facecolor=(*_OVL_COLOR, 0.8), label="Overlap"),
        mpatches.Patch(facecolor=(0.98, 0.40, 0.16, 0.8), label="XAI"),
    ]
    fig.suptitle(title, fontsize=9, color="white", y=0.98)
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        fontsize=7,
        framealpha=0.25,
        facecolor="#222222",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.01),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#101010")
    plt.close(fig)


def _save_panel_overlay(
    output_path: Path,
    title: str,
    t2w: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    heatmap: np.ndarray,
    slice_indices: np.ndarray,
    plt: Any,
    mpatches: Any,
) -> None:
    """Clear diagnostic view: separate anatomy, segmentation, XAI, and combined panels."""
    panel_labels = ("T2w", "Seg", "XAI", "Combined")
    fig, axes = plt.subplots(
        len(panel_labels),
        len(slice_indices),
        figsize=(len(slice_indices) * 1.45, len(panel_labels) * 1.45 + 0.6),
        squeeze=False,
        gridspec_kw={"wspace": 0.025, "hspace": 0.045},
    )
    fig.patch.set_facecolor("#101010")

    for col_idx, z in enumerate(slice_indices):
        z_int = int(z)
        for row_idx, label in enumerate(panel_labels):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")
            _show_t2w(ax, t2w[z_int])

            if label == "Seg":
                _draw_contours(ax, gt[z_int], pred[z_int], linewidth=0.9)
            elif label == "XAI":
                _show_heatmap(ax, heatmap[z_int], threshold=0.25, alpha=0.78)
            elif label == "Combined":
                _show_heatmap(ax, heatmap[z_int], threshold=0.25, alpha=0.46)
                _draw_contours(ax, gt[z_int], pred[z_int], linewidth=0.8)

            if row_idx == 0:
                ax.set_title(f"z={z_int}", fontsize=6, color="#dddddd", pad=1.5)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.94,
                    label,
                    transform=ax.transAxes,
                    fontsize=6,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.58, edgecolor="none", pad=1.5),
                )

    legend = [
        mpatches.Patch(facecolor=(*_GT_COLOR, 0.9), label="GT contour"),
        mpatches.Patch(facecolor=(*_PRED_COLOR, 0.9), label="Prediction contour"),
        mpatches.Patch(facecolor=(0.98, 0.40, 0.16, 0.9), label="XAI heatmap"),
    ]
    fig.suptitle(title, fontsize=9, color="white", y=0.985)
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.25,
        facecolor="#222222",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#101010")
    plt.close(fig)


def _save_saliency_summary(
    output_path: Path,
    title: str,
    t2w: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    combined_heatmap: np.ndarray,
    channel_heatmaps: dict[str, np.ndarray],
    n_cols: int = _N_VIS_COLS,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        raise ImportError("matplotlib is required for XAI visualization.") from exc

    n_cols = min(n_cols, int(t2w.shape[0]))
    slice_indices = _choose_slices(combined_heatmap, pred, gt, n_cols=n_cols)
    rows = [("T2w", None), ("Seg", None), ("Combined", combined_heatmap)]
    rows.extend((f"{modality.upper()} saliency", heatmap) for modality, heatmap in channel_heatmaps.items())

    fig, axes = plt.subplots(
        len(rows),
        len(slice_indices),
        figsize=(len(slice_indices) * 1.45, len(rows) * 1.45 + 0.6),
        squeeze=False,
        gridspec_kw={"wspace": 0.025, "hspace": 0.045},
    )
    fig.patch.set_facecolor("#101010")

    for col_idx, z in enumerate(slice_indices):
        z_int = int(z)
        for row_idx, (label, heatmap) in enumerate(rows):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")
            _show_t2w(ax, t2w[z_int])

            if label == "Seg":
                _draw_contours(ax, gt[z_int], pred[z_int], linewidth=0.9)
            elif heatmap is not None:
                _show_heatmap(ax, heatmap[z_int], threshold=0.20, alpha=0.72)
                if label == "Combined":
                    _draw_contours(ax, gt[z_int], pred[z_int], linewidth=0.65)

            if row_idx == 0:
                ax.set_title(f"z={z_int}", fontsize=6, color="#dddddd", pad=1.5)
            if col_idx == 0:
                ax.text(
                    0.02,
                    0.94,
                    label,
                    transform=ax.transAxes,
                    fontsize=6,
                    color="white",
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="black", alpha=0.58, edgecolor="none", pad=1.5),
                )

    legend = [
        mpatches.Patch(facecolor=(*_GT_COLOR, 0.9), label="GT contour"),
        mpatches.Patch(facecolor=(*_PRED_COLOR, 0.9), label="Prediction contour"),
        mpatches.Patch(facecolor=(0.98, 0.40, 0.16, 0.9), label="Saliency heatmap"),
    ]
    fig.suptitle(title, fontsize=9, color="white", y=0.987)
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        fontsize=7,
        framealpha=0.25,
        facecolor="#222222",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#101010")
    plt.close(fig)


def _volume_voxels(mask: torch.Tensor) -> int:
    return int(mask.detach().to(torch.uint8).sum().item())


def _run_modality_ablation(
    model: nn.Module,
    image_t: torch.Tensor,
    label_batched: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_pred: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    active_modalities: list[str],
) -> list[dict[str, Any]]:
    pp = _postprocess_cfg(cfg)
    baseline_probs = torch.sigmoid(baseline_logits)
    baseline_region = baseline_pred.float()
    if baseline_region.sum() <= 0:
        flat = int(torch.argmax(baseline_probs.flatten()).item())
        baseline_region = torch.zeros_like(baseline_probs)
        baseline_region.flatten()[flat] = 1.0

    rows: list[dict[str, Any]] = []
    for channel_idx, modality in enumerate(active_modalities):
        ablated = image_t.clone()
        ablated[channel_idx] = 0.0
        logits = _run_inference(model, ablated, cfg, device)
        metric_logits, pred_bin = postprocess_logits(logits=logits, **pp)
        metrics = compute_all_metrics(metric_logits, label_batched, threshold=pp["threshold"])
        probs = torch.sigmoid(logits)
        prob_drop = ((baseline_probs - probs) * baseline_region).sum() / baseline_region.sum().clamp_min(1.0)
        rows.append({
            "modality_zeroed": modality,
            "probability_drop_in_baseline_prediction": _json_float(float(prob_drop.item())),
            "predicted_voxels": _volume_voxels(pred_bin),
            "predicted_voxel_delta": _volume_voxels(pred_bin) - _volume_voxels(baseline_pred),
            "metrics": {k: _json_float(float(v)) for k, v in metrics.items()},
        })
    return rows


def _save_modality_ablation_summary(
    output_path: Path,
    case_id: str,
    baseline_voxels: int,
    baseline_metrics: dict[str, float],
    rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for XAI visualization.") from exc

    if not rows:
        return

    modalities = [str(r["modality_zeroed"]) for r in rows]
    prob_drop = [float(r["probability_drop_in_baseline_prediction"] or 0.0) for r in rows]
    voxel_delta = [int(r["predicted_voxel_delta"]) for r in rows]

    fig = plt.figure(figsize=(11.5, 6.8), facecolor="#101010")
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.15), wspace=0.24, hspace=0.30)

    ax_prob = fig.add_subplot(gs[0, 0])
    ax_vox = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    for ax in (ax_prob, ax_vox, ax_text):
        ax.set_facecolor("#161616")

    y = np.arange(len(modalities))
    colors = ["#4cc9f0", "#f72585", "#b8f2a5"]

    ax_prob.barh(y, prob_drop, color=colors[: len(modalities)], edgecolor="none")
    ax_prob.set_yticks(y, labels=[m.upper() for m in modalities], color="white")
    ax_prob.set_xlabel("Probability drop after zeroing modality", color="white")
    ax_prob.set_title("Contribution to baseline prediction", color="white", pad=8)
    ax_prob.tick_params(colors="white")
    ax_prob.grid(axis="x", color="#333333", alpha=0.65, linewidth=0.8)
    ax_prob.invert_yaxis()
    for idx, value in enumerate(prob_drop):
        ax_prob.text(value + 0.015, idx, f"{value:.3f}", va="center", color="white", fontsize=8)

    ax_vox.barh(y, voxel_delta, color=colors[: len(modalities)], edgecolor="none")
    ax_vox.axvline(0, color="#dddddd", linewidth=0.8, alpha=0.5)
    ax_vox.set_yticks(y, labels=[m.upper() for m in modalities], color="white")
    ax_vox.set_xlabel("Predicted voxel delta vs baseline", color="white")
    ax_vox.set_title("Mask change after zeroing modality", color="white", pad=8)
    ax_vox.tick_params(colors="white")
    ax_vox.grid(axis="x", color="#333333", alpha=0.65, linewidth=0.8)
    ax_vox.invert_yaxis()
    for idx, value in enumerate(voxel_delta):
        offset = 8 if value >= 0 else -40
        ax_vox.text(value + offset, idx, f"{value:+d}", va="center", color="white", fontsize=8)

    ax_text.axis("off")
    lines = [
        f"Case: {case_id}",
        f"Baseline predicted voxels: {baseline_voxels}",
        f"Baseline Dice: {_fmt_metric(baseline_metrics.get('dice'))}",
        f"Baseline IoU: {_fmt_metric(baseline_metrics.get('iou'))}",
        f"Baseline Sensitivity: {_fmt_metric(baseline_metrics.get('sensitivity'))}",
        f"Baseline Precision: {_fmt_metric(baseline_metrics.get('precision'))}",
        "",
        "Interpretation:",
    ]
    strongest = sorted(rows, key=lambda r: float(r["probability_drop_in_baseline_prediction"] or 0.0), reverse=True)
    if strongest:
        lead = strongest[0]["modality_zeroed"].upper()
        lead_drop = float(strongest[0]["probability_drop_in_baseline_prediction"] or 0.0)
        lines.append(f"- Most critical modality: {lead} (drop {lead_drop:.3f})")
    zeroed = [r["modality_zeroed"].upper() for r in rows if int(r["predicted_voxels"]) == 0]
    if zeroed:
        lines.append(f"- Removing {', '.join(zeroed)} collapses the prediction to zero voxels.")
    else:
        lines.append("- No single modality fully collapses the prediction.")

    metric_keys = ("dice", "iou", "sensitivity", "precision")
    for key in metric_keys:
        vals = []
        for row in rows:
            v = row["metrics"].get(key)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
        if vals:
            best = max(vals)
            worst = min(vals)
            lines.append(f"- {key.upper()}: {worst:.3f} to {best:.3f} after ablation.")

    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        color="white",
        fontsize=10,
        family="monospace",
        linespacing=1.5,
    )

    fig.suptitle("Modality Attribution Summary", fontsize=12, color="white", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="#101010")
    plt.close(fig)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value_f:.3f}" if math.isfinite(value_f) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XAI overlays for one trained prostate lesion segmentation case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, metavar="DIR", help="Training run directory.")
    parser.add_argument("--checkpoint", default=None, metavar="PT", help="Checkpoint path/name. Defaults to best.pt/newest.")
    parser.add_argument("--t2w", required=True, metavar="PATH", help="T2w .mha path for the case.")
    parser.add_argument("--adc", default=None, metavar="PATH", help="Optional ADC .mha path override.")
    parser.add_argument("--hbv", default=None, metavar="PATH", help="Optional HBV .mha path override.")
    parser.add_argument("--seg", default=None, metavar="PATH", help="Optional GT label path. Auto-detected when absent.")
    parser.add_argument(
        "--method",
        default="all",
        choices=("gradcam", "saliency", "modality-ablation", "all"),
        help="Explanation method to run.",
    )
    parser.add_argument("--target-layer", default="", help="Model module name for Grad-CAM.")
    parser.add_argument("--output-dir", default="visualizations/xai", metavar="DIR", help="Artifact directory.")
    parser.add_argument("--device", default=None, metavar="DEVICE", help="Override device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--sw-batch-size", type=int, default=None, help="Override sliding-window batch size.")
    parser.add_argument("--n-cols", type=int, default=_N_VIS_COLS, help="Number of slices in PNG overlays.")
    parser.add_argument(
        "--view-mode",
        choices=("panels", "overlay"),
        default="panels",
        help="PNG layout: separate diagnostic panels or old compact stacked overlay.",
    )
    parser.add_argument("--list-layers", action="store_true", help="Print Grad-CAM candidate layers and exit.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    cfg = load_config(str(run_dir / "config.yaml"))
    if args.sw_batch_size is not None:
        cfg["sw_batch_size"] = int(args.sw_batch_size)

    t2w_path = Path(args.t2w).resolve()
    if not t2w_path.is_file():
        raise FileNotFoundError(f"T2w file not found: {t2w_path}")

    try:
        seg_path: Path | None = _resolve_seg_path(t2w_path, args.seg)
    except FileNotFoundError:
        if args.seg:
            raise
        seg_path = None
        logger.warning("Ground truth label was not found; metrics and GT overlay will be omitted.")

    device = _resolve_device(args.device, cfg)
    ckpt_path = _resolve_checkpoint_path(run_dir, args.checkpoint)
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    if args.list_layers:
        print("\n".join(_candidate_module_names(model)))
        return

    image_t, label_t, t2w_display, _spacing_zyx, used_paths = _load_model_inputs(
        cfg=cfg,
        t2w_path=t2w_path,
        seg_path=seg_path,
        adc_path=args.adc,
        hbv_path=args.hbv,
    )

    pp = _postprocess_cfg(cfg)
    label_batched = label_t.unsqueeze(0)
    logits = _run_inference(model, image_t, cfg, device)
    metric_logits, pred_bin = postprocess_logits(logits=logits, **pp)
    metrics = compute_all_metrics(metric_logits, label_batched, threshold=pp["threshold"]) if seg_path else {}

    active_modalities = [key for key, _ in active_modality_pairs(cfg)]
    case_id = _case_id_from_t2w(t2w_path)
    stem = f"{_safe_filename_component(run_dir.name)}_{_safe_filename_component(case_id)}"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    center = _select_roi_center(pred_bin, label_t, logits)
    roi_size = tuple(int(v) for v in cfg.get("patch_size", [20, 128, 128]))
    roi = _roi_slices(tuple(int(v) for v in image_t.shape[1:]), center, roi_size)
    image_roi = image_t[(slice(None),) + roi]
    target_layer = args.target_layer or _default_target_layer_name(model, str(cfg.get("model", "unet3d")).lower())

    artifacts: dict[str, Any] = {}
    method_errors: dict[str, str] = {}
    methods = {args.method} if args.method != "all" else {"gradcam", "saliency", "modality-ablation"}

    # Attribution passes need gradients, so do not wrap this block in no_grad.
    grad_context = nullcontext()
    with grad_context:
        if "gradcam" in methods:
            try:
                cam_roi = _run_gradcam_roi(
                    model=model,
                    image_roi=image_roi,
                    device=device,
                    target_layer_name=target_layer,
                    threshold=pp["threshold"],
                )
                cam_full = _insert_roi(t2w_display.shape, cam_roi, roi)
                path = output_dir / f"{stem}_gradcam.png"
                _save_heatmap_overlay(
                    output_path=path,
                    title=f"{case_id} Grad-CAM ({target_layer})",
                    t2w=t2w_display,
                    gt=label_t[0].numpy().astype(np.uint8),
                    pred=pred_bin[0, 0].numpy().astype(np.uint8),
                    heatmap=cam_full,
                    n_cols=args.n_cols,
                    view_mode=args.view_mode,
                )
                artifacts["gradcam_png"] = str(path)
            except Exception as exc:
                method_errors["gradcam"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Grad-CAM failed: %s", exc)

        if "saliency" in methods:
            try:
                saliency_roi, per_channel_roi = _run_saliency_roi(
                    model=model,
                    image_roi=image_roi,
                    device=device,
                    threshold=pp["threshold"],
                )
                saliency_full = _insert_roi(t2w_display.shape, saliency_roi, roi)
                path = output_dir / f"{stem}_saliency.png"
                channel_paths: dict[str, str] = {}
                channel_heatmaps: dict[str, np.ndarray] = {}
                for idx, modality in enumerate(active_modalities):
                    if idx >= per_channel_roi.shape[0]:
                        continue
                    channel_full = _insert_roi(t2w_display.shape, per_channel_roi[idx], roi)
                    channel_heatmaps[modality] = channel_full
                    ch_path = output_dir / f"{stem}_saliency_{modality}.png"
                    _save_heatmap_overlay(
                        output_path=ch_path,
                        title=f"{case_id} {modality.upper()} saliency",
                        t2w=t2w_display,
                        gt=label_t[0].numpy().astype(np.uint8),
                        pred=pred_bin[0, 0].numpy().astype(np.uint8),
                        heatmap=channel_full,
                        n_cols=args.n_cols,
                        view_mode=args.view_mode,
                    )
                    channel_paths[modality] = str(ch_path)
                _save_saliency_summary(
                    output_path=path,
                    title=f"{case_id} input saliency by modality",
                    t2w=t2w_display,
                    gt=label_t[0].numpy().astype(np.uint8),
                    pred=pred_bin[0, 0].numpy().astype(np.uint8),
                    combined_heatmap=saliency_full,
                    channel_heatmaps=channel_heatmaps,
                    n_cols=args.n_cols,
                )
                artifacts["saliency_png"] = str(path)
                artifacts["saliency_channel_pngs"] = channel_paths
            except Exception as exc:
                method_errors["saliency"] = f"{type(exc).__name__}: {exc}"
                logger.exception("Saliency failed: %s", exc)

    modality_ablation: list[dict[str, Any]] = []
    if "modality-ablation" in methods:
        try:
            modality_ablation = _run_modality_ablation(
                model=model,
                image_t=image_t,
                label_batched=label_batched,
                baseline_logits=logits,
                baseline_pred=pred_bin,
                cfg=cfg,
                device=device,
                active_modalities=active_modalities,
            )
            ablation_path = output_dir / f"{stem}_modality_ablation.json"
            ablation_path.write_text(json.dumps(modality_ablation, indent=2, sort_keys=True), encoding="utf-8")
            artifacts["modality_ablation_json"] = str(ablation_path)
            ablation_png = output_dir / f"{stem}_modality_ablation_summary.png"
            _save_modality_ablation_summary(
                output_path=ablation_png,
                case_id=case_id,
                baseline_voxels=_volume_voxels(pred_bin),
                baseline_metrics=metrics if isinstance(metrics, dict) else {},
                rows=modality_ablation,
            )
            artifacts["modality_ablation_png"] = str(ablation_png)
        except Exception as exc:
            method_errors["modality-ablation"] = f"{type(exc).__name__}: {exc}"
            logger.exception("Modality ablation failed: %s", exc)

    if method_errors and len(method_errors) == len(methods):
        failed_methods = ", ".join(sorted(method_errors))
        raise RuntimeError(f"All requested explanation methods failed: {failed_methods}")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint": {
            "path": str(ckpt_path),
            "name": ckpt_path.name,
            "epoch": ckpt.get("epoch", None),
        },
        "case": {
            "case_id": case_id,
            "t2w": str(t2w_path),
            "seg": str(seg_path) if seg_path else None,
            "used_modalities": {k: str(v) for k, v in used_paths.items()},
        },
        "model": {
            "name": cfg.get("model", "unet3d"),
            "active_modalities": active_modalities,
            "target_layer": target_layer if "gradcam" in methods else None,
        },
        "inference": {
            "device": str(device),
            "patch_size": list(roi_size),
            "sw_batch_size": int(cfg.get("sw_batch_size", 4)),
            "sw_overlap": float(cfg.get("sw_overlap", 0.5)),
            "pred_threshold": pp["threshold"],
            "postprocess_enabled": pp["enabled"],
            "view_mode": args.view_mode,
        },
        "roi": {
            "center_zyx": list(center),
            "slices_zyx": [[int(s.start), int(s.stop)] for s in roi],
        },
        "prediction": {
            "predicted_voxels": _volume_voxels(pred_bin),
            "ground_truth_voxels": int(label_t.sum().item()) if seg_path else None,
            "metrics": {k: _json_float(float(v)) for k, v in metrics.items()},
        },
        "modality_ablation": modality_ablation,
        "requested_methods": sorted(methods),
        "method_errors": method_errors,
        "artifacts": artifacts,
    }
    summary_path = output_dir / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Summary saved: {summary_path}")
    if method_errors:
        print("method_errors:")
        for method_name, error_msg in sorted(method_errors.items()):
            print(f"  {method_name}: {error_msg}")
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
