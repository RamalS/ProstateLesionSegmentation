"""
evaluate_checkpoint.py — Load a trained checkpoint and evaluate on the
configured hold-out test set.

The test set is a permanently reserved group of 10 PI-CAI cases (5 positive,
5 negative) that are segregated from the training pool by download_dataset.sh.
Using a fixed set makes all evaluation runs directly comparable across
checkpoints and training runs. For ``dataset_type: prostate158`` runs, the
script reads the Prostate158 test CSV from ``prostate158_test_dir`` instead.

Usage
-----
python scripts/evaluate_checkpoint.py \\
    --run outputs/runs/<run> \\
    [--images-dir data/test_images] \\
    [--labels-dir data/labels]

python scripts/evaluate_checkpoint.py \\
    --external-model monai:prostate_mri_anatomy@0.3.5 \\
    --dataset-type prostate158 \\
    --prostate158-prostate-label-col t2_prostate_reader1

python scripts/evaluate_checkpoint.py \\
    --run outputs/runs/<run> \\
    --dataset-type picai \\
    --roi-mode gt_mask \\
    --picai-prostate-labels-dir data/prostate_labels

The run directory must contain:
  config.yaml       — the YAML config the model was trained with
  checkpoints/      — one or more .pt checkpoint files

An interactive arrow-key menu lets you select which checkpoint to evaluate.
In non-interactive environments (no TTY) best.pt is selected automatically;
if best.pt is absent the newest checkpoint by filename is used.

Output
------
- Per-case metrics table printed to stdout.
- Aggregate metric summary printed to stdout.
- eval_visualization.png (5 rows × 20 axial slices) saved to
  ``visualizations/<run_name>_eval_visualization.png`` by default,
  with semi-transparent colour overlays:
      Green  = ground truth only
      Red    = prediction only
      Yellow = overlap (both masks active)
"""

from __future__ import annotations

import argparse
import csv
import curses
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve src/ imports (scripts/ are not on PYTHONPATH by default)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from dataset import (  # noqa: E402
    PiCaiDataset,
    _ProstateROILocalizer,
    _prepare_case_image_tensor,
    annotate_cases_with_lesion_flags,
    active_modality_pairs,
    default_split_manifest_path,
    discover_cases,
    discover_prostate158_cases,
    load_split_manifest,
)
from external_models import (  # noqa: E402
    MonaiBundleProstateMaskAdapter,
    build_external_eval_config,
    build_external_localizer_ref,
    default_external_model_cache_root,
    list_supported_external_models,
    parse_external_localizer_ref,
    resolve_external_model_request,
)
from metrics import compute_all_metrics  # noqa: E402
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from roi import (  # noqa: E402
    ROI_PREDICTED_MASK,
    binarize_mask,
    compute_crop_bounds,
    crop_bounds_from_dict,
    keep_largest_component,
    resolve_localizer_checkpoint,
    resolve_roi_settings,
    restore_from_roi,
    validate_task_and_roi_config,
)
from transforms import get_val_transforms  # noqa: E402
from utils import load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — configured once here (entry-point only)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress MONAI class-balanced sampler warnings that fire during
# sliding-window inference on volumes that have no lesion voxels.
warnings.filterwarnings(
    "ignore",
    message=".*unable to generate class balanced samples.*",
    category=UserWarning,
    module="monai",
)

# ---------------------------------------------------------------------------
# Visualization constants
# ---------------------------------------------------------------------------

_N_VIS_COLS: int = 20                                        # axial slices per row
_GT_COLOR:   tuple[float, float, float] = (0.0, 1.0, 0.0)   # green — ground truth
_PRED_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)   # red   — prediction
_OVL_COLOR:  tuple[float, float, float] = (1.0, 1.0, 0.0)   # yellow — overlap
_OVERLAY_ALPHA: float = 0.50


@dataclass(frozen=True)
class ModelCandidate:
    """Model candidate shown in the interactive selector."""
    label: str
    model_source: str
    task: str
    dataset_type: str
    run_dir: Path | None = None
    external_model_id: str = ""
    external_model_version: str = ""


@dataclass(frozen=True)
class ROIVariant:
    """ROI override option selected in interactive batch mode."""
    label: str
    mode: str
    localizer_run: str = ""
    localizer_external_model_id: str = ""
    localizer_external_model_version: str = ""


@dataclass(frozen=True)
class BatchEvalJob:
    """One concrete evaluation job in interactive batch mode."""
    model_source: str
    dataset_type: str
    eval_split: str
    task: str
    roi_mode: str
    roi_localizer_run: str
    roi_localizer_external_model_id: str
    roi_localizer_external_model_version: str
    summary_json: Path
    vis_output: Path
    run_dir: Path | None = None
    checkpoint: Path | None = None
    external_model_id: str = ""
    external_model_version: str = ""


@dataclass(frozen=True)
class BatchWorkflow:
    """Interactive selector workflow mode."""
    key: str
    label: str


_ROI_BOUNDS_CACHE_VERSION = 1
_PICAI_GT_MASK_SENTINEL_PROSTATE_COL = "__picai_gt_mask__"


# ---------------------------------------------------------------------------
# Console formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a metric value to 4 d.p.; return 'n/a' for NaN."""
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _json_float(v: float) -> float | None:
    """Convert metric to JSON-safe float; return null for NaN/inf."""
    return float(v) if math.isfinite(v) else None


_PICAI_LABEL_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def _picai_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Official PI-CAI default overlap basis: lesion IoU."""
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return (intersection + 1e-8) / (union + 1e-8)


def _detection_map_from_probability(
    prob_zyx: np.ndarray,
    pred_mask_zyx: np.ndarray,
) -> np.ndarray:
    """
    Convert voxel probabilities into a PI-CAI detection map.

    PI-CAI expects each connected lesion candidate to have one confidence
    value. We use the max probability inside each post-threshold/postprocessed
    component, matching the official case-confidence default of max score.
    """
    mask = pred_mask_zyx.astype(bool, copy=False)
    labels, num_components = ndimage.label(mask, structure=_PICAI_LABEL_STRUCTURE)
    det = np.zeros_like(prob_zyx, dtype=np.float32)
    for component_id in range(1, num_components + 1):
        component = labels == component_id
        det[component] = float(prob_zyx[component].max())
    return det


def _picai_evaluate_case(
    det_zyx: np.ndarray,
    target_zyx: np.ndarray,
    min_overlap: float = 0.10,
) -> tuple[list[tuple[int, float, float]], float]:
    """
    PI-CAI lesion matching for one case.

    Mirrors picai_eval defaults: 26-connected components, IoU >= 0.1 hit
    criterion, Hungarian assignment prioritizing the number of matches, and
    unmatched candidates that overlap a matched GT lesion sufficiently are
    ignored rather than counted as FPs.
    """
    det = det_zyx.astype(np.float32, copy=False)
    target = (target_zyx > 0).astype(np.int32, copy=False)
    if det.min(initial=0.0) < 0.0:
        raise ValueError("PI-CAI detection confidences must be non-negative.")

    pred_labels, num_pred = ndimage.label(det > 0, structure=_PICAI_LABEL_STRUCTURE)
    pred_ids = np.arange(num_pred)
    confidences = {
        pred_id: float(det[pred_labels == (pred_id + 1)].max())
        for pred_id in pred_ids
    }
    case_confidence = float(det.max(initial=0.0))
    lesion_results: list[tuple[int, float, float]] = []

    if not target.any():
        lesion_results.extend((0, confidence, 0.0) for confidence in confidences.values())
        return lesion_results, case_confidence

    gt_labels, num_gt = ndimage.label(target, structure=_PICAI_LABEL_STRUCTURE)
    gt_ids = np.arange(num_gt)
    overlap_matrix = np.zeros((num_gt, num_pred), dtype=np.float64)

    for gt_id in gt_ids:
        gt_mask = gt_labels == (gt_id + 1)
        for pred_id in pred_ids:
            pred_mask = pred_labels == (pred_id + 1)
            overlap_matrix[gt_id, pred_id] = _picai_iou(pred_mask, gt_mask)

    match_matrix = overlap_matrix.copy()
    match_matrix[match_matrix < min_overlap] = 0.0
    match_matrix[match_matrix > 0.0] += 1.0

    if num_gt > 0 and num_pred > 0:
        matched_gt, matched_pred = linear_sum_assignment(match_matrix, maximize=True)
        keep = match_matrix[matched_gt, matched_pred] > 0.0
        matched_gt = matched_gt[keep]
        matched_pred = matched_pred[keep]
    else:
        matched_gt = np.array([], dtype=np.int64)
        matched_pred = np.array([], dtype=np.int64)

    for gt_id, pred_id in zip(matched_gt, matched_pred):
        overlap = float(match_matrix[gt_id, pred_id] - 1.0)
        lesion_results.append((1, confidences[int(pred_id)], overlap))

    unmatched_gt = set(int(gt_id) for gt_id in gt_ids) - set(int(gt_id) for gt_id in matched_gt)
    lesion_results.extend((1, 0.0, 0.0) for _ in unmatched_gt)

    candidates_sufficient_overlap = set(
        int(pred_id) for pred_id in pred_ids[(match_matrix > 0.0).any(axis=0)]
    )
    unmatched_pred = set(int(pred_id) for pred_id in pred_ids) - candidates_sufficient_overlap
    lesion_results.extend((0, confidences[pred_id], 0.0) for pred_id in unmatched_pred)

    return lesion_results, case_confidence


def _picai_average_precision(
    lesion_results: list[tuple[int, float, float]],
) -> float:
    if not lesion_results or not any(is_lesion for is_lesion, _, _ in lesion_results):
        return float("nan")
    y_true = np.asarray([is_lesion for is_lesion, _, _ in lesion_results], dtype=np.int32)
    y_score = np.asarray([confidence for _, confidence, _ in lesion_results], dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y_true=y_true, y_score=y_score)
    precision[:-1][thresholds == 0.0] = 0.0
    return float(-np.sum(np.diff(recall) * np.asarray(precision)[:-1]))


def _picai_auroc(case_targets: list[int], case_scores: list[float]) -> float:
    if len(set(case_targets)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true=case_targets, y_score=case_scores)
    return float(auc(fpr, tpr))


def _picai_ranking_metrics(per_case: list[dict]) -> dict[str, Any]:
    lesion_results = [
        item
        for row in per_case
        for item in row.get("picai_lesion_results", [])
    ]
    case_targets = [int(row["has_target"]) for row in per_case]
    case_scores = [float(row.get("picai_case_confidence", 0.0)) for row in per_case]

    ap = _picai_average_precision(lesion_results)
    auroc = _picai_auroc(case_targets, case_scores)
    score = (ap + auroc) / 2.0 if math.isfinite(ap) and math.isfinite(auroc) else float("nan")
    return {
        "AP": ap,
        "AUROC": auroc,
        "score": score,
        "num_lesions": int(sum(is_lesion for is_lesion, _, _ in lesion_results)),
        "num_candidates": int(
            sum(
                1
                for is_lesion, confidence, _ in lesion_results
                if not is_lesion and confidence > 0.0
            )
        ),
    }


def _seg_logits(outputs: object) -> torch.Tensor:
    """
    Return segmentation logits tensor from model outputs.

    Supported output structures:
    - Tensor (single-task models)
    - list[Tensor] (deep supervision; uses finest-scale logits at index 0)
    - dict with key ``"seg"`` (multi-task models)
    """
    if isinstance(outputs, dict):
        outputs = outputs.get("seg")

    if isinstance(outputs, list):
        if not outputs:
            raise ValueError("Model returned an empty segmentation output list.")
        first = outputs[0]
        if isinstance(first, torch.Tensor):
            return first
        raise TypeError(
            f"Expected first deep-supervision output to be a Tensor, got {type(first)!r}."
        )

    if isinstance(outputs, torch.Tensor):
        return outputs

    raise TypeError(f"Unsupported model output type: {type(outputs)!r}.")


def _section(title: str) -> None:
    """Print a labelled section divider to stdout."""
    bar = "─" * 68
    print(f"\n{bar}\n  {title}\n{bar}")


def _roi_triplet_from_collate(value: Any, key: str) -> tuple[int, int, int]:
    """
    Parse one ROI triplet field from DataLoader-collated batch payload.

    Supports these common collated forms:
    - list[tensor([z]), tensor([y]), tensor([x])]
    - tensor([[z, y, x]]) or tensor([z, y, x])
    """
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] == 3:
            vals = arr[0].tolist()
            return int(vals[0]), int(vals[1]), int(vals[2])
        if arr.ndim == 1 and arr.numel() == 3:
            vals = arr.tolist()
            return int(vals[0]), int(vals[1]), int(vals[2])
        raise ValueError(
            f"Unexpected tensor shape for roi.{key}: {tuple(arr.shape)}"
        )

    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"Expected 3 values for roi.{key}, got {len(value)}")
        out: list[int] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                flat = item.detach().cpu().reshape(-1)
                if flat.numel() != 1:
                    raise ValueError(
                        f"Expected scalar tensor entries for roi.{key}, got {tuple(item.shape)}"
                    )
                out.append(int(flat[0].item()))
            else:
                out.append(int(item))
        return out[0], out[1], out[2]

    raise ValueError(f"Unsupported collated type for roi.{key}: {type(value)!r}")


def _roi_bool_from_collate(value: Any) -> bool:
    """Parse boolean ROI field from collated batch payload."""
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() == 0:
            return False
        return bool(flat[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        head = value[0]
        if isinstance(head, torch.Tensor):
            flat = head.detach().cpu().reshape(-1)
            return bool(flat[0].item()) if flat.numel() else False
        return bool(head)
    return bool(value)


def _roi_bounds_from_collated_batch(roi_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize DataLoader-collated ROI metadata to restore_from_roi format."""
    required = ("start_zyx", "end_zyx", "full_shape_zyx", "used_fallback")
    missing = [k for k in required if k not in roi_payload]
    if missing:
        raise KeyError(f"ROI payload missing required key(s): {missing}")

    return {
        "start_zyx": _roi_triplet_from_collate(roi_payload["start_zyx"], "start_zyx"),
        "end_zyx": _roi_triplet_from_collate(roi_payload["end_zyx"], "end_zyx"),
        "full_shape_zyx": _roi_triplet_from_collate(roi_payload["full_shape_zyx"], "full_shape_zyx"),
        "used_fallback": _roi_bool_from_collate(roi_payload["used_fallback"]),
    }


def _roi_bounds_cache_root(repo_root: Path) -> Path:
    return (repo_root / "outputs" / "evaluation_summaries" / "roi_bounds_cache").resolve()


def _localizer_identity(localizer_run: str) -> dict[str, Any]:
    external_spec = parse_external_localizer_ref(localizer_run)
    if external_spec is not None:
        return {
            "source": external_spec.model_source,
            "external_model_id": external_spec.model_id,
            "external_model_version": external_spec.bundle_version,
        }
    cfg_path, ckpt_path = resolve_localizer_checkpoint(localizer_run)
    ckpt_stat = ckpt_path.stat()
    cfg_stat = cfg_path.stat()
    return {
        "source": "repo_run",
        "run": str(Path(localizer_run).expanduser().resolve()),
        "config_path": str(cfg_path.resolve()),
        "config_mtime_ns": int(cfg_stat.st_mtime_ns),
        "checkpoint_path": str(ckpt_path.resolve()),
        "checkpoint_mtime_ns": int(ckpt_stat.st_mtime_ns),
        "checkpoint_size": int(ckpt_stat.st_size),
    }


def _roi_bounds_cache_key(
    case: Mapping[str, Any],
    *,
    target_spacing: tuple[float, ...],
    roi_settings: Any,
    localizer_identity: Mapping[str, Any],
) -> str:
    case_paths = {
        key: str(Path(case[key]).expanduser().resolve())
        for key in ("t2w", "adc", "hbv")
        if key in case and case[key] is not None
    }
    cache_meta = {
        "version": _ROI_BOUNDS_CACHE_VERSION,
        "case_id": str(case["case_id"]),
        "case_paths": case_paths,
        "hbv_source": str(case.get("hbv_source", "hbv")),
        "target_spacing": [float(v) for v in target_spacing],
        "localizer": dict(localizer_identity),
        "roi": {
            "threshold": float(roi_settings.localizer_threshold),
            "keep_largest_component": bool(roi_settings.localizer_keep_largest_component),
            "margin_mm": [float(v) for v in roi_settings.margin_mm],
            "min_size_vox": [int(v) for v in roi_settings.min_size_vox],
            "fallback_to_full_volume": bool(roi_settings.fallback_to_full_volume),
        },
    }
    return hashlib.sha1(
        json.dumps(cache_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _roi_bounds_cache_path(
    cache_root: Path,
    case: Mapping[str, Any],
    *,
    target_spacing: tuple[float, ...],
    roi_settings: Any,
    localizer_identity: Mapping[str, Any],
) -> Path:
    digest = _roi_bounds_cache_key(
        case,
        target_spacing=target_spacing,
        roi_settings=roi_settings,
        localizer_identity=localizer_identity,
    )
    safe_case_id = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
        for ch in str(case["case_id"])
    )
    return cache_root / f"{safe_case_id}_{digest}.json"


def _load_roi_bounds_cache_entry(
    cache_path: Path,
    *,
    case_id: str,
) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        bounds_payload = payload.get("bounds", payload)
        bounds = crop_bounds_from_dict(bounds_payload).as_dict()
        payload_case_id = str(payload.get("case_id", case_id))
        if payload_case_id != case_id:
            raise ValueError(
                f"cache case_id mismatch: expected {case_id!r}, got {payload_case_id!r}"
            )
        return bounds
    except Exception as exc:
        logger.warning(
            "Unreadable ROI bounds cache entry for case '%s' at %s (%s); recomputing.",
            case_id,
            cache_path,
            exc,
        )
        return None


def _write_roi_bounds_cache_entry(
    cache_path: Path,
    *,
    case: Mapping[str, Any],
    bounds: Mapping[str, Any],
    cache_key: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _ROI_BOUNDS_CACHE_VERSION,
        "cache_key": cache_key,
        "case_id": str(case["case_id"]),
        "bounds": crop_bounds_from_dict(bounds).as_dict(),
    }
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, cache_path)


class _ExternalROILocalizer:
    def __init__(
        self,
        localizer_spec: Any,
        *,
        target_spacing: tuple[float, ...],
        device: torch.device,
        cache_root: Path,
    ) -> None:
        self.spec = localizer_spec
        self.target_spacing = target_spacing
        self.device = device
        self.adapter = MonaiBundleProstateMaskAdapter(
            spec=localizer_spec,
            device=device,
            cache_root=cache_root,
        )

    def predict_mask(self, case: dict[str, Any]) -> np.ndarray:
        image = _prepare_case_image_tensor(
            case=case,
            target_spacing=self.target_spacing,
            active_modalities=self.spec.required_modalities,
            dwi_hbv_preprocess_enabled=False,
            dwi_hbv_clip_percentiles=(1.0, 99.5),
            dwi_hbv_log1p=True,
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = self.adapter.predict_logits(image, sw_batch_size=1)
            probs = torch.sigmoid(logits.float())
        return probs[0, 0].detach().cpu().numpy()

    def close(self) -> None:
        self.adapter.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _build_roi_localizer_predictor(
    *,
    roi_settings: Any,
    target_spacing: tuple[float, ...],
    device: torch.device,
    repo_root: Path,
) -> Any:
    external_spec = parse_external_localizer_ref(roi_settings.localizer_run)
    if external_spec is not None:
        cache_root = default_external_model_cache_root(repo_root)
        try:
            return _ExternalROILocalizer(
                external_spec,
                target_spacing=target_spacing,
                device=device,
                cache_root=cache_root,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "External ROI localizer init failed for %s@%s: %s",
                external_spec.model_id,
                external_spec.bundle_version,
                exc,
            )
            logger.error(
                "Operator hint: container cannot reach MONAI Hub/hosting. "
                "Set HTTP_PROXY/HTTPS_PROXY/NO_PROXY (and lowercase variants) "
                "or seed the bundle cache at %s.",
                cache_root,
            )
            raise RuntimeError(
                "External ROI localizer initialization failed. "
                "Container cannot reach MONAI Hub/hosting; set proxy env vars "
                f"or seed cache at {cache_root}."
            ) from exc
    return _ProstateROILocalizer(
        localizer_run=roi_settings.localizer_run,
        target_spacing=target_spacing,
        device=device,
    )


def _precompute_predicted_roi_bounds(
    *,
    test_cases: list[dict[str, Any]],
    roi_settings: Any,
    target_spacing: tuple[float, ...],
    device: torch.device,
    repo_root: Path,
) -> dict[str, Any]:
    if roi_settings.mode != ROI_PREDICTED_MASK:
        return {
            "enabled": False,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_dir": None,
            "device": None,
        }

    cache_root = _roi_bounds_cache_root(repo_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    localizer_identity = _localizer_identity(roi_settings.localizer_run)

    misses: list[tuple[dict[str, Any], Path, str]] = []
    cache_hits = 0
    for case in test_cases:
        cache_path = _roi_bounds_cache_path(
            cache_root,
            case,
            target_spacing=target_spacing,
            roi_settings=roi_settings,
            localizer_identity=localizer_identity,
        )
        cache_key = _roi_bounds_cache_key(
            case,
            target_spacing=target_spacing,
            roi_settings=roi_settings,
            localizer_identity=localizer_identity,
        )
        case["_roi_bounds_predicted_path"] = str(cache_path)
        cached_bounds = _load_roi_bounds_cache_entry(
            cache_path,
            case_id=str(case["case_id"]),
        )
        if cached_bounds is not None:
            case["_roi_bounds_predicted"] = cached_bounds
            cache_hits += 1
            continue
        misses.append((case, cache_path, cache_key))

    logger.info(
        "Predicted ROI bounds cache: %d hits, %d misses (%s).",
        cache_hits,
        len(misses),
        cache_root,
    )

    if misses:
        localizer = _build_roi_localizer_predictor(
            roi_settings=roi_settings,
            target_spacing=target_spacing,
            device=device,
            repo_root=repo_root,
        )
        try:
            for case, cache_path, cache_key in tqdm(
                misses,
                desc="ROI precompute",
                unit="case",
            ):
                roi_mask = localizer.predict_mask(case)
                roi_mask = binarize_mask(
                    roi_mask,
                    threshold=roi_settings.localizer_threshold,
                )
                if roi_settings.localizer_keep_largest_component:
                    roi_mask = keep_largest_component(roi_mask)
                bounds = compute_crop_bounds(
                    mask=roi_mask,
                    spacing_zyx=target_spacing,
                    margin_mm=roi_settings.margin_mm,
                    min_size_vox=roi_settings.min_size_vox,
                    fallback_to_full_volume=roi_settings.fallback_to_full_volume,
                ).as_dict()
                case["_roi_bounds_predicted"] = bounds
                _write_roi_bounds_cache_entry(
                    cache_path,
                    case=case,
                    bounds=bounds,
                    cache_key=cache_key,
                )
        finally:
            localizer.close()
            del localizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info(
                "Predicted ROI localizer precompute finished; released localizer before lesion inference."
            )

    return {
        "enabled": True,
        "cache_hits": cache_hits,
        "cache_misses": len(misses),
        "cache_dir": str(cache_root),
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _normalize_vol_for_display(vol: np.ndarray) -> np.ndarray:
    """
    Clip a 3-D float volume to its [p1, p99] percentile range then
    scale linearly to [0, 1] for display.

    Parameters
    ----------
    vol : (D, H, W) float array

    Returns
    -------
    (D, H, W) float32 in [0, 1]
    """
    p1  = float(np.percentile(vol, 1))
    p99 = float(np.percentile(vol, 99))
    clipped = np.clip(vol, p1, p99)
    return ((clipped - p1) / max(p99 - p1, 1e-8)).astype(np.float32)


def _segmentation_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Build an RGBA (H, W, 4) overlay from binary GT and prediction masks.

    Colour coding
    -------------
    - **Green**  where only GT is 1
    - **Red**    where only Pred is 1
    - **Yellow** where both are 1  (explicit, not an alpha blend artefact)
    - Transparent (alpha = 0) where neither mask is active

    Parameters
    ----------
    gt   : (H, W) binary array {0, 1}
    pred : (H, W) binary array {0, 1}
    alpha : opacity for coloured regions in [0, 1]

    Returns
    -------
    (H, W, 4) float32 RGBA array
    """
    h, w = gt.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    gt_only   = (gt > 0) & (pred == 0)
    pred_only = (pred > 0) & (gt == 0)
    both      = (gt > 0) & (pred > 0)

    rgba[gt_only,   :3] = _GT_COLOR    # green
    rgba[pred_only, :3] = _PRED_COLOR  # red
    rgba[both,      :3] = _OVL_COLOR   # yellow

    active = gt_only | pred_only | both
    rgba[active, 3] = alpha

    return rgba


def save_visualization(
    pos_results: list[dict],
    output_path: Path,
    n_cols: int = _N_VIS_COLS,
) -> None:
    """
    Save a grid of axial overlay images to *output_path*.

    Layout
    ------
    - Rows    : one per positive case (at most 5)
    - Columns : *n_cols* (default 20) evenly-spaced axial slices
    - Per cell:
        - Greyscale T2w channel as background
        - Semi-transparent colour overlay (green / red / yellow)

    Colour key
    ----------
    Green  = ground truth lesion  (GT only)
    Red    = model prediction     (Pred only)
    Yellow = overlap of both masks

    Parameters
    ----------
    pos_results : list of dicts; each must contain:
                  "case_id"  str
                  "t2w_vol"  np.ndarray (D, H, W)  z-scored T2w channel
                  "gt_vol"   np.ndarray (D, H, W)  binary GT mask
                  "pred_vol" np.ndarray (D, H, W)  binary prediction mask
                  "roi_bounds" dict[str, Any] | None  ROI crop bounds metadata
    output_path : destination PNG file path
    n_cols      : number of axial slices per row
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend; safe for scripts
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
        return

    n_rows = len(pos_results)
    if n_rows == 0:
        logger.warning("No positive cases available — skipping visualization.")
        return

    cell_w, cell_h = 1.15, 1.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h + 0.5),
        squeeze=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )
    fig.patch.set_facecolor("#111111")

    for row_idx, res in enumerate(pos_results):
        case_id  = res["case_id"]
        t2w_norm = _normalize_vol_for_display(res["t2w_vol"])  # (D, H, W) in [0,1]
        gt_vol   = res["gt_vol"]                               # (D, H, W) binary
        pred_vol = res["pred_vol"]                             # (D, H, W) binary
        roi_bounds = res.get("roi_bounds")

        D = t2w_norm.shape[0]
        slice_indices = np.linspace(0, D - 1, n_cols, dtype=int)

        for col_idx, s in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")

            # Layer 1: T2w greyscale background
            ax.imshow(
                t2w_norm[s],
                cmap="gray", vmin=0.0, vmax=1.0,
                aspect="equal", interpolation="nearest",
            )

            # Layer 2: segmentation overlay (green / red / yellow)
            ax.imshow(
                _segmentation_overlay(gt_vol[s], pred_vol[s], _OVERLAY_ALPHA),
                aspect="equal", interpolation="nearest",
            )

            if (
                isinstance(roi_bounds, Mapping)
                and not bool(roi_bounds.get("used_fallback", False))
            ):
                start_z, start_y, start_x = (int(v) for v in roi_bounds["start_zyx"])
                end_z, end_y, end_x = (int(v) for v in roi_bounds["end_zyx"])
                if start_z <= s < end_z:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (start_x - 0.5, start_y - 0.5),
                            end_x - start_x,
                            end_y - start_y,
                            fill=False,
                            edgecolor="#00D7FF",
                            linewidth=0.9,
                            linestyle="-",
                        )
                    )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column header: slice index (top row only)
            if row_idx == 0:
                ax.set_title(str(s), fontsize=5, color="#aaaaaa", pad=1.5)

        # Case-ID label in the top-left corner of the first column
        axes[row_idx, 0].text(
            0.03, 0.97,
            case_id,
            transform=axes[row_idx, 0].transAxes,
            fontsize=5,
            color="white",
            va="top",
            ha="left",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
            ),
        )

    # Colour legend centred at the bottom of the figure
    legend_patches = [
        mpatches.Patch(facecolor=(*_GT_COLOR,   0.9), label="Ground truth (GT)"),
        mpatches.Patch(facecolor=(*_PRED_COLOR, 0.9), label="Prediction"),
        mpatches.Patch(facecolor=(*_OVL_COLOR,  0.9), label="Overlap (GT ∩ Pred)"),
        mpatches.Patch(facecolor="none", edgecolor="#00D7FF", linewidth=1.2, label="ROI crop"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=8,
        framealpha=0.3,
        facecolor="#333333",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.003),
    )

    plt.subplots_adjust(bottom=0.06, left=0.01, right=0.99, top=0.97)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\n  Visualization saved  →  {output_path}")


# ---------------------------------------------------------------------------
# Checkpoint selection
# ---------------------------------------------------------------------------

def _load_epoch(path: Path) -> str:
    """
    Read only the ``epoch`` scalar from a checkpoint file without loading
    the full tensor state.  Returns the epoch as a string, or ``"?"`` on
    any failure.

    Parameters
    ----------
    path : Path
        Absolute path to a ``.pt`` checkpoint file.

    Returns
    -------
    str
        Epoch number as a string (e.g. ``"139"``), or ``"?"`` if the key
        is absent or the file cannot be read.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        return str(epoch)
    except Exception:  # noqa: BLE001
        return "?"


def _build_checkpoint_list(ckpts_dir: Path) -> list[Path]:
    """
    Return all ``.pt`` files in *ckpts_dir* sorted for display:
    ``best.pt`` pinned first, then remaining files in descending
    lexicographic order (newest epoch filename first).

    Parameters
    ----------
    ckpts_dir : Path
        Directory that contains ``.pt`` checkpoint files.

    Returns
    -------
    list[Path]
        Sorted list of absolute checkpoint paths.  Empty list when no
        ``.pt`` files are found.
    """
    all_pts = list(ckpts_dir.glob("*.pt"))
    best    = [p for p in all_pts if p.name == "best.pt"]
    others  = sorted(
        [p for p in all_pts if p.name != "best.pt"],
        key=lambda p: p.name,
        reverse=True,   # descending: epoch_0139 before epoch_0138
    )
    return best + others


def _format_size(n_bytes: int) -> str:
    """
    Format a byte count as a human-readable string (KB / MB).

    Parameters
    ----------
    n_bytes : int
        File size in bytes.

    Returns
    -------
    str
        Formatted string such as ``"123.4 MB"`` or ``"456.7 KB"``.
    """
    mb = n_bytes / (1024 * 1024)
    if mb >= 1.0:
        return f"{mb:.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def select_checkpoint(ckpts_dir: Path) -> Path:
    """
    Interactively select a ``.pt`` checkpoint from *ckpts_dir* using a
    ``curses`` arrow-key menu.

    Each row in the menu shows:
    ``filename     <size>   [epoch <n>]``

    Controls
    --------
    ↑ / ↓          move selection up / down
    Page Up / Down  jump 5 rows
    Enter           confirm selection
    q / Ctrl-C      abort (exits the process)

    Non-TTY fallback
    ----------------
    When ``sys.stdin`` is not a TTY (e.g. piped input or Docker without
    ``-it``), the menu is skipped: ``best.pt`` is returned automatically
    if present, otherwise the first entry in the sorted list (newest
    epoch).  A warning is logged in this case.

    Parameters
    ----------
    ckpts_dir : Path
        Directory containing ``.pt`` checkpoint files.

    Returns
    -------
    Path
        Absolute path to the selected checkpoint file.

    Raises
    ------
    SystemExit
        If no ``.pt`` files are found, or if the user quits the menu.
    """
    checkpoints = _build_checkpoint_list(ckpts_dir)
    if not checkpoints:
        logger.error("No .pt checkpoint files found in %s", ckpts_dir)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Non-interactive fallback
    # ------------------------------------------------------------------ #
    if not sys.stdin.isatty():
        best_pts = [p for p in checkpoints if p.name == "best.pt"]
        chosen   = best_pts[0] if best_pts else checkpoints[0]
        logger.warning(
            "Non-interactive environment detected — auto-selecting: %s",
            chosen.name,
        )
        return chosen

    # ------------------------------------------------------------------ #
    # Pre-load epoch labels (done once before entering curses)
    # ------------------------------------------------------------------ #
    logger.info("Reading checkpoint metadata (%d files) …", len(checkpoints))
    epochs: list[str] = [_load_epoch(p) for p in checkpoints]
    sizes:  list[str] = [_format_size(p.stat().st_size) for p in checkpoints]

    # Build display rows: pad filename and size columns for alignment
    names      = [p.name for p in checkpoints]
    name_w     = max(len(n) for n in names)
    size_w     = max(len(s) for s in sizes)
    rows: list[str] = [
        f"{n:<{name_w}}   {s:>{size_w}}   [epoch {e}]"
        for n, s, e in zip(names, sizes, epochs)
    ]

    # ------------------------------------------------------------------ #
    # curses UI
    # ------------------------------------------------------------------ #
    selected_idx: int = 0

    def _draw(stdscr: curses.window, idx: int) -> None:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        title  = "Select checkpoint  (↑/↓  PgUp/PgDn  Enter to confirm  q to quit)"
        border = "─" * min(len(title), max_x - 1)

        stdscr.addstr(0, 0, title[:max_x - 1],  curses.A_BOLD)
        stdscr.addstr(1, 0, border[:max_x - 1])

        # Scrolling: keep selected row visible
        visible = max_y - 3   # rows available for the list
        start   = max(0, min(idx - visible // 2, len(rows) - visible))

        for offset, (row_text, row_idx) in enumerate(
            zip(rows[start : start + visible], range(start, start + visible))
        ):
            y = 2 + offset
            if y >= max_y - 1:
                break
            prefix = "▶ " if row_idx == idx else "  "
            line   = (prefix + row_text)[: max_x - 1]
            attr   = curses.A_REVERSE if row_idx == idx else curses.A_NORMAL
            stdscr.addstr(y, 0, line, attr)

        stdscr.refresh()

    def _run(stdscr: curses.window) -> int:
        nonlocal selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            _draw(stdscr, selected_idx)
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                selected_idx = max(0, selected_idx - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_idx = min(len(rows) - 1, selected_idx + 1)
            elif key == curses.KEY_PPAGE:       # Page Up
                selected_idx = max(0, selected_idx - 5)
            elif key == curses.KEY_NPAGE:       # Page Down
                selected_idx = min(len(rows) - 1, selected_idx + 5)
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return selected_idx
            elif key in (ord("q"), ord("Q"), 27):   # q / Esc
                return -1

        return selected_idx  # unreachable

    try:
        chosen_idx = curses.wrapper(_run)
    except KeyboardInterrupt:
        chosen_idx = -1

    if chosen_idx == -1:
        print("Aborted.")
        sys.exit(0)

    return checkpoints[chosen_idx]


# ---------------------------------------------------------------------------
# Interactive batch selection helpers
# ---------------------------------------------------------------------------


def _default_runs_root() -> str:
    """Pick a sensible default runs root for local and Docker environments."""
    docker_root = Path("/outputs/runs")
    if docker_root.is_dir():
        return str(docker_root)
    return str((Path(__file__).resolve().parent.parent / "outputs" / "runs").resolve())


def _slugify(value: str) -> str:
    """Convert an arbitrary string to a filesystem-safe ASCII-ish token."""
    token = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in value)
    token = token.strip("._")
    return token or "item"


def _directory_is_writable(path: Path) -> bool:
    """Return True when *path* can be created and written by this process."""
    probe: Path | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write_probe_{os.getpid()}"
        with probe.open("w", encoding="utf-8") as fh:
            fh.write("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
        return False


def _default_vis_output_path(run_dir: Path, vis_name: str, repo_root: Path) -> Path:
    """
    Resolve a writable default visualization path.

    Preference order:
    1) <repo_root>/visualizations/<vis_name>
    2) <outputs>/evaluation_summaries/<vis_name>
    3) <run_dir>/<vis_name>
    """
    preferred_dir = (repo_root / "visualizations").resolve()
    if _directory_is_writable(preferred_dir):
        return preferred_dir / vis_name

    fallback_dir = run_dir.resolve()
    outputs_root = fallback_dir.parent.parent if fallback_dir.parent.name == "runs" else repo_root / "outputs"
    shared_dir = (outputs_root / "evaluation_summaries").resolve()
    logger.warning(
        "Visualization directory is not writable (%s). Trying shared fallback locations.",
        preferred_dir,
    )

    if _directory_is_writable(shared_dir):
        logger.warning(
            "Writing visualization to shared summaries directory: %s",
            shared_dir,
        )
        return shared_dir / vis_name

    logger.warning(
        "Shared summaries directory is not writable (%s). Falling back to %s",
        shared_dir,
        fallback_dir,
    )
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir / vis_name


def _default_eval_summary_path(run_dir: Path, summary_name: str, repo_root: Path) -> Path:
    """
    Resolve a writable default evaluation summary path.

    Evaluation commonly runs from Docker as a UID that can read historical run
    directories but cannot write into them. Keep summaries beside the run when
    possible; otherwise use a shared outputs-level summaries directory.
    """
    run_dir = run_dir.resolve()
    if _directory_is_writable(run_dir):
        return run_dir / summary_name

    outputs_root = run_dir.parent.parent if run_dir.parent.name == "runs" else repo_root / "outputs"
    fallback_dir = (outputs_root / "evaluation_summaries").resolve()
    if _directory_is_writable(fallback_dir):
        logger.warning(
            "Run directory is not writable (%s). Writing summary to %s",
            run_dir,
            fallback_dir,
        )
        return fallback_dir / f"{run_dir.name}_{summary_name}"

    logger.warning(
        "Neither run directory nor evaluation_summaries is writable. Falling back to %s",
        run_dir,
    )
    return run_dir / summary_name


def _format_run_candidate_label(run_dir: Path) -> str:
    """Build a compact run label with config metadata for selector display."""
    cfg_path = run_dir / "config.yaml"
    model = "?"
    dataset_type = "?"
    task = "?"
    try:
        cfg = load_config(str(cfg_path))
        model = str(cfg.get("model", "?")).strip() or "?"
        dataset_type = str(cfg.get("dataset_type", "?")).strip() or "?"
        task = str(cfg.get("task", "lesion_segmentation")).strip() or "?"
    except Exception:  # noqa: BLE001
        pass
    return f"{run_dir.name}   [repo | {model} | {dataset_type} | {task}]"


def _discover_run_candidates(runs_root: Path) -> list[ModelCandidate]:
    """Discover runnable model directories from runs_root."""
    if not runs_root.is_dir():
        return []

    run_dirs = sorted(
        [d for d in runs_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    out: list[ModelCandidate] = []
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        ckpt_dir = run_dir / "checkpoints"
        if not cfg_path.exists() or not ckpt_dir.is_dir():
            continue
        if not any(ckpt_dir.glob("*.pt")):
            continue
        dataset_type = "?"
        task = "?"
        try:
            cfg = load_config(str(cfg_path))
            dataset_type = str(cfg.get("dataset_type", "?")).strip() or "?"
            task = str(cfg.get("task", "lesion_segmentation")).strip().lower()
        except Exception:  # noqa: BLE001
            pass
        out.append(
            ModelCandidate(
                label=_format_run_candidate_label(run_dir),
                model_source="repo_run",
                task=task,
                dataset_type=dataset_type,
                run_dir=run_dir.resolve(),
            )
        )
    return out


def _discover_external_model_candidates() -> list[ModelCandidate]:
    out: list[ModelCandidate] = []
    for spec in list_supported_external_models():
        out.append(
            ModelCandidate(
                label=(
                    f"{spec.display_name}@{spec.bundle_version}   "
                    f"[external | {spec.model_source} | {spec.dataset_type} | {spec.task}]"
                ),
                model_source=spec.model_source,
                task=spec.task,
                dataset_type=spec.dataset_type,
                external_model_id=spec.model_id,
                external_model_version=spec.bundle_version,
            )
        )
    return out


def _discover_roi_localizer_candidates(runs_root: Path) -> list[ModelCandidate]:
    """Discover ROI-localizer runs (task=prostate_localization) for predicted ROI mode."""
    candidates = _discover_run_candidates(runs_root)
    out: list[ModelCandidate] = []
    for candidate in candidates:
        if candidate.run_dir is None:
            continue
        cfg_path = candidate.run_dir / "config.yaml"
        try:
            cfg = load_config(str(cfg_path))
            task = str(cfg.get("task", "lesion_segmentation")).strip().lower()
        except Exception:  # noqa: BLE001
            continue
        if task != "prostate_localization":
            continue
        out.append(candidate)
    return out


def _discover_external_roi_localizer_candidates() -> list[ModelCandidate]:
    return [
        c
        for c in _discover_external_model_candidates()
        if c.task == "prostate_localization" and c.dataset_type == "prostate158"
    ]


def _checkbox_menu(
    title: str,
    options: list[str],
    *,
    include_all: bool = True,
    all_label: str = "All",
    preselected: set[int] | None = None,
) -> list[int]:
    """
    Curses multi-select UI.

    Returns selected option indices in ``options`` order, excluding the synthetic
    "All" row when ``include_all=True``.
    """
    if not options:
        return []
    if not sys.stdin.isatty():
        return sorted(preselected or set(range(len(options))))

    if preselected is None:
        selected_opts = set(range(len(options)))
    else:
        selected_opts = {i for i in preselected if 0 <= i < len(options)}

    rows = [all_label, *options] if include_all else list(options)
    selected: set[int] = (
        {idx + 1 for idx in selected_opts}
        if include_all
        else set(selected_opts)
    )
    if include_all and len(selected_opts) == len(options):
        selected.add(0)

    selected_idx = 0
    scroll = 0

    def _sync_all_row() -> None:
        if not include_all:
            return
        option_rows = set(range(1, len(rows)))
        if option_rows and option_rows.issubset(selected):
            selected.add(0)
        else:
            selected.discard(0)

    def _draw(stdscr: curses.window) -> None:
        nonlocal scroll
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        help_line = "↑/↓ move   Space toggle   PgUp/PgDn jump   Enter confirm   q quit"
        stdscr.addstr(0, 0, title[: max_x - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, help_line[: max_x - 1], curses.A_DIM)
        stdscr.addstr(2, 0, ("─" * (max_x - 1))[: max_x - 1])

        visible = max(1, max_y - 4)
        if selected_idx < scroll:
            scroll = selected_idx
        elif selected_idx >= scroll + visible:
            scroll = selected_idx - visible + 1

        for row_offset in range(visible):
            row_idx = scroll + row_offset
            if row_idx >= len(rows):
                break
            y = 3 + row_offset
            mark = "[x]" if row_idx in selected else "[ ]"
            prefix = "▶ " if row_idx == selected_idx else "  "
            line = f"{prefix}{mark} {rows[row_idx]}"
            attr = curses.A_REVERSE if row_idx == selected_idx else curses.A_NORMAL
            stdscr.addstr(y, 0, line[: max_x - 1], attr)

        stdscr.refresh()

    def _run(stdscr: curses.window) -> list[int]:
        nonlocal selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            _draw(stdscr)
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                selected_idx = max(0, selected_idx - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_idx = min(len(rows) - 1, selected_idx + 1)
            elif key == curses.KEY_PPAGE:
                selected_idx = max(0, selected_idx - 5)
            elif key == curses.KEY_NPAGE:
                selected_idx = min(len(rows) - 1, selected_idx + 5)
            elif key == ord(" "):
                if include_all and selected_idx == 0:
                    if 0 in selected:
                        selected.clear()
                    else:
                        selected.update(range(len(rows)))
                else:
                    if selected_idx in selected:
                        selected.discard(selected_idx)
                    else:
                        selected.add(selected_idx)
                _sync_all_row()
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                if include_all:
                    chosen = sorted(i - 1 for i in selected if i > 0)
                else:
                    chosen = sorted(selected)
                if chosen:
                    return chosen
                curses.beep()
            elif key in (ord("q"), ord("Q"), 27):
                return []

    try:
        return curses.wrapper(_run)
    except KeyboardInterrupt:
        return []


def _single_choice_menu(
    title: str,
    options: list[str],
    *,
    default_idx: int = 0,
) -> int:
    """Curses single-choice UI. Returns -1 on abort."""
    if not options:
        return -1
    if not sys.stdin.isatty():
        return max(0, min(default_idx, len(options) - 1))

    selected_idx = max(0, min(default_idx, len(options) - 1))

    def _draw(stdscr: curses.window) -> None:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        help_line = "↑/↓ move   Enter confirm   q quit"
        stdscr.addstr(0, 0, title[: max_x - 1], curses.A_BOLD)
        stdscr.addstr(1, 0, help_line[: max_x - 1], curses.A_DIM)
        stdscr.addstr(2, 0, ("─" * (max_x - 1))[: max_x - 1])
        visible = max(1, max_y - 4)
        start = max(0, min(selected_idx - visible // 2, len(options) - visible))
        for offset, row_idx in enumerate(range(start, min(len(options), start + visible))):
            y = 3 + offset
            mark = "(*)"
            prefix = "▶ " if row_idx == selected_idx else "  "
            line = f"{prefix}{mark} {options[row_idx]}"
            attr = curses.A_REVERSE if row_idx == selected_idx else curses.A_NORMAL
            stdscr.addstr(y, 0, line[: max_x - 1], attr)
        stdscr.refresh()

    def _run(stdscr: curses.window) -> int:
        nonlocal selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)
        while True:
            _draw(stdscr)
            key = stdscr.getch()
            if key in (curses.KEY_UP, ord("k")):
                selected_idx = max(0, selected_idx - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_idx = min(len(options) - 1, selected_idx + 1)
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return selected_idx
            elif key in (ord("q"), ord("Q"), 27):
                return -1

    try:
        return curses.wrapper(_run)
    except KeyboardInterrupt:
        return -1


def _select_datasets_for_batch() -> list[str]:
    """Interactive dataset checkbox step."""
    keys = ["picai", "prostate158"]
    labels = ["PI-CAI", "Prostate158"]
    chosen = _checkbox_menu(
        title="Dataset test",
        options=labels,
        include_all=True,
        all_label="All datasets",
    )
    return [keys[i] for i in chosen]


def _select_eval_split_for_batch(current_split: str = "test") -> str:
    """Interactive evaluation split selection step."""
    splits = [
        ("test", "Fixed hold-out test set"),
        ("val", "Run validation split"),
    ]
    default_idx = 1 if current_split == "val" else 0
    chosen = _single_choice_menu(
        title="Evaluation split",
        options=[label for _, label in splits],
        default_idx=default_idx,
    )
    return splits[chosen][0] if chosen >= 0 else ""


def _select_workflow_for_batch() -> str:
    """Interactive workflow selection step."""
    workflows = [
        BatchWorkflow("evaluate", "Evaluate only"),
        BatchWorkflow("tune", "Tune postprocess only"),
        BatchWorkflow("both", "Evaluate + tune postprocess"),
    ]
    chosen = _single_choice_menu(
        title="Workflow",
        options=[w.label for w in workflows],
        default_idx=0,
    )
    return workflows[chosen].key if chosen >= 0 else ""


def _select_models_for_batch(candidates: list[ModelCandidate]) -> list[ModelCandidate]:
    """Interactive model checkbox step."""
    labels = [c.label for c in candidates]
    chosen = _checkbox_menu(
        title="Models",
        options=labels,
        include_all=True,
        all_label="All models",
    )
    return [candidates[i] for i in chosen]


def _select_roi_variants_for_batch(
    localizer_candidates: list[ModelCandidate],
    external_localizer_candidates: list[ModelCandidate],
) -> list[ROIVariant]:
    """Interactive ROI checkbox step."""
    variants: list[ROIVariant] = [
        ROIVariant(label="Use each model config ROI setting", mode=""),
        ROIVariant(label="Disabled ROI", mode="disabled"),
        ROIVariant(
            label=(
                "GT prostate ROI (roi.mode=gt_mask; Prostate158 needs prostate "
                "label column, PI-CAI needs --picai-prostate-labels-dir)"
            ),
            mode="gt_mask",
        ),
    ]
    variants.extend(
        ROIVariant(
            label=f"Predicted ROI via {c.run_dir.name}",
            mode="predicted_mask",
            localizer_run=str(c.run_dir),
        )
        for c in localizer_candidates
    )
    variants.extend(
        ROIVariant(
            label=(
                "Predicted ROI via "
                f"{c.external_model_id}@{c.external_model_version} (external)"
            ),
            mode="predicted_mask",
            localizer_run=build_external_localizer_ref(
                c.external_model_id,
                c.external_model_version,
            ),
            localizer_external_model_id=c.external_model_id,
            localizer_external_model_version=c.external_model_version,
        )
        for c in external_localizer_candidates
    )

    chosen = _checkbox_menu(
        title="ROI model",
        options=[v.label for v in variants],
        include_all=True,
        all_label="All ROI variants",
        preselected={0},
    )
    return [variants[i] for i in chosen]


def _roi_tag(mode: str, localizer_run: str) -> str:
    if not mode:
        return "roi_cfg"
    if mode == "disabled":
        return "roi_disabled"
    if mode == "gt_mask":
        return "roi_gt"
    if mode == "predicted_mask":
        external_spec = parse_external_localizer_ref(localizer_run)
        if external_spec is not None:
            return f"roi_pred_{_slugify(external_spec.versioned_id.replace(':', '_'))}"
        return f"roi_pred_{_slugify(Path(localizer_run).name)}"
    return f"roi_{_slugify(mode)}"


def _build_batch_jobs(
    selected_models: list[ModelCandidate],
    checkpoints_by_run: dict[Path, Path],
    datasets: list[str],
    eval_split: str,
    roi_variants: list[ROIVariant],
    repo_root: Path,
    prostate158_prostate_label_col: str,
    prostate158_root_override: str,
    visualization_enabled: bool,
    picai_prostate_labels_dir: str = "",
) -> list[BatchEvalJob]:
    """Create the cross-product jobs for selected datasets/models/ROI variants."""
    jobs: list[BatchEvalJob] = []
    for candidate in selected_models:
        if candidate.model_source == "repo_run":
            if candidate.run_dir is None:
                continue
            run_dir = candidate.run_dir
            checkpoint = checkpoints_by_run[run_dir]
            cfg_path = run_dir / "config.yaml"
            base_cfg = load_config(str(cfg_path))
            ckpt_tag = _slugify(checkpoint.stem)
            artifact_stem = run_dir.name
        else:
            if eval_split != "test":
                logger.warning(
                    "Skipping external model %s for eval_split=%s; validation splits are run-specific.",
                    candidate.label,
                    eval_split,
                )
                continue
            run_dir = None
            checkpoint = None
            spec = resolve_external_model_request(
                candidate.external_model_id,
                candidate.external_model_version,
            )
            base_cfg = build_external_eval_config(
                spec,
                prostate158_root=prostate158_root_override,
            )
            ckpt_tag = _slugify(spec.bundle_version)
            artifact_stem = _slugify(spec.model_id.replace(":", "_"))
        for dataset_type in datasets:
            if candidate.dataset_type not in {"", "?"} and dataset_type != candidate.dataset_type:
                logger.warning(
                    "Skipping incompatible combo model=%s dataset=%s (requires %s)",
                    candidate.label,
                    dataset_type,
                    candidate.dataset_type,
                )
                continue
            for roi in roi_variants:
                try:
                    prostate158_root_for_col = prostate158_root_override
                    prostate158_split_for_col = ""
                    if dataset_type == "prostate158" and eval_split == "val":
                        prostate158_root_for_col = (
                            prostate158_root_override
                            or str(base_cfg.get("prostate158_train_dir", "data/prostate158_train"))
                        )
                        prostate158_split_for_col = str(
                            base_cfg.get("prostate158_val_split", "valid")
                        )
                    cfg_for_combo = _apply_roi_overrides(
                        base_cfg,
                        roi.mode,
                        roi.localizer_run,
                        roi.localizer_external_model_id,
                        roi.localizer_external_model_version,
                    )
                    cfg_for_combo, _ = _ensure_prostate_label_col(
                        cfg_for_combo,
                        dataset_type=dataset_type,
                        explicit_col=prostate158_prostate_label_col,
                        prostate158_root_override=prostate158_root_for_col,
                        prostate158_split_override=prostate158_split_for_col,
                    )
                    cfg_for_combo, _ = _ensure_picai_gt_roi_config(
                        cfg_for_combo,
                        dataset_type=dataset_type,
                        picai_prostate_labels_dir=picai_prostate_labels_dir,
                    )
                    _ = validate_task_and_roi_config(cfg_for_combo, dataset_type)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Skipping incompatible combo model=%s dataset=%s roi=%s (%s)",
                        candidate.label,
                        dataset_type,
                        roi.mode or "config",
                        exc,
                    )
                    continue

                roi_tag = _roi_tag(roi.mode, roi.localizer_run)
                split_tag = "val" if eval_split == "val" else "test"
                summary_name = (
                    f"evaluation_summary_{split_tag}_{dataset_type}_{roi_tag}_{ckpt_tag}.json"
                )
                vis_name = (
                    f"{artifact_stem}_{split_tag}_{dataset_type}_{roi_tag}_{ckpt_tag}_"
                    "eval_visualization.png"
                )
                if run_dir is None:
                    eval_dir = (repo_root / "outputs" / "evaluation_summaries").resolve()
                    summary_path = (eval_dir / f"{artifact_stem}_{summary_name}").resolve()
                    vis_output = (
                        ((repo_root / "visualizations").resolve() / vis_name)
                        if visualization_enabled
                        else (eval_dir / vis_name)
                    ).resolve()
                else:
                    summary_path = _default_eval_summary_path(
                        run_dir=run_dir,
                        summary_name=summary_name,
                        repo_root=repo_root,
                    ).resolve()
                    vis_output = (
                        _default_vis_output_path(
                            run_dir=run_dir,
                            vis_name=vis_name,
                            repo_root=repo_root,
                        ).resolve()
                        if visualization_enabled
                        else (run_dir / vis_name).resolve()
                    )
                jobs.append(
                    BatchEvalJob(
                        model_source=candidate.model_source,
                        dataset_type=dataset_type,
                        eval_split=eval_split,
                        task=str(cfg_for_combo.get("task", candidate.task)).strip().lower(),
                        roi_mode=roi.mode,
                        roi_localizer_run=roi.localizer_run,
                        roi_localizer_external_model_id=roi.localizer_external_model_id,
                        roi_localizer_external_model_version=roi.localizer_external_model_version,
                        summary_json=summary_path,
                        vis_output=vis_output,
                        run_dir=run_dir,
                        checkpoint=checkpoint,
                        external_model_id=candidate.external_model_id,
                        external_model_version=candidate.external_model_version,
                    )
                )
    return jobs


def _batch_command_for_job(job: BatchEvalJob, args: argparse.Namespace) -> list[str]:
    """Build subprocess argv for one batch job."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset-type", job.dataset_type,
        "--images-dir", args.images_dir,
        "--labels-dir", args.labels_dir,
        "--summary-json", str(job.summary_json),
    ]
    if job.eval_split != "test":
        cmd.extend(["--eval-split", job.eval_split])
    if job.run_dir is not None:
        cmd.extend(["--run", str(job.run_dir)])
    if job.checkpoint is not None:
        cmd.extend(["--checkpoint", str(job.checkpoint)])
    if job.external_model_id:
        cmd.extend(["--external-model", job.external_model_id])
    if job.external_model_version:
        cmd.extend(["--external-model-version", job.external_model_version])
    if args.visualize:
        cmd.extend(["--visualize", "--vis-output", str(job.vis_output)])
    if job.roi_mode:
        cmd.extend(["--roi-mode", job.roi_mode])
    if job.roi_localizer_run and not job.roi_localizer_external_model_id:
        cmd.extend(["--roi-localizer-run", job.roi_localizer_run])
    if job.roi_localizer_external_model_id:
        cmd.extend(
            [
                "--roi-localizer-external-model",
                job.roi_localizer_external_model_id,
            ]
        )
    if job.roi_localizer_external_model_version:
        cmd.extend(
            [
                "--roi-localizer-external-model-version",
                job.roi_localizer_external_model_version,
            ]
        )
    if args.prostate158_root:
        cmd.extend(["--prostate158-root", args.prostate158_root])
    if args.prostate158_label_reader:
        cmd.extend(["--prostate158-label-reader", args.prostate158_label_reader])
    if args.prostate158_prostate_label_col:
        cmd.extend(["--prostate158-prostate-label-col", args.prostate158_prostate_label_col])
    picai_prostate_labels_dir = str(
        getattr(args, "picai_prostate_labels_dir", "")
    ).strip()
    if picai_prostate_labels_dir:
        cmd.extend(["--picai-prostate-labels-dir", picai_prostate_labels_dir])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.sw_batch_size is not None:
        cmd.extend(["--sw-batch-size", str(args.sw_batch_size)])
    return cmd


def _batch_tune_command_for_job(job: BatchEvalJob, args: argparse.Namespace) -> list[str] | None:
    """Build subprocess argv for one post-processing tuning job."""
    if job.run_dir is None or job.checkpoint is None:
        return None

    tune_script = Path(__file__).resolve().parent / "tune_postprocess.py"
    summary_name = job.summary_json.name.replace(
        "evaluation_summary",
        "postprocess_tuning_summary",
        1,
    )
    if summary_name == job.summary_json.name:
        summary_name = f"{job.run_dir.name}_{job.dataset_type}_{_roi_tag(job.roi_mode, job.roi_localizer_run)}_postprocess_tuning_summary.json"
    summary_path = (job.summary_json.parent / summary_name).resolve()

    cmd = [
        sys.executable,
        str(tune_script),
        "--run", str(job.run_dir),
        "--checkpoint", str(job.checkpoint),
        "--config", str(Path(getattr(args, "tune_config", "") or "configs/postprocess_tuning.yaml")),
        "--dataset-type", job.dataset_type,
        "--summary-json", str(summary_path),
    ]
    if job.roi_mode:
        cmd.extend(["--roi-mode", job.roi_mode])
    if job.roi_localizer_run and not job.roi_localizer_external_model_id:
        cmd.extend(["--roi-localizer-run", job.roi_localizer_run])
    if job.roi_localizer_external_model_id:
        cmd.extend(["--roi-localizer-external-model", job.roi_localizer_external_model_id])
    if job.roi_localizer_external_model_version:
        cmd.extend(["--roi-localizer-external-model-version", job.roi_localizer_external_model_version])
    if args.prostate158_root:
        cmd.extend(["--prostate158-root", args.prostate158_root])
    if args.prostate158_label_reader:
        cmd.extend(["--prostate158-label-reader", args.prostate158_label_reader])
    if args.prostate158_prostate_label_col:
        cmd.extend(["--prostate158-prostate-label-col", args.prostate158_prostate_label_col])
    picai_prostate_labels_dir = str(getattr(args, "picai_prostate_labels_dir", "")).strip()
    if picai_prostate_labels_dir:
        cmd.extend(["--picai-prostate-labels-dir", picai_prostate_labels_dir])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.sw_batch_size is not None:
        cmd.extend(["--sw-batch-size", str(args.sw_batch_size)])
    return cmd


def _run_interactive_batch(args: argparse.Namespace) -> None:
    """Top-level interactive selector flow for dataset/model/ROI batch evaluation."""
    if not sys.stdin.isatty():
        logger.error("Interactive selector requires a TTY. Provide --run in non-interactive mode.")
        sys.exit(1)

    runs_root = Path(args.runs_root).expanduser().resolve()
    candidates = _discover_run_candidates(runs_root) + _discover_external_model_candidates()
    if not candidates:
        logger.error("No repo runs or external baselines available for selection.")
        sys.exit(1)
    roi_localizer_candidates = _discover_roi_localizer_candidates(runs_root)
    external_roi_localizer_candidates = _discover_external_roi_localizer_candidates()

    workflow = _select_workflow_for_batch()
    if not workflow:
        print("Aborted.")
        sys.exit(0)

    datasets = _select_datasets_for_batch()
    if not datasets:
        print("Aborted.")
        sys.exit(0)

    eval_split = args.eval_split
    if workflow in {"evaluate", "both"}:
        eval_split = _select_eval_split_for_batch(args.eval_split)
        if not eval_split:
            print("Aborted.")
            sys.exit(0)
    args.eval_split = eval_split

    selected_models = _select_models_for_batch(candidates)
    if not selected_models:
        print("Aborted.")
        sys.exit(0)

    checkpoints_by_run: dict[Path, Path] = {}
    for candidate in selected_models:
        if candidate.model_source != "repo_run" or candidate.run_dir is None:
            continue
        print(f"\nSelect checkpoint for model: {candidate.run_dir.name}")
        checkpoints_by_run[candidate.run_dir] = select_checkpoint(candidate.run_dir / "checkpoints")

    roi_variants = _select_roi_variants_for_batch(
        roi_localizer_candidates,
        external_roi_localizer_candidates,
    )
    if not roi_variants:
        print("Aborted.")
        sys.exit(0)

    if args.summary_json or args.vis_output:
        logger.warning(
            "--summary-json/--vis-output are ignored in selector mode; "
            "per-job unique paths are generated automatically."
        )

    repo_root = Path(__file__).resolve().parent.parent
    jobs = _build_batch_jobs(
        selected_models=selected_models,
        checkpoints_by_run=checkpoints_by_run,
        datasets=datasets,
        eval_split=eval_split,
        roi_variants=roi_variants,
        repo_root=repo_root,
        prostate158_prostate_label_col=args.prostate158_prostate_label_col,
        prostate158_root_override=args.prostate158_root,
        picai_prostate_labels_dir=args.picai_prostate_labels_dir,
        visualization_enabled=bool(args.visualize),
    )
    if not jobs:
        logger.error("No valid jobs remain after dataset/model/ROI compatibility checks.")
        logger.error(
            "Hint: roi.mode='gt_mask' needs dataset_type='prostate158' and "
            "'prostate158_prostate_label_col' (from config or --prostate158-prostate-label-col). "
            "For PI-CAI gt_mask, defaults are /data/prostate_labels or data/prostate_labels "
            "(or pass --picai-prostate-labels-dir)."
        )
        sys.exit(1)

    _section("Batch Selector Plan")
    print(
        "  Workflow         : "
        + {
            "evaluate": "Evaluate only",
            "tune": "Tune postprocess only",
            "both": "Evaluate + tune postprocess",
        }[workflow]
    )
    print(f"  Models selected  : {len(selected_models)}")
    print(f"  Datasets         : {datasets}")
    if workflow in {"evaluate", "both"}:
        print(f"  Eval split       : {eval_split}")
    print(f"  ROI variants     : {len(roi_variants)}")
    print(f"  Eval jobs        : {len(jobs) if workflow in {'evaluate', 'both'} else 0}")
    tune_jobs = [job for job in jobs if job.run_dir is not None]
    skipped_tune_jobs = len(jobs) - len(tune_jobs)
    print(f"  Tune jobs        : {len(tune_jobs) if workflow in {'tune', 'both'} else 0}")
    if workflow in {"tune", "both"} and skipped_tune_jobs:
        print(f"  Tune skipped     : {skipped_tune_jobs} external model job(s)")

    preview_n = min(8, len(jobs))
    if preview_n:
        print("\n  Job preview:")
        for job in jobs[:preview_n]:
            roi_desc = _roi_variant_desc(
                job.roi_mode,
                job.roi_localizer_run,
                job.roi_localizer_external_model_id,
                job.roi_localizer_external_model_version,
            )
            model_desc = (
                job.run_dir.name
                if job.run_dir is not None
                else f"{job.external_model_id}@{job.external_model_version}"
            )
            ckpt_desc = job.checkpoint.name if job.checkpoint is not None else "-"
            print(
                f"    - model={model_desc}  ckpt={ckpt_desc}  "
                f"dataset={job.dataset_type}  split={job.eval_split}  roi={roi_desc}"
            )
        if len(jobs) > preview_n:
            print(f"    ... ({len(jobs) - preview_n} more)")

    requested_jobs = (
        (len(jobs) if workflow in {"evaluate", "both"} else 0)
        + (len(tune_jobs) if workflow in {"tune", "both"} else 0)
    )
    proceed = input(f"\nRun {requested_jobs} selected job(s)? [y/N]: ").strip().lower()
    if proceed not in {"y", "yes"}:
        print("Aborted.")
        sys.exit(0)

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        _section(f"Batch job {idx}/{len(jobs)}")
        roi_desc = _roi_variant_desc(
            job.roi_mode,
            job.roi_localizer_run,
            job.roi_localizer_external_model_id,
            job.roi_localizer_external_model_version,
        )
        model_desc = (
            str(job.run_dir)
            if job.run_dir is not None
            else f"{job.external_model_id}@{job.external_model_version}"
        )
        ckpt_desc = job.checkpoint.name if job.checkpoint is not None else "-"
        print(f"  Model      : {model_desc}")
        print(f"  Checkpoint : {ckpt_desc}")
        print(f"  Dataset    : {job.dataset_type}")
        print(f"  Eval split : {job.eval_split}")
        print(f"  Task       : {job.task}")
        print(f"  ROI        : {roi_desc}")

        commands: list[tuple[str, list[str]]] = []
        if workflow in {"evaluate", "both"}:
            commands.append(("evaluation", _batch_command_for_job(job, args)))
        if workflow in {"tune", "both"}:
            tune_cmd = _batch_tune_command_for_job(job, args)
            if tune_cmd is None:
                logger.info(
                    "Skipping postprocess tuning for external model: %s",
                    model_desc,
                )
            else:
                commands.append(("postprocess tuning", tune_cmd))

        for command_label, cmd in commands:
            print(f"\n  Running {command_label}...")
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                failures += 1
                logger.error(
                    "Batch %s job failed (rc=%d): model=%s checkpoint=%s dataset=%s roi=%s",
                    command_label,
                    rc,
                    model_desc,
                    ckpt_desc,
                    job.dataset_type,
                    roi_desc,
                )

    if failures:
        logger.error("Batch run finished with %d failed job(s).", failures)
        sys.exit(1)

    logger.info("Batch run complete: all %d job(s) succeeded.", len(jobs))


def _apply_roi_overrides(
    cfg: dict[str, Any],
    roi_mode: str,
    roi_localizer_run: str,
    roi_localizer_external_model_id: str = "",
    roi_localizer_external_model_version: str = "",
) -> dict[str, Any]:
    """Apply CLI ROI overrides on top of the run config."""
    if (
        not roi_mode
        and not roi_localizer_run
        and not roi_localizer_external_model_id
        and not roi_localizer_external_model_version
    ):
        return cfg
    merged = deepcopy(cfg)
    roi_cfg = dict(merged.get("roi", {}) or {})
    if roi_mode:
        roi_cfg["mode"] = roi_mode
    localizer_ref = roi_localizer_run
    if roi_localizer_external_model_id:
        localizer_ref = build_external_localizer_ref(
            roi_localizer_external_model_id,
            roi_localizer_external_model_version,
        )
    if localizer_ref:
        roi_cfg["localizer_run"] = localizer_ref
    if roi_localizer_external_model_id:
        roi_cfg["localizer_external_model_id"] = roi_localizer_external_model_id
    if roi_localizer_external_model_version:
        roi_cfg["localizer_external_model_version"] = roi_localizer_external_model_version
    merged["roi"] = roi_cfg
    return merged


def _roi_variant_desc(
    roi_mode: str,
    roi_localizer_run: str,
    roi_localizer_external_model_id: str = "",
    roi_localizer_external_model_version: str = "",
) -> str:
    if not roi_mode:
        return "config"
    if roi_mode != "predicted_mask":
        return roi_mode
    if roi_localizer_external_model_id:
        return (
            f"predicted_mask:{roi_localizer_external_model_id}"
            f"@{roi_localizer_external_model_version}"
        )
    external_spec = parse_external_localizer_ref(roi_localizer_run)
    if external_spec is not None:
        return f"predicted_mask:{external_spec.versioned_id}"
    if roi_localizer_run:
        return f"predicted_mask:{Path(roi_localizer_run).name}"
    return "predicted_mask"


def _default_prostate_label_col(label_reader: Any) -> str:
    """Build fallback Prostate158 prostate-label column from label-reader id."""
    text = str(label_reader).strip()
    if not text:
        return "t2_anatomy_reader1"
    match = re.search(r"\d+", text)
    suffix = match.group(0) if match else "1"
    return f"t2_anatomy_reader{suffix}"


def _resolve_prostate158_csv_path(root_dir: Path, split: str) -> Path:
    """Resolve Prostate158 split CSV path using the dataset loader's conventions."""
    root_dir = root_dir.expanduser().resolve()
    csv_path = root_dir / f"{split}.csv"
    if csv_path.exists():
        return csv_path
    nested = root_dir / root_dir.name / f"{split}.csv"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Prostate158 {split}.csv not found under {root_dir} "
        f"(checked {csv_path} and {nested})."
    )


def _read_csv_columns(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or [])


def _infer_prostate_label_col(columns: list[str], label_reader: Any) -> str | None:
    """Infer prostate-mask column from CSV headers."""
    text = str(label_reader).strip()
    match = re.search(r"\d+", text)
    suffix = match.group(0) if match else "1"

    preferred = [
        f"t2_prostate_reader{suffix}",
        f"t2_anatomy_reader{suffix}",
    ]
    for col in preferred:
        if col in columns:
            return col

    prostate_like = [
        col
        for col in columns
        if col.startswith("t2_")
        and "reader" in col
        and ("anatomy" in col or "prostate" in col)
        and "tumor" not in col
    ]
    with_reader = [col for col in prostate_like if col.endswith(f"reader{suffix}")]
    if with_reader:
        return sorted(with_reader)[0]
    if prostate_like:
        return sorted(prostate_like)[0]
    return None


def _ensure_prostate_label_col(
    cfg: dict[str, Any],
    dataset_type: str,
    explicit_col: str,
    prostate158_root_override: str = "",
    prostate158_split_override: str = "",
) -> tuple[dict[str, Any], bool]:
    """
    Ensure prostate158_prostate_label_col exists when ROI/task requires it.

    Returns
    -------
    (cfg, was_auto_filled)
    """
    merged = deepcopy(cfg)

    dataset_norm = str(dataset_type).strip().lower()
    task = str(merged.get("task", "lesion_segmentation")).strip().lower()
    roi_mode = str((merged.get("roi", {}) or {}).get("mode", "disabled")).strip().lower()

    needs_col = (
        dataset_norm == "prostate158"
        and (task == "prostate_localization" or roi_mode == "gt_mask")
    )
    has_col = bool(str(merged.get("prostate158_prostate_label_col", "")).strip())
    if not needs_col:
        return merged, False

    chosen = str(explicit_col).strip() or str(merged.get("prostate158_prostate_label_col", "")).strip()
    root_dir = Path(
        prostate158_root_override
        or merged.get("prostate158_test_dir", "data/prostate158_test")
    ).expanduser()
    split = (
        str(prostate158_split_override).strip().lower()
        or str(merged.get("prostate158_test_split", "test")).strip().lower()
        or "test"
    )
    columns: list[str] = []
    csv_path: Path | None = None
    try:
        csv_path = _resolve_prostate158_csv_path(root_dir, split)
        columns = _read_csv_columns(csv_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not inspect Prostate158 CSV headers for prostate label inference "
            "(root=%s split=%s): %s",
            root_dir,
            split,
            exc,
        )

    if chosen and columns and chosen not in columns:
        raise ValueError(
            "Requested prostate label column "
            f"'{chosen}' was not found in {csv_path}. "
            f"Available columns: {', '.join(columns)}"
        )
    if chosen:
        merged["prostate158_prostate_label_col"] = chosen
        return merged, False

    inferred = _infer_prostate_label_col(columns, merged.get("prostate158_label_reader", 1))
    if inferred:
        merged["prostate158_prostate_label_col"] = inferred
        return merged, True

    if has_col:
        return merged, False

    fallback = _default_prostate_label_col(merged.get("prostate158_label_reader", 1))
    merged["prostate158_prostate_label_col"] = fallback
    return merged, True


def _resolve_picai_prostate_labels_dir(
    explicit_dir: str,
    cfg: Mapping[str, Any],
) -> Path | None:
    explicit_raw = str(explicit_dir).strip()
    if explicit_raw:
        return Path(explicit_raw).expanduser().resolve()

    cfg_raw = str(cfg.get("picai_prostate_labels_dir", "")).strip()
    if cfg_raw:
        return Path(cfg_raw).expanduser().resolve()

    # Zero-flag defaults aligned with scripts/download_dataset.sh output.
    for candidate in (Path("/data/prostate_labels"), Path("data/prostate_labels")):
        if candidate.is_dir():
            return candidate.resolve()

    return None


def _ensure_picai_gt_roi_config(
    cfg: dict[str, Any],
    dataset_type: str,
    picai_prostate_labels_dir: str,
) -> tuple[dict[str, Any], Path | None]:
    """
    Enable PI-CAI GT prostate ROI mode when a prostate-label directory is available.

    The shared ROI validator currently also checks for
    ``prostate158_prostate_label_col`` in gt_mask mode; for PI-CAI we provide a
    sentinel value because this field is not used to discover PI-CAI labels.
    """
    merged = deepcopy(cfg)
    dataset_norm = str(dataset_type).strip().lower()
    roi_cfg = dict(merged.get("roi", {}) or {})
    roi_modes = {
        resolve_roi_settings(merged, stage="train").mode,
        resolve_roi_settings(merged, stage="val").mode,
    }
    picai_labels_dir = _resolve_picai_prostate_labels_dir(picai_prostate_labels_dir, merged)

    if dataset_norm != "picai" or "gt_mask" not in roi_modes:
        return merged, picai_labels_dir

    if picai_labels_dir is None:
        raise ValueError(
            "roi.mode='gt_mask' with dataset_type='picai' requires PI-CAI prostate labels. "
            "Looked for defaults in /data/prostate_labels and data/prostate_labels. "
            "Pass --picai-prostate-labels-dir or set picai_prostate_labels_dir in config."
        )
    if not picai_labels_dir.is_dir():
        raise NotADirectoryError(
            f"PI-CAI prostate labels directory not found: {picai_labels_dir}"
        )

    gt_dataset_types = [
        str(v).strip().lower()
        for v in roi_cfg.get("gt_dataset_types", ["prostate158"])
    ]
    if "picai" not in gt_dataset_types:
        gt_dataset_types.append("picai")
    roi_cfg["gt_dataset_types"] = gt_dataset_types
    merged["roi"] = roi_cfg
    merged["picai_prostate_labels_dir"] = str(picai_labels_dir)

    if not str(merged.get("prostate158_prostate_label_col", "")).strip():
        merged["prostate158_prostate_label_col"] = _PICAI_GT_MASK_SENTINEL_PROSTATE_COL

    return merged, picai_labels_dir


def _resolve_picai_prostate_label_path(
    prostate_labels_dir: Path,
    case_id: str,
) -> Path | None:
    candidates = [
        prostate_labels_dir / f"{case_id}.nii.gz",
        prostate_labels_dir / f"{case_id}_prostate.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _attach_picai_prostate_labels(
    cases: list[dict[str, Any]],
    prostate_labels_dir: Path,
    *,
    require_all: bool,
) -> None:
    if not prostate_labels_dir.is_dir():
        raise NotADirectoryError(
            f"PI-CAI prostate labels directory not found: {prostate_labels_dir}"
        )
    missing: list[str] = []
    for case in cases:
        case_id = str(case["case_id"])
        prostate_path = _resolve_picai_prostate_label_path(prostate_labels_dir, case_id)
        if prostate_path is None:
            if require_all:
                missing.append(case_id)
            continue
        case["prostate_label"] = prostate_path

    if require_all and missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Missing PI-CAI prostate labels for {len(missing)} case(s) in {prostate_labels_dir}: "
            f"{preview}{suffix}"
        )


def _select_cases_by_manifest_ids(
    *,
    all_cases: list[dict[str, Any]],
    case_ids: list[str],
    manifest_path: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    case_map = {str(case["case_id"]): case for case in all_cases}
    missing = [cid for cid in case_ids if cid not in case_map]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"{split_name} split manifest references {len(missing)} case(s) "
            f"not found in current data: {preview}{suffix} (manifest={manifest_path})"
        )
    return [case_map[cid] for cid in case_ids]


def _resolve_picai_validation_cases(
    *,
    cfg: Mapping[str, Any],
    run_dir: Path,
    active_modalities: list[str],
    task: str,
    roi_settings: Any,
    picai_prostate_labels_dir: Path | None,
) -> tuple[list[dict[str, Any]], Path, Path, str]:
    images_dir = Path(str(cfg.get("images_dir", "data/images"))).expanduser()
    labels_dir = Path(str(cfg.get("labels_dir", "data/labels"))).expanduser()

    manifest_candidates = [run_dir / "train_val_split_manifest.json"]
    split_manifest_raw = cfg.get("split_manifest_path", "")
    split_manifest_cfg = str(split_manifest_raw).strip() if split_manifest_raw is not None else ""
    if split_manifest_cfg:
        manifest_candidates.append(Path(split_manifest_cfg).expanduser())
    base_output_dir = cfg.get("base_output_dir")
    if base_output_dir:
        manifest_candidates.append(default_split_manifest_path(base_output_dir))

    manifest_path = next((path for path in manifest_candidates if path.exists()), None)
    if manifest_path is None:
        checked = ", ".join(str(path) for path in manifest_candidates)
        raise FileNotFoundError(
            "Could not find PI-CAI train/validation split manifest for this run. "
            f"Checked: {checked}"
        )

    all_cases = discover_cases(images_dir, labels_dir, active_keys=active_modalities)
    if task == "prostate_localization" or roi_settings.mode == "gt_mask":
        if picai_prostate_labels_dir is None:
            raise ValueError(
                "PI-CAI prostate labels are required for validation evaluation "
                f"with task={task} / roi.mode={roi_settings.mode}."
            )
        _attach_picai_prostate_labels(
            all_cases,
            picai_prostate_labels_dir,
            require_all=True,
        )

    manifest = load_split_manifest(manifest_path)
    val_cases = _select_cases_by_manifest_ids(
        all_cases=all_cases,
        case_ids=[str(v) for v in manifest["val_case_ids"]],
        manifest_path=manifest_path,
        split_name="Validation",
    )
    return val_cases, images_dir, labels_dir, str(manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run evaluation on the fixed test set, print results, save visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on the fixed hold-out test set "
            "(data/test_images/) and produce an axial overlay visualization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run",
        default="",
        metavar="DIR",
        help=(
            "Path to a training run directory "
            "(e.g. /outputs/20260418_224246_deconver). "
            "Must contain config.yaml and a checkpoints/ subdirectory. "
            "If omitted (or with --selector), an interactive checkbox selector opens."
        ),
    )
    parser.add_argument(
        "--selector",
        action="store_true",
        help=(
            "Open interactive checkbox selection for dataset/model/ROI and run "
            "the selected evaluations in batch."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=_default_runs_root(),
        metavar="DIR",
        help=(
            "Root directory that contains run folders used by --selector "
            "(each run must have config.yaml and checkpoints/*.pt)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        metavar="NAME_OR_PATH",
        help=(
            "Checkpoint to evaluate. Accepts a filename inside <run>/checkpoints "
            "(e.g. best.pt) or an explicit path. Default opens the selector."
        ),
    )
    parser.add_argument(
        "--external-model",
        type=str,
        default="",
        metavar="ID[@VERSION]",
        help=(
            "Evaluate a supported external baseline instead of a repo run. "
            "Currently supports monai:prostate_mri_anatomy@0.3.5."
        ),
    )
    parser.add_argument(
        "--external-model-version",
        type=str,
        default="",
        metavar="VERSION",
        help="Optional explicit version for --external-model.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/test_images",
        metavar="DIR",
        help="Root directory of PI-CAI .mha image files for the test set",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        metavar="DIR",
        help="Directory containing .nii.gz label masks",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="",
        choices=["", "picai", "prostate158"],
        help="Dataset adapter to use. Default reads dataset_type from the run config.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["test", "val"],
        help=(
            "Dataset split to evaluate. 'test' keeps the fixed hold-out behavior; "
            "'val' evaluates the run's original validation split."
        ),
    )
    parser.add_argument(
        "--roi-mode",
        type=str,
        default="",
        choices=["", "disabled", "gt_mask", "predicted_mask"],
        help=(
            "Override roi.mode from config.yaml. "
            "Default keeps the run config setting."
        ),
    )
    parser.add_argument(
        "--roi-localizer-run",
        type=str,
        default="",
        metavar="DIR_OR_PT",
        help=(
            "Override roi.localizer_run from config.yaml (used with "
            "--roi-mode predicted_mask)."
        ),
    )
    parser.add_argument(
        "--roi-localizer-external-model",
        type=str,
        default="",
        metavar="ID[@VERSION]",
        help=(
            "Use a supported external prostate localizer as the ROI source "
            "for roi.mode=predicted_mask."
        ),
    )
    parser.add_argument(
        "--roi-localizer-external-model-version",
        type=str,
        default="",
        metavar="VERSION",
        help="Optional explicit version for --roi-localizer-external-model.",
    )
    parser.add_argument(
        "--prostate158-root",
        type=str,
        default="",
        metavar="DIR",
        help=(
            "Extracted Prostate158 root. Default reads prostate158_test_dir for "
            "test evaluation and prostate158_train_dir for validation evaluation."
        ),
    )
    parser.add_argument(
        "--prostate158-label-reader",
        type=str,
        default="",
        metavar="N",
        help="Prostate158 label reader to evaluate against. Default reads config or uses 1.",
    )
    parser.add_argument(
        "--prostate158-prostate-label-col",
        type=str,
        default="",
        metavar="COL",
        help=(
            "Prostate158 prostate-mask column (e.g. t2_prostate_reader1). "
            "Needed for roi.mode=gt_mask or task=prostate_localization "
            "when missing in run config."
        ),
    )
    parser.add_argument(
        "--picai-prostate-labels-dir",
        type=str,
        default="",
        metavar="DIR",
        help=(
            "Directory containing PI-CAI whole-prostate masks used as GT ROI "
            "for roi.mode=gt_mask (expected files: <case_id>.nii.gz or "
            "<case_id>_prostate.nii.gz). If omitted, defaults to "
            "/data/prostate_labels then data/prostate_labels when present."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Override the compute device from config "
            "(e.g. 'cpu', 'cuda', 'cuda:0'). "
            "Useful when the GPU is not compatible with the installed PyTorch build."
        ),
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Sliding-window batch size used for evaluation inference. "
            "Lower values reduce peak memory."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable evaluation visualization PNG export (disabled by default).",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Write aggregate evaluation summary JSON to this path. "
            "Default writes to <run>/evaluation_summary.json, or to "
            "<outputs>/evaluation_summaries/ when <run> is not writable."
        ),
    )
    parser.add_argument(
        "--vis-output",
        type=str,
        default="",
        metavar="PATH",
        help=(
            "Write eval visualization PNG to this path (used only with --visualize). "
            "Default writes to <repo_root>/visualizations/<run_name>_eval_visualization.png, "
            "or to <outputs>/evaluation_summaries/ when the preferred directory is not writable."
        ),
    )
    args = parser.parse_args()

    if args.vis_output and not args.visualize:
        logger.warning("--vis-output ignored because --visualize was not enabled.")

    if args.selector or (not args.run and not args.external_model):
        _run_interactive_batch(args)
        return

    if args.run and args.external_model:
        logger.error("--run and --external-model are mutually exclusive.")
        sys.exit(1)
    if args.external_model and args.checkpoint:
        logger.error("--checkpoint cannot be used with --external-model.")
        sys.exit(1)
    if args.external_model and args.eval_split != "test":
        logger.error("--eval-split val requires --run because validation splits are run-specific.")
        sys.exit(1)
    if args.roi_localizer_run and args.roi_localizer_external_model:
        logger.error(
            "--roi-localizer-run and --roi-localizer-external-model are mutually exclusive."
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parent.parent
    run_dir: Path | None = None
    ckpt_path: Path | None = None
    ckpt_epoch: str | int = "external"
    best_val_dice = float("nan")
    external_model_id = ""
    external_model_version = ""
    model_source = "repo_run"

    if args.external_model:
        try:
            spec = resolve_external_model_request(
                args.external_model,
                args.external_model_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("%s", exc)
            sys.exit(1)
        model_source = spec.model_source
        external_model_id = spec.model_id
        external_model_version = spec.bundle_version
        cfg = build_external_eval_config(
            spec,
            prostate158_root=args.prostate158_root,
            prostate158_label_reader=args.prostate158_label_reader,
        )
        logger.info(
            "Using external model baseline: %s@%s",
            spec.model_id,
            spec.bundle_version,
        )
    else:
        # ---- Validate run directory ---------------------------------------------
        run_dir = Path(args.run).resolve()
        ckpts_dir = run_dir / "checkpoints"
        cfg_path = run_dir / "config.yaml"

        if not run_dir.is_dir():
            logger.error("Run directory not found: %s", run_dir)
            sys.exit(1)
        if not cfg_path.exists():
            logger.error("config.yaml not found in run directory: %s", cfg_path)
            sys.exit(1)
        if not ckpts_dir.is_dir():
            logger.error("checkpoints/ subdirectory not found in: %s", run_dir)
            sys.exit(1)

        # ---- Load config --------------------------------------------------------
        cfg = load_config(str(cfg_path))
        logger.info("Config loaded from %s", cfg_path)

        # ---- Select checkpoint --------------------------------------------------
        if args.checkpoint:
            requested = Path(args.checkpoint)
            ckpt_path = requested if requested.is_absolute() else ckpts_dir / requested
            ckpt_path = ckpt_path.resolve()
            if not ckpt_path.exists():
                logger.error("Requested checkpoint not found: %s", ckpt_path)
                sys.exit(1)
        else:
            ckpt_path = select_checkpoint(ckpts_dir)
        logger.info("Selected checkpoint: %s", ckpt_path.name)

    if args.roi_mode:
        logger.info("ROI override from CLI: roi.mode=%s", args.roi_mode)
    if args.roi_localizer_run:
        logger.info("ROI override from CLI: roi.localizer_run=%s", args.roi_localizer_run)
    if args.roi_localizer_external_model:
        logger.info(
            "ROI override from CLI: external localizer=%s version=%s",
            args.roi_localizer_external_model,
            args.roi_localizer_external_model_version or "<default>",
        )
    try:
        cfg = _apply_roi_overrides(
            cfg,
            args.roi_mode,
            args.roi_localizer_run,
            args.roi_localizer_external_model,
            args.roi_localizer_external_model_version,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid ROI override configuration: %s", exc)
        sys.exit(1)
    if args.prostate158_prostate_label_col:
        logger.info(
            "Prostate label column override from CLI: prostate158_prostate_label_col=%s",
            args.prostate158_prostate_label_col,
        )
    if args.picai_prostate_labels_dir:
        logger.info(
            "PI-CAI prostate labels directory override from CLI: %s",
            Path(args.picai_prostate_labels_dir).expanduser().resolve(),
        )
    pred_threshold: float = float(cfg.get("pred_threshold", 0.5))
    postprocess_enabled: bool = bool(cfg.get("postprocess_enabled", False))
    postprocess_min_component_volume_mm3: float = float(
        cfg.get("postprocess_min_component_volume_mm3", 30.0)
    )
    postprocess_connectivity: int = int(cfg.get("postprocess_connectivity", 26))

    if not (0.0 <= pred_threshold <= 1.0):
        logger.error("pred_threshold must be in [0,1], got %s", pred_threshold)
        sys.exit(1)
    if postprocess_min_component_volume_mm3 < 0.0:
        logger.error(
            "postprocess_min_component_volume_mm3 must be >= 0, got %s",
            postprocess_min_component_volume_mm3,
        )
        sys.exit(1)
    if postprocess_connectivity not in (6, 18, 26):
        logger.error(
            "postprocess_connectivity must be one of {6,18,26}, got %s",
            postprocess_connectivity,
        )
        sys.exit(1)

    # ---- Device -----------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
        logger.info("Device overridden via --device flag: %s", device)
    else:
        use_cuda = torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
        device   = torch.device("cuda" if use_cuda else "cpu")

    # ---- Model / adapter --------------------------------------------------------
    external_adapter: MonaiBundleProstateMaskAdapter | None = None
    if args.external_model:
        external_adapter = MonaiBundleProstateMaskAdapter(
            spec=spec,
            device=device,
            cache_root=default_external_model_cache_root(repo_root),
        )
        model = None
    else:
        model = build_model(cfg).to(device)
        assert ckpt_path is not None
        ckpt = load_checkpoint(ckpt_path, model, device=device)
        ckpt_epoch = ckpt.get("epoch", "?")
        best_val_dice = float(ckpt.get("best_val_dice", float("nan")))
        model.eval()

    # ---- Discover + annotate cases ----------------------------------------------
    dataset_type = (
        args.dataset_type
        or str(cfg.get("dataset_type", "picai")).strip().lower()
        or "picai"
    )
    eval_split = str(args.eval_split).strip().lower()
    prostate158_root_for_col = args.prostate158_root
    prostate158_split_for_col = ""
    if dataset_type == "prostate158" and eval_split == "val":
        prostate158_root_for_col = (
            args.prostate158_root
            or str(cfg.get("prostate158_train_dir", "data/prostate158_train"))
        )
        prostate158_split_for_col = str(cfg.get("prostate158_val_split", "valid"))
    cfg, auto_prostate_col = _ensure_prostate_label_col(
        cfg,
        dataset_type=dataset_type,
        explicit_col=args.prostate158_prostate_label_col,
        prostate158_root_override=prostate158_root_for_col,
        prostate158_split_override=prostate158_split_for_col,
    )
    if auto_prostate_col:
        logger.info(
            "Auto-filled prostate158_prostate_label_col=%s from prostate158_label_reader=%s",
            cfg.get("prostate158_prostate_label_col"),
            cfg.get("prostate158_label_reader", 1),
        )
    try:
        cfg, picai_prostate_labels_dir = _ensure_picai_gt_roi_config(
            cfg,
            dataset_type=dataset_type,
            picai_prostate_labels_dir=args.picai_prostate_labels_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid PI-CAI GT ROI configuration: %s", exc)
        sys.exit(1)

    if picai_prostate_labels_dir is not None:
        logger.info("PI-CAI prostate labels directory: %s", picai_prostate_labels_dir)

    try:
        task, _ = validate_task_and_roi_config(cfg, dataset_type)
        # Evaluation uses inference/validation behavior, so resolve the val-stage ROI.
        roi_settings = resolve_roi_settings(cfg, stage="val")
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid evaluation configuration: %s", exc)
        sys.exit(1)
    active_modalities = [key for key, _ in active_modality_pairs(cfg)]
    split_source = "fixed hold-out test set"

    if dataset_type == "prostate158":
        if eval_split == "val":
            images_dir = Path(
                args.prostate158_root
                or cfg.get("prostate158_train_dir", "data/prostate158_train")
            )
            prostate158_split = str(cfg.get("prostate158_val_split", "valid")).strip().lower() or "valid"
            split_source = f"Prostate158 {prostate158_split}.csv"
        else:
            images_dir = Path(
                args.prostate158_root
                or cfg.get("prostate158_test_dir", "data/prostate158_test")
            )
            prostate158_split = str(cfg.get("prostate158_test_split", "test")).strip().lower() or "test"
            split_source = f"Prostate158 {prostate158_split}.csv"
        labels_dir = images_dir
        label_reader = (
            args.prostate158_label_reader
            or cfg.get("prostate158_label_reader", 1)
        )
        prostate_label_col = str(cfg.get("prostate158_prostate_label_col", "")).strip()
        logger.info("Discovering Prostate158 %s cases in %s ...", eval_split, images_dir)
        test_cases = discover_prostate158_cases(
            root_dir=images_dir,
            split=prostate158_split,
            active_keys=active_modalities,
            label_target=str(cfg.get("prostate158_label_target", "tumor")),
            label_reader=label_reader,
            label_modality=cfg.get("prostate158_label_modality"),
            prostate_label_col=prostate_label_col if (prostate_label_col and (task == "prostate_localization" or roi_settings.mode == "gt_mask")) else None,
        )
    elif dataset_type == "picai":
        if eval_split == "val":
            if run_dir is None:
                logger.error("--eval-split val requires --run for PI-CAI evaluation.")
                sys.exit(1)
            try:
                test_cases, images_dir, labels_dir, split_source = _resolve_picai_validation_cases(
                    cfg=cfg,
                    run_dir=run_dir,
                    active_modalities=active_modalities,
                    task=task,
                    roi_settings=roi_settings,
                    picai_prostate_labels_dir=picai_prostate_labels_dir,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not resolve PI-CAI validation split: %s", exc)
                sys.exit(1)
            logger.info(
                "Resolved PI-CAI validation split from %s (%d case(s)).",
                split_source,
                len(test_cases),
            )
        else:
            images_dir = Path(args.images_dir)
            labels_dir = Path(args.labels_dir)
            logger.info("Discovering cases in %s ...", images_dir)
            test_cases = discover_cases(images_dir, labels_dir, active_keys=active_modalities)
            if task == "prostate_localization" or roi_settings.mode == "gt_mask":
                if picai_prostate_labels_dir is None:
                    logger.error(
                        "PI-CAI prostate labels are required for task=%s / roi.mode=%s. "
                        "Defaults are /data/prostate_labels or data/prostate_labels; "
                        "otherwise pass --picai-prostate-labels-dir or set picai_prostate_labels_dir in config.",
                        task,
                        roi_settings.mode,
                    )
                    sys.exit(1)
                _attach_picai_prostate_labels(
                    test_cases,
                    picai_prostate_labels_dir,
                    require_all=True,
                )
    else:
        logger.error("Unsupported dataset_type: %s", dataset_type)
        sys.exit(1)

    if not test_cases:
        logger.error(
            "No cases found in %s. Check that the data directory is correct.",
            images_dir,
        )
        sys.exit(1)

    logger.info("Annotating %d test cases with has_lesion ...", len(test_cases))
    annotate_cases_with_lesion_flags(test_cases)

    if task == "prostate_localization":
        pos_count = len(test_cases)
    else:
        pos_count = sum(1 for c in test_cases if c.get("has_lesion", False))
    neg_count  = len(test_cases) - pos_count

    # ---- Dataset + loader -------------------------------------------------------
    target_spacing: tuple[float, ...] = tuple(
        float(v) for v in cfg.get("target_spacing", [3.0, 0.5, 0.5])
    )
    patch_size: tuple[int, ...] = tuple(
        int(v) for v in cfg.get("patch_size", [20, 128, 128])
    )
    sw_overlap    = float(cfg.get("sw_overlap", 0.5))
    sw_batch_size = int(args.sw_batch_size)
    roi_precompute = _precompute_predicted_roi_bounds(
        test_cases=test_cases,
        roi_settings=roi_settings,
        target_spacing=target_spacing,
        device=device,
        repo_root=repo_root,
    )
    ds = PiCaiDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        target_spacing=target_spacing,
        transform=get_val_transforms(),
        cases=test_cases,
        active_modalities=active_modalities,
        dwi_hbv_preprocess=cfg.get("dwi_hbv_preprocess", {}),
        task=task,
        roi_settings=roi_settings,
        include_full_resampled=roi_settings.enabled and task == "lesion_segmentation",
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,           # fixed small set — no worker overhead needed
        pin_memory=(device.type == "cuda"),
    )

    # ---- Header -----------------------------------------------------------------
    split_label = "validation split" if eval_split == "val" else "fixed test set"
    _section(f"Checkpoint Evaluation  ({split_label})")
    model_label = (
        f"{external_model_id}@{external_model_version}"
        if external_model_id
        else str(run_dir)
    )
    print(f"  Model       : {model_label}")
    print(f"  Source      : {model_source}")
    print(f"  Checkpoint  : {ckpt_path if ckpt_path is not None else '-'}")
    print(f"  Epoch       : {ckpt_epoch}   |   Best val Dice (training): {_fmt(best_val_dice)}")
    print(f"  Device      : {device}")
    print(f"  Eval split  : {eval_split} ({split_source})")
    print(f"  Cases       : {len(test_cases)}  ({pos_count} positive, {neg_count} negative)")
    print(f"  Dataset     : {dataset_type}")
    print(f"  Task        : {task}")
    print(f"  ROI mode    : {roi_settings.mode}")
    if roi_settings.mode == ROI_PREDICTED_MASK:
        print(
            "  ROI source  : "
            f"{_roi_variant_desc('predicted_mask', roi_settings.localizer_run)}"
        )
    if dataset_type == "picai" and picai_prostate_labels_dir is not None:
        print(f"  PI-CAI prostate labels: {picai_prostate_labels_dir}")
    if roi_precompute["enabled"]:
        print(
            "  ROI cache   : "
            f"{roi_precompute['cache_hits']} hit(s), {roi_precompute['cache_misses']} miss(es)"
        )
    if external_adapter is not None:
        print(f"  Bundle cache: {external_adapter.cache_root}")
    print(f"  Images dir  : {images_dir}")
    print(f"  Modalities  : {active_modalities}")
    print(f"  Patch size  : {patch_size}   SW overlap: {sw_overlap}")
    print(f"  Threshold   : {pred_threshold:.3f}")
    print(
        "  Postprocess : "
        f"{'on' if postprocess_enabled else 'off'} "
        f"(min_component_volume_mm3={postprocess_min_component_volume_mm3:.1f}, "
        f"connectivity={postprocess_connectivity})"
    )
    # ---- Inference loop ---------------------------------------------------------
    per_case: list[dict] = []
    vis_data: list[dict] = []          # volumetric arrays for up to 5 positive cases

    # Fast case_id → has_lesion lookup (avoids re-searching test_cases per batch)
    lesion_map: dict[str, bool] = {
        c["case_id"]: c.get("has_lesion", False) for c in test_cases
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="vol"):
            images   = batch["image"].to(device)    # (1, 3, D, H, W)
            labels   = batch.get("full_label", batch["label"]).to(device)
            case_id: str = batch["case_id"][0]
            if external_adapter is not None:
                logits = external_adapter.predict_logits(
                    images,
                    sw_batch_size=sw_batch_size,
                )
            else:
                assert model is not None
                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=patch_size,
                    sw_batch_size=sw_batch_size,
                    predictor=lambda x: _seg_logits(model(x)),
                    overlap=sw_overlap,
                )   # (1, 1, D, H, W) — raw logits

            logits = logits.float()
            if "roi" in batch:
                roi_batch = _roi_bounds_from_collated_batch(batch["roi"])
                logits = restore_from_roi(logits, roi_batch)

            metric_logits, pred_bin = postprocess_logits(
                logits=logits,
                threshold=pred_threshold,
                enabled=postprocess_enabled,
                spacing_zyx=target_spacing,
                min_component_volume_mm3=postprocess_min_component_volume_mm3,
                connectivity=postprocess_connectivity,
            )
            m = compute_all_metrics(
                metric_logits,
                labels,
                threshold=pred_threshold,
            )
            has_target = bool(labels[0, 0].detach().sum().item() > 0)
            has_lesion = lesion_map.get(case_id, False)
            prob_zyx = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
            pred_mask_zyx = pred_bin[0, 0].detach().cpu().numpy() > 0
            target_zyx = labels[0, 0].detach().cpu().numpy() > 0
            detection_map = _detection_map_from_probability(
                prob_zyx=prob_zyx,
                pred_mask_zyx=pred_mask_zyx,
            )
            picai_lesion_results, picai_case_confidence = _picai_evaluate_case(
                det_zyx=detection_map,
                target_zyx=target_zyx,
            )

            per_case.append({
                "case_id":    case_id,
                "has_target": has_target,
                "has_lesion": has_lesion,
                "picai_case_confidence": picai_case_confidence,
                "picai_lesion_results": picai_lesion_results,
                **m,
            })

            # Store volumetric data for the first 5 positive cases (visualization)
            if args.visualize and has_target and len(vis_data) < 5:
                t2w_source = batch.get("full_image", batch["image"])
                roi_payload = None
                if "roi" in batch:
                    roi_payload = _roi_bounds_from_collated_batch(batch["roi"])
                vis_data.append({
                    "case_id":  case_id,
                    "t2w_vol":  t2w_source[0, 0].cpu().numpy(),
                    "gt_vol":   labels[0, 0].cpu().numpy(),      # (D, H, W)
                    "pred_vol": pred_bin[0, 0].cpu().numpy(),    # (D, H, W)
                    "roi_bounds": roi_payload,
                })

    # ---- Per-case table ---------------------------------------------------------
    _section("Per-Case Results")
    cid_w  = max(len(r["case_id"]) for r in per_case)
    case_flag_label = "target" if task == "prostate_localization" else "lesion"
    header = (
        f"  {'case_id':<{cid_w}}  {case_flag_label:<7}"
        f"  {'dice':>7}  {'iou':>7}  {'sens':>7}  {'prec':>7}  {'hd95':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in per_case:
        tag = "yes" if r["has_target"] else "no"
        print(
            f"  {r['case_id']:<{cid_w}}  {tag:<7}"
            f"  {_fmt(r['dice']):>7}  {_fmt(r['iou']):>7}"
            f"  {_fmt(r['sensitivity']):>7}  {_fmt(r['precision']):>7}"
            f"  {_fmt(r['hd95']):>8}"
        )

    # ---- Aggregate metrics ------------------------------------------------------
    pos_rows = [r for r in per_case if r["has_target"]]
    neg_rows = [r for r in per_case if not r["has_target"]]

    # dice / iou / sensitivity: positive cases only (nan-guard)
    dice_vals = [r["dice"]        for r in pos_rows if not math.isnan(r["dice"])]
    iou_vals  = [r["iou"]         for r in pos_rows if not math.isnan(r["iou"])]
    sens_vals = [r["sensitivity"] for r in pos_rows if not math.isnan(r["sensitivity"])]

    # precision: positive cases only (nan when target is empty)
    prec_vals = [r["precision"] for r in pos_rows if not math.isnan(r["precision"])]

    # hd95: non-empty pairs only (nan when either mask is empty)
    hd95_vals = [r["hd95"] for r in per_case if not math.isnan(r["hd95"])]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    _section(
        f"Aggregate Metrics  "
        f"({len(pos_rows)} positive | {len(neg_rows)} negative | {len(per_case)} total)"
    )
    agg_dice = _mean(dice_vals)
    agg_iou = _mean(iou_vals)
    agg_sensitivity = _mean(sens_vals)
    agg_precision = _mean(prec_vals)
    agg_hd95 = _mean(hd95_vals)
    picai_metrics = _picai_ranking_metrics(per_case)

    print(f"  Dice         (positive cases only) : {_fmt(agg_dice)}")
    print(f"  IoU          (positive cases only) : {_fmt(agg_iou)}")
    print(f"  Sensitivity  (positive cases only) : {_fmt(agg_sensitivity)}")
    print(f"  Precision    (positive cases only) : {_fmt(agg_precision)}")
    print(f"  HD95         (non-empty pairs)     : {_fmt(agg_hd95)} voxels")
    print(f"  PI-CAI AP    (lesion ranking)      : {_fmt(picai_metrics['AP'])}")
    print(f"  PI-CAI AUROC (case ranking)        : {_fmt(picai_metrics['AUROC'])}")
    print(f"  PI-CAI score ((AP+AUROC)/2)        : {_fmt(picai_metrics['score'])}")

    # ---- Visualization ----------------------------------------------------------
    vis_path: Path | None = None
    if args.visualize:
        _section("Visualization  (5 rows × 20 axial slices)")
        if args.vis_output:
            vis_path = Path(args.vis_output).expanduser().resolve()
        else:
            vis_stem = (
                run_dir.name
                if run_dir is not None
                else _slugify(f"{external_model_id}_{external_model_version}")
            )
            split_suffix = "_val" if eval_split == "val" else ""
            vis_name = f"{vis_stem}_eval{split_suffix}_visualization.png"
            if run_dir is not None:
                vis_path = _default_vis_output_path(
                    run_dir=run_dir,
                    vis_name=vis_name,
                    repo_root=repo_root,
                ).resolve()
            else:
                vis_path = (repo_root / "visualizations" / vis_name).resolve()
        vis_path.parent.mkdir(parents=True, exist_ok=True)
        save_visualization(vis_data, vis_path, n_cols=_N_VIS_COLS)
        print(
            "\n  Colour key:\n"
            "    Green  = ground truth only\n"
            "    Red    = prediction only\n"
            "    Yellow = overlap (GT ∩ Pred)\n"
        )
    else:
        _section("Visualization")
        print("  Skipped (disabled by default; pass --visualize to enable).")

    # ---- Machine-readable summary ----------------------------------------------
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
    elif run_dir is not None:
        summary_name = "evaluation_summary_val.json" if eval_split == "val" else "evaluation_summary.json"
        summary_path = _default_eval_summary_path(
            run_dir=run_dir,
            summary_name=summary_name,
            repo_root=repo_root,
        ).resolve()
    else:
        summary_path = (
            repo_root
            / "outputs"
            / "evaluation_summaries"
            / f"{_slugify(f'{external_model_id}_{external_model_version}')}_evaluation_summary.json"
        ).resolve()
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "model_source": model_source,
        "external_model_id": external_model_id or None,
        "external_model_version": external_model_version or None,
        "task": task,
        "checkpoint": {
            "path": str(ckpt_path) if ckpt_path is not None else None,
            "name": ckpt_path.name if ckpt_path is not None else None,
            "epoch": ckpt_epoch,
            "best_val_dice_training": _json_float(best_val_dice),
        },
        "dataset": {
            "dataset_type": dataset_type,
            "eval_split": eval_split,
            "split_source": split_source,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "picai_prostate_labels_dir": (
                str(picai_prostate_labels_dir)
                if dataset_type == "picai" and picai_prostate_labels_dir is not None
                else None
            ),
            "total_cases": len(per_case),
            "positive_cases": len(pos_rows),
            "negative_cases": len(neg_rows),
        },
        "inference": {
            "task": task,
            "roi_mode": roi_settings.mode,
            "roi_localizer": (
                {
                    "ref": roi_settings.localizer_run,
                    "description": _roi_variant_desc(
                        "predicted_mask",
                        roi_settings.localizer_run,
                    ),
                }
                if roi_settings.mode == ROI_PREDICTED_MASK
                else None
            ),
            "device": str(device),
            "active_modalities": list(active_modalities),
            "patch_size": list(patch_size),
            "sw_overlap": sw_overlap,
            "sw_batch_size": sw_batch_size,
            "roi_precompute": roi_precompute,
            "external_bundle_cache_dir": (
                str(external_adapter.cache_root) if external_adapter is not None else None
            ),
            "pred_threshold": pred_threshold,
            "postprocess_enabled": postprocess_enabled,
            "postprocess_min_component_volume_mm3": postprocess_min_component_volume_mm3,
            "postprocess_connectivity": postprocess_connectivity,
        },
        "aggregate_metrics": {
            "dice_pos_only": _json_float(agg_dice),
            "iou_pos_only": _json_float(agg_iou),
            "sensitivity_pos_only": _json_float(agg_sensitivity),
            "precision_pos_only": _json_float(agg_precision),
            "hd95_non_empty_pairs_voxels": _json_float(agg_hd95),
            "picai_AP": _json_float(picai_metrics["AP"]),
            "picai_AUROC": _json_float(picai_metrics["AUROC"]),
            "picai_score": _json_float(picai_metrics["score"]),
            "picai_num_lesions": picai_metrics["num_lesions"],
            "picai_num_candidates": picai_metrics["num_candidates"],
        },
        "picai_ranking": {
            "AP": _json_float(picai_metrics["AP"]),
            "AUROC": _json_float(picai_metrics["AUROC"]),
            "score": _json_float(picai_metrics["score"]),
            "score_formula": "(AP + AUROC) / 2",
            "min_overlap": 0.10,
            "overlap_func": "IoU",
            "connectivity": 26,
            "case_confidence_func": "max",
            "num_lesions": picai_metrics["num_lesions"],
            "num_candidates": picai_metrics["num_candidates"],
            "case_pred": {
                row["case_id"]: _json_float(float(row["picai_case_confidence"]))
                for row in per_case
            },
            "case_target": {
                row["case_id"]: int(row["has_target"])
                for row in per_case
            },
            "lesion_results": {
                row["case_id"]: [
                    [int(is_lesion), float(confidence), float(overlap)]
                    for is_lesion, confidence, overlap in row["picai_lesion_results"]
                ]
                for row in per_case
            },
        },
        "artifacts": {
            "visualization_enabled": bool(args.visualize),
            "eval_visualization_png": str(vis_path) if vis_path is not None else None,
            "eval_visualization_png_exists": bool(vis_path is not None and vis_path.exists()),
            "visualized_positive_cases": len(vis_data) if args.visualize else 0,
            "visualization_cols": _N_VIS_COLS if args.visualize else 0,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"  Summary saved       →  {summary_path}")


if __name__ == "__main__":
    main()
