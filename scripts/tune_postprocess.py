"""
tune_postprocess.py — tune frozen-model post-processing on validation logits.

This script intentionally uses the training validation split, not the held-out
test set used by evaluate_checkpoint.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from dataset import (  # noqa: E402
    PiCaiDataset,
    active_modality_pairs,
    annotate_cases_with_lesion_flags,
    default_split_manifest_path,
    discover_cases,
    discover_prostate158_cases,
    load_split_manifest,
)
from metrics import compute_all_metrics  # noqa: E402
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from roi import ROI_PREDICTED_MASK, resolve_roi_settings, restore_from_roi, validate_task_and_roi_config  # noqa: E402
from transforms import get_val_transforms  # noqa: E402
from utils import load_checkpoint  # noqa: E402

from evaluate_checkpoint import (  # noqa: E402
    _apply_roi_overrides,
    _attach_picai_prostate_labels,
    _default_eval_summary_path,
    _ensure_picai_gt_roi_config,
    _ensure_prostate_label_col,
    _fmt,
    _precompute_predicted_roi_bounds,
    _roi_bounds_from_collated_batch,
    _roi_variant_desc,
    _section,
    _seg_logits,
    select_checkpoint,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_float_grid(raw: str, *, name: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            bits = [b.strip() for b in token.split(":")]
            if len(bits) != 3:
                raise ValueError(f"{name} range must be start:stop:step, got {token!r}")
            start, stop, step = (float(b) for b in bits)
            if step == 0.0:
                raise ValueError(f"{name} range step must be non-zero: {token!r}")
            if (stop - start) * step < 0.0:
                raise ValueError(f"{name} range step moves away from stop: {token!r}")
            x = start
            eps = abs(step) * 1e-6
            if step > 0:
                while x <= stop + eps:
                    values.append(float(x))
                    x += step
            else:
                while x >= stop - eps:
                    values.append(float(x))
                    x += step
            continue
        values.append(float(token))

    if not values:
        raise ValueError(f"{name} grid is empty.")
    return [float(v) for v in sorted({round(v, 10) for v in values})]


def _grid_config_to_text(raw: Any, *, name: str) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (list, tuple)):
        return ",".join(str(v) for v in raw)
    raise TypeError(f"{name} must be a string range or a list of numbers, got {type(raw)!r}")


def _score_tuned_postprocess(row: Mapping[str, Any], objective: str) -> float:
    dice = float(row.get("dice_pos_only", float("nan")))
    precision = float(row.get("precision_pos_only", float("nan")))
    sensitivity = float(row.get("sensitivity_pos_only", float("nan")))
    neg_pred_rate = float(row.get("negative_pred_case_rate", float("nan")))

    dice = dice if math.isfinite(dice) else 0.0
    precision = precision if math.isfinite(precision) else 0.0
    sensitivity = sensitivity if math.isfinite(sensitivity) else 0.0
    neg_pred_rate = neg_pred_rate if math.isfinite(neg_pred_rate) else 0.0

    if objective == "dice":
        return dice
    if objective == "precision":
        return precision
    if objective == "sensitivity":
        return sensitivity
    if objective == "balanced":
        return 0.50 * dice + 0.25 * precision + 0.25 * sensitivity
    if objective == "fp_penalized":
        return 0.55 * dice + 0.25 * precision + 0.20 * sensitivity - 0.20 * neg_pred_rate
    raise ValueError(f"Unsupported tune objective: {objective}")


def _select_cases_by_manifest_ids(
    *,
    all_cases: list[dict[str, Any]],
    case_ids: list[str],
    manifest_path: Path,
) -> list[dict[str, Any]]:
    case_map = {str(case["case_id"]): case for case in all_cases}
    missing = [cid for cid in case_ids if cid not in case_map]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(
            f"Validation split manifest references {len(missing)} case(s) "
            f"not found in current data: {preview}{suffix} (manifest={manifest_path})"
        )
    return [case_map[cid] for cid in case_ids]


def _resolve_picai_val_cases_for_tuning(
    *,
    cfg: Mapping[str, Any],
    run_dir: Path,
    active_modalities: list[str],
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

    manifest_path = next((p for p in manifest_candidates if p.exists()), None)
    if manifest_path is None:
        checked = ", ".join(str(p) for p in manifest_candidates)
        raise FileNotFoundError(
            "Could not find PI-CAI train/validation split manifest for tuning. "
            f"Checked: {checked}"
        )

    all_cases = discover_cases(images_dir, labels_dir, active_keys=active_modalities)
    if roi_settings.mode == "gt_mask":
        if picai_prostate_labels_dir is None:
            raise ValueError("PI-CAI prostate labels are required for tuning with roi.mode=gt_mask.")
        _attach_picai_prostate_labels(all_cases, picai_prostate_labels_dir, require_all=True)

    manifest = load_split_manifest(manifest_path)
    val_cases = _select_cases_by_manifest_ids(
        all_cases=all_cases,
        case_ids=[str(v) for v in manifest["val_case_ids"]],
        manifest_path=manifest_path,
    )
    return val_cases, images_dir, labels_dir, str(manifest_path)


def _resolve_val_cases_for_tuning(
    *,
    cfg: Mapping[str, Any],
    run_dir: Path,
    dataset_type: str,
    task: str,
    roi_settings: Any,
    active_modalities: list[str],
    picai_prostate_labels_dir: Path | None,
    prostate158_root_override: str,
    prostate158_label_reader_override: str,
) -> tuple[list[dict[str, Any]], Path, Path, str]:
    if dataset_type == "picai":
        return _resolve_picai_val_cases_for_tuning(
            cfg=cfg,
            run_dir=run_dir,
            active_modalities=active_modalities,
            roi_settings=roi_settings,
            picai_prostate_labels_dir=picai_prostate_labels_dir,
        )

    if dataset_type == "prostate158":
        images_dir = Path(
            prostate158_root_override
            or cfg.get("prostate158_train_dir", cfg.get("prostate158_test_dir", "data/prostate158_train"))
        )
        labels_dir = images_dir
        label_reader = prostate158_label_reader_override or cfg.get("prostate158_label_reader", 1)
        prostate_label_col = str(cfg.get("prostate158_prostate_label_col", "")).strip()
        split = str(cfg.get("prostate158_val_split", "valid")).strip().lower() or "valid"
        val_cases = discover_prostate158_cases(
            root_dir=images_dir,
            split=split,
            active_keys=active_modalities,
            label_target=str(cfg.get("prostate158_label_target", "tumor")),
            label_reader=label_reader,
            label_modality=cfg.get("prostate158_label_modality"),
            prostate_label_col=prostate_label_col if (prostate_label_col and (task == "prostate_localization" or roi_settings.mode == "gt_mask")) else None,
        )
        return val_cases, images_dir, labels_dir, f"Prostate158 {split}.csv"

    raise ValueError(f"Unsupported dataset_type for postprocess tuning: {dataset_type}")


def _cache_logits_for_postprocess_tuning(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: tuple[int, ...],
    sw_overlap: float,
    sw_batch_size: int,
    lesion_map: Mapping[str, bool],
) -> list[dict[str, Any]]:
    cached_cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Tune inference", unit="vol"):
            images = batch["image"].to(device)
            labels = batch.get("full_label", batch["label"]).to(device)
            case_id: str = batch["case_id"][0]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=lambda x: _seg_logits(model(x)),
                overlap=sw_overlap,
            ).float()
            if "roi" in batch:
                logits = restore_from_roi(logits, _roi_bounds_from_collated_batch(batch["roi"]))

            cached_cases.append(
                {
                    "case_id": case_id,
                    "has_target": bool(labels[0, 0].detach().sum().item() > 0),
                    "has_lesion": bool(lesion_map.get(case_id, False)),
                    "logits": logits.detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )
    return cached_cases


def _summarize_postprocess_metrics(
    *,
    cached_cases: list[dict[str, Any]],
    threshold: float,
    postprocess_enabled: bool,
    min_component_volume_mm3: float,
    connectivity: int,
    spacing_zyx: tuple[float, ...],
    compute_hd95: bool,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    for item in cached_cases:
        metric_logits, pred_bin = postprocess_logits(
            logits=item["logits"],
            threshold=threshold,
            enabled=postprocess_enabled,
            spacing_zyx=spacing_zyx,  # type: ignore[arg-type]
            min_component_volume_mm3=min_component_volume_mm3,
            connectivity=connectivity,
        )
        metrics = compute_all_metrics(
            metric_logits,
            item["labels"],
            threshold=threshold,
            compute_hd95=compute_hd95,
        )
        per_case.append(
            {
                "case_id": item["case_id"],
                "has_target": bool(item["has_target"]),
                "has_lesion": bool(item["has_lesion"]),
                "pred_positive": bool(pred_bin.detach().sum().item() > 0),
                **metrics,
            }
        )

    pos_rows = [r for r in per_case if r["has_target"]]
    neg_rows = [r for r in per_case if not r["has_target"]]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    neg_pred_cases = sum(1 for r in neg_rows if r["pred_positive"])
    pos_pred_cases = sum(1 for r in pos_rows if r["pred_positive"])
    return {
        "threshold": float(threshold),
        "postprocess_enabled": bool(postprocess_enabled),
        "postprocess_min_component_volume_mm3": float(min_component_volume_mm3),
        "postprocess_connectivity": int(connectivity),
        "dice_pos_only": _mean([r["dice"] for r in pos_rows if not math.isnan(r["dice"])]),
        "iou_pos_only": _mean([r["iou"] for r in pos_rows if not math.isnan(r["iou"])]),
        "sensitivity_pos_only": _mean([r["sensitivity"] for r in pos_rows if not math.isnan(r["sensitivity"])]),
        "precision_pos_only": _mean([r["precision"] for r in pos_rows if not math.isnan(r["precision"])]),
        "hd95_non_empty_pairs_voxels": _mean([r["hd95"] for r in per_case if not math.isnan(r["hd95"])]),
        "positive_pred_case_rate": pos_pred_cases / len(pos_rows) if pos_rows else float("nan"),
        "negative_pred_case_rate": neg_pred_cases / len(neg_rows) if neg_rows else float("nan"),
        "negative_pred_cases": neg_pred_cases,
        "positive_cases": len(pos_rows),
        "negative_cases": len(neg_rows),
        "total_cases": len(per_case),
    }


def _tune_postprocess_grid(
    *,
    cached_cases: list[dict[str, Any]],
    threshold_grid: list[float],
    min_component_volume_grid: list[float],
    connectivity: int,
    spacing_zyx: tuple[float, ...],
    objective: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total = len(threshold_grid) * len(min_component_volume_grid)
    with tqdm(total=total, desc="Tune postprocess", unit="cfg") as progress:
        for threshold in threshold_grid:
            for min_volume in min_component_volume_grid:
                row = _summarize_postprocess_metrics(
                    cached_cases=cached_cases,
                    threshold=threshold,
                    postprocess_enabled=min_volume > 0.0,
                    min_component_volume_mm3=min_volume,
                    connectivity=connectivity,
                    spacing_zyx=spacing_zyx,
                    compute_hd95=True,
                )
                row["objective"] = objective
                row["objective_score"] = _score_tuned_postprocess(row, objective)
                rows.append(row)
                progress.update(1)

    rows.sort(
        key=lambda r: (
            float(r["objective_score"]),
            float(r["dice_pos_only"]) if math.isfinite(float(r["dice_pos_only"])) else -1.0,
            -float(r["negative_pred_case_rate"]) if math.isfinite(float(r["negative_pred_case_rate"])) else -1.0,
        ),
        reverse=True,
    )
    return rows[0], rows


def _resolve_tune_config_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute() and not path.exists():
        repo_relative = _REPO_ROOT / path
        if repo_relative.exists():
            return repo_relative.resolve()
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune frozen-model post-processing on the original validation split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", required=True, metavar="DIR", help="Training run directory.")
    parser.add_argument("--checkpoint", default="best.pt", metavar="NAME_OR_PATH", help="Checkpoint filename in <run>/checkpoints or explicit path.")
    parser.add_argument("--config", default="configs/postprocess_tuning.yaml", metavar="PATH", help="Post-processing tuning YAML preset.")
    parser.add_argument("--dataset-type", default="", choices=["", "picai", "prostate158"], help="Dataset adapter override.")
    parser.add_argument("--roi-mode", default="", choices=["", "disabled", "gt_mask", "predicted_mask"], help="Override roi.mode from config.yaml.")
    parser.add_argument("--roi-localizer-run", default="", metavar="DIR_OR_PT", help="Override roi.localizer_run for predicted ROI.")
    parser.add_argument("--roi-localizer-external-model", default="", metavar="ID[@VERSION]", help="Use a supported external prostate localizer for predicted ROI.")
    parser.add_argument("--roi-localizer-external-model-version", default="", metavar="VERSION", help="Optional explicit external localizer version.")
    parser.add_argument("--prostate158-root", default="", metavar="DIR", help="Extracted Prostate158 train root for validation tuning.")
    parser.add_argument("--prostate158-label-reader", default="", metavar="N", help="Prostate158 label reader override.")
    parser.add_argument("--prostate158-prostate-label-col", default="", metavar="COL", help="Prostate158 prostate-mask column for GT ROI/localizer tasks.")
    parser.add_argument("--picai-prostate-labels-dir", default="", metavar="DIR", help="PI-CAI whole-prostate masks for roi.mode=gt_mask.")
    parser.add_argument("--device", default=None, metavar="DEVICE", help="Compute device override.")
    parser.add_argument("--sw-batch-size", type=int, default=2, metavar="N", help="Sliding-window inference batch size.")
    parser.add_argument("--threshold-grid", default=None, metavar="GRID", help="Override threshold grid.")
    parser.add_argument("--min-component-volume-grid", default=None, metavar="GRID", help="Override min component volume grid in mm^3.")
    parser.add_argument("--objective", default=None, choices=["dice", "precision", "sensitivity", "balanced", "fp_penalized"], help="Override objective used to rank settings.")
    parser.add_argument("--connectivity", type=int, default=None, choices=[6, 18, 26], help="Override connected-component connectivity.")
    parser.add_argument("--summary-json", default="", metavar="PATH", help="Write full tuning summary to this path.")
    args = parser.parse_args()

    if args.roi_localizer_run and args.roi_localizer_external_model:
        logger.error("--roi-localizer-run and --roi-localizer-external-model are mutually exclusive.")
        sys.exit(1)

    run_dir = Path(args.run).expanduser().resolve()
    ckpts_dir = run_dir / "checkpoints"
    cfg_path = run_dir / "config.yaml"
    if not run_dir.is_dir() or not cfg_path.exists() or not ckpts_dir.is_dir():
        logger.error("Invalid run directory: %s", run_dir)
        sys.exit(1)

    requested = Path(args.checkpoint)
    ckpt_path = requested if requested.is_absolute() else ckpts_dir / requested
    if not ckpt_path.exists():
        ckpt_path = select_checkpoint(ckpts_dir)
    ckpt_path = ckpt_path.resolve()

    cfg = load_config(str(cfg_path))
    tune_cfg_path = _resolve_tune_config_path(args.config)
    try:
        loaded_tune_cfg = load_config(str(tune_cfg_path))
        block = loaded_tune_cfg.get("postprocess_tuning", loaded_tune_cfg)
        if not isinstance(block, dict):
            raise TypeError("postprocess_tuning must be a mapping")
        tune_cfg = dict(block)
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid post-processing tuning config: %s", exc)
        sys.exit(1)

    threshold_grid_raw = args.threshold_grid if args.threshold_grid is not None else _grid_config_to_text(tune_cfg.get("threshold_grid", "0.30:0.80:0.05"), name="threshold_grid")
    min_component_grid_raw = args.min_component_volume_grid if args.min_component_volume_grid is not None else _grid_config_to_text(tune_cfg.get("min_component_volume_mm3_grid", tune_cfg.get("min_component_volume_grid", "0,10,30,50,100,200,300,500")), name="min_component_volume_mm3_grid")
    objective = str(args.objective or tune_cfg.get("objective", "fp_penalized"))
    if objective not in {"dice", "precision", "sensitivity", "balanced", "fp_penalized"}:
        logger.error("Unsupported tuning objective: %s", objective)
        sys.exit(1)
    try:
        threshold_grid = _parse_float_grid(threshold_grid_raw, name="threshold-grid")
        min_component_volume_grid = _parse_float_grid(min_component_grid_raw, name="min-component-volume-grid")
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid post-processing tuning grid: %s", exc)
        sys.exit(1)
    if any(v < 0.0 or v > 1.0 for v in threshold_grid):
        logger.error("All threshold-grid values must be in [0,1].")
        sys.exit(1)
    if any(v < 0.0 for v in min_component_volume_grid):
        logger.error("All min-component-volume-grid values must be >= 0.")
        sys.exit(1)

    cfg = _apply_roi_overrides(
        cfg,
        args.roi_mode,
        args.roi_localizer_run,
        args.roi_localizer_external_model,
        args.roi_localizer_external_model_version,
    )
    dataset_type = args.dataset_type or str(cfg.get("dataset_type", "picai")).strip().lower() or "picai"
    cfg, _ = _ensure_prostate_label_col(
        cfg,
        dataset_type=dataset_type,
        explicit_col=args.prostate158_prostate_label_col,
        prostate158_root_override=args.prostate158_root,
    )
    try:
        cfg, picai_prostate_labels_dir = _ensure_picai_gt_roi_config(
            cfg,
            dataset_type=dataset_type,
            picai_prostate_labels_dir=args.picai_prostate_labels_dir,
        )
        task, _ = validate_task_and_roi_config(cfg, dataset_type)
        roi_settings = resolve_roi_settings(cfg, stage="val")
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid tuning configuration: %s", exc)
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    active_modalities = [key for key, _ in active_modality_pairs(cfg)]
    target_spacing = tuple(float(v) for v in cfg.get("target_spacing", [3.0, 0.5, 0.5]))
    patch_size = tuple(int(v) for v in cfg.get("patch_size", [20, 128, 128]))
    sw_overlap = float(cfg.get("sw_overlap", 0.5))
    connectivity = int(args.connectivity if args.connectivity is not None else cfg.get("postprocess_connectivity", 26))

    try:
        val_cases, images_dir, labels_dir, split_source = _resolve_val_cases_for_tuning(
            cfg=cfg,
            run_dir=run_dir,
            dataset_type=dataset_type,
            task=task,
            roi_settings=roi_settings,
            active_modalities=active_modalities,
            picai_prostate_labels_dir=picai_prostate_labels_dir,
            prostate158_root_override=args.prostate158_root,
            prostate158_label_reader_override=args.prostate158_label_reader,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not resolve validation set for post-processing tuning: %s", exc)
        sys.exit(1)
    if not val_cases:
        logger.error("Validation set for post-processing tuning is empty.")
        sys.exit(1)

    annotate_cases_with_lesion_flags(val_cases)
    pos_count = sum(1 for c in val_cases if c.get("has_lesion", False))
    neg_count = len(val_cases) - pos_count

    _section("Post-Processing Tuning  (validation split)")
    print(f"  Run         : {run_dir}")
    print(f"  Checkpoint  : {ckpt_path.name}")
    print(f"  Epoch       : {ckpt.get('epoch', '?')}")
    print(f"  Device      : {device}")
    print(f"  Dataset     : {dataset_type}")
    print(f"  Task        : {task}")
    print(f"  ROI mode    : {roi_settings.mode}")
    if roi_settings.mode == ROI_PREDICTED_MASK:
        print(f"  ROI source  : {_roi_variant_desc('predicted_mask', roi_settings.localizer_run)}")
    print(f"  Tuning data : {len(val_cases)} validation case(s) ({pos_count} positive, {neg_count} negative)")
    print(f"  Split source: {split_source}")
    print(f"  Images dir  : {images_dir}")
    print(f"  Grid        : {len(threshold_grid)} threshold(s), {len(min_component_volume_grid)} component-volume setting(s)")
    print(f"  Objective   : {objective}")

    roi_precompute = _precompute_predicted_roi_bounds(
        test_cases=val_cases,
        roi_settings=roi_settings,
        target_spacing=target_spacing,
        device=device,
        repo_root=_REPO_ROOT,
    )
    if roi_precompute["enabled"]:
        print(f"  ROI cache   : {roi_precompute['cache_hits']} hit(s), {roi_precompute['cache_misses']} miss(es)")

    ds = PiCaiDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        target_spacing=target_spacing,
        transform=get_val_transforms(),
        cases=val_cases,
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
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    lesion_map = {str(c["case_id"]): bool(c.get("has_lesion", False)) for c in val_cases}
    cached_cases = _cache_logits_for_postprocess_tuning(
        loader=loader,
        model=model,
        device=device,
        patch_size=patch_size,
        sw_overlap=sw_overlap,
        sw_batch_size=int(args.sw_batch_size),
        lesion_map=lesion_map,
    )
    best, rows = _tune_postprocess_grid(
        cached_cases=cached_cases,
        threshold_grid=threshold_grid,
        min_component_volume_grid=min_component_volume_grid,
        connectivity=connectivity,
        spacing_zyx=target_spacing,
        objective=objective,
    )

    print("\n  Best setting:")
    print(f"    pred_threshold: {_fmt(float(best['threshold']))}")
    print(f"    postprocess_enabled: {bool(best['postprocess_enabled'])}")
    print(f"    postprocess_min_component_volume_mm3: {float(best['postprocess_min_component_volume_mm3']):.1f}")
    print(f"    postprocess_connectivity: {int(best['postprocess_connectivity'])}")
    print(f"    objective_score: {_fmt(float(best['objective_score']))}")
    print(
        "    metrics: "
        f"dice={_fmt(float(best['dice_pos_only']))}, "
        f"sens={_fmt(float(best['sensitivity_pos_only']))}, "
        f"prec={_fmt(float(best['precision_pos_only']))}, "
        f"neg_pred_rate={_fmt(float(best['negative_pred_case_rate']))}, "
        f"hd95={_fmt(float(best['hd95_non_empty_pairs_voxels']))}"
    )

    print("\n  Top 5 settings:")
    print("    " f"{'thr':>5}  {'min_mm3':>8}  {'score':>7}  {'dice':>7}  {'sens':>7}  {'prec':>7}  {'neg+':>7}")
    for row in rows[:5]:
        print(
            "    "
            f"{float(row['threshold']):5.2f}  "
            f"{float(row['postprocess_min_component_volume_mm3']):8.1f}  "
            f"{_fmt(float(row['objective_score'])):>7}  "
            f"{_fmt(float(row['dice_pos_only'])):>7}  "
            f"{_fmt(float(row['sensitivity_pos_only'])):>7}  "
            f"{_fmt(float(row['precision_pos_only'])):>7}  "
            f"{_fmt(float(row['negative_pred_case_rate'])):>7}"
        )

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
    else:
        summary_path = _default_eval_summary_path(
            run_dir=run_dir,
            summary_name="postprocess_tuning_summary.json",
            repo_root=_REPO_ROOT,
        ).resolve()

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "model_source": "repo_run",
        "checkpoint": str(ckpt_path),
        "dataset_type": dataset_type,
        "task": task,
        "tuning_dataset": {
            "source": split_source,
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "total_cases": len(cached_cases),
            "positive_cases": pos_count,
            "negative_cases": neg_count,
            "roi_precompute": roi_precompute,
        },
        "objective": objective,
        "grid": {
            "threshold": threshold_grid,
            "postprocess_min_component_volume_mm3": min_component_volume_grid,
            "postprocess_connectivity": connectivity,
        },
        "best": best,
        "results": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\n  Tuning summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
