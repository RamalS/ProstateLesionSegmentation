#!/usr/bin/env python3
"""
Evaluate a 2D Deconver checkpoint on JPG/PNG segmentation pairs.

Usage:
  PYTHONPATH=. python scripts/evaluate_checkpoint_2d.py \
      --run outputs/runs/<run_name> \
      [--images-dir data/test_images_2d] \
      [--labels-dir data/test_labels_2d]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from dataset_2d import ImageMask2DDataset, discover_cases_2d  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from transforms_2d import get_val_transforms_2d  # noqa: E402
from utils import ensure_cuda_binary_compatibility, load_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _fmt(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _json_float(v: float) -> float | None:
    return float(v) if math.isfinite(v) else None


def _resolve_checkpoint(run_dir: Path, ckpt_arg: str | None) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory missing: {ckpt_dir}")

    if ckpt_arg:
        direct = Path(ckpt_arg)
        if direct.exists():
            return direct.resolve()
        candidate = ckpt_dir / ckpt_arg
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_arg} (checked direct path and {ckpt_dir})"
        )

    best = ckpt_dir / "best.pt"
    if best.exists():
        return best.resolve()

    epoch_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not epoch_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    return epoch_files[-1].resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Deconver 2D checkpoint on JPG/PNG pairs.",
    )
    parser.add_argument(
        "--run",
        required=True,
        type=str,
        help="Run directory containing config.yaml and checkpoints/",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path or filename inside run/checkpoints (default: best.pt or latest epoch).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Override images_dir from run config.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Override labels_dir from run config.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Output JSON path (default: <run>/evaluation_2d_summary.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Run config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))

    model_name = str(cfg.get("model", "")).lower()
    spatial_dims = int(cfg.get("spatial_dims", 3))
    if model_name != "deconver" or spatial_dims != 2:
        raise ValueError(
            "This evaluator is for Deconver 2D runs only (model='deconver', spatial_dims=2). "
            f"Got model={model_name!r}, spatial_dims={spatial_dims}."
        )

    images_dir = Path(args.images_dir or cfg["images_dir"]).resolve()
    labels_dir = Path(args.labels_dir or cfg["labels_dir"]).resolve()
    image_ext = str(cfg.get("image_ext", ".jpg"))
    label_ext = str(cfg.get("label_ext", ".png"))
    recursive_discovery = bool(cfg.get("recursive_discovery", False))
    input_channels = int(cfg.get("input_channels", 1))

    ckpt_path = _resolve_checkpoint(run_dir, args.checkpoint)
    logger.info("Using checkpoint: %s", ckpt_path)

    cases = discover_cases_2d(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_ext=image_ext,
        label_ext=label_ext,
        recursive=recursive_discovery,
        strict=True,
    )
    if not cases:
        raise RuntimeError(f"No JPG/PNG pairs found in {images_dir} / {labels_dir}")

    patch_size = tuple(int(v) for v in cfg.get("patch_size", [256, 256]))
    if len(patch_size) != 2:
        raise ValueError(f"patch_size must be [H, W], got {patch_size}")
    sw_overlap = float(cfg.get("sw_overlap", 0.5))
    sw_batch_size = int(cfg.get("sw_batch_size", 8))
    pred_threshold = float(cfg.get("pred_threshold", 0.5))
    postprocess_enabled = bool(cfg.get("postprocess_enabled", False))
    postprocess_min_component_volume_mm3 = float(
        cfg.get("postprocess_min_component_volume_mm3", 30.0)
    )
    postprocess_connectivity = int(cfg.get("postprocess_connectivity", 8))
    spacing_yx = tuple(float(v) for v in cfg.get("target_spacing", [0.5, 0.5]))

    if postprocess_connectivity not in (4, 8):
        raise ValueError(
            f"postprocess_connectivity must be 4 or 8 for 2D, got {postprocess_connectivity}"
        )
    if len(spacing_yx) != 2:
        raise ValueError(f"target_spacing must contain 2 values [y, x], got {spacing_yx}")

    ds = ImageMask2DDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        transform=get_val_transforms_2d(),
        cases=cases,
        use_cache=bool(cfg.get("cache_dataset", False)),
        cache_rate=float(cfg.get("cache_rate", 1.0)),
        input_channels=input_channels,
        image_ext=image_ext,
        label_ext=label_ext,
        recursive=recursive_discovery,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu"
    )
    ensure_cuda_binary_compatibility(device)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    per_case: list[dict[str, object]] = []
    pos_sums: dict[str, float] = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0, "precision": 0.0}
    n_pos = 0
    n_all = 0
    hd95_values: list[float] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluate", unit="img"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            case_id = str(batch["case_id"][0])

            def _predictor(x):
                out = model(x)
                return out[0] if isinstance(out, list) else out

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=_predictor,
                overlap=sw_overlap,
            )
            metric_logits, _ = postprocess_logits(
                logits=logits.float(),
                threshold=pred_threshold,
                enabled=postprocess_enabled,
                spacing_zyx=spacing_yx,
                min_component_volume_mm3=postprocess_min_component_volume_mm3,
                connectivity=postprocess_connectivity,
            )
            metrics = compute_all_metrics(
                metric_logits,
                labels,
                threshold=pred_threshold,
                compute_hd95=True,
            )

            n_all += 1
            if not math.isnan(metrics["dice"]):
                for k in pos_sums:
                    pos_sums[k] += metrics[k]
                n_pos += 1
            if not math.isnan(metrics["hd95"]):
                hd95_values.append(metrics["hd95"])

            per_case.append(
                {
                    "case_id": case_id,
                    "dice": _json_float(metrics["dice"]),
                    "iou": _json_float(metrics["iou"]),
                    "sensitivity": _json_float(metrics["sensitivity"]),
                    "precision": _json_float(metrics["precision"]),
                    "hd95": _json_float(metrics["hd95"]),
                }
            )

    aggregate: dict[str, float] = {
        "hd95": float(sum(hd95_values) / len(hd95_values)) if hd95_values else float("nan"),
        "n_pos": float(n_pos),
        "n_all": float(n_all),
    }
    if n_pos > 0:
        for key, val in pos_sums.items():
            aggregate[key] = val / n_pos
    else:
        aggregate["dice"] = float("nan")
        aggregate["iou"] = float("nan")
        aggregate["sensitivity"] = float("nan")
        aggregate["precision"] = float("nan")

    logger.info(
        "Aggregate | dice=%s iou=%s sens=%s prec=%s hd95=%s | pos_cases=%d/%d",
        _fmt(aggregate["dice"]),
        _fmt(aggregate["iou"]),
        _fmt(aggregate["sensitivity"]),
        _fmt(aggregate["precision"]),
        _fmt(aggregate["hd95"]),
        int(aggregate["n_pos"]),
        int(aggregate["n_all"]),
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "patch_size": list(patch_size),
        "sw_overlap": sw_overlap,
        "sw_batch_size": sw_batch_size,
        "pred_threshold": pred_threshold,
        "postprocess_enabled": postprocess_enabled,
        "postprocess_min_component_volume_mm3": postprocess_min_component_volume_mm3,
        "postprocess_connectivity": postprocess_connectivity,
        "target_spacing": list(spacing_yx),
        "aggregate": {k: _json_float(v) for k, v in aggregate.items()},
        "per_case": per_case,
    }

    out_path = (
        Path(args.output_json).resolve()
        if args.output_json is not None
        else run_dir / "evaluation_2d_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    logger.info("Saved evaluation summary to %s", out_path)


if __name__ == "__main__":
    main()

