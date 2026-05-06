#!/usr/bin/env python3
"""
report_runs.py — Generate a Markdown report for supervised training runs.

Default behavior scans all runs under ``outputs/runs`` and prints:
1) a compact comparison table (one row per run),
2) a detailed section for each run.

Primary metric source is TensorBoard event files in ``<run>/tensorboard``.
If TensorBoard parsing is unavailable, the script still reports config and
checkpoint-derived metadata, but validation metrics will be ``n/a``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}_.+$")
EPOCH_FILE_RE = re.compile(r"^epoch_(\d{4})\.pt$")

VAL_TAGS: dict[str, str] = {
    "dice": "val/dice",
    "iou": "val/iou",
    "sensitivity": "val/sensitivity",
    "precision": "val/precision",
    "hd95": "val/hd95",
    "composite_score": "val/composite_score",
}
TRAIN_TAGS: dict[str, str] = {
    "loss": "train/loss",
    "lr": "train/lr",
}

DEFAULT_SORT_BY = "best_composite_score"
SORT_CHOICES = (
    "best_composite_score",
    "best_dice",
    "duration_hours",
    "start_time",
)

EVAL_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("dice_pos_only", "dice"),
    ("iou_pos_only", "iou"),
    ("sensitivity_pos_only", "sensitivity"),
    ("precision_pos_only", "precision"),
    ("hd95_non_empty_pairs_voxels", "hd95"),
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    run_dir: Path
    run_name: str
    warnings: list[str] = field(default_factory=list)

    experiment_name: str = "unknown"
    git_commit: str = "unknown"
    config_source: str = "none"
    config: dict[str, Any] = field(default_factory=dict)

    start_ts: float | None = None
    end_ts: float | None = None
    duration_sec: float | None = None

    configured_epochs: int | None = None
    stopped_epoch: int | None = None
    best_epoch: int | None = None
    early_stopped: bool | None = None

    best_composite_score: float | None = None
    best_dice: float | None = None
    best_iou: float | None = None
    best_sensitivity: float | None = None
    best_precision: float | None = None
    best_hd95: float | None = None

    last_val_epoch: int | None = None
    last_val: dict[str, float | None] = field(default_factory=dict)
    best_epoch_val: dict[str, float | None] = field(default_factory=dict)

    n_params_total: int | None = None
    n_params_trainable: int | None = None
    orbit_gif_path: Path | None = None
    eval_summary_path: Path | None = None
    eval_visualization_path: Path | None = None
    eval_checkpoint_name: str | None = None
    eval_metrics: dict[str, float | None] = field(default_factory=dict)
    eval_total_cases: int | None = None
    eval_positive_cases: int | None = None
    eval_negative_cases: int | None = None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _fmt_float(val: float | None, ndigits: int = 4) -> str:
    if val is None or not math.isfinite(val):
        return "n/a"
    return f"{val:.{ndigits}f}"


def _fmt_int(val: int | None) -> str:
    return "n/a" if val is None else str(val)


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _fmt_duration(duration_sec: float | None) -> str:
    if duration_sec is None or duration_sec < 0:
        return "n/a"
    total = int(round(duration_sec))
    hours, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours:d}:{mins:02d}:{secs:02d}"


def _duration_hours(duration_sec: float | None) -> float | None:
    if duration_sec is None or duration_sec < 0:
        return None
    return duration_sec / 3600.0


def _md_escape(text: str) -> str:
    return text.replace("|", r"\|")


def _to_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _md_relpath(path: Path, report_dir: Path) -> str:
    """Return a POSIX relative path suitable for Markdown links."""
    rel = os.path.relpath(str(path), start=str(report_dir))
    return rel.replace(os.sep, "/")


def _resolve_artifact_path(path_value: str, run_dir: Path) -> Path:
    """
    Resolve artifact paths written by local or containerized evaluators.
    Container paths under /outputs are mapped back to <cwd>/outputs when present.
    """
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        return (run_dir / p).resolve()
    if p.exists():
        return p
    try:
        rel_from_workspace = p.relative_to("/workspace")
    except ValueError:
        rel_from_workspace = None
    if rel_from_workspace is not None:
        candidate = (Path.cwd() / rel_from_workspace).resolve()
        if candidate.exists():
            return candidate
    try:
        rel_from_outputs = p.relative_to("/outputs")
    except ValueError:
        return p
    candidate = (Path.cwd() / "outputs" / rel_from_outputs).resolve()
    return candidate if candidate.exists() else p


def _find_orbit_gif(
    run_name: str,
    visualizations_dir: Path,
    warnings: list[str],
) -> Path | None:
    if not visualizations_dir.exists():
        return None
    matches = sorted(visualizations_dir.glob(f"{run_name}*.gif"))
    if not matches:
        return None
    if len(matches) > 1:
        warnings.append(
            f"Multiple orbit GIFs found under {visualizations_dir}; using {matches[0].name}."
        )
    return matches[0]


def _canonical_eval_png(run_name: str, visualizations_dir: Path | None) -> Path | None:
    if visualizations_dir is None:
        return None
    return visualizations_dir / f"{run_name}_eval_visualization.png"


def _load_evaluation_summary(run_dir: Path, warnings: list[str]) -> dict[str, Any] | None:
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.exists():
        return None
    try:
        data = _load_json(summary_path)
    except Exception as exc:
        warnings.append(
            "Failed to parse evaluation_summary.json: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    if data is None:
        warnings.append("evaluation_summary.json is missing or not a JSON object.")
        return None
    return data


# ---------------------------------------------------------------------------
# Config / metadata loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _load_yaml_optional(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore
    except Exception:
        warnings.append("PyYAML not available; config.yaml fallback skipped.")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else None


def _load_run_config(run_dir: Path, warnings: list[str]) -> tuple[dict[str, Any], str, str, str]:
    metadata = _load_json(run_dir / "metadata.json")
    metadata_cfg = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(metadata_cfg, dict):
        metadata_cfg = None

    yaml_cfg = _load_yaml_optional(run_dir / "config.yaml", warnings)

    if metadata_cfg is not None:
        cfg = metadata_cfg
        source = "metadata.json"
    elif yaml_cfg is not None:
        cfg = yaml_cfg
        source = "config.yaml"
    else:
        cfg = {}
        source = "none"

    if metadata_cfg is not None and yaml_cfg is not None:
        differing = [
            k for k in sorted(set(metadata_cfg.keys()) | set(yaml_cfg.keys()))
            if metadata_cfg.get(k) != yaml_cfg.get(k)
        ]
        if differing:
            preview = ", ".join(differing[:8])
            more = "" if len(differing) <= 8 else f" (+{len(differing) - 8} more)"
            warnings.append(
                "metadata.json config differs from config.yaml; using metadata.json "
                f"(mismatched keys: {preview}{more})."
            )

    exp = str(
        (metadata or {}).get("experiment_name")
        or cfg.get("experiment_name")
        or "unknown"
    )
    git_commit = str((metadata or {}).get("git_commit", "unknown"))

    return cfg, source, exp, git_commit


# ---------------------------------------------------------------------------
# TensorBoard parsing
# ---------------------------------------------------------------------------

def _parse_scalar_events(events: list[Any]) -> dict[int, tuple[float, float]]:
    """
    Convert tensorboard ScalarEvent list into:
        step -> (value, wall_time)
    Duplicate steps keep the newest wall_time.
    """
    step_map: dict[int, tuple[float, float]] = {}
    for ev in events:
        step = int(ev.step)
        wall = float(ev.wall_time)
        value = float(ev.value)

        prev = step_map.get(step)
        if prev is None or wall >= prev[1]:
            step_map[step] = (value, wall)
    return step_map


def _parse_tensor_events(events: list[Any], warnings: list[str]) -> dict[int, tuple[float, float]]:
    """
    Convert tensorboard TensorEvent list into:
        step -> (value, wall_time)
    Only scalar-like tensors are accepted.
    """
    try:
        from tensorboard.util import tensor_util  # type: ignore
    except Exception:
        warnings.append("TensorBoard tensor decoding unavailable (tensor_util import failed).")
        return {}

    step_map: dict[int, tuple[float, float]] = {}
    for ev in events:
        step = int(ev.step)
        wall = float(ev.wall_time)
        arr = tensor_util.make_ndarray(ev.tensor_proto)
        try:
            value = float(arr.reshape(-1)[0])
        except Exception:
            continue
        if not math.isfinite(value):
            continue

        prev = step_map.get(step)
        if prev is None or wall >= prev[1]:
            step_map[step] = (value, wall)
    return step_map


def _load_tb_series(tb_dir: Path, warnings: list[str]) -> dict[str, dict[int, tuple[float, float]]]:
    """
    Return:
        tag -> { step -> (value, wall_time) }
    """
    if not tb_dir.exists():
        warnings.append("Missing tensorboard directory.")
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import (  # type: ignore
            EventAccumulator,
        )
    except Exception:
        warnings.append(
            "TensorBoard package not available; cannot parse validation metrics "
            "(install `tensorboard` or run inside trainer environment)."
        )
        return {}

    try:
        ea = EventAccumulator(
            str(tb_dir),
            size_guidance={"scalars": 0, "tensors": 0},
        )
        ea.Reload()
    except Exception as exc:
        warnings.append(f"TensorBoard event parsing failed: {type(exc).__name__}: {exc}")
        return {}

    tags = ea.Tags()
    scalar_tags = set(tags.get("scalars", []))
    tensor_tags = set(tags.get("tensors", []))
    all_tags = scalar_tags | tensor_tags

    series: dict[str, dict[int, tuple[float, float]]] = {}
    for tag in sorted(all_tags):
        step_map: dict[int, tuple[float, float]] = {}
        if tag in scalar_tags:
            try:
                step_map = _parse_scalar_events(ea.Scalars(tag))
            except Exception as exc:
                warnings.append(f"Failed reading scalar tag '{tag}': {type(exc).__name__}: {exc}")
                step_map = {}
        elif tag in tensor_tags:
            try:
                step_map = _parse_tensor_events(ea.Tensors(tag), warnings)
            except Exception as exc:
                warnings.append(f"Failed reading tensor tag '{tag}': {type(exc).__name__}: {exc}")
                step_map = {}

        if step_map:
            series[tag] = step_map

    return series


def _collect_wall_times(series: dict[str, dict[int, tuple[float, float]]]) -> list[float]:
    out: list[float] = []
    for tag_map in series.values():
        for _, (_, wall_time) in tag_map.items():
            out.append(wall_time)
    return out


def _value_at_step(
    series: dict[str, dict[int, tuple[float, float]]],
    tag: str,
    step: int,
) -> float | None:
    tag_map = series.get(tag, {})
    pair = tag_map.get(step)
    if pair is None:
        return None
    val = pair[0]
    return val if math.isfinite(val) else None


# ---------------------------------------------------------------------------
# Run metric derivation
# ---------------------------------------------------------------------------

def _max_epoch_from_checkpoints(ckpt_dir: Path) -> int | None:
    if not ckpt_dir.exists():
        return None
    max_epoch: int | None = None
    for f in ckpt_dir.iterdir():
        m = EPOCH_FILE_RE.match(f.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if max_epoch is None or epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def _derive_stopped_epoch(
    series: dict[str, dict[int, tuple[float, float]]],
    ckpt_dir: Path,
) -> int | None:
    candidates: list[int] = []
    train_loss = series.get(TRAIN_TAGS["loss"], {})
    if train_loss:
        candidates.append(max(train_loss.keys()))

    val_candidates: list[int] = []
    for tag in VAL_TAGS.values():
        tag_map = series.get(tag, {})
        if tag_map:
            val_candidates.append(max(tag_map.keys()))
    if val_candidates:
        candidates.append(max(val_candidates))

    ckpt_epoch = _max_epoch_from_checkpoints(ckpt_dir)
    if ckpt_epoch is not None:
        candidates.append(ckpt_epoch)

    if not candidates:
        return None
    return max(candidates)


def _best_of_tag(
    series: dict[str, dict[int, tuple[float, float]]],
    tag: str,
    mode: str = "max",
) -> tuple[float | None, int | None]:
    tag_map = series.get(tag, {})
    best_val: float | None = None
    best_epoch: int | None = None

    for epoch, (value, _) in tag_map.items():
        if not math.isfinite(value):
            continue
        if best_val is None:
            best_val = value
            best_epoch = epoch
            continue
        if mode == "max" and value > best_val:
            best_val = value
            best_epoch = epoch
        elif mode == "min" and value < best_val:
            best_val = value
            best_epoch = epoch

    return best_val, best_epoch


def _best_epoch_by_composite(
    series: dict[str, dict[int, tuple[float, float]]],
    min_delta: float,
) -> tuple[int | None, float | None]:
    """
    Reconstruct best checkpoint update logic from train.py:
      update when score > best + min_delta
    with initial best = 0.0
    """
    tag = VAL_TAGS["composite_score"]
    tag_map = series.get(tag, {})
    if not tag_map:
        return None, None

    best = 0.0
    best_epoch: int | None = None
    best_score: float | None = None
    for epoch in sorted(tag_map):
        score = tag_map[epoch][0]
        if not math.isfinite(score):
            continue
        if score > best + min_delta:
            best = score
            best_epoch = epoch
            best_score = score

    if best_epoch is not None:
        return best_epoch, best_score

    # Fallback if no score surpassed 0.0 + min_delta:
    return _best_of_tag(series, tag, mode="max")[1], _best_of_tag(series, tag, mode="max")[0]


# ---------------------------------------------------------------------------
# Model parameter counting
# ---------------------------------------------------------------------------

def _count_model_params(cfg: dict[str, Any], warnings: list[str]) -> tuple[int | None, int | None]:
    if not cfg:
        return None, None

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from src.models import build_model  # type: ignore
    except Exception as exc:
        warnings.append(f"Could not import model factory for parameter counting: {type(exc).__name__}: {exc}")
        return None, None

    try:
        model = build_model(cfg)
    except Exception as exc:
        warnings.append(f"Could not instantiate model for parameter counting: {type(exc).__name__}: {exc}")
        return None, None

    total = 0
    trainable = 0
    try:
        for p in model.parameters():
            n = int(getattr(p, "numel")())
            total += n
            if bool(getattr(p, "requires_grad", False)):
                trainable += n
    except Exception as exc:
        warnings.append(f"Parameter counting failed: {type(exc).__name__}: {exc}")
        return None, None

    return total, trainable


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _run_start_fallback_from_name(run_name: str) -> float | None:
    """
    Parse `YYYYMMDD_HHMMSS_*` into local timestamp.
    """
    parts = run_name.split("_", 2)
    if len(parts) < 3:
        return None
    dt_str = f"{parts[0]}_{parts[1]}"
    try:
        dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    return dt.timestamp()


def _artifact_end_fallback(run_dir: Path) -> float | None:
    """
    Best-effort training end timestamp without TensorBoard parsing.

    Restrict to core training artifacts to avoid inflation from later files
    (for example manual evaluation images saved in the run directory).
    """
    candidates: list[Path] = []

    tb_dir = run_dir / "tensorboard"
    if tb_dir.exists():
        candidates.extend(sorted(tb_dir.glob("events.out.tfevents*")))

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        candidates.extend(sorted(ckpt_dir.glob("epoch_*.pt")))
        best_pt = ckpt_dir / "best.pt"
        if best_pt.exists():
            candidates.append(best_pt)

    for name in ("metadata.json", "config.yaml", "train_val_split_manifest.json"):
        p = run_dir / name
        if p.exists():
            candidates.append(p)

    latest: float | None = None
    for p in candidates:
        if not p.is_file():
            continue
        ts = p.stat().st_mtime
        if latest is None or ts > latest:
            latest = ts
    return latest


def _build_report_for_run(
    run_dir: Path,
    visualizations_dir: Path | None,
) -> RunReport:
    report = RunReport(run_dir=run_dir, run_name=run_dir.name)
    cfg, cfg_source, exp, git_commit = _load_run_config(run_dir, report.warnings)
    report.config = cfg
    report.config_source = cfg_source
    report.experiment_name = exp
    report.git_commit = git_commit
    report.configured_epochs = _to_int(cfg.get("epochs"))

    tb_series = _load_tb_series(run_dir / "tensorboard", report.warnings)

    wall_times = _collect_wall_times(tb_series)
    if wall_times:
        report.start_ts = min(wall_times)
        report.end_ts = max(wall_times)
    else:
        report.start_ts = _run_start_fallback_from_name(report.run_name)
        report.end_ts = _artifact_end_fallback(run_dir)
        report.warnings.append(
            "No TensorBoard wall-time series found; duration derived from filesystem timestamps."
        )
    if report.start_ts is not None and report.end_ts is not None:
        report.duration_sec = max(0.0, report.end_ts - report.start_ts)

    ckpt_dir = run_dir / "checkpoints"
    report.stopped_epoch = _derive_stopped_epoch(tb_series, ckpt_dir)

    min_delta = _to_float(cfg.get("early_stopping_min_delta"))
    min_delta = min_delta if min_delta is not None else 0.001
    best_epoch, best_comp = _best_epoch_by_composite(tb_series, min_delta=min_delta)
    report.best_epoch = best_epoch
    report.best_composite_score = best_comp

    report.best_dice, _ = _best_of_tag(tb_series, VAL_TAGS["dice"], mode="max")
    report.best_iou, _ = _best_of_tag(tb_series, VAL_TAGS["iou"], mode="max")
    report.best_sensitivity, _ = _best_of_tag(tb_series, VAL_TAGS["sensitivity"], mode="max")
    report.best_precision, _ = _best_of_tag(tb_series, VAL_TAGS["precision"], mode="max")
    report.best_hd95, _ = _best_of_tag(tb_series, VAL_TAGS["hd95"], mode="min")

    # Best-epoch metric snapshot
    if report.best_epoch is not None:
        for key, tag in VAL_TAGS.items():
            report.best_epoch_val[key] = _value_at_step(tb_series, tag, report.best_epoch)

    # Last validation epoch snapshot
    val_epochs: list[int] = []
    for tag in VAL_TAGS.values():
        val_epochs.extend(tb_series.get(tag, {}).keys())
    if val_epochs:
        report.last_val_epoch = max(val_epochs)
        for key, tag in VAL_TAGS.items():
            report.last_val[key] = _value_at_step(tb_series, tag, report.last_val_epoch)
    else:
        report.last_val_epoch = None
        for key in VAL_TAGS:
            report.last_val[key] = None

    if report.configured_epochs is not None and report.stopped_epoch is not None:
        report.early_stopped = report.stopped_epoch < report.configured_epochs

    n_total, n_trainable = _count_model_params(cfg, report.warnings)
    report.n_params_total = n_total
    report.n_params_trainable = n_trainable

    if visualizations_dir is not None:
        report.orbit_gif_path = _find_orbit_gif(
            run_name=report.run_name,
            visualizations_dir=visualizations_dir,
            warnings=report.warnings,
        )

    eval_summary = _load_evaluation_summary(run_dir, report.warnings)
    if isinstance(eval_summary, dict):
        report.eval_summary_path = run_dir / "evaluation_summary.json"
        ckpt = eval_summary.get("checkpoint")
        if isinstance(ckpt, dict):
            report.eval_checkpoint_name = str(ckpt.get("name", "")) or None

        dataset = eval_summary.get("dataset")
        if isinstance(dataset, dict):
            report.eval_total_cases = _to_int(dataset.get("total_cases"))
            report.eval_positive_cases = _to_int(dataset.get("positive_cases"))
            report.eval_negative_cases = _to_int(dataset.get("negative_cases"))

        aggs = eval_summary.get("aggregate_metrics")
        if isinstance(aggs, dict):
            for raw_key, key in EVAL_METRIC_KEYS:
                report.eval_metrics[key] = _to_float(aggs.get(raw_key))

        artifacts = eval_summary.get("artifacts")
        if isinstance(artifacts, dict):
            vis = artifacts.get("eval_visualization_png")
            if isinstance(vis, str) and vis:
                resolved = _resolve_artifact_path(vis, run_dir)
                if resolved.exists():
                    report.eval_visualization_path = resolved
                else:
                    report.warnings.append(
                        "evaluation_summary.json references missing eval PNG: "
                        f"{resolved}"
                    )

    canonical_eval_png = _canonical_eval_png(report.run_name, visualizations_dir)
    if canonical_eval_png is not None and canonical_eval_png.exists():
        if (
            report.eval_visualization_path is not None
            and report.eval_visualization_path.resolve() != canonical_eval_png.resolve()
        ):
            report.warnings.append(
                "Using centralized eval PNG from visualizations directory: "
                f"{canonical_eval_png.name}"
            )
        report.eval_visualization_path = canonical_eval_png

    fallback_eval_png = run_dir / "eval_visualization.png"
    if report.eval_visualization_path is None and fallback_eval_png.exists():
        report.eval_visualization_path = fallback_eval_png

    return report


def _find_runs(base_dir: Path, run_args: list[str]) -> list[Path]:
    if run_args:
        paths: list[Path] = []
        for raw in run_args:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Run path not found or not a directory: {p}")
            paths.append(p)
        return sorted(paths, key=lambda p: p.name, reverse=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")

    runs = [
        d for d in base_dir.iterdir()
        if d.is_dir()
        and d.name != "latest"
        and RUN_NAME_RE.match(d.name)
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _sort_reports(reports: list[RunReport], sort_by: str) -> list[RunReport]:
    def key(report: RunReport) -> tuple[int, float]:
        if sort_by == "best_composite_score":
            val = report.best_composite_score
        elif sort_by == "best_dice":
            val = report.best_dice
        elif sort_by == "duration_hours":
            val = _duration_hours(report.duration_sec)
        elif sort_by == "start_time":
            val = report.start_ts
        else:
            val = report.best_composite_score

        if val is None:
            return (1, 0.0)  # always last
        return (0, -float(val))  # descending for non-missing values

    return sorted(reports, key=key)


def _comparison_row(report: RunReport) -> list[str]:
    cfg = report.config
    patch = cfg.get("patch_size")
    patch_s = (
        str(tuple(patch))
        if isinstance(patch, (list, tuple)) and len(patch) == 3
        else str(patch) if patch is not None else "n/a"
    )
    model = str(cfg.get("model", "n/a"))
    loss_fn = str(cfg.get("loss_fn", "n/a"))

    return [
        report.run_name,
        _fmt_duration(report.duration_sec),
        _fmt_int(report.stopped_epoch),
        _fmt_int(report.best_epoch),
        _fmt_float(report.best_composite_score),
        _fmt_float(report.best_dice),
        _fmt_float(report.best_iou),
        _fmt_float(report.best_sensitivity),
        _fmt_float(report.best_precision),
        _fmt_float(report.best_hd95),
        model,
        loss_fn,
        patch_s,
    ]


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out: list[str] = []
    out.append("| " + " | ".join(_md_escape(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
    return "\n".join(out)


def _config_highlights(cfg: dict[str, Any], report: RunReport) -> list[str]:
    lines: list[str] = []

    model = cfg.get("model", "n/a")
    lines.append(f"- model: `{model}`")

    features = cfg.get("features", "n/a")
    lines.append(f"- features: `{features}`")

    loss_fn = cfg.get("loss_fn", "n/a")
    if loss_fn == "tversky_bce":
        a = cfg.get("tversky_alpha", "n/a")
        b = cfg.get("tversky_beta", "n/a")
        dw = cfg.get("dice_weight", "n/a")
        bw = cfg.get("bce_weight", "n/a")
        pw = cfg.get("bce_pos_weight", "n/a")
        lines.append(
            f"- loss: `{loss_fn}` (alpha={a}, beta={b}, dice_weight={dw}, bce_weight={bw}, bce_pos_weight={pw})"
        )
    elif loss_fn == "dice_bce":
        dw = cfg.get("dice_weight", "n/a")
        bw = cfg.get("bce_weight", "n/a")
        pw = cfg.get("bce_pos_weight", "n/a")
        lines.append(f"- loss: `{loss_fn}` (dice_weight={dw}, bce_weight={bw}, bce_pos_weight={pw})")
    else:
        lines.append(f"- loss: `{loss_fn}`")

    lines.append(f"- patch_size: `{cfg.get('patch_size', 'n/a')}`")
    lines.append(f"- target_spacing: `{cfg.get('target_spacing', 'n/a')}`")
    lines.append(
        "- modalities: "
        f"use_t2w={cfg.get('use_t2w', 'n/a')}, "
        f"use_adc={cfg.get('use_adc', 'n/a')}, "
        f"use_hbv={cfg.get('use_hbv', 'n/a')}"
    )
    lines.append(
        "- optimizer/schedule: "
        f"lr={cfg.get('learning_rate', 'n/a')}, "
        f"weight_decay={cfg.get('weight_decay', 'n/a')}, "
        f"warmup_epochs={cfg.get('warmup_epochs', 'n/a')}"
    )
    lines.append(
        "- train/val cadence: "
        f"batch_size={cfg.get('batch_size', 'n/a')}, "
        f"epochs={cfg.get('epochs', 'n/a')}, "
        f"val_every={cfg.get('val_every', 'n/a')}, "
        f"val_start_epoch={cfg.get('val_start_epoch', 'n/a')}, "
        f"val_compute_hd95_every={cfg.get('val_compute_hd95_every', 'n/a')}"
    )
    lines.append(
        "- best-checkpoint score weights: "
        f"w_sensitivity={cfg.get('best_ckpt_w_sensitivity', 'n/a')}, "
        f"w_dice={cfg.get('best_ckpt_w_dice', 'n/a')}, "
        f"w_hd95={cfg.get('best_ckpt_w_hd95', 'n/a')}, "
        f"hd95_scale={cfg.get('best_ckpt_hd95_scale', 'n/a')}"
    )
    lines.append(
        "- early stopping: "
        f"patience={cfg.get('early_stopping_patience', 'n/a')}, "
        f"min_delta={cfg.get('early_stopping_min_delta', 'n/a')}"
    )
    lines.append(
        "- runtime: "
        f"use_amp={cfg.get('use_amp', 'n/a')}, "
        f"amp_dtype={cfg.get('amp_dtype', 'n/a')}, "
        f"use_compile={cfg.get('use_compile', 'n/a')}"
    )
    lines.append(
        "- encoder init: "
        f"pretrained_encoder_checkpoint={cfg.get('pretrained_encoder_checkpoint', 'n/a')}, "
        f"freeze_encoder_epochs={cfg.get('freeze_encoder_epochs', 'n/a')}"
    )

    if report.n_params_total is not None:
        lines.append(
            f"- parameters: total={report.n_params_total:,}, trainable={report.n_params_trainable:,}"
        )
    else:
        lines.append("- parameters: n/a")

    return lines


def _run_section(report: RunReport, report_dir: Path) -> str:
    lines: list[str] = []

    lines.append(f"## {report.run_name}")
    lines.append("")
    lines.append(f"- run_dir: `{report.run_dir}`")
    lines.append(f"- experiment_name: `{report.experiment_name}`")
    lines.append(f"- git_commit: `{report.git_commit}`")
    lines.append(f"- config_source: `{report.config_source}`")
    lines.append(f"- start_time: `{_fmt_ts(report.start_ts)}`")
    lines.append(f"- end_time: `{_fmt_ts(report.end_ts)}`")
    lines.append(f"- duration: `{_fmt_duration(report.duration_sec)}`")
    lines.append(
        "- epochs: "
        f"stopped={_fmt_int(report.stopped_epoch)}, "
        f"configured={_fmt_int(report.configured_epochs)}, "
        f"best={_fmt_int(report.best_epoch)}"
    )
    if report.early_stopped is not None:
        lines.append(f"- early_stopped: `{report.early_stopped}`")
    lines.append("")

    lines.append("### Visualization")
    if report.orbit_gif_path is not None and report.orbit_gif_path.exists():
        orbit_rel = _md_relpath(report.orbit_gif_path, report_dir)
        lines.append("Loading orbit GIF (large file, may take a moment)...")
        lines.append(
            f'<img src="{orbit_rel}" alt="{report.run_name} orbit" '
            'loading="lazy" decoding="async">'
        )
    else:
        lines.append("- orbit_gif: `n/a`")
    lines.append("")
    if report.eval_visualization_path is not None and report.eval_visualization_path.exists():
        eval_rel = _md_relpath(report.eval_visualization_path, report_dir)
        lines.append("Loading evaluation image...")
        lines.append(
            f'<img src="{eval_rel}" alt="{report.run_name} evaluation" '
            'loading="lazy" decoding="async">'
        )
    else:
        lines.append("- eval_visualization_png: `n/a`")
    lines.append("")

    lines.append("### Evaluation")
    if report.eval_metrics:
        if report.eval_checkpoint_name:
            lines.append(f"- checkpoint: `{report.eval_checkpoint_name}`")
        if (
            report.eval_total_cases is not None
            and report.eval_positive_cases is not None
            and report.eval_negative_cases is not None
        ):
            lines.append(
                "- test_cases: "
                f"`{report.eval_total_cases}` "
                f"(`{report.eval_positive_cases}` positive, `{report.eval_negative_cases}` negative)"
            )
        eval_headers = ["Metric", "Value"]
        eval_rows: list[list[str]] = []
        for key in ("dice", "iou", "sensitivity", "precision", "hd95"):
            eval_rows.append([key, _fmt_float(report.eval_metrics.get(key))])
        # Keep a blank line before tables so GitHub renders them outside lists.
        lines.append("")
        lines.append(_markdown_table(eval_headers, eval_rows))
    else:
        lines.append("- aggregate_test_metrics: `n/a`")
    lines.append("")

    lines.append("### Training Validation Metrics")
    headers = ["Metric", "Best", "At Best Epoch", "Last Val"]
    metric_rows: list[list[str]] = []
    for key in ("composite_score", "dice", "iou", "sensitivity", "precision", "hd95"):
        if key == "composite_score":
            best_val = report.best_composite_score
        elif key == "dice":
            best_val = report.best_dice
        elif key == "iou":
            best_val = report.best_iou
        elif key == "sensitivity":
            best_val = report.best_sensitivity
        elif key == "precision":
            best_val = report.best_precision
        else:
            best_val = report.best_hd95

        at_best = report.best_epoch_val.get(key)
        last_v = report.last_val.get(key)
        metric_rows.append(
            [
                key,
                _fmt_float(best_val),
                _fmt_float(at_best),
                _fmt_float(last_v),
            ]
        )
    lines.append("")
    lines.append(_markdown_table(headers, metric_rows))
    lines.append("")
    lines.append("### Config Highlights")
    lines.extend(_config_highlights(report.config, report))
    lines.append("")

    if report.warnings:
        lines.append("### Warnings")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _build_markdown_report(
    reports: list[RunReport],
    base_dir: Path,
    sort_by: str,
    report_dir: Path,
) -> str:
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# Training Run Report")
    lines.append("")
    lines.append(f"- generated_at: `{now}`")
    lines.append(f"- base_dir: `{base_dir}`")
    lines.append(f"- runs: `{len(reports)}`")
    lines.append(f"- sort_by: `{sort_by}`")
    lines.append("")

    headers = [
        "run",
        "duration",
        "stopped_epoch",
        "best_epoch",
        "best_composite",
        "best_dice",
        "best_iou",
        "best_sens",
        "best_prec",
        "best_hd95",
        "model",
        "loss",
        "patch_size",
    ]
    rows = [_comparison_row(r) for r in reports]
    lines.append("## Comparison")
    lines.append("")
    lines.append(_markdown_table(headers, rows))
    lines.append("")

    for report in reports:
        lines.append(_run_section(report, report_dir=report_dir))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown report for training runs."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/runs",
        help="Directory containing run folders (default: outputs/runs).",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit run directory. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Write report to file (default: stdout).",
    )
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        default="visualizations",
        help=(
            "Directory containing per-run orbit GIFs named with run-name prefix "
            "(default: visualizations)."
        ),
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default=DEFAULT_SORT_BY,
        choices=SORT_CHOICES,
        help=f"Sort key for comparison rows (default: {DEFAULT_SORT_BY}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of runs after sorting (0 = no limit).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    visualizations_dir = Path(args.visualizations_dir).expanduser()
    if not visualizations_dir.is_absolute():
        visualizations_dir = (Path.cwd() / visualizations_dir).resolve()

    run_dirs = _find_runs(base_dir=base_dir, run_args=args.run)
    if not run_dirs:
        raise RuntimeError(
            "No runs found. Check --base-dir or pass explicit --run path(s)."
        )

    reports = [
        _build_report_for_run(run_dir, visualizations_dir=visualizations_dir)
        for run_dir in run_dirs
    ]
    reports = _sort_reports(reports, sort_by=args.sort_by)

    if args.limit > 0:
        reports = reports[:args.limit]

    if args.output:
        out_path = Path(args.output).expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        report_dir = out_path.parent
    else:
        out_path = None
        report_dir = Path.cwd()

    report_md = _build_markdown_report(
        reports,
        base_dir=base_dir,
        sort_by=args.sort_by,
        report_dir=report_dir,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Report written to: {out_path}")
    else:
        print(report_md)


if __name__ == "__main__":
    main()
