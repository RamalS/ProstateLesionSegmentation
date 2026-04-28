"""
visualize_3d.py - Interactive 3-D visualisation for a single PI-CAI case.

Two modes
---------
1) Image / GT visualisation (no model inference):
   - Load T2w MRI volume and optional segmentation mask
   - If provided, align segmentation into T2w grid (nearest-neighbour)
   - Save rotatable Plotly HTML scene

2) Model prediction visualisation (optional):
   - Load run config + checkpoint
   - Preprocess case exactly like training/evaluation pipeline
   - Run sliding-window inference
   - Visualise GT-only / Pred-only / Overlap (or prediction-only when GT absent)
   - Print per-case metrics when GT is available

Usage
-----
Local (auto label lookup):
    PYTHONPATH=. python scripts/visualize_3d.py \
        --t2w data/test_images/10028_1000771_t2w.mha

With explicit label path:
    PYTHONPATH=. python scripts/visualize_3d.py \
        --t2w data/test_images/10028_1000771_t2w.mha \
        --seg data/labels/10028_1000771.nii.gz

With model comparison (checkpoint defaults to best.pt):
    PYTHONPATH=. python scripts/visualize_3d.py \
        --t2w data/test_images/10028_1000771_t2w.mha \
        --run outputs/runs/20260420_123456_baseline_run

With orbit GIF export (writes HTML and GIF):
    PYTHONPATH=. python scripts/visualize_3d.py \
        --t2w data/test_images/10028_1000771_t2w.mha \
        --gif
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from io import BytesIO
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference

# ---------------------------------------------------------------------------
# Resolve src/ imports (scripts/ are not on PYTHONPATH by default)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from dataset import PiCaiDataset, active_modality_pairs  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
from utils import load_checkpoint  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: image I/O + preprocessing
# ---------------------------------------------------------------------------

def _to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image -> float32 numpy array shaped (D, H, W)."""
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _normalize_for_display(vol: np.ndarray) -> np.ndarray:
    """Percentile clip [p1, p99] then scale to [0, 1] for rendering."""
    p1 = float(np.percentile(vol, 1))
    p99 = float(np.percentile(vol, 99))
    clipped = np.clip(vol, p1, p99)
    return ((clipped - p1) / max(p99 - p1, 1e-8)).astype(np.float32)


def _resample(
    image: sitk.Image,
    target_spacing_zyx: tuple[float, float, float],
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample *image* to target spacing in (z, y, x) mm order.

    SimpleITK internally uses (x, y, z); this helper accepts the (z, y, x)
    convention used in this repository.
    """
    orig_spacing_xyz = np.array(image.GetSpacing(), dtype=np.float64)
    orig_size_xyz = np.array(image.GetSize(), dtype=np.int64)
    target_spacing_xyz = np.array(target_spacing_zyx[::-1], dtype=np.float64)

    new_size_xyz = np.round(orig_size_xyz * orig_spacing_xyz / target_spacing_xyz)
    new_size_xyz = np.maximum(new_size_xyz, 1).astype(int)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz.tolist())
    resampler.SetSize(new_size_xyz.tolist())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def _resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample *moving* image into physical grid of *reference*."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def _fmt(v: float) -> str:
    """Format metric value to 4 d.p.; return 'n/a' for NaN."""
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _case_id_from_t2w(t2w_path: Path) -> str:
    """Infer case id from a T2w filename."""
    if t2w_path.name.endswith("_t2w.mha"):
        return t2w_path.name[: -len("_t2w.mha")]
    return t2w_path.stem


def _safe_filename_component(raw: str) -> str:
    """Convert arbitrary label to filesystem-safe token."""
    token = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in raw)
    token = token.strip("_")
    return token or "item"


def _default_export_stem(t2w_path: Path, run_dir_arg: str | None) -> str:
    """
    Build default filename stem.

    Uses image name (T2w stem) and prepends model run name when --run is set.
    """
    image_name = _safe_filename_component(t2w_path.stem)
    if run_dir_arg is None:
        return image_name

    run_name = _safe_filename_component(Path(run_dir_arg).name)
    if run_name:
        return f"{run_name}_{image_name}"
    return image_name


def _resolve_seg_path(t2w_path: Path, seg_arg: str | None) -> Path:
    """
    Resolve segmentation path.

    - If --seg provided: use it.
    - Else auto-detect from case id in common labels locations.
    """
    if seg_arg is not None:
        seg_path = Path(seg_arg).resolve()
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
        return seg_path

    case_id = _case_id_from_t2w(t2w_path)
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        t2w_path.parent.parent / "labels" / f"{case_id}.nii.gz",
        repo_root / "data" / "labels" / f"{case_id}.nii.gz",
        Path("data") / "labels" / f"{case_id}.nii.gz",
    ]

    checked: list[Path] = []
    for candidate in candidates:
        c = candidate.resolve()
        checked.append(c)
        if c.exists():
            logger.info("Auto-detected segmentation: %s", c)
            return c

    tried = "\n  - " + "\n  - ".join(str(p) for p in checked)
    raise FileNotFoundError(
        "--seg not provided and auto-detection failed for case "
        f"'{case_id}'. Tried:{tried}\n"
        "Provide explicit label path with --seg <path>."
    )


def resolve_run_dir(run_dir_arg: str | Path) -> Path:
    """Resolve and validate a run directory path."""
    run_dir = Path(run_dir_arg).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found in run directory: {cfg_path}")
    return run_dir


def list_available_checkpoints(run_dir: Path) -> list[Path]:
    """Return run checkpoints with best.pt first, then newest epoch filenames."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"checkpoints/ directory not found: {ckpt_dir}")

    all_pts = [p.resolve() for p in ckpt_dir.glob("*.pt") if p.is_file()]
    if not all_pts:
        raise FileNotFoundError(f"No .pt checkpoints found in {ckpt_dir}")

    best = [p for p in all_pts if p.name == "best.pt"]
    others = sorted(
        [p for p in all_pts if p.name != "best.pt"],
        key=lambda p: p.name,
        reverse=True,
    )
    return best + others


def _resolve_checkpoint_path(run_dir: Path, checkpoint_arg: str | None) -> Path:
    """Resolve checkpoint path from --run and optional --checkpoint argument."""
    ckpt_paths = list_available_checkpoints(run_dir)
    ckpt_dir = run_dir / "checkpoints"

    if checkpoint_arg:
        raw = Path(checkpoint_arg)
        if raw.is_file():
            return raw.resolve()

        candidate = ckpt_dir / checkpoint_arg
        if candidate.is_file():
            return candidate.resolve()

        raise FileNotFoundError(
            f"Checkpoint not found: '{checkpoint_arg}'. "
            f"Checked as absolute/relative path and under {ckpt_dir}."
        )

    return ckpt_paths[0]


def _maybe_autodetect_modality(
    modality_key: str,
    explicit: str | None,
    t2w_path: Path,
    case_id: str,
) -> Path | None:
    """
    Resolve modality path.

    Priority:
    1) explicit CLI path (--adc / --hbv)
    2) sibling file inferred from --t2w path (<case_id>_<modality>.mha)
    """
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"{modality_key.upper()} file not found: {p}")
        return p

    candidate = t2w_path.with_name(f"{case_id}_{modality_key}.mha")
    if candidate.exists():
        return candidate.resolve()
    return None


def _load_native_t2w_and_optional_gt(
    t2w_path: Path,
    seg_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], bool]:
    """
    Load native-space T2w and optional GT mask for visualisation mode.

    When *seg_path* is ``None``, returns an all-zero GT mask and
    ``has_ground_truth=False``.
    """
    t2w_img = sitk.ReadImage(str(t2w_path))
    t2w_np = _normalize_for_display(_to_numpy(t2w_img))

    has_ground_truth = seg_path is not None
    if has_ground_truth:
        seg_img = sitk.ReadImage(str(seg_path))
        seg_on_t2w = _resample_to_reference(
            moving=seg_img,
            reference=t2w_img,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
        )
        gt_np = (_to_numpy(seg_on_t2w) > 0).astype(np.uint8)
    else:
        gt_np = np.zeros(t2w_np.shape, dtype=np.uint8)

    sx, sy, sz = t2w_img.GetSpacing()  # SITK: (x, y, z)
    spacing_zyx = (float(sz), float(sy), float(sx))
    return t2w_np, gt_np, spacing_zyx, has_ground_truth


def _load_native_t2w_and_gt(
    t2w_path: Path,
    seg_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Backwards-compatible wrapper for GT-available native-space loading."""
    t2w_np, gt_np, spacing_zyx, _ = _load_native_t2w_and_optional_gt(t2w_path, seg_path)
    return t2w_np, gt_np, spacing_zyx


def _load_model_inputs(
    cfg: dict,
    t2w_path: Path,
    seg_path: Path | None,
    adc_path: str | None,
    hbv_path: str | None,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, tuple[float, float, float], dict[str, Path]]:
    """
    Build single-case tensors using exact training/eval preprocessing pipeline.

    Returns
    -------
    image_t      : (C, D, H, W) model input tensor
    label_t      : (1, D, H, W) GT tensor (binary). All-zero when GT unavailable.
    t2w_display  : (D, H, W) normalized T2w volume in model grid
    spacing_zyx  : voxel spacing of returned arrays
    used_paths   : modality file paths used for model input
    """
    case_id = _case_id_from_t2w(t2w_path)
    active_pairs = active_modality_pairs(cfg)
    active_keys = [k for k, _ in active_pairs]

    used_paths: dict[str, Path] = {"t2w": t2w_path}

    if "adc" in active_keys:
        adc = _maybe_autodetect_modality("adc", adc_path, t2w_path, case_id)
        if adc is None:
            raise FileNotFoundError(
                "ADC required by model config (use_adc=true) but no file found. "
                "Pass --adc or place '<case_id>_adc.mha' next to --t2w."
            )
        used_paths["adc"] = adc

    if "hbv" in active_keys:
        hbv = _maybe_autodetect_modality("hbv", hbv_path, t2w_path, case_id)
        if hbv is None:
            raise FileNotFoundError(
                "HBV required by model config (use_hbv=true) but no file found. "
                "Pass --hbv or place '<case_id>_hbv.mha' next to --t2w."
            )
        used_paths["hbv"] = hbv

    # PiCaiDataset always expects "t2w" in the case dict because T2w is used
    # internally as registration + label-grid reference even when use_t2w=false.
    case: dict[str, str | Path | None] = {
        "case_id": case_id,
        "t2w": t2w_path,
        "label": seg_path,
    }
    for key in ("adc", "hbv"):
        if key in used_paths:
            case[key] = used_paths[key]

    target_spacing = tuple(float(v) for v in cfg.get("target_spacing", [3.0, 0.5, 0.5]))

    ds = PiCaiDataset(
        images_dir=t2w_path.parent,
        labels_dir=seg_path.parent if seg_path is not None else t2w_path.parent,
        target_spacing=target_spacing,
        transform=None,
        cases=[case],
        use_cache=False,
        active_modalities=tuple(active_keys),
    )

    sample = ds[0]
    image_t: torch.Tensor = sample["image"]
    label_t: torch.Tensor = sample["label"]

    # Build display T2w volume in exact model grid (same target_spacing).
    t2w_img = sitk.ReadImage(str(t2w_path))
    t2w_resampled = _resample(
        t2w_img,
        target_spacing_zyx=target_spacing,  # type: ignore[arg-type]
        interpolator=sitk.sitkLinear,
    )
    t2w_display = _normalize_for_display(_to_numpy(t2w_resampled))

    spacing_zyx = (
        float(target_spacing[0]),
        float(target_spacing[1]),
        float(target_spacing[2]),
    )
    return image_t, label_t, t2w_display, spacing_zyx, used_paths


# ---------------------------------------------------------------------------
# Helpers: model inference
# ---------------------------------------------------------------------------

def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_inference(
    model: torch.nn.Module,
    image_t: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run sliding-window inference and return raw logits on CPU."""
    patch_size = tuple(int(v) for v in cfg.get("patch_size", [20, 128, 128]))
    sw_overlap = float(cfg.get("sw_overlap", 0.5))
    sw_batch_size = int(cfg.get("sw_batch_size", 4))

    use_amp = bool(cfg.get("use_amp", False)) and device.type == "cuda"
    amp_dtype_key = str(cfg.get("amp_dtype", "bf16")).lower()
    amp_dtype = torch.float16 if amp_dtype_key == "fp16" else torch.bfloat16

    def predictor(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if use_amp
        else nullcontext()
    )

    with torch.no_grad():
        with autocast_ctx:
            logits = sliding_window_inference(
                inputs=image_t.unsqueeze(0).to(device),
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=predictor,
                overlap=sw_overlap,
            )
    return logits.float().cpu()


# ---------------------------------------------------------------------------
# Helpers: Plotly rendering
# ---------------------------------------------------------------------------

def _compute_uniform_stride(shape: tuple[int, int, int], max_voxels: int) -> tuple[int, int, int]:
    """Compute uniform (z, y, x) stride so resulting volume <= max_voxels."""
    n_voxels = int(np.prod(shape))
    if n_voxels <= max_voxels:
        return (1, 1, 1)
    s = int(math.ceil((n_voxels / max_voxels) ** (1.0 / 3.0)))
    s = max(1, s)
    return (s, s, s)


def _downsample_for_render(
    t2w_vol: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[float, float, float], tuple[int, int, int]]:
    """Downsample arrays for rendering performance while preserving alignment."""
    stride_zyx = _compute_uniform_stride(t2w_vol.shape, max_voxels)
    sz, sy, sx = stride_zyx

    if stride_zyx == (1, 1, 1):
        return t2w_vol, gt_mask, pred_mask, spacing_zyx, stride_zyx

    t2w_ds = t2w_vol[::sz, ::sy, ::sx]
    gt_ds = gt_mask[::sz, ::sy, ::sx]
    pred_ds = pred_mask[::sz, ::sy, ::sx] if pred_mask is not None else None
    spacing_ds = (
        spacing_zyx[0] * sz,
        spacing_zyx[1] * sy,
        spacing_zyx[2] * sx,
    )
    return t2w_ds, gt_ds, pred_ds, spacing_ds, stride_zyx


def _grid_coordinates(shape_zyx: tuple[int, int, int], spacing_zyx: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened x/y/z coordinates in millimeters for a 3-D grid."""
    d, h, w = shape_zyx
    z_mm = np.arange(d, dtype=np.float32) * spacing_zyx[0]
    y_mm = np.arange(h, dtype=np.float32) * spacing_zyx[1]
    x_mm = np.arange(w, dtype=np.float32) * spacing_zyx[2]

    zz, yy, xx = np.meshgrid(z_mm, y_mm, x_mm, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


def _make_mask_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    mask: np.ndarray,
    color: str,
    name: str,
    opacity: float,
):
    """Build a Plotly Isosurface trace for a binary mask, or None if empty."""
    if not np.any(mask):
        return None

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for 3-D visualisation. Install with: pip install plotly"
        ) from exc

    values = mask.astype(np.float32).ravel()
    return go.Isosurface(
        x=x,
        y=y,
        z=z,
        value=values,
        isomin=0.5,
        isomax=1.0,
        surface_count=1,
        colorscale=((0.0, color), (1.0, color)),
        showscale=False,
        opacity=opacity,
        name=name,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )


def _build_3d_figure(
    t2w_vol: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    case_id: str,
    has_ground_truth: bool = True,
):
    """Build Plotly 3-D figure with MRI volume + segmentation surfaces."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for 3-D visualisation. Install with: pip install plotly"
        ) from exc

    x, y, z = _grid_coordinates(t2w_vol.shape, spacing_zyx)

    fig = go.Figure()

    # MRI intensity context
    fig.add_trace(
        go.Volume(
            x=x,
            y=y,
            z=z,
            value=t2w_vol.ravel(),
            isomin=0.20,
            isomax=1.00,
            opacity=0.06,
            surface_count=12,
            colorscale="Gray",
            showscale=False,
            name="T2w",
        )
    )

    if pred_mask is None and has_ground_truth:
        gt_trace = _make_mask_trace(
            x=x,
            y=y,
            z=z,
            mask=(gt_mask > 0),
            color="#00FF00",
            name="Ground truth",
            opacity=0.55,
        )
        if gt_trace is not None:
            fig.add_trace(gt_trace)
    elif pred_mask is not None and has_ground_truth:
        gt_bool = gt_mask > 0
        pred_bool = pred_mask > 0
        gt_only = gt_bool & ~pred_bool
        pred_only = pred_bool & ~gt_bool
        overlap = gt_bool & pred_bool

        for trace in (
            _make_mask_trace(x, y, z, gt_only, "#00FF00", "GT only", 0.55),
            _make_mask_trace(x, y, z, pred_only, "#FF0000", "Prediction only", 0.55),
            _make_mask_trace(x, y, z, overlap, "#FFFF00", "Overlap", 0.65),
        ):
            if trace is not None:
                fig.add_trace(trace)
    elif pred_mask is not None:
        pred_trace = _make_mask_trace(
            x=x,
            y=y,
            z=z,
            mask=(pred_mask > 0),
            color="#FF0000",
            name="Prediction",
            opacity=0.55,
        )
        if pred_trace is not None:
            fig.add_trace(pred_trace)

    title = f"{case_id} - 3D lesion view"
    if pred_mask is None and not has_ground_truth:
        title += " (image only)"
    elif pred_mask is not None and has_ground_truth:
        title += " (GT vs prediction)"
    elif pred_mask is not None:
        title += " (prediction only)"

    fig.update_layout(
        title=title,
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font=dict(color="#f0f0f0"),
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(bgcolor="rgba(20,20,20,0.65)"),
        scene=dict(
            aspectmode="data",
            bgcolor="#0f0f0f",
            xaxis=dict(title="x (mm)", showbackground=False),
            yaxis=dict(title="y (mm)", showbackground=False),
            zaxis=dict(title="z (mm)", showbackground=False),
        ),
    )
    return fig


def _prepare_3d_figure_for_export(
    t2w_vol: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    case_id: str,
    max_voxels: int,
    has_ground_truth: bool,
):
    """Build Plotly figure after render-time downsampling."""
    t2w_ds, gt_ds, pred_ds, spacing_ds, stride_zyx = _downsample_for_render(
        t2w_vol=t2w_vol,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        spacing_zyx=spacing_zyx,
        max_voxels=max_voxels,
    )

    if stride_zyx != (1, 1, 1):
        logger.info(
            "Render downsampling applied: stride(z,y,x)=%s -> shape %s",
            stride_zyx,
            t2w_ds.shape,
        )

    fig = _build_3d_figure(
        t2w_vol=t2w_ds,
        gt_mask=gt_ds,
        pred_mask=pred_ds,
        spacing_zyx=spacing_ds,
        case_id=case_id,
        has_ground_truth=has_ground_truth,
    )
    return fig


def save_3d_visualization_html(
    t2w_vol: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    output_path: Path,
    case_id: str,
    max_voxels: int = 250_000,
    has_ground_truth: bool = True,
) -> Path:
    """Render and save standalone interactive Plotly HTML visualisation."""
    fig = _prepare_3d_figure_for_export(
        t2w_vol=t2w_vol,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        spacing_zyx=spacing_zyx,
        case_id=case_id,
        max_voxels=max_voxels,
        has_ground_truth=has_ground_truth,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def _camera_xyz(raw: Any, *, fallback: tuple[float, float, float]) -> dict[str, float]:
    out = {"x": fallback[0], "y": fallback[1], "z": fallback[2]}
    if raw is None:
        return out

    for axis in ("x", "y", "z"):
        if isinstance(raw, dict):
            value = raw.get(axis)
        else:
            value = getattr(raw, axis, None)
        if value is None:
            continue
        try:
            out[axis] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _extract_scene_camera(figure: Any) -> dict[str, dict[str, float]]:
    defaults = {
        "eye": (1.25, 1.25, 1.25),
        "up": (0.0, 0.0, 1.0),
        "center": (0.0, 0.0, 0.0),
    }
    scene = getattr(getattr(figure, "layout", None), "scene", None)
    camera = getattr(scene, "camera", None) if scene is not None else None

    if isinstance(camera, dict):
        eye_raw = camera.get("eye")
        up_raw = camera.get("up")
        center_raw = camera.get("center")
    else:
        eye_raw = getattr(camera, "eye", None)
        up_raw = getattr(camera, "up", None)
        center_raw = getattr(camera, "center", None)

    return {
        "eye": _camera_xyz(eye_raw, fallback=defaults["eye"]),
        "up": _camera_xyz(up_raw, fallback=defaults["up"]),
        "center": _camera_xyz(center_raw, fallback=defaults["center"]),
    }


def _is_missing_kaleido_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return 'requires the "kaleido" engine' in msg or "requires the kaleido package" in msg


def _is_missing_chrome_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "kaleido requires google chrome" in msg or ("kaleido" in msg and "requires chrome" in msg)


def _plotly_chrome_install_dir() -> Path | None:
    # Prefer /cache in Docker to persist Chrome across container runs.
    candidates = [
        Path("/cache/plotly_chrome"),
        Path.home() / ".cache" / "plotly_chrome",
    ]
    for candidate in candidates:
        parent = candidate.parent
        if parent.exists() and os.access(parent, os.W_OK):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    return None


def _chrome_setup_message(*, attempted_auto_install: bool, install_detail: str | None = None) -> str:
    lines = [
        "Kaleido needs a Chrome/Chromium binary for GIF export.",
        "Install Chrome for Plotly with: plotly_get_chrome",
        "You can also call plotly.io.get_chrome() from Python.",
    ]
    if attempted_auto_install:
        lines.append("Automatic Chrome install was attempted but did not complete.")
    if install_detail:
        lines.append(f"Install error: {install_detail}")
    return "\n".join(lines)


def _render_plotly_png_bytes(fig: Any) -> bytes:
    try:
        return fig.to_image(format="png")
    except Exception as exc:  # noqa: BLE001
        if _is_missing_kaleido_error(exc):
            raise RuntimeError(
                "GIF export requires kaleido. Install with: pip install --upgrade kaleido"
            ) from exc
        if not _is_missing_chrome_error(exc):
            raise

    chrome_dir = _plotly_chrome_install_dir()
    try:
        import plotly.io as pio

        if chrome_dir is not None:
            pio.get_chrome(path=chrome_dir)
        else:
            pio.get_chrome()
    except Exception as install_exc:  # noqa: BLE001
        raise RuntimeError(
            _chrome_setup_message(
                attempted_auto_install=True,
                install_detail=str(install_exc).strip() or None,
            )
        ) from install_exc

    try:
        return fig.to_image(format="png")
    except Exception as retry_exc:  # noqa: BLE001
        if _is_missing_chrome_error(retry_exc):
            raise RuntimeError(_chrome_setup_message(attempted_auto_install=True)) from retry_exc
        raise


def _export_orbit_gif_bytes(
    figure: Any,
    *,
    frame_count: int,
    fps: int,
    turns: float,
    width_px: int,
    height_px: int,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> bytes:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for GIF export. Install with: pip install plotly"
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for GIF export. Install with: pip install Pillow"
        ) from exc

    if frame_count < 2:
        raise ValueError("frame_count must be >= 2")
    if fps < 1:
        raise ValueError("fps must be >= 1")
    if turns <= 0:
        raise ValueError("turns must be > 0")
    if width_px < 320:
        raise ValueError("width_px must be >= 320")
    if height_px < 320:
        raise ValueError("height_px must be >= 320")

    base_camera = _extract_scene_camera(figure)
    base_eye = base_camera["eye"]
    radius_xy = float(np.hypot(base_eye["x"], base_eye["y"]))
    if radius_xy < 1e-6:
        radius_xy = 1.25
    start_angle = float(np.arctan2(base_eye["y"], base_eye["x"]))

    fig = go.Figure(figure)
    fig.update_layout(width=int(width_px), height=int(height_px))

    duration_ms = max(10, int(round(1000.0 / float(fps))))
    step = (2.0 * np.pi * float(turns)) / float(frame_count)
    frames: list[Image.Image] = []
    if on_progress is not None:
        on_progress(0, frame_count, "rendering")
    for idx in range(frame_count):
        angle = start_angle + float(idx) * step
        eye = {
            "x": radius_xy * float(np.cos(angle)),
            "y": radius_xy * float(np.sin(angle)),
            "z": float(base_eye["z"]),
        }
        fig.update_scenes(camera={"eye": eye, "up": base_camera["up"], "center": base_camera["center"]})
        png_bytes = _render_plotly_png_bytes(fig)
        image = Image.open(BytesIO(png_bytes)).convert("RGB")
        frames.append(image)
        if on_progress is not None:
            on_progress(idx + 1, frame_count, "rendering")

    if not frames:
        raise RuntimeError("No frames were generated for GIF export.")

    out = BytesIO()
    if on_progress is not None:
        on_progress(frame_count, frame_count, "finalizing")
    frames[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return out.getvalue()


def save_3d_visualization_gif(
    t2w_vol: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray | None,
    spacing_zyx: tuple[float, float, float],
    output_path: Path,
    case_id: str,
    *,
    max_voxels: int = 250_000,
    has_ground_truth: bool = True,
    frame_count: int = 48,
    fps: int = 12,
    turns: float = 1.0,
    width_px: int = 960,
    height_px: int = 900,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Render and save orbit animation GIF from the same 3-D figure as HTML export."""
    fig = _prepare_3d_figure_for_export(
        t2w_vol=t2w_vol,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        spacing_zyx=spacing_zyx,
        case_id=case_id,
        max_voxels=max_voxels,
        has_ground_truth=has_ground_truth,
    )
    gif_bytes = _export_orbit_gif_bytes(
        figure=fig,
        frame_count=frame_count,
        fps=fps,
        turns=turns,
        width_px=width_px,
        height_px=height_px,
        on_progress=on_progress,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(gif_bytes)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive 3-D visualizer for a PI-CAI case. "
            "Loads T2w + segmentation (optional auto-detect), optionally runs "
            "model inference from a run directory, "
            "and saves a rotatable Plotly HTML scene (with optional orbit GIF export)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--t2w", required=True, type=str, help="Path to T2w .mha volume")
    parser.add_argument(
        "--seg",
        required=False,
        type=str,
        default=None,
        help=(
            "Optional segmentation mask path (.nii.gz). "
            "If omitted, auto-detected as data/labels/<case_id>.nii.gz. "
            "If not found, visualization proceeds without GT."
        ),
    )
    parser.add_argument("--adc", type=str, default=None, help="Optional ADC .mha path")
    parser.add_argument("--hbv", type=str, default=None, help="Optional HBV .mha path")

    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Optional run directory. When provided, script loads run config + checkpoint, "
            "runs model inference, and overlays prediction vs GT."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint filename/path to use with --run. "
            "If omitted, best.pt is used (fallback: newest epoch_*.pt)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device when --run is set (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for binarizing model logits",
    )

    parser.add_argument(
        "--max-voxels",
        type=int,
        default=250_000,
        help=(
            "Maximum voxel count used for interactive rendering. "
            "Larger volumes are uniformly downsampled for browser performance."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output HTML path. Default: outputs/visualizations/<run_name>_<image_name>_3d.html "
            "when --run is set; otherwise outputs/visualizations/<image_name>_3d.html."
        ),
    )
    parser.add_argument(
        "--gif",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Also export orbit GIF. Optionally provide output path; "
            "default when flag is present: outputs/visualizations/<run_name>_<image_name>_orbit.gif "
            "when --run is set; otherwise outputs/visualizations/<image_name>_orbit.gif."
        ),
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=48,
        help="Number of frames for GIF orbit animation",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Frames per second for GIF animation",
    )
    parser.add_argument(
        "--gif-turns",
        type=float,
        default=1.0,
        help="Number of full camera turns around the scene",
    )
    parser.add_argument(
        "--gif-width",
        type=int,
        default=960,
        help="GIF frame width in pixels",
    )
    parser.add_argument(
        "--gif-height",
        type=int,
        default=900,
        help="GIF frame height in pixels",
    )

    args = parser.parse_args()
    quiet_for_gif = args.gif is not None
    if quiet_for_gif:
        logging.getLogger().setLevel(logging.WARNING)
        logger.setLevel(logging.WARNING)

    t2w_path = Path(args.t2w).resolve()

    if not t2w_path.exists():
        raise FileNotFoundError(f"T2w file not found: {t2w_path}")

    case_id = _case_id_from_t2w(t2w_path)
    default_export_stem = _default_export_stem(t2w_path, args.run)
    try:
        seg_path: Path | None = _resolve_seg_path(t2w_path, args.seg)
    except FileNotFoundError:
        if args.seg is not None:
            raise
        seg_path = None
        logger.warning("No segmentation found for case '%s' - continuing without GT.", case_id)

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")
    if args.max_voxels <= 0:
        raise ValueError(f"--max-voxels must be > 0, got {args.max_voxels}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = (Path("outputs") / "visualizations" / f"{default_export_stem}_3d.html").resolve()
    gif_output_path: Path | None
    if args.gif is None:
        gif_output_path = None
    elif args.gif == "":
        gif_output_path = (Path("outputs") / "visualizations" / f"{default_export_stem}_orbit.gif").resolve()
    else:
        gif_output_path = Path(args.gif).resolve()

    pred_mask: np.ndarray | None = None
    has_ground_truth = seg_path is not None

    if args.run is None:
        logger.info(
            "Mode: %s visualisation (no model inference)",
            "GT-only" if has_ground_truth else "image-only",
        )
        t2w_vol, gt_mask, spacing_zyx, has_ground_truth = _load_native_t2w_and_optional_gt(
            t2w_path,
            seg_path,
        )
    else:
        run_dir = resolve_run_dir(args.run)
        cfg_path = run_dir / "config.yaml"

        ckpt_path = _resolve_checkpoint_path(run_dir, args.checkpoint)
        cfg = load_config(str(cfg_path))
        device = _resolve_device(args.device)
        postprocess_enabled = bool(cfg.get("postprocess_enabled", False))
        postprocess_min_component_volume_mm3 = float(
            cfg.get("postprocess_min_component_volume_mm3", 30.0)
        )
        postprocess_connectivity = int(cfg.get("postprocess_connectivity", 26))
        if postprocess_min_component_volume_mm3 < 0.0:
            raise ValueError(
                "postprocess_min_component_volume_mm3 must be >= 0, "
                f"got {postprocess_min_component_volume_mm3}"
            )
        if postprocess_connectivity not in (6, 18, 26):
            raise ValueError(
                "postprocess_connectivity must be one of {6, 18, 26}, "
                f"got {postprocess_connectivity}"
            )

        logger.info(
            "Mode: %s",
            "GT + model comparison" if has_ground_truth else "model prediction (no GT)",
        )
        logger.info("Run directory : %s", run_dir)
        logger.info("Checkpoint    : %s", ckpt_path)
        logger.info("Device        : %s", device)
        logger.info(
            "Postprocess   : %s (min_component_volume_mm3=%.1f, connectivity=%d)",
            "on" if postprocess_enabled else "off",
            postprocess_min_component_volume_mm3,
            postprocess_connectivity,
        )

        image_t, label_t, t2w_vol, spacing_zyx, used_paths = _load_model_inputs(
            cfg=cfg,
            t2w_path=t2w_path,
            seg_path=seg_path,
            adc_path=args.adc,
            hbv_path=args.hbv,
        )

        logger.info(
            "Model input modalities: %s",
            ", ".join(f"{k}={v.name}" for k, v in used_paths.items() if k != "t2w")
            if any(k != "t2w" for k in used_paths)
            else "t2w only",
        )

        model = build_model(cfg).to(device)
        _ = load_checkpoint(ckpt_path, model, device=device)
        model.eval()

        logits = _run_inference(model, image_t=image_t, cfg=cfg, device=device)
        metric_logits, pred_bin = postprocess_logits(
            logits=logits,
            threshold=args.threshold,
            enabled=postprocess_enabled,
            spacing_zyx=spacing_zyx,
            min_component_volume_mm3=postprocess_min_component_volume_mm3,
            connectivity=postprocess_connectivity,
        )
        pred_mask = pred_bin[0, 0].numpy().astype(np.uint8)
        gt_mask = label_t[0].numpy().astype(np.uint8)

        if has_ground_truth:
            metrics = compute_all_metrics(
                preds=metric_logits,
                targets=label_t.unsqueeze(0),
                threshold=args.threshold,
                compute_hd95=True,
            )

            if not quiet_for_gif:
                print("\nMetrics (single case):")
                print(f"  case_id      : {case_id}")
                print(f"  dice         : {_fmt(metrics['dice'])}")
                print(f"  iou          : {_fmt(metrics['iou'])}")
                print(f"  sensitivity  : {_fmt(metrics['sensitivity'])}")
                print(f"  precision    : {_fmt(metrics['precision'])}")
                print(f"  hd95         : {_fmt(metrics['hd95'])} voxels")
        elif not quiet_for_gif:
            print("\nMetrics skipped: no ground-truth label provided.")

    out = save_3d_visualization_html(
        t2w_vol=t2w_vol,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        spacing_zyx=spacing_zyx,
        output_path=output_path,
        case_id=case_id,
        max_voxels=args.max_voxels,
        has_ground_truth=has_ground_truth,
    )

    if not quiet_for_gif:
        print(f"\n3D visualization saved: {out}")

    if gif_output_path is not None:
        if args.gif_frames < 2:
            raise ValueError(f"--gif-frames must be >= 2, got {args.gif_frames}")
        if args.gif_fps < 1:
            raise ValueError(f"--gif-fps must be >= 1, got {args.gif_fps}")
        if args.gif_turns <= 0.0:
            raise ValueError(f"--gif-turns must be > 0, got {args.gif_turns}")
        if args.gif_width < 320:
            raise ValueError(f"--gif-width must be >= 320, got {args.gif_width}")
        if args.gif_height < 320:
            raise ValueError(f"--gif-height must be >= 320, got {args.gif_height}")

        def _print_gif_progress(done: int, total: int, stage: str) -> None:
            safe_total = max(1, int(total))
            safe_done = max(0, min(int(done), safe_total))
            bar_width = 30
            if stage == "rendering":
                ratio = float(safe_done) / float(safe_total)
                filled = int(round(ratio * float(bar_width)))
                bar = "#" * filled + "-" * (bar_width - filled)
                percent = int(round(ratio * 100.0))
                sys.stdout.write(f"\r[{bar}] {percent:3d}% {safe_done}/{safe_total}")
            else:
                bar = "#" * bar_width
                sys.stdout.write(f"\r[{bar}] 100% Finalizing GIF...")
            sys.stdout.flush()

        try:
            gif_out = save_3d_visualization_gif(
                t2w_vol=t2w_vol,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                spacing_zyx=spacing_zyx,
                output_path=gif_output_path,
                case_id=case_id,
                max_voxels=args.max_voxels,
                has_ground_truth=has_ground_truth,
                frame_count=args.gif_frames,
                fps=args.gif_fps,
                turns=args.gif_turns,
                width_px=args.gif_width,
                height_px=args.gif_height,
                on_progress=_print_gif_progress,
            )
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()

        print(f"3D orbit GIF saved: {gif_out}")


if __name__ == "__main__":
    main()
