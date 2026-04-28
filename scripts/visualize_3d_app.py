"""
visualize_3d_app.py - Localhost Streamlit app for PI-CAI 3-D visualisation.

UI controls
-----------
- Browse/drag-drop: modality files (.mha) with auto-detection by suffix
  (<case_id>_t2w.mha, <case_id>_adc.mha, <case_id>_hbv.mha)
- Browse/drag-drop: label file (optional .nii/.nii.gz)
- Browse/drag-drop: run folder (optional)
- Button: Run (render image-only, GT, or prediction overlay in-app)
- Dropdown: model checkpoint (defaults to best.pt when available)
"""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

import numpy as np
import streamlit as st

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import visualize_3d as vis


def _init_session_state() -> None:
    defaults = {
        "device_input": "",
        "renderer_height": 900,
        "upload_root": None,
        "uploaded_modalities": {},
        "uploaded_modality_signature": None,
        "selected_case_id": None,
        "uploaded_seg_path": None,
        "uploaded_seg_signature": None,
        "uploaded_run_dir": None,
        "uploaded_run_signature": None,
        "loaded_t2w_path": None,
        "loaded_adc_path": None,
        "loaded_hbv_path": None,
        "loaded_seg_path": None,
        "last_result": None,
        "last_orbit_gif": None,
        "last_orbit_gif_filename": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _ensure_upload_root() -> Path:
    raw = st.session_state.get("upload_root")
    if isinstance(raw, str) and raw:
        root = Path(raw)
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    root = Path(tempfile.mkdtemp(prefix="picai_3d_uploads_"))
    st.session_state.upload_root = str(root.resolve())
    return root.resolve()


def _uploaded_signature(uploaded_file: Any) -> tuple[str, int]:
    name = str(getattr(uploaded_file, "name", ""))
    size = int(getattr(uploaded_file, "size", -1))
    return name, size


def _sanitize_uploaded_relative_path(raw_name: str) -> Path:
    parts = [p for p in raw_name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts:
        raise ValueError("Uploaded file has an invalid path.")
    return Path(*parts)


def _persist_uploaded_single_file(
    uploaded_file: Any | None,
    *,
    path_state_key: str,
    signature_state_key: str,
    subdir: str,
) -> Path | None:
    if uploaded_file is None:
        st.session_state[path_state_key] = None
        st.session_state[signature_state_key] = None
        return None

    signature = _uploaded_signature(uploaded_file)
    existing_signature = st.session_state.get(signature_state_key)
    existing_path_raw = st.session_state.get(path_state_key)
    existing_ok = isinstance(existing_path_raw, str) and Path(existing_path_raw).is_file()

    if existing_signature != signature or not existing_ok:
        root = _ensure_upload_root()
        filename = Path(str(uploaded_file.name)).name
        dest = (root / subdir / filename).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            handle.write(uploaded_file.getbuffer())
        st.session_state[path_state_key] = str(dest)
        st.session_state[signature_state_key] = signature

    path_raw = st.session_state.get(path_state_key)
    if not isinstance(path_raw, str) or not path_raw:
        return None
    return Path(path_raw)


def _parse_case_modality_from_filename(filename: str) -> tuple[str, str] | None:
    name = Path(filename).name
    if not name.lower().endswith(".mha"):
        return None

    stem = name[: -len(".mha")]
    stem_lower = stem.lower()
    for modality in ("t2w", "adc", "hbv"):
        suffix = f"_{modality}"
        if stem_lower.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], modality
    return None


def _deserialize_uploaded_modalities(raw: Any) -> dict[str, dict[str, Path]]:
    if not isinstance(raw, dict):
        return {}

    out: dict[str, dict[str, Path]] = {}
    for case_id, modality_map in raw.items():
        if not isinstance(case_id, str) or not isinstance(modality_map, dict):
            continue
        for modality, path_raw in modality_map.items():
            if modality not in ("t2w", "adc", "hbv") or not isinstance(path_raw, str):
                continue
            path = Path(path_raw)
            if path.is_file():
                out.setdefault(case_id, {})[modality] = path.resolve()
    return out


def _persist_uploaded_modality_files(uploaded_files: list[Any]) -> tuple[dict[str, dict[str, Path]], list[str]]:
    if not uploaded_files:
        st.session_state.uploaded_modalities = {}
        st.session_state.uploaded_modality_signature = None
        return {}, []

    signature = tuple(sorted(_uploaded_signature(f) for f in uploaded_files))
    existing_signature = st.session_state.get("uploaded_modality_signature")
    if existing_signature == signature:
        restored = _deserialize_uploaded_modalities(st.session_state.get("uploaded_modalities"))
        if restored:
            return restored, []

    modality_root = (_ensure_upload_root() / "modalities").resolve()
    if modality_root.exists():
        shutil.rmtree(modality_root)
    modality_root.mkdir(parents=True, exist_ok=True)

    modalities_by_case: dict[str, dict[str, Path]] = {}
    skipped: list[str] = []
    replaced: list[str] = []

    for uploaded in uploaded_files:
        original_name = str(uploaded.name)
        parsed = _parse_case_modality_from_filename(original_name)
        if parsed is None:
            skipped.append(Path(original_name).name)
            continue

        case_id, modality = parsed
        filename = Path(original_name).name
        dest = (modality_root / filename).resolve()
        if modality_root not in dest.parents and dest != modality_root:
            raise ValueError(f"Unsafe upload path: {original_name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            handle.write(uploaded.getbuffer())

        existing = modalities_by_case.get(case_id, {}).get(modality)
        if existing is not None:
            replaced.append(f"{case_id}_{modality}.mha")
        modalities_by_case.setdefault(case_id, {})[modality] = dest

    st.session_state.uploaded_modality_signature = signature
    st.session_state.uploaded_modalities = {
        case_id: {modality: str(path) for modality, path in modality_map.items()}
        for case_id, modality_map in modalities_by_case.items()
    }

    notes: list[str] = []
    if skipped:
        skipped_preview = ", ".join(sorted(set(skipped))[:5])
        notes.append(f"Ignored non-modality files: {skipped_preview}")
    if replaced:
        replaced_preview = ", ".join(sorted(set(replaced))[:5])
        notes.append(f"Duplicate modality uploads replaced: {replaced_preview}")

    return modalities_by_case, notes


def _discover_run_dir_candidates(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for cfg_path in root.rglob("config.yaml"):
        run_dir = cfg_path.parent.resolve()
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.is_dir() and any(checkpoints_dir.glob("*.pt")):
            candidates.add(run_dir)

    return sorted(
        candidates,
        key=lambda p: (len(p.relative_to(root).parts), str(p)),
    )


def _persist_uploaded_run_directory(uploaded_files: list[Any]) -> tuple[Path | None, str | None]:
    if not uploaded_files:
        st.session_state.uploaded_run_dir = None
        st.session_state.uploaded_run_signature = None
        return None, None

    signature = tuple(sorted(_uploaded_signature(f) for f in uploaded_files))
    existing_signature = st.session_state.get("uploaded_run_signature")
    existing_dir_raw = st.session_state.get("uploaded_run_dir")
    existing_ok = isinstance(existing_dir_raw, str) and Path(existing_dir_raw).is_dir()
    if existing_signature == signature and existing_ok:
        return Path(existing_dir_raw), None

    run_upload_root = (_ensure_upload_root() / "run_upload").resolve()
    if run_upload_root.exists():
        shutil.rmtree(run_upload_root)
    run_upload_root.mkdir(parents=True, exist_ok=True)

    for uploaded in uploaded_files:
        rel_path = _sanitize_uploaded_relative_path(str(uploaded.name))
        dest = (run_upload_root / rel_path).resolve()
        if run_upload_root not in dest.parents and dest != run_upload_root:
            raise ValueError(f"Unsafe upload path: {uploaded.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as handle:
            handle.write(uploaded.getbuffer())

    run_candidates = _discover_run_dir_candidates(run_upload_root)
    if not run_candidates:
        st.session_state.uploaded_run_dir = None
        st.session_state.uploaded_run_signature = signature
        raise ValueError(
            "Uploaded folder is not a valid run directory. Expected config.yaml and checkpoints/*.pt."
        )

    selected = run_candidates[0]
    st.session_state.uploaded_run_dir = str(selected)
    st.session_state.uploaded_run_signature = signature

    note = None
    if len(run_candidates) > 1:
        note = f"Multiple run directories found; using: {selected}"
    return selected, note


def _run_folder_uploader() -> list[Any]:
    label = "Browse or drop run folder (optional)"
    help_text = "Drop a run directory that contains config.yaml and checkpoints/*.pt"
    try:
        uploaded = st.file_uploader(
            label,
            accept_multiple_files="directory",
            key="run_dir_upload",
            help=help_text,
        )
    except TypeError:
        uploaded = st.file_uploader(
            label,
            accept_multiple_files=True,
            key="run_dir_upload",
            help=(
                "Directory upload is unsupported by this Streamlit version. "
                "Upload all run files while preserving folder structure."
            ),
        )
    if isinstance(uploaded, list):
        return uploaded
    return []


def _resolve_run_ui_state(run_input: str) -> tuple[Path | None, list[str], str | None]:
    raw = run_input.strip()
    if not raw:
        return None, [], None

    try:
        run_dir = vis.resolve_run_dir(raw)
        ckpt_paths = vis.list_available_checkpoints(run_dir)
    except Exception as exc:  # noqa: BLE001
        return None, [], str(exc)

    return run_dir, [p.name for p in ckpt_paths], None


def _render_pipeline(
    t2w_path: Path,
    seg_path: Path | None,
    adc_path: Path | None,
    hbv_path: Path | None,
    run_dir: Path | None,
    checkpoint_name: str | None,
    threshold: float,
    max_voxels: int,
    device_arg: str | None,
) -> dict:
    case_id = vis._case_id_from_t2w(t2w_path)

    pred_mask: np.ndarray | None = None
    metrics: dict[str, float] | None = None
    checkpoint_path: Path | None = None
    used_paths: dict[str, Path] = {"t2w": t2w_path}
    has_ground_truth = seg_path is not None

    if run_dir is None:
        t2w_vol, gt_mask, spacing_zyx, has_ground_truth = vis._load_native_t2w_and_optional_gt(
            t2w_path=t2w_path,
            seg_path=seg_path,
        )
    else:
        cfg = vis.load_config(str(run_dir / "config.yaml"))
        checkpoint_path = vis._resolve_checkpoint_path(run_dir, checkpoint_name)
        device = vis._resolve_device(device_arg)

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

        image_t, label_t, t2w_vol, spacing_zyx, used_paths = vis._load_model_inputs(
            cfg=cfg,
            t2w_path=t2w_path,
            seg_path=seg_path,
            adc_path=str(adc_path) if adc_path is not None else None,
            hbv_path=str(hbv_path) if hbv_path is not None else None,
        )

        model = vis.build_model(cfg).to(device)
        _ = vis.load_checkpoint(checkpoint_path, model, device=device)
        model.eval()

        logits = vis._run_inference(model, image_t=image_t, cfg=cfg, device=device)
        metric_logits, pred_bin = vis.postprocess_logits(
            logits=logits,
            threshold=threshold,
            enabled=postprocess_enabled,
            spacing_zyx=spacing_zyx,
            min_component_volume_mm3=postprocess_min_component_volume_mm3,
            connectivity=postprocess_connectivity,
        )
        pred_mask = pred_bin[0, 0].numpy().astype(np.uint8)
        gt_mask = label_t[0].numpy().astype(np.uint8)

        if has_ground_truth:
            metrics = vis.compute_all_metrics(
                preds=metric_logits,
                targets=label_t.unsqueeze(0),
                threshold=threshold,
                compute_hd95=True,
            )

    t2w_ds, gt_ds, pred_ds, spacing_ds, stride_zyx = vis._downsample_for_render(
        t2w_vol=t2w_vol,
        gt_mask=gt_mask,
        pred_mask=pred_mask,
        spacing_zyx=spacing_zyx,
        max_voxels=max_voxels,
    )

    figure = vis._build_3d_figure(
        t2w_vol=t2w_ds,
        gt_mask=gt_ds,
        pred_mask=pred_ds,
        spacing_zyx=spacing_ds,
        case_id=case_id,
        has_ground_truth=has_ground_truth,
    )

    return {
        "case_id": case_id,
        "has_ground_truth": has_ground_truth,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "used_paths": {k: str(v) for k, v in used_paths.items()},
        "render_stride": stride_zyx,
        "render_shape": tuple(int(v) for v in t2w_ds.shape),
        "figure": figure,
    }


def _show_metrics(metrics: dict[str, float] | None) -> None:
    if metrics is None:
        st.info("Metrics skipped (no ground-truth label loaded).")
        return

    cols = st.columns(5)
    cols[0].metric("Dice", vis._fmt(metrics["dice"]))
    cols[1].metric("IoU", vis._fmt(metrics["iou"]))
    cols[2].metric("Sensitivity", vis._fmt(metrics["sensitivity"]))
    cols[3].metric("Precision", vis._fmt(metrics["precision"]))
    cols[4].metric("HD95 (vox)", vis._fmt(metrics["hd95"]))


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


def _safe_case_slug(case_id: str) -> str:
    slug = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in case_id)
    slug = slug.strip("_")
    return slug or "case"


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


def main() -> None:
    st.set_page_config(page_title="Prostate Lesion 3-D Visualizer", layout="wide")
    _init_session_state()

    st.title("Prostate Lesion 3-D Visualizer")
    st.caption(
        "Browse/drag files, choose a checkpoint, and render an interactive 3-D lesion view."
    )

    controls_col, renderer_col = st.columns([20, 80], gap="large")
    run_request: dict[str, Any] | None = None

    with controls_col:
        st.subheader("Controls")

        uploaded_modality_files = st.file_uploader(
            "Browse or drop modality files (.mha)",
            type=["mha"],
            accept_multiple_files=True,
            key="modality_files_upload",
            help=(
                "Upload one or more files named <case_id>_t2w.mha, <case_id>_adc.mha, "
                "and <case_id>_hbv.mha."
            ),
        )

        uploaded_seg = st.file_uploader(
            "Browse or drop label (optional .nii/.nii.gz)",
            key="seg_file_upload",
        )

        uploaded_run_files = _run_folder_uploader()
        modalities_by_case: dict[str, dict[str, Path]] = {}
        modality_notes: list[str] = []
        try:
            if isinstance(uploaded_modality_files, list):
                modalities_by_case, modality_notes = _persist_uploaded_modality_files(uploaded_modality_files)
            else:
                modalities_by_case, modality_notes = _persist_uploaded_modality_files([])
        except Exception as exc:  # noqa: BLE001
            st.session_state.uploaded_modalities = {}
            st.session_state.uploaded_modality_signature = None
            st.session_state.selected_case_id = None
            st.session_state.loaded_t2w_path = None
            st.session_state.loaded_adc_path = None
            st.session_state.loaded_hbv_path = None
            st.error(f"Modality upload error: {exc}")

        for note in modality_notes:
            st.info(note)

        if modalities_by_case:
            case_ids = sorted(modalities_by_case)
            selected_case_id = st.session_state.get("selected_case_id")

            if selected_case_id not in case_ids:
                with_t2w = [cid for cid in case_ids if "t2w" in modalities_by_case.get(cid, {})]
                st.session_state.selected_case_id = with_t2w[0] if with_t2w else case_ids[0]

            if len(case_ids) > 1:
                st.selectbox(
                    "Detected case ID",
                    options=case_ids,
                    key="selected_case_id",
                    help="Choose which case to run when multiple case IDs are uploaded.",
                )
            else:
                st.session_state.selected_case_id = case_ids[0]
                st.caption(f"Detected case ID: `{case_ids[0]}`")

            selected_case_id = st.session_state.selected_case_id
            selected_modalities = modalities_by_case.get(selected_case_id, {})

            t2w_path = selected_modalities.get("t2w")
            adc_path = selected_modalities.get("adc")
            hbv_path = selected_modalities.get("hbv")
            st.session_state.loaded_t2w_path = str(t2w_path) if t2w_path is not None else None
            st.session_state.loaded_adc_path = str(adc_path) if adc_path is not None else None
            st.session_state.loaded_hbv_path = str(hbv_path) if hbv_path is not None else None

            if t2w_path is None:
                st.error(
                    "Detected case is missing T2w. Include a file named <case_id>_t2w.mha in modality uploads."
                )
        else:
            st.session_state.selected_case_id = None
            st.session_state.loaded_t2w_path = None
            st.session_state.loaded_adc_path = None
            st.session_state.loaded_hbv_path = None

            if isinstance(uploaded_modality_files, list) and uploaded_modality_files:
                st.error(
                    "No valid modality files detected. Expected names like <case_id>_t2w.mha, "
                    "<case_id>_adc.mha, <case_id>_hbv.mha."
                )

        if uploaded_seg is not None:
            seg_name = str(uploaded_seg.name).lower()
            if not (seg_name.endswith(".nii") or seg_name.endswith(".nii.gz")):
                st.session_state.uploaded_seg_path = None
                st.session_state.uploaded_seg_signature = None
                st.session_state.loaded_seg_path = None
                st.error("Label upload error: expected .nii or .nii.gz file.")
            else:
                try:
                    uploaded_seg_path = _persist_uploaded_single_file(
                        uploaded_seg,
                        path_state_key="uploaded_seg_path",
                        signature_state_key="uploaded_seg_signature",
                        subdir="label",
                    )
                    st.session_state.loaded_seg_path = str(uploaded_seg_path) if uploaded_seg_path else None
                except Exception as exc:  # noqa: BLE001
                    st.session_state.uploaded_seg_path = None
                    st.session_state.loaded_seg_path = None
                    st.error(f"Label upload error: {exc}")
        else:
            st.session_state.uploaded_seg_path = None
            st.session_state.uploaded_seg_signature = None
            st.session_state.loaded_seg_path = None

        run_upload_note: str | None = None
        try:
            _, run_upload_note = _persist_uploaded_run_directory(uploaded_run_files)
        except Exception as exc:  # noqa: BLE001
            st.session_state.uploaded_run_dir = None
            st.error(f"Run folder upload error: {exc}")
        if run_upload_note:
            st.info(run_upload_note)

        effective_run_input = st.session_state.uploaded_run_dir or ""
        run_dir, checkpoint_options, run_error = _resolve_run_ui_state(effective_run_input)
        if run_error is not None:
            st.warning(f"Run directory validation: {run_error}")

        if checkpoint_options:
            default_index = checkpoint_options.index("best.pt") if "best.pt" in checkpoint_options else 0
            checkpoint_name = st.selectbox(
                "Model checkpoint",
                options=checkpoint_options,
                index=default_index,
            )
        else:
            checkpoint_name = st.selectbox(
                "Model checkpoint",
                options=["best.pt"],
                index=0,
                disabled=True,
                help="Upload a valid run folder to list checkpoints.",
            )

        with st.expander("Advanced options", expanded=False):
            threshold = st.slider(
                "Prediction threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )
            max_voxels = int(
                st.number_input(
                    "Max voxels for render downsampling",
                    min_value=10_000,
                    max_value=2_000_000,
                    value=250_000,
                    step=10_000,
                )
            )
            st.number_input(
                "Renderer height (px)",
                min_value=400,
                max_value=1600,
                step=50,
                key="renderer_height",
            )
            st.text_input(
                "Device override (optional)",
                key="device_input",
                placeholder="cpu, cuda, cuda:0",
            )

        selected_t2w = st.session_state.loaded_t2w_path
        selected_adc = st.session_state.loaded_adc_path
        selected_hbv = st.session_state.loaded_hbv_path
        selected_label = st.session_state.loaded_seg_path
        selected_run = effective_run_input
        with st.expander("Selected paths", expanded=False):
            st.write(
                f"Selected T2w path: `{selected_t2w}`"
                if selected_t2w
                else "Selected T2w path: not set"
            )
            st.write(
                f"Selected ADC path: `{selected_adc}`"
                if selected_adc
                else "Selected ADC path: not set"
            )
            st.write(
                f"Selected HBV path: `{selected_hbv}`"
                if selected_hbv
                else "Selected HBV path: not set"
            )
            st.write(
                f"Selected label path: `{selected_label}`"
                if selected_label
                else "Selected label path: not set"
            )
            st.write(
                f"Selected run path: `{selected_run}`"
                if selected_run
                else "Selected run path: not set"
            )

        if st.button("Run", type="primary", use_container_width=True):
            try:
                if st.session_state.loaded_t2w_path is None:
                    raise ValueError("Upload modality files that include <case_id>_t2w.mha first.")

                t2w_path = Path(st.session_state.loaded_t2w_path)
                adc_path = (
                    Path(st.session_state.loaded_adc_path)
                    if st.session_state.loaded_adc_path is not None
                    else None
                )
                hbv_path = (
                    Path(st.session_state.loaded_hbv_path)
                    if st.session_state.loaded_hbv_path is not None
                    else None
                )

                seg_path: Path | None
                if st.session_state.loaded_seg_path is not None:
                    seg_path = Path(st.session_state.loaded_seg_path)
                else:
                    try:
                        seg_path = vis._resolve_seg_path(t2w_path, None)
                        st.info(f"Auto-detected label: {seg_path}")
                    except FileNotFoundError:
                        seg_path = None
                        st.info("No label detected. Running without ground truth.")

                active_run_dir = run_dir
                if effective_run_input and active_run_dir is None:
                    raise ValueError(
                        "Uploaded run folder is invalid. Re-upload it or clear run-folder upload for image-only mode."
                    )

                run_request = {
                    "t2w_path": t2w_path,
                    "seg_path": seg_path,
                    "adc_path": adc_path,
                    "hbv_path": hbv_path,
                    "run_dir": active_run_dir,
                    "checkpoint_name": checkpoint_name if active_run_dir is not None else None,
                    "threshold": threshold,
                    "max_voxels": max_voxels,
                    "device_arg": st.session_state.device_input.strip() or None,
                }
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with renderer_col:
        if run_request is not None:
            try:
                with st.spinner("Rendering 3-D visualization..."):
                    result = _render_pipeline(**run_request)
                st.session_state.last_result = result
                st.session_state.last_orbit_gif = None
                st.session_state.last_orbit_gif_filename = None
                st.success("Visualization ready (rendered in Streamlit).")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        result = st.session_state.last_result
        if isinstance(result, dict):
            figure = result.get("figure")
            if figure is not None:
                st.subheader(f"Case: {result['case_id']}")
                if result.get("checkpoint_path"):
                    st.write(f"Checkpoint: `{result['checkpoint_path']}`")

                used_paths = result.get("used_paths", {})
                if isinstance(used_paths, dict) and used_paths:
                    modality_items = [
                        f"{k}={Path(v).name}" for k, v in used_paths.items() if k != "t2w"
                    ]
                    if modality_items:
                        st.write("Modalities: " + ", ".join(modality_items))
                    else:
                        st.write("Modalities: t2w only")

                stride = result.get("render_stride")
                shape = result.get("render_shape")
                if stride and shape:
                    st.caption(f"Render grid shape: {shape}, stride(z,y,x): {stride}")

                _show_metrics(result.get("metrics"))
                figure.update_layout(height=int(st.session_state.renderer_height))
                st.plotly_chart(figure, use_container_width=True, theme=None)

                with st.expander("Export orbit GIF", expanded=False):
                    gif_col_a, gif_col_b = st.columns(2)
                    frame_count = int(
                        gif_col_a.slider(
                            "Frames",
                            min_value=16,
                            max_value=120,
                            value=48,
                            step=4,
                            key="orbit_frame_count",
                        )
                    )
                    fps = int(
                        gif_col_b.slider(
                            "FPS",
                            min_value=4,
                            max_value=30,
                            value=12,
                            step=1,
                            key="orbit_fps",
                        )
                    )

                    gif_col_c, gif_col_d = st.columns(2)
                    turns = float(
                        gif_col_c.slider(
                            "Turns",
                            min_value=0.25,
                            max_value=3.0,
                            value=1.0,
                            step=0.25,
                            key="orbit_turns",
                        )
                    )
                    width_px = int(
                        gif_col_d.number_input(
                            "Width (px)",
                            min_value=480,
                            max_value=1920,
                            value=960,
                            step=80,
                            key="orbit_width_px",
                        )
                    )

                    if st.button("Export GIF", use_container_width=True):
                        progress_slot = st.empty()
                        status_slot = st.empty()
                        progress_bar = progress_slot.progress(0)

                        def _update_gif_export_progress(done: int, total: int, stage: str) -> None:
                            safe_total = max(1, int(total))
                            safe_done = max(0, min(int(done), safe_total))
                            if stage == "rendering":
                                percent = int(round((100.0 * float(safe_done)) / float(safe_total)))
                                progress_bar.progress(percent)
                                status_slot.caption(f"Rendering frames: {safe_done}/{safe_total}")
                            else:
                                progress_bar.progress(100)
                                status_slot.caption("Finalizing GIF...")

                        try:
                            gif_bytes = _export_orbit_gif_bytes(
                                figure=figure,
                                frame_count=frame_count,
                                fps=fps,
                                turns=turns,
                                width_px=width_px,
                                height_px=int(st.session_state.renderer_height),
                                on_progress=_update_gif_export_progress,
                            )
                            case_slug = _safe_case_slug(str(result.get("case_id", "case")))
                            st.session_state.last_orbit_gif = gif_bytes
                            st.session_state.last_orbit_gif_filename = f"{case_slug}_orbit.gif"
                            progress_slot.empty()
                            status_slot.empty()
                            st.success("GIF export ready.")
                        except Exception as exc:  # noqa: BLE001
                            progress_slot.empty()
                            status_slot.empty()
                            st.error(f"GIF export failed: {exc}")

                    gif_data = st.session_state.get("last_orbit_gif")
                    gif_name = st.session_state.get("last_orbit_gif_filename")
                    if isinstance(gif_data, (bytes, bytearray)) and isinstance(gif_name, str):
                        st.download_button(
                            "Download GIF",
                            data=bytes(gif_data),
                            file_name=gif_name,
                            mime="image/gif",
                            use_container_width=True,
                        )
        else:
            st.info("Renderer output appears here after you click Run.")


if __name__ == "__main__":
    main()
