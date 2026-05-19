"""
PI-CAI prostate lesion segmentation dataset.

Expected directory layout
--------------------------
Two layouts are supported and detected automatically.

Flat (PI-CAI default download)::

    <images_dir>/
        <patient_id>_<study_id>_t2w.mha
        <patient_id>_<study_id>_adc.mha
        <patient_id>_<study_id>_hbv.mha
        ...   (extra modalities such as _cor/_sag are ignored)

Nested::

    <images_dir>/
        <patient_id>/
            <patient_id>_<study_id>/
                <patient_id>_<study_id>_t2w.mha
                <patient_id>_<study_id>_adc.mha
                <patient_id>_<study_id>_hbv.mha

Labels (same for both image layouts)::

    <labels_dir>/
        <patient_id>_<study_id>.nii.gz   (csPCa lesion mask, grades 0-5)

Labels encode PI-RADS grade per voxel (0 = background/benign, >=1 = lesion).
The dataset binarises labels by default: 0 = background, 1 = any lesion.

Prostate158 cases are discovered from the upstream train/valid/test CSV files
via ``discover_prostate158_cases`` and then loaded through the same dataset
class.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import threading
import hashlib
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import Dataset

from src.config import load_config
from src.models import build_model
from src.roi import (
    ROISettings,
    TASK_LESION_SEGMENTATION,
    TASK_PROSTATE_LOCALIZATION,
    binarize_mask,
    compute_crop_bounds,
    crop_bounds_from_dict,
    crop_tensor,
    keep_largest_component,
    resolve_localizer_checkpoint,
    resolve_roi_settings,
    resolve_task,
)
from src.utils import load_checkpoint

logger = logging.getLogger(__name__)

# PI-CAI modality suffixes in channel order [T2w, ADC, HBV]
MODALITY_SUFFIXES = ("_t2w.mha", "_adc.mha", "_hbv.mha")
MODALITY_KEYS = ("t2w", "adc", "hbv")
LABEL_SUFFIX = ".nii.gz"
SPLIT_MANIFEST_VERSION = 1


def active_modality_pairs(cfg: dict) -> list[tuple[str, str]]:
    """
    Return ``(key, suffix)`` pairs for the modalities enabled in *cfg*.

    Reads the ``use_t2w``, ``use_adc``, and ``use_hbv`` boolean flags
    (all default to ``True`` when absent so that existing configs without
    the flags continue to behave as before).  Pairs are returned in the
    canonical channel order ``[T2w, ADC, HBV]``.

    Parameters
    ----------
    cfg : dict
        Training configuration dict with optional keys
        ``use_t2w``, ``use_adc``, ``use_hbv``.

    Returns
    -------
    list[tuple[str, str]]
        Each element is ``(modality_key, file_suffix)``, e.g.
        ``[("t2w", "_t2w.mha"), ("adc", "_adc.mha")]``.

    Raises
    ------
    ValueError
        If all three flags are ``False`` (no modality would be loaded).
    """
    pairs = [
        (key, suffix)
        for key, suffix in zip(MODALITY_KEYS, MODALITY_SUFFIXES)
        if cfg.get(f"use_{key}", True)
    ]
    if not pairs:
        raise ValueError(
            "No modalities enabled. At least one of use_t2w, use_adc, "
            "use_hbv must be true in the config."
        )
    return pairs


# ---------------------------------------------------------------------------
# SimpleITK helpers
# ---------------------------------------------------------------------------

def _load_volume(path: Path) -> sitk.Image:
    """Load a medical image volume (.mha or .nii.gz)."""
    return sitk.ReadImage(str(path))


def _resample(
    image: sitk.Image,
    target_spacing: tuple[float, ...],
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample *image* to *target_spacing* (z, y, x) in mm.

    SimpleITK uses (x, y, z) convention internally; this function accepts
    the (z, y, x) convention used elsewhere in the codebase and converts.
    """
    orig_spacing = np.array(image.GetSpacing())    # (x, y, z)
    orig_size = np.array(image.GetSize())          # (x, y, z)
    tgt_sp = np.array(target_spacing[::-1])        # (z,y,x) -> (x,y,z)

    new_size = np.round(orig_size * orig_spacing / tgt_sp).astype(int).tolist()

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tgt_sp.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def _resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample *moving* image into the physical space of *reference*.

    Used to co-register ADC/HBV into the T2w voxel grid before any
    global resampling step.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def _to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image → float32 numpy array shaped (D, H, W)."""
    # sitk.GetArrayFromImage returns (z, y, x) which equals (D, H, W)
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _zscore_normalize(arr: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """
    Z-score normalise a 3-D float array and clip to [-clip, +clip].

    Normalisation is computed over all non-zero voxels to avoid background
    bias; falls back to whole-volume stats if the foreground is empty.
    """
    foreground = arr[arr > 0]
    if foreground.size < 10:
        foreground = arr  # fallback: use everything

    mean = foreground.mean()
    std = foreground.std()
    if std < 1e-8:
        return np.zeros_like(arr)

    arr = (arr - mean) / std
    return np.clip(arr, -clip, clip).astype(np.float32)


def _preprocess_dwi_as_hbv(
    arr: np.ndarray,
    clip_percentiles: tuple[float, float] = (1.0, 99.5),
    use_log1p: bool = True,
) -> np.ndarray:
    """
    Preprocess Prostate158 DWI (b>=1000) when used as an HBV proxy.

    Steps:
      1) NaN/inf guard
      2) clamp to non-negative values
      3) percentile clipping
      4) optional log1p dynamic-range compression
    """
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)

    lo, hi = clip_percentiles
    if 0.0 <= lo < hi <= 100.0:
        p_lo, p_hi = np.percentile(arr, [lo, hi])
        arr = np.clip(arr, p_lo, p_hi)

    if use_log1p:
        arr = np.log1p(arr)

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _is_flat_layout(images_dir: Path) -> bool:
    """
    Return True if *images_dir* uses a flat layout (all .mha files directly
    inside the directory), False if it uses the nested layout
    (<patient_id>/<case_id>/ sub-directories).

    Detection: if the first non-hidden child entry is a regular file (not a
    directory) we treat the whole directory as flat.
    """
    for entry in images_dir.iterdir():
        if entry.name.startswith("."):
            continue
        return entry.is_file()
    return False  # empty directory — default to nested


def _discover_cases_flat(
    images_dir: Path,
    labels_dir: Path,
    active_keys: Sequence[str],
) -> list[dict]:
    """
    Discover cases from a flat layout::

        <images_dir>/
            <case_id>_t2w.mha
            <case_id>_adc.mha
            <case_id>_hbv.mha
            ...   (other modalities such as _cor/_sag are ignored)

    Case IDs are derived by stripping ``_t2w.mha`` from each T2w filename.
    T2w is always required on disk (it is the co-registration reference
    frame).  ADC and HBV files are only required when their respective
    keys appear in *active_keys*.
    """
    t2w_suffix = MODALITY_SUFFIXES[0]  # "_t2w.mha"
    cases: list[dict] = []

    for t2w_path in sorted(images_dir.glob(f"*{t2w_suffix}")):
        case_id = t2w_path.name[: -len(t2w_suffix)]
        # T2w is always included — it is the reference frame for
        # co-registration and label resampling.
        paths: dict = {"case_id": case_id, "t2w": t2w_path}
        complete = True

        for key, suffix in zip(MODALITY_KEYS[1:], MODALITY_SUFFIXES[1:]):
            if key not in active_keys:
                continue  # modality disabled — skip existence check
            p = images_dir / f"{case_id}{suffix}"
            if not p.exists():
                logger.warning("Case %s: missing modality '%s' (%s)", case_id, key, p)
                complete = False
                break
            paths[key] = p

        if not complete:
            continue

        label_path = labels_dir / f"{case_id}{LABEL_SUFFIX}"
        paths["label"] = label_path if label_path.exists() else None

        if paths["label"] is None:
            logger.debug("Case %s: no label found at %s", case_id, label_path)

        cases.append(paths)

    return cases


def _discover_cases_nested(
    images_dir: Path,
    labels_dir: Path,
    active_keys: Sequence[str],
) -> list[dict]:
    """
    Discover cases from a nested layout::

        <images_dir>/
            <patient_id>/
                <case_id>/
                    <case_id>_t2w.mha
                    <case_id>_adc.mha
                    <case_id>_hbv.mha

    T2w is always required on disk (it is the co-registration reference
    frame).  ADC and HBV files are only required when their respective
    keys appear in *active_keys*.
    """
    cases: list[dict] = []

    for patient_dir in sorted(images_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for study_dir in sorted(patient_dir.iterdir()):
            if not study_dir.is_dir():
                continue

            case_id = study_dir.name
            paths: dict = {"case_id": case_id}
            complete = True

            for key, suffix in zip(MODALITY_KEYS, MODALITY_SUFFIXES):
                if key != "t2w" and key not in active_keys:
                    continue  # modality disabled — skip existence check
                p = study_dir / f"{case_id}{suffix}"
                if not p.exists():
                    logger.warning("Case %s: missing modality '%s' (%s)", case_id, key, p)
                    complete = False
                    break
                paths[key] = p

            if not complete:
                continue

            label_path = labels_dir / f"{case_id}{LABEL_SUFFIX}"
            paths["label"] = label_path if label_path.exists() else None

            if paths["label"] is None:
                logger.debug("Case %s: no label found at %s", case_id, label_path)

            cases.append(paths)

    return cases


def discover_cases(
    images_dir: Path,
    labels_dir: Path,
    active_keys: Sequence[str] = MODALITY_KEYS,
) -> list[dict]:
    """
    Walk *images_dir* and return a list of case dicts.

    Supports two directory layouts automatically:

    **Flat** (all .mha files directly in *images_dir*)::

        <images_dir>/
            10000_1000000_t2w.mha
            10000_1000000_adc.mha
            10000_1000000_hbv.mha
            ...

    **Nested** (<patient>/<case> sub-directories)::

        <images_dir>/
            10000/
                10000_1000000/
                    10000_1000000_t2w.mha
                    ...

    T2w is always required on disk because it serves as the
    co-registration reference frame for ADC/HBV and as the reference
    grid for label resampling.  ADC and HBV files are required only when
    their keys appear in *active_keys*.

    Each returned dict always contains:
        case_id : str          e.g. ``"10000_1000000"``
        t2w     : Path
        label   : Path | None  (None for unlabelled cases)

    Plus, for each key in *active_keys* that is not ``"t2w"``:
        adc     : Path         (present only when ``"adc"`` in active_keys)
        hbv     : Path         (present only when ``"hbv"`` in active_keys)

    Parameters
    ----------
    images_dir  : directory containing PI-CAI .mha image files.
    labels_dir  : directory containing .nii.gz label masks.
    active_keys : modality keys to load; defaults to all three
                  ``("t2w", "adc", "hbv")``.  Files for inactive
                  modalities are not checked for existence.
    """
    flat = _is_flat_layout(images_dir)
    layout = "flat" if flat else "nested"
    logger.info("Detected %s image layout in %s", layout, images_dir)

    if flat:
        cases = _discover_cases_flat(images_dir, labels_dir, active_keys)
    else:
        cases = _discover_cases_nested(images_dir, labels_dir, active_keys)

    logger.info("Discovered %d cases (%s layout)", len(cases), layout)
    return cases


def discover_unlabeled_cases(
    images_dir: Path,
    active_keys: Sequence[str] = MODALITY_KEYS,
) -> list[dict]:
    """
    Discover Prostate158 unlabeled cases in flattened layout::

        <images_dir>/
            <case_id>_t2.nii.gz
            <case_id>_adc.nii.gz
            <case_id>_dwi.nii.gz

    Returned dicts are PiCaiDataset-compatible:
      - ``t2w`` points to ``*_t2.nii.gz``
      - ``adc`` points to ``*_adc.nii.gz`` when ADC is active
      - ``hbv`` points to ``*_dwi.nii.gz`` when HBV is active
      - ``hbv_source`` is set to ``"dwi"`` for optional DWI-specific preprocessing
      - ``label`` is always ``None``
    """
    images_dir = Path(images_dir)
    t2_suffix = "_t2.nii.gz"
    adc_suffix = "_adc.nii.gz"
    dwi_suffix = "_dwi.nii.gz"

    cases: list[dict] = []
    for t2_path in sorted(images_dir.glob(f"*{t2_suffix}")):
        case_id = t2_path.name[: -len(t2_suffix)]
        paths: dict = {
            "case_id": case_id,
            "t2w": t2_path,
            "label": None,
        }
        complete = True

        if "adc" in active_keys:
            adc_path = images_dir / f"{case_id}{adc_suffix}"
            if not adc_path.exists():
                logger.warning("Unlabeled case %s: missing ADC (%s)", case_id, adc_path)
                complete = False
            else:
                paths["adc"] = adc_path

        if "hbv" in active_keys:
            dwi_path = images_dir / f"{case_id}{dwi_suffix}"
            if not dwi_path.exists():
                logger.warning("Unlabeled case %s: missing DWI (%s)", case_id, dwi_path)
                complete = False
            else:
                paths["hbv"] = dwi_path
                paths["hbv_source"] = "dwi"

        if complete:
            cases.append(paths)

    logger.info("Discovered %d unlabeled cases in %s", len(cases), images_dir)
    return cases


def _resolve_prostate158_csv(root_dir: Path, split: str) -> Path:
    csv_path = root_dir / f"{split}.csv"
    if csv_path.exists():
        return csv_path

    nested = root_dir / root_dir.name / f"{split}.csv"
    if nested.exists():
        return nested

    raise FileNotFoundError(
        f"Prostate158 {split}.csv not found under {root_dir}. "
        "Expected the extracted archive root, e.g. data/prostate158_train."
    )


def _resolve_prostate158_path(root_dir: Path, csv_path: Path, value: str) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path

    candidates = [
        root_dir / path,
        csv_path.parent / path,
        root_dir / root_dir.name / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _prostate158_label_column(
    columns: Sequence[str],
    label_target: str,
    label_reader: int | str,
    label_modality: str | None,
) -> str:
    target = str(label_target).strip().lower()
    reader = str(label_reader).strip().lower()
    if reader.startswith("reader"):
        reader = reader.removeprefix("reader")

    if label_modality is None:
        modality = "adc" if target == "tumor" else "t2"
    else:
        modality = str(label_modality).strip().lower()

    preferred = f"{modality}_{target}_reader{reader}"
    if preferred in columns:
        return preferred

    suffix = f"_{target}_reader{reader}"
    matches = [col for col in columns if col.endswith(suffix)]
    if matches:
        return matches[0]

    raise ValueError(
        f"Could not find Prostate158 label column '{preferred}' in CSV. "
        f"Available columns: {', '.join(columns)}"
    )


def _require_prostate158_column(
    columns: Sequence[str],
    column_name: str,
    description: str,
) -> str:
    column = str(column_name).strip()
    if not column:
        raise ValueError(f"Missing Prostate158 {description} column name.")
    if column not in columns:
        raise ValueError(
            f"Could not find Prostate158 {description} column '{column}' in CSV. "
            f"Available columns: {', '.join(columns)}"
        )
    return column


def discover_prostate158_cases(
    root_dir: str | Path,
    split: str = "train",
    active_keys: Sequence[str] = MODALITY_KEYS,
    label_target: str = "tumor",
    label_reader: int | str = 1,
    label_modality: str | None = None,
    prostate_label_col: str | None = None,
) -> list[dict]:
    """
    Discover Prostate158 cases from the upstream CSV files.

    The official Prostate158 loader reads ``train.csv`` / ``valid.csv`` /
    ``test.csv`` and turns each row into a MONAI-style dict.  This helper
    adapts the same CSV contract to this repo's case-dict schema:

      - ``t2`` -> ``t2w``
      - ``adc`` -> ``adc``
      - ``dwi`` -> ``hbv`` with ``hbv_source="dwi"``
      - default label -> ``adc_tumor_reader1``

    The returned dicts can be passed directly to :class:`PiCaiDataset`.
    """
    root_dir = Path(root_dir)
    split = str(split).strip().lower()
    csv_path = _resolve_prostate158_csv(root_dir, split)

    cases: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        required = {"t2"}
        if "adc" in active_keys:
            required.add("adc")
        if "hbv" in active_keys:
            required.add("dwi")

        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(
                f"Prostate158 CSV {csv_path} is missing columns: {', '.join(missing)}"
            )

        label_col = _prostate158_label_column(
            columns=columns,
            label_target=label_target,
            label_reader=label_reader,
            label_modality=label_modality,
        )
        prostate_col: str | None = None
        if prostate_label_col not in (None, ""):
            prostate_col = _require_prostate158_column(
                columns=columns,
                column_name=str(prostate_label_col),
                description="prostate mask",
            )

        for row_idx, row in enumerate(reader, 1):
            t2_path = _resolve_prostate158_path(root_dir, csv_path, row["t2"])
            case_id = t2_path.parent.name or f"{split}_{row_idx:03d}"
            paths: dict = {
                "case_id": case_id,
                "t2w": t2_path,
                "label": _resolve_prostate158_path(root_dir, csv_path, row[label_col]),
                "source_dataset": "prostate158",
                "prostate158_split": split,
                "prostate158_label_col": label_col,
            }
            if prostate_col is not None:
                paths["prostate_label"] = _resolve_prostate158_path(
                    root_dir, csv_path, row[prostate_col]
                )
                paths["prostate158_prostate_label_col"] = prostate_col

            if "adc" in active_keys:
                paths["adc"] = _resolve_prostate158_path(root_dir, csv_path, row["adc"])
            if "hbv" in active_keys:
                paths["hbv"] = _resolve_prostate158_path(root_dir, csv_path, row["dwi"])
                paths["hbv_source"] = "dwi"

            missing_paths = [
                str(paths[key])
                for key in ("t2w", "adc", "hbv", "label", "prostate_label")
                if key in paths and paths[key] is not None and not Path(paths[key]).exists()
            ]
            if missing_paths:
                logger.warning(
                    "Prostate158 case %s skipped; missing file(s): %s",
                    case_id,
                    ", ".join(missing_paths),
                )
                continue

            cases.append(paths)

    logger.info(
        "Discovered %d Prostate158 %s cases in %s using label column '%s'",
        len(cases),
        split,
        root_dir,
        label_col,
    )
    return cases


def train_val_split(
    cases: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Randomly split *cases* into train and validation subsets.

    Parameters
    ----------
    cases        : full list of case dicts from `discover_cases`
    val_fraction : fraction of cases to reserve for validation [0, 1)
    seed         : random seed for reproducibility

    Returns
    -------
    (train_cases, val_cases)
    """
    rng = random.Random(seed)
    shuffled = cases.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


# ---------------------------------------------------------------------------
# Stratified split helpers
# ---------------------------------------------------------------------------

def _case_has_lesion(case: dict) -> bool:
    """
    Return True if *case* has a label file that contains at least one
    positive (non-zero) voxel.

    Reads the label via SimpleITK.  Returns False if the label path is
    None, the file is unreadable, or the file is empty (e.g. synthetic
    touch-files used in unit tests).
    """
    if case.get("label") is None:
        return False
    try:
        img = sitk.ReadImage(str(case["label"]))
        arr = sitk.GetArrayViewFromImage(img)
        return bool(arr.max() > 0)
    except Exception:
        return False


def _load_lesion_flag_cache(cache_path: Path | None) -> dict[str, bool]:
    """
    Load ``{case_id: has_lesion}`` flags from JSON sidecar.

    Returns an empty dict when no cache is configured, the cache file does
    not exist, or parsing fails.
    """
    if cache_path is None or not cache_path.exists():
        return {}

    try:
        with cache_path.open() as fh:
            flags = {str(k): bool(v) for k, v in json.load(fh).items()}
        logger.info("Loaded %d cached lesion flags from %s", len(flags), cache_path)
        return flags
    except Exception as exc:
        logger.warning("Could not read lesion flag cache %s: %s", cache_path, exc)
        return {}


def _persist_lesion_flag_cache(cache_path: Path | None, flags: dict[str, bool]) -> None:
    """
    Persist lesion-flag cache to disk.
    """
    if cache_path is None:
        return

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump(flags, fh)
        logger.info("Saved %d lesion flags to %s", len(flags), cache_path)
    except Exception as exc:
        logger.warning("Could not write lesion flag cache %s: %s", cache_path, exc)


def annotate_cases_with_lesion_flags(
    cases: list[dict],
    cache_path: Path | None = None,
) -> None:
    """
    Annotate each case dict in-place with ``has_lesion``.

    When *cache_path* is provided, known labels are reused from
    ``{case_id: has_lesion}`` JSON cache and newly-computed flags are written
    back.
    """
    flag_cache = _load_lesion_flag_cache(cache_path)

    new_flags = False
    seen_case_ids: set[str] = set()

    for case in cases:
        cid = str(case["case_id"])
        if cid in seen_case_ids:
            raise ValueError(f"Duplicate case_id in case list: {cid}")
        seen_case_ids.add(cid)

        if "has_lesion" in case:
            flag = bool(case["has_lesion"])
            case["has_lesion"] = flag
            if flag_cache.get(cid) != flag:
                flag_cache[cid] = flag
                new_flags = True
            continue

        if cid in flag_cache:
            case["has_lesion"] = flag_cache[cid]
            continue

        flag = _case_has_lesion(case)
        case["has_lesion"] = flag
        flag_cache[cid] = flag
        new_flags = True

    if new_flags:
        _persist_lesion_flag_cache(cache_path, flag_cache)


def default_split_manifest_path(base_output_dir: str | Path) -> Path:
    """
    Return the shared default split-manifest path for a run root.

    Examples:
      - ``/outputs/runs``         -> ``/outputs/splits/picai_train_val_split.json``
      - ``/outputs/pretrain_runs`` -> ``/outputs/splits/picai_train_val_split.json``
    """
    return Path(base_output_dir).parent / "splits" / "picai_train_val_split.json"


def _build_split_manifest(
    train_cases: list[dict],
    val_cases: list[dict],
    seed: int,
    val_fraction: float,
) -> dict[str, object]:
    return {
        "version": SPLIT_MANIFEST_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "num_total": len(train_cases) + len(val_cases),
        "num_train": len(train_cases),
        "num_val": len(val_cases),
        "train_case_ids": [str(c["case_id"]) for c in train_cases],
        "val_case_ids": [str(c["case_id"]) for c in val_cases],
    }


def save_split_manifest(path: Path, manifest: dict[str, object]) -> None:
    """
    Save split manifest JSON to *path*.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


def load_split_manifest(path: Path) -> dict[str, object]:
    """
    Load and minimally validate a split-manifest JSON file.
    """
    with path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    if not isinstance(manifest, dict):
        raise ValueError(f"Split manifest at {path} must contain a JSON object.")

    for key in ("train_case_ids", "val_case_ids"):
        value = manifest.get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(
                f"Split manifest at {path} must contain '{key}' as a list of strings."
            )

    return manifest


def _cases_from_split_manifest(
    cases: list[dict],
    manifest: dict[str, object],
    manifest_path: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Resolve train/val case lists from a split manifest.

    Raises when the manifest is inconsistent with the current discovered case
    list (e.g. stale IDs, overlaps, duplicates).
    """
    train_case_ids = [str(v) for v in manifest["train_case_ids"]]
    val_case_ids = [str(v) for v in manifest["val_case_ids"]]

    if len(train_case_ids) != len(set(train_case_ids)):
        raise ValueError(f"Split manifest has duplicate train IDs: {manifest_path}")
    if len(val_case_ids) != len(set(val_case_ids)):
        raise ValueError(f"Split manifest has duplicate val IDs: {manifest_path}")

    overlap = set(train_case_ids) & set(val_case_ids)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise ValueError(
            "Split manifest has case IDs in both train and val: "
            f"{sample} (path={manifest_path})"
        )

    case_map: dict[str, dict] = {}
    for case in cases:
        cid = str(case["case_id"])
        if cid in case_map:
            raise ValueError(f"Duplicate case_id discovered in dataset: {cid}")
        case_map[cid] = case

    manifest_ids = set(train_case_ids) | set(val_case_ids)
    discovered_ids = set(case_map.keys())

    missing_in_data = sorted(manifest_ids - discovered_ids)
    missing_in_manifest = sorted(discovered_ids - manifest_ids)

    if missing_in_data or missing_in_manifest:
        details: list[str] = []
        if missing_in_data:
            details.append(
                "manifest IDs missing in current data="
                f"{len(missing_in_data)} (e.g. {', '.join(missing_in_data[:5])})"
            )
        if missing_in_manifest:
            details.append(
                "current data IDs missing in manifest="
                f"{len(missing_in_manifest)} (e.g. {', '.join(missing_in_manifest[:5])})"
            )
        joined = " | ".join(details)
        raise RuntimeError(
            f"Split manifest does not match current dataset: {joined}. "
            "Use --new-split-manifest to regenerate."
        )

    train_cases = [case_map[cid] for cid in train_case_ids]
    val_cases = [case_map[cid] for cid in val_case_ids]
    return train_cases, val_cases


def resolve_train_val_split_from_manifest(
    cases: list[dict],
    val_fraction: float,
    seed: int,
    manifest_path: Path,
    new_split_manifest: bool = False,
    cache_path: Path | None = None,
) -> tuple[list[dict], list[dict], dict[str, object], bool]:
    """
    Resolve train/val split using a persisted manifest.

    Behavior:
      - If ``manifest_path`` does not exist, it is created automatically.
      - If ``new_split_manifest`` is True, the manifest is regenerated.
      - Otherwise, existing manifest IDs are reused exactly.

    Returns
    -------
    (train_cases, val_cases, manifest, was_created)
    """
    manifest_path = Path(manifest_path)
    existed_before = manifest_path.exists()
    should_create = new_split_manifest or not existed_before

    if should_create:
        train_cases, val_cases = stratified_train_val_split(
            cases,
            val_fraction=val_fraction,
            seed=seed,
            cache_path=cache_path,
        )
        manifest = _build_split_manifest(
            train_cases=train_cases,
            val_cases=val_cases,
            seed=seed,
            val_fraction=val_fraction,
        )
        save_split_manifest(manifest_path, manifest)
        logger.info(
            "%s split manifest at %s",
            "Regenerated" if new_split_manifest and existed_before else "Created",
            manifest_path,
        )
        return train_cases, val_cases, manifest, True

    manifest = load_split_manifest(manifest_path)
    annotate_cases_with_lesion_flags(cases, cache_path=cache_path)

    train_cases, val_cases = _cases_from_split_manifest(
        cases=cases,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    manifest_seed = manifest.get("seed")
    manifest_val_fraction = manifest.get("val_fraction")
    if (
        manifest_seed is not None
        and int(manifest_seed) != int(seed)
    ) or (
        manifest_val_fraction is not None
        and float(manifest_val_fraction) != float(val_fraction)
    ):
        logger.warning(
            "Using existing split manifest %s (seed=%s, val_fraction=%s) "
            "which differs from current config (seed=%s, val_fraction=%s).",
            manifest_path,
            manifest_seed,
            manifest_val_fraction,
            seed,
            val_fraction,
        )

    logger.info("Loaded existing split manifest from %s", manifest_path)
    return train_cases, val_cases, manifest, False


def stratified_train_val_split(
    cases: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
    cache_path: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Split *cases* into train and validation subsets while preserving the
    positive-to-negative case ratio in both splits.

    Each case dict is annotated in-place with a ``has_lesion`` boolean key
    so callers (e.g. the training loop's WeightedRandomSampler) can use it
    without re-reading any files.

    Parameters
    ----------
    cases        : full list of case dicts from `discover_cases`
    val_fraction : fraction of cases to reserve for validation [0, 1)
    seed         : random seed for reproducibility
    cache_path   : optional path to a JSON sidecar file that stores
                   ``{case_id: has_lesion}`` flags across runs.  When
                   provided, the file is read on startup (avoiding
                   ``sitk.ReadImage`` for already-seen cases) and written
                   back after any new cases are processed.

    Returns
    -------
    (train_cases, val_cases) — both lists preserve the dataset's
    positive/negative ratio within ±1 case.
    """
    annotate_cases_with_lesion_flags(cases, cache_path=cache_path)

    pos_cases = [c for c in cases if c["has_lesion"]]
    neg_cases = [c for c in cases if not c["has_lesion"]]

    logger.info(
        "Stratified split: %d positive / %d negative cases",
        len(pos_cases), len(neg_cases),
    )

    rng = random.Random(seed)

    def _split(subset: list[dict]) -> tuple[list[dict], list[dict]]:
        shuffled = subset.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
        return shuffled[n_val:], shuffled[:n_val]

    pos_train, pos_val = _split(pos_cases)
    neg_train, neg_val = _split(neg_cases)

    train_cases = pos_train + neg_train
    val_cases   = pos_val   + neg_val

    # Shuffle each split so the DataLoader sees interleaved pos/neg batches.
    rng.shuffle(train_cases)
    rng.shuffle(val_cases)

    logger.info(
        "Train: %d total (%d pos / %d neg) | "
        "Val: %d total (%d pos / %d neg)",
        len(train_cases), len(pos_train), len(neg_train),
        len(val_cases),   len(pos_val),   len(neg_val),
    )

    return train_cases, val_cases


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class _PreprocessedCase:
    image: torch.Tensor
    lesion_label: torch.Tensor
    prostate_label: torch.Tensor | None = None


class _ProstateROILocalizer:
    def __init__(
        self,
        localizer_run: str,
        target_spacing: tuple[float, ...],
        device: torch.device | str | None = None,
    ) -> None:
        cfg_path, ckpt_path = resolve_localizer_checkpoint(localizer_run)
        self.cfg = load_config(str(cfg_path))
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model: Any | None = build_model(self.cfg).to(self.device)
        load_checkpoint(ckpt_path, self.model, device=self.device)
        self.model.eval()
        self.target_spacing = target_spacing
        self.patch_size = tuple(int(v) for v in self.cfg.get("patch_size", [16, 128, 128]))
        self.sw_overlap = float(self.cfg.get("sw_overlap", 0.5))
        self.active_modalities = tuple(
            k for k in MODALITY_KEYS if self.cfg.get(f"use_{k}", True)
        )
        if not self.active_modalities:
            raise ValueError("ROI localizer config disables all modalities.")
        dwi_hbv_preprocess = self.cfg.get("dwi_hbv_preprocess", {}) or {}
        self.dwi_hbv_preprocess_enabled = bool(dwi_hbv_preprocess.get("enabled", False))
        clip = dwi_hbv_preprocess.get("clip_percentiles", (1.0, 99.5))
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            clip = (1.0, 99.5)
        self.dwi_hbv_clip_percentiles = (float(clip[0]), float(clip[1]))
        self.dwi_hbv_log1p = bool(dwi_hbv_preprocess.get("log1p", True))

    @staticmethod
    def _seg_logits(outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            outputs = outputs["seg"]
        if isinstance(outputs, list):
            outputs = outputs[0]
        if not isinstance(outputs, torch.Tensor):
            raise TypeError(
                "ROI localizer model output must be a Tensor, list[Tensor], or {'seg': Tensor}."
            )
        return outputs

    def predict_mask(self, case: dict) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ROI localizer has been released and can no longer run inference.")
        image = _prepare_case_image_tensor(
            case=case,
            target_spacing=self.target_spacing,
            active_modalities=self.active_modalities,
            dwi_hbv_preprocess_enabled=self.dwi_hbv_preprocess_enabled,
            dwi_hbv_clip_percentiles=self.dwi_hbv_clip_percentiles,
            dwi_hbv_log1p=self.dwi_hbv_log1p,
        ).to(self.device)
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=image.unsqueeze(0),
                roi_size=self.patch_size,
                sw_batch_size=1,
                predictor=lambda x: self._seg_logits(self.model(x)),
                overlap=self.sw_overlap,
            )
            probs = torch.sigmoid(logits.float())
        return probs[0, 0].cpu().numpy()

    def close(self) -> None:
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def _prepare_case_image_tensor(
    case: dict,
    target_spacing: tuple[float, ...],
    active_modalities: Sequence[str],
    dwi_hbv_preprocess_enabled: bool,
    dwi_hbv_clip_percentiles: tuple[float, float],
    dwi_hbv_log1p: bool,
) -> torch.Tensor:
    t2w_sitk = _load_volume(case["t2w"])
    secondary: dict[str, sitk.Image] = {}
    for key in active_modalities:
        if key == "t2w":
            continue
        secondary[key] = _resample_to_reference(
            _load_volume(case[key]), t2w_sitk, sitk.sitkLinear
        )

    t2w_sitk = _resample(t2w_sitk, target_spacing, sitk.sitkLinear)

    arrays: list[np.ndarray] = []
    for key in MODALITY_KEYS:
        if key not in active_modalities:
            continue
        if key == "t2w":
            arrays.append(_zscore_normalize(_to_numpy(t2w_sitk)))
        else:
            resampled = _resample(secondary[key], target_spacing, sitk.sitkLinear)
            arr = _to_numpy(resampled)
            if (
                key == "hbv"
                and case.get("hbv_source", "hbv") == "dwi"
                and dwi_hbv_preprocess_enabled
            ):
                arr = _preprocess_dwi_as_hbv(
                    arr,
                    clip_percentiles=dwi_hbv_clip_percentiles,
                    use_log1p=dwi_hbv_log1p,
                )
            arrays.append(_zscore_normalize(arr))

    image_np = np.stack(arrays, axis=0)
    return torch.from_numpy(image_np)


class PiCaiDataset(Dataset):
    """
    PyTorch Dataset for PI-CAI biparametric MRI prostate lesion segmentation.

    Each sample is loaded on-the-fly:
      1. T2w volume is always read from disk (co-registration reference).
      2. Active non-T2w modalities (ADC, HBV) are co-registered into the
         T2w physical space.
      3. All active modalities are resampled to *target_spacing*.
      4. Each modality is z-score normalised independently.
      5. Active modalities are stacked into a ``(C, D, H, W)`` image tensor
         in canonical order ``[T2w, ADC, HBV]``, where ``C`` equals the
         number of enabled modalities.
      6. The label mask (.nii.gz) is resampled with nearest-neighbour
         interpolation and binarised (any grade > 0 → 1).

    Returns
    -------
    dict with keys:
        "image"   : float32 tensor  (C, D, H, W)   active modalities in order
        "label"   : float32 tensor  (1, D, H, W)   binary lesion mask
        "case_id" : str
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        target_spacing: tuple[float, ...] = (3.0, 0.5, 0.5),
        transform: Optional[Callable] = None,
        cases: Optional[Sequence[dict]] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        active_modalities: Optional[Sequence[str]] = None,
        dwi_hbv_preprocess: Optional[dict[str, Any]] = None,
        cache_mode: str | None = None,
        cache_dir: str | Path | None = None,
        task: str = TASK_LESION_SEGMENTATION,
        roi_settings: ROISettings | None = None,
        include_full_resampled: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        images_dir        Root directory of PI-CAI images.
        labels_dir        Directory containing .nii.gz label masks.
        target_spacing    Desired voxel spacing in (z, y, x) mm order.
                          Default (3.0, 0.5, 0.5) preserves T2w in-plane
                          resolution at typical PI-CAI slice thickness.
        transform         Optional callable applied to the output dict.
                          Receives and returns {"image": Tensor, "label": Tensor, ...}.
                          Intended for MONAI-style random augmentations.
        cases             Pre-computed case list (skips discovery if provided).
                          Useful for passing pre-split train/val subsets.
        use_cache         Legacy compatibility flag.  When ``cache_mode`` is
                          not provided, ``True`` maps to ``cache_mode="ram"``
                          and ``False`` maps to ``cache_mode="none"``.
        cache_rate        Fraction of cases to cache, in [0.0, 1.0].  Cases are
                          selected deterministically (first ``ceil(N * rate)``
                          cases by index).  Default 1.0 caches the full dataset
                          in ``ram``/``storage`` modes.
        active_modalities Ordered sequence of modality keys to include in the
                           output image tensor.  Must be a subset of
                           ``("t2w", "adc", "hbv")``.  Defaults to all three
                           when ``None``.  The canonical channel order
                           ``[T2w, ADC, HBV]`` is always preserved regardless
                           of the order supplied here.
        dwi_hbv_preprocess Optional configuration for DWI-as-HBV preprocessing
                           when a case sets ``hbv_source="dwi"``.
                           Supported keys: ``enabled`` (bool),
                           ``clip_percentiles`` ([low, high]), ``log1p`` (bool).
        cache_mode        Dataset cache backend: ``"none"``, ``"ram"``, or
                          ``"storage"``.  ``ram`` keeps preprocessed tensors in
                          process memory; ``storage`` persists tensors on disk.
        cache_dir         Root directory for ``cache_mode="storage"``.  Ignored
                          for ``ram``/``none``.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.target_spacing = target_spacing
        self.transform = transform
        self.task = resolve_task({"task": task})
        self.roi_settings = roi_settings or resolve_roi_settings({})
        self.include_full_resampled = bool(include_full_resampled)
        self.cache_rate = float(cache_rate)
        if not 0.0 <= self.cache_rate <= 1.0:
            raise ValueError(
                f"cache_rate must be in [0.0, 1.0], got {self.cache_rate}."
            )

        if cache_mode is None:
            resolved_cache_mode = "ram" if use_cache else "none"
        else:
            resolved_cache_mode = str(cache_mode).strip().lower()
            if resolved_cache_mode not in {"none", "ram", "storage"}:
                raise ValueError(
                    f"Unsupported cache_mode='{cache_mode}'. "
                    "Expected one of: none, ram, storage."
                )
            if use_cache and resolved_cache_mode != "ram":
                logger.warning(
                    "use_cache=%s ignored because cache_mode='%s' takes precedence.",
                    use_cache,
                    resolved_cache_mode,
                )

        self.cache_mode = resolved_cache_mode
        self.use_cache = self.cache_mode == "ram"  # legacy compatibility alias
        self.cache_enabled = self.cache_mode in {"ram", "storage"}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_mode == "storage":
            if self.cache_dir is None:
                self.cache_dir = Path("cache") / "dataset_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Resolve active modalities: preserve canonical order, default to all.
        self.active_modalities: tuple[str, ...] = (
            tuple(k for k in MODALITY_KEYS if k in active_modalities)
            if active_modalities is not None
            else MODALITY_KEYS
        )
        self.requires_prostate_label = (
            self.task == TASK_PROSTATE_LOCALIZATION
            or self.roi_settings.mode == "gt_mask"
        )
        self._roi_localizer: _ProstateROILocalizer | None = None
        dwi_hbv_preprocess = dwi_hbv_preprocess or {}
        self.dwi_hbv_preprocess_enabled: bool = bool(
            dwi_hbv_preprocess.get("enabled", False)
        )
        clip = dwi_hbv_preprocess.get("clip_percentiles", (1.0, 99.5))
        if not isinstance(clip, (list, tuple)) or len(clip) != 2:
            clip = (1.0, 99.5)
        self.dwi_hbv_clip_percentiles: tuple[float, float] = (
            float(clip[0]),
            float(clip[1]),
        )
        self.dwi_hbv_log1p: bool = bool(dwi_hbv_preprocess.get("log1p", True))

        if cases is not None:
            self.cases = list(cases)
        else:
            self.cases = discover_cases(
                self.images_dir, self.labels_dir,
                active_keys=self.active_modalities,
            )

        # Determine which indices are eligible for caching.
        n_cache = math.ceil(len(self.cases) * self.cache_rate) if self.cache_enabled else 0
        self._cache_indices: set[int] = set(range(n_cache))
        # Mapping from case index → preprocessed tensors.
        # Used only for cache_mode="ram".  Pre-populated eagerly in the main
        # process so all DataLoader workers inherit a fully-warmed cache via
        # fork() — 100% hit rate from epoch 1.
        self._cache: dict[int, _PreprocessedCase] = {}

        cache_dir_msg = str(self.cache_dir) if self.cache_mode == "storage" else "-"
        logger.info(
            "PiCaiDataset ready: %d cases, modalities=%s, spacing=%s, "
            "transform=%s, task=%s, roi_mode=%s, cache_mode=%s "
            "(%.0f%% = %d cases), cache_dir=%s, dwi_hbv_preprocess=%s",
            len(self.cases),
            list(self.active_modalities),
            target_spacing,
            type(transform).__name__ if transform is not None else "None",
            self.task,
            self.roi_settings.mode,
            self.cache_mode,
            self.cache_rate * 100,
            n_cache,
            cache_dir_msg,
            "on" if self.dwi_hbv_preprocess_enabled else "off",
        )

        # Eager cache warmup: load all cacheable cases now, in the main process,
        # before DataLoader forks workers.  Workers inherit the populated dict and
        # achieve a 100% hit rate from the very first epoch without any per-worker
        # duplication of I/O.
        # ThreadPoolExecutor overlaps I/O and resampling across cases — SimpleITK
        # and numpy both release the GIL during their compute-heavy operations.
        if self.cache_mode == "ram" and n_cache > 0:
            warmup_workers = min(n_cache, os.cpu_count() or 4)
            logger.info(
                "Warming cache (%d threads): loading %d/%d cases into RAM …",
                warmup_workers, n_cache, len(self.cases),
            )

            def _load_one(i: int) -> tuple[int, _PreprocessedCase]:
                return i, self._load_and_preprocess(self.cases[i])

            with ThreadPoolExecutor(max_workers=warmup_workers) as pool:
                for idx, tensors in tqdm(
                    pool.map(_load_one, range(n_cache)),
                    total=n_cache,
                    desc="Cache warmup",
                    unit="case",
                    disable=not sys.stdout.isatty(),
                ):
                    self._cache[idx] = tensors

            logger.info("Cache warmup complete (%d cases loaded).", n_cache)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict | list[dict]:
        case = self.cases[idx]
        case_id: str = case["case_id"]
        cache_candidate = idx in self._cache_indices

        # ---------------------------------------------------------------------------
        # Dataset cache lookup
        # ---------------------------------------------------------------------------
        if self.cache_mode == "ram" and cache_candidate:
            if idx in self._cache:
                # Cache hit: return cloned tensors to prevent in-place transform
                # mutations from corrupting the stored originals.
                cached = self._clone_preprocessed(self._cache[idx])
            else:
                # Defensive fallback — should never reach here after eager warmup.
                logger.debug(
                    "Cache miss for idx=%d after eager warmup — loading from disk.", idx
                )
                cached = self._load_and_preprocess(case)
                self._cache[idx] = self._clone_preprocessed(cached)
        elif self.cache_mode == "storage" and cache_candidate:
            cache_path = self._storage_cache_path(case)
            cached = self._load_from_storage_cache(cache_path)
            if cached is None:
                cached = self._load_and_preprocess(case)
                self._save_to_storage_cache(cache_path, cached)
        else:
            cached = self._load_and_preprocess(case)

        sample = self._build_sample(case, cached)
        sample["case_id"] = case_id

        if self.transform is not None:
            result = self.transform(sample)
            # RandCropByPosNegLabeld with num_samples > 1 always returns a list
            # of dicts.  Return the full list so list_data_collate in the
            # DataLoader can assemble a proper batched tensor with shape
            # (B * num_samples, C, D, H, W).
            return result  # type: ignore[return-value]

        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _storage_cache_path(self, case: dict) -> Path:
        """
        Build a deterministic path for one storage-cache entry.
        """
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured for storage cache mode.")

        cache_meta: dict[str, Any] = {
            "case_id": str(case["case_id"]),
            "target_spacing": [float(v) for v in self.target_spacing],
            "active_modalities": list(self.active_modalities),
            "task": self.task,
            "roi_mode": self.roi_settings.mode,
            "dwi_hbv_preprocess": {
                "enabled": self.dwi_hbv_preprocess_enabled,
                "clip_percentiles": list(self.dwi_hbv_clip_percentiles),
                "log1p": self.dwi_hbv_log1p,
            },
            "hbv_source": str(case.get("hbv_source", "hbv")),
            "paths": {
                key: str(Path(case[key]).resolve())
                for key in ("t2w", "adc", "hbv")
                if key in case and case[key] is not None
            },
            "label_path": (
                str(Path(case["label"]).resolve())
                if case.get("label") is not None
                else None
            ),
            "prostate_label_path": (
                str(Path(case["prostate_label"]).resolve())
                if case.get("prostate_label") is not None
                else None
            ),
        }
        digest = hashlib.sha1(
            json.dumps(cache_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        safe_case_id = "".join(
            ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
            for ch in str(case["case_id"])
        )
        return self.cache_dir / f"{safe_case_id}_{digest}.pt"

    def _load_from_storage_cache(
        self,
        cache_path: Path,
    ) -> _PreprocessedCase | None:
        if not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path, map_location="cpu")
            image = payload.get("image")
            lesion_label = payload.get("lesion_label")
            prostate_label = payload.get("prostate_label")
            if not isinstance(image, torch.Tensor) or not isinstance(lesion_label, torch.Tensor):
                raise TypeError(
                    "cache payload is missing tensor keys 'image'/'lesion_label'"
                )
            if prostate_label is not None and not isinstance(prostate_label, torch.Tensor):
                raise TypeError("cache payload 'prostate_label' must be a Tensor or None")
            return _PreprocessedCase(
                image=image,
                lesion_label=lesion_label,
                prostate_label=prostate_label,
            )
        except Exception as exc:
            logger.warning(
                "Could not read storage cache entry %s (%s); recomputing.",
                cache_path,
                exc,
            )
            return None

    def _save_to_storage_cache(
        self,
        cache_path: Path,
        payload: _PreprocessedCase,
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(
            f"{cache_path.suffix}.tmp.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            torch.save(
                {
                    "image": payload.image,
                    "lesion_label": payload.lesion_label,
                    "prostate_label": payload.prostate_label,
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning("Could not write storage cache entry %s: %s", cache_path, exc)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def _clone_preprocessed(self, payload: _PreprocessedCase) -> _PreprocessedCase:
        return _PreprocessedCase(
            image=payload.image.clone(),
            lesion_label=payload.lesion_label.clone(),
            prostate_label=(
                payload.prostate_label.clone()
                if payload.prostate_label is not None
                else None
            ),
        )

    def _get_roi_localizer(self) -> _ProstateROILocalizer:
        if self._roi_localizer is None:
            self._roi_localizer = _ProstateROILocalizer(
                localizer_run=self.roi_settings.localizer_run,
                target_spacing=self.target_spacing,
            )
        return self._roi_localizer

    def release_roi_localizer(self) -> None:
        if self._roi_localizer is None:
            return
        self._roi_localizer.close()
        self._roi_localizer = None

    @staticmethod
    def _normalize_predicted_roi_bounds(
        payload: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        bounds_payload = payload.get("bounds", payload)
        if not isinstance(bounds_payload, dict):
            return None
        crop = crop_bounds_from_dict(bounds_payload)
        return crop.as_dict()

    def _load_predicted_roi_bounds_from_cache_path(
        self,
        case: dict,
    ) -> dict[str, Any] | None:
        cache_path_raw = case.get("_roi_bounds_predicted_path")
        if cache_path_raw in {None, ""}:
            return None
        cache_path = Path(cache_path_raw)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            bounds = self._normalize_predicted_roi_bounds(payload)
            if bounds is None:
                raise TypeError("cache payload does not contain a valid ROI bounds dict")
            case["_roi_bounds_predicted"] = bounds
            return bounds
        except Exception as exc:
            logger.warning(
                "Could not read predicted ROI bounds cache for case '%s' from %s (%s).",
                case.get("case_id", "<unknown>"),
                cache_path,
                exc,
            )
            return None

    def get_predicted_roi_bounds(
        self,
        case: dict,
        *,
        localizer: _ProstateROILocalizer | None = None,
    ) -> dict[str, Any]:
        cached_bounds = self._normalize_predicted_roi_bounds(
            case.get("_roi_bounds_predicted")
        )
        if cached_bounds is not None:
            case["_roi_bounds_predicted"] = cached_bounds
            return cached_bounds

        path_bounds = self._load_predicted_roi_bounds_from_cache_path(case)
        if path_bounds is not None:
            return path_bounds

        roi_mask = (
            localizer.predict_mask(case)
            if localizer is not None
            else self._get_roi_localizer().predict_mask(case)
        )
        roi_mask = binarize_mask(
            roi_mask,
            threshold=self.roi_settings.localizer_threshold,
        )
        if self.roi_settings.localizer_keep_largest_component:
            roi_mask = keep_largest_component(roi_mask)
        bounds = compute_crop_bounds(
            mask=roi_mask,
            spacing_zyx=self.target_spacing,
            margin_mm=self.roi_settings.margin_mm,
            min_size_vox=self.roi_settings.min_size_vox,
            fallback_to_full_volume=self.roi_settings.fallback_to_full_volume,
        ).as_dict()
        case["_roi_bounds_predicted"] = bounds
        return bounds

    def _build_sample(self, case: dict, payload: _PreprocessedCase) -> dict:
        image = payload.image.clone()
        lesion_label = payload.lesion_label.clone()
        prostate_label = (
            payload.prostate_label.clone() if payload.prostate_label is not None else None
        )

        if self.task == TASK_PROSTATE_LOCALIZATION:
            if prostate_label is None:
                raise ValueError(
                    f"Case '{case['case_id']}' is missing prostate_label for prostate localization."
                )
            return {
                "image": image,
                "label": prostate_label,
            }

        sample: dict[str, Any] = {
            "image": image,
            "label": lesion_label,
        }

        if not self.roi_settings.enabled:
            return sample

        if self.roi_settings.mode == "gt_mask":
            if prostate_label is None:
                raise ValueError(
                    f"Case '{case['case_id']}' is missing prostate_label required for roi.mode='gt_mask'."
                )
            roi_mask = prostate_label[0].cpu().numpy()
            bounds = compute_crop_bounds(
                mask=roi_mask,
                spacing_zyx=self.target_spacing,
                margin_mm=self.roi_settings.margin_mm,
                min_size_vox=self.roi_settings.min_size_vox,
                fallback_to_full_volume=self.roi_settings.fallback_to_full_volume,
            )
        else:
            bounds = self.get_predicted_roi_bounds(case)
        if not isinstance(bounds, dict):
            bounds = bounds.as_dict()
        sample["roi"] = bounds
        if self.include_full_resampled:
            sample["full_image"] = image.clone()
            sample["full_label"] = lesion_label.clone()

        sample["image"] = crop_tensor(image, bounds)
        sample["label"] = crop_tensor(lesion_label, bounds)
        return sample

    def _load_and_preprocess(
        self, case: dict
    ) -> _PreprocessedCase:
        """
        Run full SimpleITK I/O, co-registration, resampling, and
        z-score normalisation for one case.

        T2w is always loaded as the co-registration reference frame for
        any secondary modality and as the reference grid for label
        resampling, regardless of whether ``"t2w"`` is in
        ``self.active_modalities``.

        Returns
        -------
        (image, label) as float32 tensors shaped (C, D, H, W) and
        (1, D, H, W), where C = ``len(self.active_modalities)``.
        """
        image = _prepare_case_image_tensor(
            case=case,
            target_spacing=self.target_spacing,
            active_modalities=self.active_modalities,
            dwi_hbv_preprocess_enabled=self.dwi_hbv_preprocess_enabled,
            dwi_hbv_clip_percentiles=self.dwi_hbv_clip_percentiles,
            dwi_hbv_log1p=self.dwi_hbv_log1p,
        )
        image_np = image.numpy()
        t2w_sitk = _resample(_load_volume(case["t2w"]), self.target_spacing, sitk.sitkLinear)

        # 6. Load and binarise label.
        # Resample the label into the *resampled* T2w's exact voxel grid rather
        # than performing an independent global resample.  Independent resampling
        # can produce a slightly different integer output size (due to rounding in
        # np.round) even when the target spacing is identical, causing a shape
        # mismatch that crashes RandCropByPosNegLabeld.  Using t2w_sitk as the
        # reference image guarantees the label and image always share the same
        # (D, H, W) dimensions.
        if case["label"] is not None:
            lbl_sitk = _load_volume(case["label"])
            lbl_sitk = _resample_to_reference(
                lbl_sitk,
                t2w_sitk,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0.0,
            )
            label_np = _to_numpy(lbl_sitk)
            label_np = (label_np > 0).astype(np.float32)  # binarise
        else:
            # Inference / unlabelled case: return all-zero mask
            label_np = np.zeros(image_np.shape[1:], dtype=np.float32)

        label_np = label_np[np.newaxis]  # (1, D, H, W)

        prostate_label_t: torch.Tensor | None = None
        if self.requires_prostate_label:
            prostate_path = case.get("prostate_label")
            if prostate_path is None:
                raise ValueError(
                    f"Case '{case['case_id']}' is missing prostate_label but the current task/ROI requires it."
                )
            prostate_sitk = _load_volume(prostate_path)
            prostate_sitk = _resample_to_reference(
                prostate_sitk,
                t2w_sitk,
                interpolator=sitk.sitkNearestNeighbor,
                default_value=0.0,
            )
            prostate_np = (_to_numpy(prostate_sitk) > 0).astype(np.float32)[np.newaxis]
            prostate_label_t = torch.from_numpy(prostate_np)

        return _PreprocessedCase(
            image=torch.from_numpy(image_np),
            lesion_label=torch.from_numpy(label_np),
            prostate_label=prostate_label_t,
        )
