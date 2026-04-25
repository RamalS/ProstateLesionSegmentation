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
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from tqdm import tqdm

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# PI-CAI modality suffixes in channel order [T2w, ADC, HBV]
MODALITY_SUFFIXES = ("_t2w.mha", "_adc.mha", "_hbv.mha")
MODALITY_KEYS = ("t2w", "adc", "hbv")
LABEL_SUFFIX = ".nii.gz"


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
    # Load cached lesion flags from JSON sidecar (if available).
    flag_cache: dict[str, bool] = {}
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open() as fh:
                flag_cache = {k: bool(v) for k, v in json.load(fh).items()}
            logger.info(
                "Loaded %d cached lesion flags from %s", len(flag_cache), cache_path
            )
        except Exception as exc:
            logger.warning("Could not read lesion flag cache %s: %s", cache_path, exc)

    # Annotate cases with positivity flag (reads label files only for cache misses).
    new_flags = False
    for case in cases:
        if "has_lesion" not in case:
            cid = case["case_id"]
            if cid in flag_cache:
                case["has_lesion"] = flag_cache[cid]
            else:
                case["has_lesion"] = _case_has_lesion(case)
                flag_cache[cid] = case["has_lesion"]
                new_flags = True

    # Persist updated flag cache if anything new was computed.
    if new_flags and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w") as fh:
                json.dump(flag_cache, fh)
            logger.info("Saved %d lesion flags to %s", len(flag_cache), cache_path)
        except Exception as exc:
            logger.warning("Could not write lesion flag cache %s: %s", cache_path, exc)

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
        use_cache         If True, cache preprocessed (image, label) tensor pairs
                          in worker memory after the first access.  Subsequent
                          epochs skip all SimpleITK I/O and resampling for cached
                          cases.  Requires ``persistent_workers=True`` in
                          DataLoader so the worker processes (and their caches)
                          survive across epochs.
        cache_rate        Fraction of cases to cache, in [0.0, 1.0].  Cases are
                          selected deterministically (first ``ceil(N * rate)``
                          cases by index).  Default 1.0 caches the full dataset.
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
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.target_spacing = target_spacing
        self.transform = transform
        self.use_cache = use_cache
        self.cache_rate = float(cache_rate)
        # Resolve active modalities: preserve canonical order, default to all.
        self.active_modalities: tuple[str, ...] = (
            tuple(k for k in MODALITY_KEYS if k in active_modalities)
            if active_modalities is not None
            else MODALITY_KEYS
        )
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
        n_cache = math.ceil(len(self.cases) * self.cache_rate) if use_cache else 0
        self._cache_indices: set[int] = set(range(n_cache))
        # Mapping from case index → (image_tensor, label_tensor).
        # Pre-populated eagerly in the main process so all DataLoader workers
        # inherit a fully-warmed cache via fork() — 100% hit rate from epoch 1.
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        logger.info(
            "PiCaiDataset ready: %d cases, modalities=%s, spacing=%s, "
            "transform=%s, cache=%s (%.0f%% = %d cases), dwi_hbv_preprocess=%s",
            len(self.cases),
            list(self.active_modalities),
            target_spacing,
            type(transform).__name__ if transform is not None else "None",
            use_cache,
            self.cache_rate * 100,
            n_cache,
            "on" if self.dwi_hbv_preprocess_enabled else "off",
        )

        # Eager cache warmup: load all cacheable cases now, in the main process,
        # before DataLoader forks workers.  Workers inherit the populated dict and
        # achieve a 100% hit rate from the very first epoch without any per-worker
        # duplication of I/O.
        # ThreadPoolExecutor overlaps I/O and resampling across cases — SimpleITK
        # and numpy both release the GIL during their compute-heavy operations.
        if use_cache and n_cache > 0:
            warmup_workers = min(n_cache, os.cpu_count() or 4)
            logger.info(
                "Warming cache (%d threads): loading %d/%d cases into RAM …",
                warmup_workers, n_cache, len(self.cases),
            )

            def _load_one(i: int) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
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

        # ---------------------------------------------------------------------------
        # In-memory cache lookup
        # ---------------------------------------------------------------------------
        if self.use_cache and idx in self._cache_indices:
            if idx in self._cache:
                # Cache hit: return cloned tensors to prevent in-place transform
                # mutations from corrupting the stored originals.
                image, label = self._cache[idx]
                image = image.clone()
                label = label.clone()
            else:
                # Defensive fallback — should never reach here after eager warmup.
                logger.debug(
                    "Cache miss for idx=%d after eager warmup — loading from disk.", idx
                )
                image, label = self._load_and_preprocess(case)
                self._cache[idx] = (image.clone(), label.clone())
        else:
            image, label = self._load_and_preprocess(case)

        sample: dict = {
            "image": image,
            "label": label,
            "case_id": case_id,
        }

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

    def _load_and_preprocess(
        self, case: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        # 1. Always load T2w — co-registration + label-resampling reference.
        t2w_sitk = _load_volume(case["t2w"])

        # 2. Load active non-T2w modalities and co-register to T2w space.
        secondary: dict[str, sitk.Image] = {}
        for key in self.active_modalities:
            if key == "t2w":
                continue
            secondary[key] = _resample_to_reference(
                _load_volume(case[key]), t2w_sitk, sitk.sitkLinear
            )

        # 3. Resample T2w to target spacing (always needed for label grid).
        t2w_sitk = _resample(t2w_sitk, self.target_spacing, sitk.sitkLinear)

        # 4. Resample + normalise each active modality; collect in canonical
        #    order (T2w → ADC → HBV) regardless of how active_modalities is
        #    ordered.
        arrays: list[np.ndarray] = []
        for key in MODALITY_KEYS:
            if key not in self.active_modalities:
                continue
            if key == "t2w":
                arrays.append(_zscore_normalize(_to_numpy(t2w_sitk)))
            else:
                resampled = _resample(
                    secondary[key], self.target_spacing, sitk.sitkLinear
                )
                arr = _to_numpy(resampled)
                if (
                    key == "hbv"
                    and case.get("hbv_source", "hbv") == "dwi"
                    and self.dwi_hbv_preprocess_enabled
                ):
                    arr = _preprocess_dwi_as_hbv(
                        arr,
                        clip_percentiles=self.dwi_hbv_clip_percentiles,
                        use_log1p=self.dwi_hbv_log1p,
                    )
                arrays.append(_zscore_normalize(arr))

        # 5. Stack → (C, D, H, W)
        image_np = np.stack(arrays, axis=0)

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

        return torch.from_numpy(image_np), torch.from_numpy(label_np)
