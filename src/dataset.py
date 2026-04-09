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

import logging
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# PI-CAI modality suffixes in channel order [T2w, ADC, HBV]
MODALITY_SUFFIXES = ("_t2w.mha", "_adc.mha", "_hbv.mha")
MODALITY_KEYS = ("t2w", "adc", "hbv")
LABEL_SUFFIX = ".nii.gz"


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
) -> list[dict]:
    """
    Discover cases from a flat layout::

        <images_dir>/
            <case_id>_t2w.mha
            <case_id>_adc.mha
            <case_id>_hbv.mha
            ...   (other modalities such as _cor/_sag are ignored)

    Case IDs are derived by stripping ``_t2w.mha`` from each T2w filename.
    """
    t2w_suffix = MODALITY_SUFFIXES[0]  # "_t2w.mha"
    cases: list[dict] = []

    for t2w_path in sorted(images_dir.glob(f"*{t2w_suffix}")):
        case_id = t2w_path.name[: -len(t2w_suffix)]
        paths: dict = {"case_id": case_id, "t2w": t2w_path}
        complete = True

        for suffix, key in zip(MODALITY_SUFFIXES[1:], MODALITY_KEYS[1:]):
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
) -> list[dict]:
    """
    Discover cases from a nested layout::

        <images_dir>/
            <patient_id>/
                <case_id>/
                    <case_id>_t2w.mha
                    <case_id>_adc.mha
                    <case_id>_hbv.mha
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

            for suffix, key in zip(MODALITY_SUFFIXES, MODALITY_KEYS):
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

    Each returned dict has keys:
        case_id : str          e.g. ``"10000_1000000"``
        t2w     : Path
        adc     : Path
        hbv     : Path
        label   : Path | None  (None for unlabelled cases)
    """
    flat = _is_flat_layout(images_dir)
    layout = "flat" if flat else "nested"
    logger.info("Detected %s image layout in %s", layout, images_dir)

    if flat:
        cases = _discover_cases_flat(images_dir, labels_dir)
    else:
        cases = _discover_cases_nested(images_dir, labels_dir)

    logger.info("Discovered %d cases (%s layout)", len(cases), layout)
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

    Returns
    -------
    (train_cases, val_cases) — both lists preserve the dataset's
    positive/negative ratio within ±1 case.
    """
    # Annotate cases with positivity flag (reads label files once).
    for case in cases:
        if "has_lesion" not in case:
            case["has_lesion"] = _case_has_lesion(case)

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
      1. T2w, ADC, HBV volumes are read from .mha files.
      2. ADC and HBV are resampled into the T2w physical space (registration).
      3. All modalities are resampled to *target_spacing*.
      4. Each modality is z-score normalised independently.
      5. Modalities are stacked into a (3, D, H, W) image tensor.
      6. The label mask (.nii.gz) is resampled with nearest-neighbour
         interpolation and binarised (any grade > 0 → 1).

    Returns
    -------
    dict with keys:
        "image"   : float32 tensor  (3, D, H, W)   [T2w, ADC, HBV]
        "label"   : float32 tensor  (1, D, H, W)   [binary lesion mask]
        "case_id" : str
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        target_spacing: tuple[float, ...] = (3.0, 0.5, 0.5),
        transform: Optional[Callable] = None,
        cases: Optional[Sequence[dict]] = None,
    ) -> None:
        """
        Parameters
        ----------
        images_dir      Root directory of PI-CAI images.
        labels_dir      Directory containing .nii.gz label masks.
        target_spacing  Desired voxel spacing in (z, y, x) mm order.
                        Default (3.0, 0.5, 0.5) preserves T2w in-plane
                        resolution at typical PI-CAI slice thickness.
        transform       Optional callable applied to the output dict.
                        Receives and returns {"image": Tensor, "label": Tensor, ...}.
                        Intended for MONAI-style random augmentations.
        cases           Pre-computed case list (skips discovery if provided).
                        Useful for passing pre-split train/val subsets.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.target_spacing = target_spacing
        self.transform = transform

        if cases is not None:
            self.cases = list(cases)
        else:
            self.cases = discover_cases(self.images_dir, self.labels_dir)

        logger.info(
            "PiCaiDataset ready: %d cases, spacing=%s, transform=%s",
            len(self.cases),
            target_spacing,
            type(transform).__name__ if transform is not None else "None",
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.cases[idx]
        case_id: str = case["case_id"]

        # 1. Load raw volumes
        t2w_sitk = _load_volume(case["t2w"])
        adc_sitk = _load_volume(case["adc"])
        hbv_sitk = _load_volume(case["hbv"])

        # 2. Co-register ADC + HBV into T2w physical space
        adc_sitk = _resample_to_reference(adc_sitk, t2w_sitk, sitk.sitkLinear)
        hbv_sitk = _resample_to_reference(hbv_sitk, t2w_sitk, sitk.sitkLinear)

        # 3. Resample all modalities to target spacing
        t2w_sitk = _resample(t2w_sitk, self.target_spacing, sitk.sitkLinear)
        adc_sitk = _resample(adc_sitk, self.target_spacing, sitk.sitkLinear)
        hbv_sitk = _resample(hbv_sitk, self.target_spacing, sitk.sitkLinear)

        # 4. Convert to numpy and z-score normalise each modality
        t2w = _zscore_normalize(_to_numpy(t2w_sitk))
        adc = _zscore_normalize(_to_numpy(adc_sitk))
        hbv = _zscore_normalize(_to_numpy(hbv_sitk))

        # 5. Stack → (3, D, H, W)
        image = np.stack([t2w, adc, hbv], axis=0)

        # 6. Load and binarise label
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
            label = _to_numpy(lbl_sitk)
            label = (label > 0).astype(np.float32)  # binarise
        else:
            # Inference / unlabelled case: return all-zero mask
            label = np.zeros(image.shape[1:], dtype=np.float32)

        label = label[np.newaxis]  # (1, D, H, W)

        sample: dict = {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "case_id": case_id,
        }

        if self.transform is not None:
            result = self.transform(sample)
            # RandCropByPosNegLabeld always returns a list regardless of
            # num_samples; pick the first patch so __getitem__ always
            # returns a single dict for the standard PyTorch DataLoader.
            sample = result[0] if isinstance(result, list) else result

        return sample
