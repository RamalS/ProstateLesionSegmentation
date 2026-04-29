"""
2D JPG/PNG dataset for Deconver training.

Expected layout
---------------
images/
  <case_id>.jpg
labels/
  <case_id>.png

Pairs are matched by basename (``case_id``). Labels are binarized as
``mask > 0``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _normalize_ext(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _iter_files(root: Path, ext: str, recursive: bool) -> list[Path]:
    pattern = f"*{ext}"
    if recursive:
        return sorted(p for p in root.rglob(pattern) if p.is_file())
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _mask_has_lesion(mask_path: Path) -> bool:
    with Image.open(mask_path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return bool((arr > 0).any())


def discover_cases_2d(
    images_dir: str | Path,
    labels_dir: str | Path,
    image_ext: str = ".jpg",
    label_ext: str = ".png",
    recursive: bool = False,
    strict: bool = True,
) -> list[dict]:
    """
    Discover 2D image/mask pairs and return case dicts.

    Each case dict contains:
      - ``case_id``: basename without extension
      - ``image``: path to JPG image
      - ``label``: path to PNG label
      - ``has_lesion``: True if any mask pixel is >0
    """
    images_root = Path(images_dir)
    labels_root = Path(labels_dir)
    image_ext = _normalize_ext(image_ext)
    label_ext = _normalize_ext(label_ext)

    image_paths = _iter_files(images_root, image_ext, recursive=recursive)
    label_paths = _iter_files(labels_root, label_ext, recursive=recursive)

    label_by_id: dict[str, Path] = {}
    for path in label_paths:
        case_id = path.stem
        if case_id in label_by_id:
            raise ValueError(f"Duplicate label basename detected: {case_id}")
        label_by_id[case_id] = path

    cases: list[dict] = []
    missing_labels: list[str] = []
    seen_ids: set[str] = set()

    for image_path in image_paths:
        case_id = image_path.stem
        if case_id in seen_ids:
            raise ValueError(f"Duplicate image basename detected: {case_id}")
        seen_ids.add(case_id)

        label_path = label_by_id.get(case_id)
        if label_path is None:
            missing_labels.append(case_id)
            continue

        cases.append(
            {
                "case_id": case_id,
                "image": image_path,
                "label": label_path,
                "has_lesion": _mask_has_lesion(label_path),
            }
        )

    orphan_labels = sorted(set(label_by_id) - seen_ids)

    if strict and missing_labels:
        sample = ", ".join(missing_labels[:10])
        raise RuntimeError(
            f"Missing labels for {len(missing_labels)} image(s): {sample}. "
            "Expected basename-matched PNG masks."
        )
    if strict and orphan_labels:
        sample = ", ".join(orphan_labels[:10])
        raise RuntimeError(
            f"Found {len(orphan_labels)} label(s) without images: {sample}. "
            "Expected basename-matched JPG images."
        )

    if not strict and missing_labels:
        logger.warning(
            "Skipping %d image(s) without matching labels.",
            len(missing_labels),
        )
    if not strict and orphan_labels:
        logger.warning(
            "Found %d orphan label(s) without matching images.",
            len(orphan_labels),
        )

    logger.info(
        "Discovered %d 2D cases in %s (labels: %s)",
        len(cases),
        images_root,
        labels_root,
    )
    return cases


def _zscore_normalize_2d(arr: np.ndarray, clip: float = 5.0) -> np.ndarray:
    foreground = arr[arr > 0]
    if foreground.size < 10:
        foreground = arr

    mean = foreground.mean()
    std = foreground.std()
    if std < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - mean) / std
    return np.clip(out, -clip, clip).astype(np.float32)


class ImageMask2DDataset(Dataset):
    """
    2D image segmentation dataset backed by JPG images and PNG labels.

    Returns sample dicts with:
      - ``image``: float32 tensor (C, H, W)
      - ``label``: float32 tensor (1, H, W)
      - ``case_id``: str
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        transform: Optional[Callable] = None,
        cases: Optional[Sequence[dict]] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        input_channels: int = 1,
        image_ext: str = ".jpg",
        label_ext: str = ".png",
        recursive: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.use_cache = bool(use_cache)
        self.cache_rate = float(cache_rate)
        self.input_channels = int(input_channels)
        self.image_ext = _normalize_ext(image_ext)
        self.label_ext = _normalize_ext(label_ext)
        self.recursive = bool(recursive)

        if self.input_channels not in (1, 3):
            raise ValueError(
                f"input_channels must be 1 or 3, got {self.input_channels}"
            )

        if cases is not None:
            self.cases = list(cases)
        else:
            self.cases = discover_cases_2d(
                self.images_dir,
                self.labels_dir,
                image_ext=self.image_ext,
                label_ext=self.label_ext,
                recursive=self.recursive,
                strict=True,
            )

        n_cache = math.ceil(len(self.cases) * self.cache_rate) if self.use_cache else 0
        self._cache_indices: set[int] = set(range(n_cache))
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        logger.info(
            "ImageMask2DDataset ready: %d cases, input_channels=%d, transform=%s, "
            "cache=%s (%.0f%% = %d cases)",
            len(self.cases),
            self.input_channels,
            type(transform).__name__ if transform is not None else "None",
            self.use_cache,
            self.cache_rate * 100.0,
            n_cache,
        )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict | list[dict]:
        case = self.cases[idx]
        case_id = str(case["case_id"])

        if self.use_cache and idx in self._cache_indices:
            if idx in self._cache:
                image, label = self._cache[idx]
                image = image.clone()
                label = label.clone()
            else:
                image, label = self._load_and_preprocess(case)
                self._cache[idx] = (image.clone(), label.clone())
        else:
            image, label = self._load_and_preprocess(case)

        sample: dict = {"image": image, "label": label, "case_id": case_id}
        if self.transform is not None:
            return self.transform(sample)  # type: ignore[return-value]
        return sample

    def _load_and_preprocess(self, case: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = Path(case["image"])
        label_path = Path(case["label"])

        with Image.open(image_path) as img:
            if self.input_channels == 1:
                img_np = np.asarray(img.convert("L"), dtype=np.float32)
                img_np = _zscore_normalize_2d(img_np)
                image_np = img_np[np.newaxis, ...]  # (1, H, W)
            else:
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
                channels = [
                    _zscore_normalize_2d(rgb[..., c])
                    for c in range(3)
                ]
                image_np = np.stack(channels, axis=0)  # (3, H, W)

        with Image.open(label_path) as lbl:
            label_np = np.asarray(lbl.convert("L"), dtype=np.uint8)
        label_np = (label_np > 0).astype(np.float32)  # (H, W)

        if image_np.shape[1:] != label_np.shape:
            raise ValueError(
                "Image/label shape mismatch for case "
                f"{case['case_id']}: image={image_np.shape[1:]}, label={label_np.shape}"
            )

        label_np = label_np[np.newaxis, ...]  # (1, H, W)
        return torch.from_numpy(image_np), torch.from_numpy(label_np)


__all__ = [
    "ImageMask2DDataset",
    "discover_cases_2d",
]

