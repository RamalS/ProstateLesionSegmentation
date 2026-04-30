from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects
from torch.utils.data import Dataset

_PIL_RESAMPLING = getattr(Image, "Resampling", Image)


def build_tissue_mask_from_image(
    image_rgb: np.ndarray,
    close_radius: int = 3,
    min_object_size: int = 4096,
    min_hole_size: int = 4096,
) -> np.ndarray:
    """Return binary tissue mask where 1=tissue, 0=background."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape [H, W, 3].")

    gray = image_rgb.astype(np.float32).mean(axis=2)
    th = threshold_otsu(gray)
    tissue = gray < th
    tissue = closing(tissue, footprint=disk(close_radius))
    # skimage>=0.26 deprecates min_size/area_threshold in favor of max_size
    # with <= semantics; use (threshold - 1) to keep prior strict-< behavior.
    obj_max_size = max(0, int(min_object_size) - 1)
    hole_max_size = max(0, int(min_hole_size) - 1)
    tissue = remove_small_objects(tissue, max_size=obj_max_size)
    tissue = remove_small_holes(tissue, max_size=hole_max_size)
    return tissue.astype(np.uint8)


def clean_ignore_mask(
    ignore_mask: np.ndarray,
    tissue_mask: np.ndarray,
    enforce_background_ignore: bool = True,
) -> np.ndarray:
    """Clean ignore mask and optionally force non-tissue/background to ignore."""
    if ignore_mask.shape != tissue_mask.shape:
        raise ValueError("ignore_mask and tissue_mask must have same shape.")
    out = (ignore_mask > 0).astype(np.uint8)
    if enforce_background_ignore:
        out[tissue_mask == 0] = 1
    return out


class GleasonConsensusDataset(Dataset):
    """
    Dataset for consensus outputs + source histology image.

    Expects:
      data_root/
        Trains_imgs|Test_imgs/<image_id>.jpg
      consensus_root/<image_id>/
        consensus_hard_mask.png
        consensus_probs_compact.npz  (key: 'probs', [4,H,W], float16/float32)
        ignore_mask.png
        qc_report.json (optional)
    """

    def __init__(
        self,
        data_root: str | Path,
        consensus_root: str | Path,
        image_subdirs: tuple[str, ...] = ("Trains_imgs", "Test_imgs"),
        transform: Callable | None = None,
        renormalize_probs: bool = True,
        enforce_background_ignore: bool = True,
        otsu_close_radius: int = 3,
        otsu_min_object_size: int = 4096,
        otsu_min_hole_size: int = 4096,
        probs_eps: float = 1e-8,
        load_qc_report: bool = False,
        max_long_side: int | None = None,
        resize_divisor: int = 1,
    ) -> None:
        self.data_root = Path(data_root)
        self.consensus_root = Path(consensus_root)
        self.image_subdirs = image_subdirs
        self.transform = transform
        self.renormalize_probs = renormalize_probs
        self.enforce_background_ignore = enforce_background_ignore
        self.otsu_close_radius = int(otsu_close_radius)
        self.otsu_min_object_size = int(otsu_min_object_size)
        self.otsu_min_hole_size = int(otsu_min_hole_size)
        self.probs_eps = float(probs_eps)
        self.load_qc_report = bool(load_qc_report)
        self.max_long_side = int(max_long_side) if max_long_side and int(max_long_side) > 0 else None
        self.resize_divisor = max(1, int(resize_divisor))

        self.items = self._discover_items()
        if not self.items:
            raise RuntimeError("No valid consensus samples found.")

    def _find_image_path(self, image_id: str) -> tuple[Path, str] | None:
        for sub in self.image_subdirs:
            for ext in (".jpg", ".png", ".jpeg"):
                p = self.data_root / sub / f"{image_id}{ext}"
                if p.exists():
                    return p, sub
        return None

    def _discover_items(self) -> list[dict]:
        items: list[dict] = []
        if not self.consensus_root.exists():
            return items

        for d in sorted(self.consensus_root.iterdir()):
            if not d.is_dir():
                continue
            image_id = d.name
            image_path_meta = self._find_image_path(image_id)
            if image_path_meta is None:
                continue
            image_path, image_subdir = image_path_meta

            hard = d / "consensus_hard_mask.png"
            soft = d / "consensus_probs_compact.npz"
            ign = d / "ignore_mask.png"
            qc = d / "qc_report.json"
            if not (hard.exists() and soft.exists() and ign.exists()):
                continue

            items.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_subdir": image_subdir,
                    "hard_path": hard,
                    "soft_path": soft,
                    "ignore_path": ign,
                    "qc_path": qc,
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _load_probs(self, path: Path, image_id: str) -> np.ndarray:
        probs = np.load(path)["probs"].astype(np.float32)
        if probs.ndim != 3 or probs.shape[0] != 4:
            raise ValueError(f"Invalid probs shape for {image_id}: {probs.shape}")

        probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)

        if self.renormalize_probs:
            probs_sum = probs.sum(axis=0, keepdims=True)
            nonzero = probs_sum >= self.probs_eps
            probs = np.divide(
                probs,
                np.clip(probs_sum, self.probs_eps, None),
                out=np.zeros_like(probs, dtype=np.float32),
                where=nonzero,
            )
            # Fallback pixels with degenerate all-zero distributions to benign.
            zero_mask = ~nonzero[0]
            if zero_mask.any():
                probs[0, zero_mask] = 1.0

        return probs

    def _maybe_resize_sample(
        self,
        image: np.ndarray,
        hard: np.ndarray,
        ignore: np.ndarray,
        probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.max_long_side is None:
            return image, hard, ignore, probs

        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.max_long_side:
            return image, hard, ignore, probs

        scale = float(self.max_long_side) / float(longest)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if self.resize_divisor > 1:
            # Deconver's U-Net skip concatenation expects aligned shapes after
            # repeated stride-2 down/up sampling. Snap to divisor to prevent
            # off-by-one shape mismatches in the decoder.
            new_h = max(self.resize_divisor, (new_h // self.resize_divisor) * self.resize_divisor)
            new_w = max(self.resize_divisor, (new_w // self.resize_divisor) * self.resize_divisor)

        image_rs = np.array(
            Image.fromarray(image).resize(
                (new_w, new_h),
                resample=_PIL_RESAMPLING.BILINEAR,
            ),
            dtype=np.uint8,
            copy=True,
        )
        hard_rs = np.array(
            Image.fromarray(hard).resize(
                (new_w, new_h),
                resample=_PIL_RESAMPLING.NEAREST,
            ),
            dtype=np.uint8,
            copy=True,
        )
        ignore_rs = np.array(
            Image.fromarray(ignore).resize(
                (new_w, new_h),
                resample=_PIL_RESAMPLING.NEAREST,
            ),
            dtype=np.uint8,
            copy=True,
        )

        probs_t = torch.from_numpy(probs).unsqueeze(0)
        probs_rs = F.interpolate(
            probs_t,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).numpy()
        probs_rs = np.clip(np.nan_to_num(probs_rs, nan=0.0, posinf=1.0, neginf=0.0), 0.0, None)

        if self.renormalize_probs:
            probs_sum = probs_rs.sum(axis=0, keepdims=True)
            nonzero = probs_sum >= self.probs_eps
            probs_rs = np.divide(
                probs_rs,
                np.clip(probs_sum, self.probs_eps, None),
                out=np.zeros_like(probs_rs, dtype=np.float32),
                where=nonzero,
            )
            zero_mask = ~nonzero[0]
            if zero_mask.any():
                probs_rs[0, zero_mask] = 1.0
        else:
            probs_rs = probs_rs.astype(np.float32, copy=False)

        return image_rs, hard_rs, ignore_rs, probs_rs

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]

        image = np.array(Image.open(item["image_path"]).convert("RGB"), dtype=np.uint8)
        hard = np.array(Image.open(item["hard_path"]), dtype=np.uint8)
        ignore = np.array(Image.open(item["ignore_path"]), dtype=np.uint8)
        probs = self._load_probs(item["soft_path"], image_id=str(item["image_id"]))
        image, hard, ignore, probs = self._maybe_resize_sample(
            image=image,
            hard=hard,
            ignore=ignore,
            probs=probs,
        )

        if hard.shape != image.shape[:2] or ignore.shape != image.shape[:2]:
            raise ValueError(f"Shape mismatch for {item['image_id']}")

        tissue = build_tissue_mask_from_image(
            image,
            close_radius=self.otsu_close_radius,
            min_object_size=self.otsu_min_object_size,
            min_hole_size=self.otsu_min_hole_size,
        )
        ignore_clean = clean_ignore_mask(
            ignore,
            tissue,
            enforce_background_ignore=self.enforce_background_ignore,
        )

        sample = {
            "image_id": item["image_id"],
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "soft_probs": torch.from_numpy(probs).float(),
            "hard_mask": torch.from_numpy(hard.astype(np.int64)),
            "ignore_mask": torch.from_numpy(ignore_clean.astype(np.uint8)),
            "tissue_mask": torch.from_numpy(tissue.astype(np.uint8)),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


__all__ = [
    "GleasonConsensusDataset",
    "build_tissue_mask_from_image",
    "clean_ignore_mask",
]
