"""
2D augmentation and preprocessing transforms for Deconver JPG/PNG training.

Uses MONAI dictionary transforms with keys:
  - ``image``: (C, H, W)
  - ``label``: (1, H, W)
"""

from __future__ import annotations

from monai import transforms as T

IMAGE_KEY = "image"
LABEL_KEY = "label"
_BOTH = (IMAGE_KEY, LABEL_KEY)


def get_train_transforms_2d(
    patch_size: tuple[int, int] = (256, 256),
    pos_fraction: float = 0.75,
    num_samples: int = 1,
) -> T.Compose:
    """
    Build MONAI training transforms for 2D segmentation.
    """
    return T.Compose(
        [
            T.SpatialPadd(
                keys=_BOTH,
                spatial_size=patch_size,
                mode="constant",
            ),
            T.RandCropByPosNegLabeld(
                keys=_BOTH,
                label_key=LABEL_KEY,
                spatial_size=patch_size,
                pos=pos_fraction,
                neg=1.0 - pos_fraction,
                num_samples=num_samples,
                image_key=IMAGE_KEY,
                image_threshold=0.0,
            ),
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=0),
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=1),
            T.RandRotate90d(
                keys=_BOTH,
                prob=0.5,
                max_k=3,
                spatial_axes=(0, 1),
            ),
            T.RandAffined(
                keys=_BOTH,
                mode=("bilinear", "nearest"),
                prob=0.5,
                rotate_range=(0.26,),
                scale_range=(0.1, 0.1),
                padding_mode="border",
            ),
            T.RandGaussianNoised(
                keys=[IMAGE_KEY],
                prob=0.3,
                mean=0.0,
                std=0.1,
            ),
            T.RandGaussianSmoothd(
                keys=[IMAGE_KEY],
                prob=0.3,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
            ),
            T.RandScaleIntensityd(
                keys=[IMAGE_KEY],
                factors=0.2,
                prob=0.4,
            ),
            T.RandShiftIntensityd(
                keys=[IMAGE_KEY],
                offsets=0.2,
                prob=0.4,
            ),
            T.RandAdjustContrastd(
                keys=[IMAGE_KEY],
                prob=0.3,
                gamma=(0.7, 1.5),
            ),
            T.RandBiasFieldd(
                keys=[IMAGE_KEY],
                prob=0.3,
                coeff_range=(0.0, 0.3),
            ),
            T.RandGibbsNoised(
                keys=[IMAGE_KEY],
                prob=0.2,
                alpha=(0.0, 0.5),
            ),
            T.RandCoarseDropoutd(
                keys=[IMAGE_KEY],
                holes=3,
                spatial_size=(16, 16),
                fill_value=0.0,
                prob=0.2,
            ),
        ]
    )


def get_val_transforms_2d() -> T.Compose:
    """
    Validation transform for 2D segmentation.
    """
    return T.Compose([])


__all__ = [
    "get_train_transforms_2d",
    "get_val_transforms_2d",
]

