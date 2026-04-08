"""
3D data augmentation and preprocessing transforms for prostate lesion segmentation.

Uses MONAI's dictionary-based transform API (keys: "image", "label").
All transforms that modify geometry must be applied identically to both
the image and the label; intensity transforms are applied to the image only.

Training pipeline
-----------------
1. RandCropByPosNegLabeld  — balanced patch sampling (favours lesion voxels)
2. RandFlipd (x3)          — independent left-right, anterior-posterior, superior-inferior flips
3. RandRotate90d           — random 90° rotations in the axial plane
4. RandAffined             — small random rotations + scale changes
5. RandGaussianNoised      — additive Gaussian noise
6. RandGaussianSmoothd     — Gaussian blur
7. RandScaleIntensityd     — multiplicative intensity scaling
8. RandShiftIntensityd     — additive intensity shift

Validation pipeline
-------------------
Identity transform — the full resampled volume is returned as-is.
Sliding-window inference is performed in the training loop.
"""

from __future__ import annotations

from monai import transforms as T

IMAGE_KEY = "image"
LABEL_KEY = "label"
_BOTH = (IMAGE_KEY, LABEL_KEY)


def get_train_transforms(
    patch_size: tuple[int, ...] = (20, 128, 128),
    pos_fraction: float = 0.75,
    num_samples: int = 1,
) -> T.Compose:
    """
    Return a composed MONAI transform for training.

    Parameters
    ----------
    patch_size    : (D, H, W) patch dimensions in voxels.
                    Default (20, 128, 128) matches (3.0, 0.5, 0.5) mm spacing
                    with ~60mm z coverage and ~64mm in-plane coverage.
    pos_fraction  : fraction of extracted patches guaranteed to contain at
                    least one lesion voxel (remainder are background patches).
                    Higher values help with class imbalance; 0.75 is typical.
    num_samples   : number of patches to extract per __getitem__ call.
                    Increase alongside monai.data.list_data_collate for higher
                    GPU utilisation per volume load.
    """
    return T.Compose(
        [
            # ----------------------------------------------------------
            # Ensure every volume is at least patch_size in each axis so
            # that RandCropByPosNegLabeld never sees a volume smaller than
            # the requested crop ROI.  Volumes that are already large
            # enough are unaffected; smaller ones get zero-padded.
            # ----------------------------------------------------------
            T.SpatialPadd(
                keys=_BOTH,
                spatial_size=patch_size,
                mode="constant",
            ),
            # ----------------------------------------------------------
            # Patch sampling
            # Positive patches: centred on a foreground (lesion) voxel.
            # Negative patches: random crop from the background region.
            # ----------------------------------------------------------
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
            # ----------------------------------------------------------
            # Spatial augmentations (applied to image AND label)
            # ----------------------------------------------------------
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=0),   # z / slice axis
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=1),   # y axis
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=2),   # x axis
            T.RandRotate90d(
                keys=_BOTH,
                prob=0.5,
                max_k=3,
                spatial_axes=(1, 2),  # rotate in the axial (y-x) plane only
            ),
            T.RandAffined(
                keys=_BOTH,
                mode=("bilinear", "nearest"),  # bilinear for image, NN for label
                prob=0.3,
                rotate_range=(0.15, 0.15, 0.15),  # ±~8.6° per axis
                scale_range=(0.1, 0.1, 0.1),      # ±10% isotropic scaling
                padding_mode="border",
            ),
            # ----------------------------------------------------------
            # Intensity augmentations (applied to image only)
            # ----------------------------------------------------------
            T.RandGaussianNoised(
                keys=[IMAGE_KEY],
                prob=0.2,
                mean=0.0,
                std=0.1,
            ),
            T.RandGaussianSmoothd(
                keys=[IMAGE_KEY],
                prob=0.2,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            T.RandScaleIntensityd(
                keys=[IMAGE_KEY],
                factors=0.2,   # scales by U(1-0.2, 1+0.2)
                prob=0.3,
            ),
            T.RandShiftIntensityd(
                keys=[IMAGE_KEY],
                offsets=0.2,   # shifts by U(-0.2, 0.2)
                prob=0.3,
            ),
        ]
    )


def get_val_transforms() -> T.Compose:
    """
    Return a no-op composed MONAI transform for validation.

    The full resampled volume is returned unchanged; sliding-window
    inference is applied in the evaluation loop.
    """
    return T.Compose([])
