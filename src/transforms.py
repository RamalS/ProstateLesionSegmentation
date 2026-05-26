"""
3D data augmentation and preprocessing transforms for prostate lesion segmentation.

Uses MONAI's dictionary-based transform API (keys: "image", "label").
All transforms that modify geometry must be applied identically to both
the image and the label; intensity transforms are applied to the image only.

Training pipeline
-----------------
1.  RandCropByPosNegLabeld  — balanced patch sampling (favours lesion voxels)
2.  RandFlipd (x3)          — independent left-right, anterior-posterior, superior-inferior flips
3.  RandRotate90d           — random 90° rotations in the axial plane
4.  RandAffined             — random rotations (±15°) + scale changes (±10%)
5.  Rand3DElasticd          — elastic tissue deformation (simulates prostate shape variability)
6.  RandGaussianNoised      — additive Gaussian noise
7.  RandGaussianSmoothd     — Gaussian blur
8.  RandScaleIntensityd     — multiplicative intensity scaling
9.  RandShiftIntensityd     — additive intensity shift
10. RandAdjustContrastd     — gamma contrast adjustment (simulates scanner/protocol variability)
11. RandBiasFieldd          — MRI B1 field inhomogeneity per channel
12. RandGibbsNoised         — MRI ringing/truncation artifact
13. RandCoarseDropoutd      — random patch dropout (regularization)

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
    augmentation_profile: str = "default",
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
    augmentation_profile : augmentation preset. ``"default"`` keeps the
                    existing stronger pipeline; ``"light"`` reduces geometric
                    and intensity perturbations for transformer-style models
                    that are prone to collapsing into near-background
                    predictions early in training.
    """
    profile = str(augmentation_profile).strip().lower()
    if profile not in {"default", "light"}:
        raise ValueError(
            "augmentation_profile must be 'default' or 'light', "
            f"got {augmentation_profile!r}."
        )

    if profile == "light":
        rotate90_prob = 0.3
        affine_prob = 0.2
        affine_rotate = (0.10, 0.10, 0.10)
        affine_scale = (0.05, 0.05, 0.05)
        elastic_prob = 0.05
        elastic_sigma = (6, 8)
        elastic_magnitude = (20, 60)
        noise_prob = 0.15
        smooth_prob = 0.15
        intensity_scale_prob = 0.2
        intensity_shift_prob = 0.2
        contrast_prob = 0.15
        bias_field_prob = 0.15
        gibbs_prob = 0.1
        coarse_dropout_prob = 0.1
    else:
        rotate90_prob = 0.5
        affine_prob = 0.5
        affine_rotate = (0.26, 0.26, 0.26)
        affine_scale = (0.1, 0.1, 0.1)
        elastic_prob = 0.2
        elastic_sigma = (5, 7)
        elastic_magnitude = (50, 150)
        noise_prob = 0.3
        smooth_prob = 0.3
        intensity_scale_prob = 0.4
        intensity_shift_prob = 0.4
        contrast_prob = 0.3
        bias_field_prob = 0.3
        gibbs_prob = 0.2
        coarse_dropout_prob = 0.2

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
            ),
            # ----------------------------------------------------------
            # Spatial augmentations (applied to image AND label)
            # ----------------------------------------------------------
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=0),   # z / slice axis
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=1),   # y axis
            T.RandFlipd(keys=_BOTH, prob=0.5, spatial_axis=2),   # x axis
            T.RandRotate90d(
                keys=_BOTH,
                prob=rotate90_prob,
                max_k=3,
                spatial_axes=(1, 2),  # rotate in the axial (y-x) plane only
            ),
            T.RandAffined(
                keys=_BOTH,
                mode=("bilinear", "nearest"),  # bilinear for image, NN for label
                prob=affine_prob,
                rotate_range=affine_rotate,
                scale_range=affine_scale,
                padding_mode="border",
            ),
            # Elastic deformation: simulates prostate shape changes due to
            # rectal filling, patient positioning, and breathing motion.
            # sigma controls smoothness of the deformation field; magnitude
            # controls displacement amplitude in voxels.
            T.Rand3DElasticd(
                keys=_BOTH,
                mode=("bilinear", "nearest"),
                prob=elastic_prob,
                sigma_range=elastic_sigma,
                magnitude_range=elastic_magnitude,
                padding_mode="border",
            ),
            # ----------------------------------------------------------
            # Intensity augmentations (applied to image only)
            # ----------------------------------------------------------
            T.RandGaussianNoised(
                keys=[IMAGE_KEY],
                prob=noise_prob,
                mean=0.0,
                std=0.1,
            ),
            T.RandGaussianSmoothd(
                keys=[IMAGE_KEY],
                prob=smooth_prob,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            T.RandScaleIntensityd(
                keys=[IMAGE_KEY],
                factors=0.2,   # scales by U(1-0.2, 1+0.2)
                prob=intensity_scale_prob,
            ),
            T.RandShiftIntensityd(
                keys=[IMAGE_KEY],
                offsets=0.2,   # shifts by U(-0.2, 0.2)
                prob=intensity_shift_prob,
            ),
            # Gamma contrast: simulates intensity response differences between
            # scanner vendors (Siemens vs Philips) and acquisition protocols
            # across the 4 PI-CAI acquisition centers.
            T.RandAdjustContrastd(
                keys=[IMAGE_KEY],
                prob=contrast_prob,
                gamma=(0.7, 1.5),
            ),
            # MRI B1 field inhomogeneity: simulates spatially varying receive
            # coil sensitivity.  MONAI applies the bias field independently
            # per channel, which is physically correct — each modality (T2w,
            # ADC, HBV) has its own coil geometry and field profile.
            T.RandBiasFieldd(
                keys=[IMAGE_KEY],
                prob=bias_field_prob,
                coeff_range=(0.0, 0.3),
            ),
            # Gibbs/ringing artifact: simulates MRI k-space truncation, which
            # produces ringing at high-contrast tissue boundaries.  Common in
            # T2w and high-b-value DWI acquisitions.
            T.RandGibbsNoised(
                keys=[IMAGE_KEY],
                prob=gibbs_prob,
                alpha=(0.0, 0.5),
            ),
            # Coarse dropout: randomly zeros rectangular 3-D regions, forcing
            # the model to rely on broader spatial context rather than
            # memorising local intensity patterns.  Acts as a regularizer.
            T.RandCoarseDropoutd(
                keys=[IMAGE_KEY],
                holes=3,
                spatial_size=(4, 16, 16),
                fill_value=0.0,
                prob=coarse_dropout_prob,
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
