"""
smoke_test.py — Sanity-check the full PI-CAI segmentation pipeline.

What this tests (no real data required):
  1. PyTorch + CUDA availability
  2. Optional dependency imports (SimpleITK, MONAI, nibabel, scipy)
  3. UNet3D: instantiation, parameter count, forward pass shape
  4. DiceBCELoss: forward pass with random logits/targets
  5. All metrics: dice, iou, sensitivity, specificity, hd95
  6. Dataset helpers: discover_cases + train_val_split on a synthetic fixture
  7. Transforms: get_train_transforms / get_val_transforms on a dummy batch
  8. Checkpoint save/load round-trip (save_checkpoint + load_checkpoint)

Run inside the Docker container:
    python scripts/smoke_test.py

Run locally (only PyTorch-only tests will pass; medical libs may be absent):
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
INFO = "[INFO]"

_all_passed = True


def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")


def skip(msg: str) -> None:
    print(f"  {SKIP}  {msg}")


def fail(msg: str, exc: BaseException | None = None) -> None:
    global _all_passed
    _all_passed = False
    print(f"  {FAIL}  {msg}")
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        for line in tb.splitlines():
            print(f"         {line}")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# 1. PyTorch / CUDA
# ---------------------------------------------------------------------------
section("1. PyTorch + CUDA")

print(f"  {INFO}  PyTorch version : {torch.__version__}")
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    print(f"  {INFO}  CUDA version    : {torch.version.cuda}")
    print(f"  {INFO}  GPU count       : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  {INFO}  GPU {i}           : {torch.cuda.get_device_name(i)}")
    ok("CUDA available")
else:
    print(f"  {INFO}  CUDA: not available — running on CPU")
    ok("CPU-only mode (CUDA not detected)")

DEVICE = torch.device("cuda" if cuda_ok else "cpu")

# ---------------------------------------------------------------------------
# 2. Optional dependency imports
# ---------------------------------------------------------------------------
section("2. Optional dependencies")

_sitk: Any = None
_monai: Any = None

try:
    import SimpleITK as sitk  # noqa: N813
    _sitk = sitk
    ok(f"SimpleITK {sitk.Version_VersionString()}")
except ImportError as e:
    skip(f"SimpleITK not installed ({e}) — dataset loading tests skipped")

try:
    import monai
    _monai = monai
    ok(f"MONAI {monai.__version__}")
except ImportError as e:
    skip(f"MONAI not installed ({e}) — transform tests skipped")

try:
    import nibabel
    ok(f"nibabel {nibabel.__version__}")
except ImportError as e:
    skip(f"nibabel not installed ({e})")

try:
    import scipy
    ok(f"scipy {scipy.__version__}")
except ImportError as e:
    skip(f"scipy not installed ({e}) — HD95 will return nan")

# ---------------------------------------------------------------------------
# 3. UNet3D: instantiation + forward pass
# ---------------------------------------------------------------------------
section("3. UNet3D model")

# Add repo src/ to path so imports work from the scripts/ directory
_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_src))

try:
    from models import UNet3D

    model = UNet3D(in_channels=3, out_channels=1, features=(16, 32, 64, 128))
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"UNet3D instantiated — {n_params:,} parameters")

    # Forward pass: small patch (2, 3, 20, 64, 64) to keep memory low
    B, C, D, H, W = 2, 3, 20, 64, 64
    dummy_input = torch.randn(B, C, D, H, W, device=DEVICE)
    with torch.no_grad():
        out = model(dummy_input)

    expected = (B, 1, D, H, W)
    if out.shape == expected:
        ok(f"Forward pass OK — output shape {tuple(out.shape)}")
    else:
        fail(f"Output shape {tuple(out.shape)} != expected {expected}")

except Exception as exc:
    fail("UNet3D test failed", exc)

# ---------------------------------------------------------------------------
# 4. DiceBCELoss
# ---------------------------------------------------------------------------
section("4. DiceBCELoss")

try:
    from losses import DiceBCELoss

    criterion = DiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    logits = torch.randn(2, 1, 20, 64, 64, device=DEVICE)
    targets = (torch.rand(2, 1, 20, 64, 64, device=DEVICE) > 0.8).float()

    loss_val = criterion(logits, targets)

    if torch.isfinite(loss_val):
        ok(f"DiceBCELoss forward pass — loss={loss_val.item():.4f}")
    else:
        fail(f"DiceBCELoss returned non-finite value: {loss_val.item()}")

except Exception as exc:
    fail("DiceBCELoss test failed", exc)

# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------
section("5. Segmentation metrics")

try:
    from metrics import compute_all_metrics

    logits_m = torch.randn(2, 1, 20, 64, 64, device=DEVICE)
    targets_m = (torch.rand(2, 1, 20, 64, 64, device=DEVICE) > 0.8).float()

    metrics = compute_all_metrics(logits_m, targets_m)

    for name, val in metrics.items():
        ok(f"{name:<14s} = {val:.4f}")

except Exception as exc:
    fail("Metrics test failed", exc)

# ---------------------------------------------------------------------------
# 6. Dataset helpers (no real files needed)
# ---------------------------------------------------------------------------
section("6. Dataset helpers: discover_cases + train_val_split")

try:
    import tempfile

    from dataset import discover_cases, train_val_split

    N_CASES = 6

    # ---- 6a. Nested layout ----
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir(parents=True)

        for i in range(N_CASES):
            patient_id = f"1000{i}"
            study_id   = f"100000{i}"
            case_id    = f"{patient_id}_{study_id}"
            study_dir  = images_dir / patient_id / case_id
            study_dir.mkdir(parents=True)
            for suffix in ("_t2w.mha", "_adc.mha", "_hbv.mha"):
                (study_dir / f"{case_id}{suffix}").touch()
            if i < 4:
                (labels_dir / f"{case_id}.nii.gz").touch()

        cases = discover_cases(images_dir, labels_dir)
        if len(cases) == N_CASES:
            ok(f"nested layout: found {len(cases)} cases")
        else:
            fail(f"nested layout: expected {N_CASES}, got {len(cases)}")

    # ---- 6b. Flat layout (mirrors actual PI-CAI download) ----
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        for i in range(N_CASES):
            patient_id = f"1000{i}"
            study_id   = f"100000{i}"
            case_id    = f"{patient_id}_{study_id}"
            for suffix in ("_t2w.mha", "_adc.mha", "_hbv.mha", "_cor.mha", "_sag.mha"):
                (images_dir / f"{case_id}{suffix}").touch()
            if i < 4:
                (labels_dir / f"{case_id}.nii.gz").touch()

        cases = discover_cases(images_dir, labels_dir)
        if len(cases) == N_CASES:
            ok(f"flat layout  : found {len(cases)} cases")
        else:
            fail(f"flat layout: expected {N_CASES}, got {len(cases)}")

        labelled = sum(1 for c in cases if c["label"] is not None)
        ok(f"  {labelled} labelled, {len(cases) - labelled} unlabelled")

        train_c, val_c = train_val_split(cases, val_fraction=0.33, seed=0)
        ok(f"train_val_split → {len(train_c)} train / {len(val_c)} val")

except Exception as exc:
    fail("Dataset helper test failed", exc)

# ---------------------------------------------------------------------------
# 7. Transforms (requires MONAI)
# ---------------------------------------------------------------------------
section("7. MONAI transforms")

if _monai is None:
    skip("MONAI not installed — skipping transform tests")
else:
    try:
        from transforms import get_train_transforms, get_val_transforms

        train_tfm = get_train_transforms(patch_size=(20, 64, 64), num_samples=1)
        val_tfm = get_val_transforms()
        ok("get_train_transforms instantiated")
        ok("get_val_transforms instantiated")

        # Run val transform on a dummy batch (identity — should return unchanged)
        dummy_batch = {
            "image": torch.randn(3, 20, 64, 64),
            "label": (torch.rand(1, 20, 64, 64) > 0.8).float(),
        }
        result = val_tfm(dummy_batch)
        if isinstance(result, dict) and "image" in result and "label" in result:
            ok(f"val_transforms forward pass OK — image shape {tuple(result['image'].shape)}")
        else:
            fail(f"Unexpected val_transforms output type: {type(result)}")

    except Exception as exc:
        fail("Transforms test failed", exc)

# ---------------------------------------------------------------------------
# 8. Checkpoint save / load round-trip
# ---------------------------------------------------------------------------
section("8. Checkpoint save/load (save_checkpoint + load_checkpoint)")

try:
    import tempfile

    from utils import load_checkpoint, save_checkpoint

    # Build a minimal model + optimizer + scheduler to round-trip
    _ckpt_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16))
    _ckpt_model = _ckpt_model.to(DEVICE)
    _ckpt_opt = torch.optim.AdamW(_ckpt_model.parameters(), lr=1e-4)
    _ckpt_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        _ckpt_opt, T_max=10, eta_min=1e-6
    )

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "test_epoch_0001.pt"

        # Save
        save_checkpoint(
            _ckpt_model, _ckpt_opt, epoch=1,
            path=str(ckpt_path),
            scheduler=_ckpt_sched,
            best_val_dice=0.42,
        )
        ok(f"save_checkpoint wrote {ckpt_path.name}")

        # Load into a fresh model/optimizer/scheduler
        _new_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16)).to(DEVICE)
        _new_opt = torch.optim.AdamW(_new_model.parameters(), lr=1e-4)
        _new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            _new_opt, T_max=10, eta_min=1e-6
        )

        ckpt = load_checkpoint(
            ckpt_path, _new_model, _new_opt, _new_sched, device=DEVICE
        )

        assert ckpt["epoch"] == 1, f"epoch mismatch: {ckpt['epoch']}"
        assert abs(ckpt.get("best_val_dice", -1) - 0.42) < 1e-6, "best_val_dice mismatch"
        assert "scheduler_state_dict" in ckpt, "scheduler_state_dict missing"
        ok(f"load_checkpoint restored epoch={ckpt['epoch']}, "
           f"best_val_dice={ckpt['best_val_dice']:.2f}, scheduler state present")

except Exception as exc:
    fail("Checkpoint round-trip test failed", exc)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Summary")

if _all_passed:
    print("  All checks passed.\n")
    sys.exit(0)
else:
    print("  One or more checks FAILED — see above.\n")
    sys.exit(1)
