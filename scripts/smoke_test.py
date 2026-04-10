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
  8. Checkpoint save/load round-trip (save_checkpoint + load_checkpoint),
     including best_composite_score persistence
  9. evaluate_checkpoint helpers: _normalize_vol_for_display, _segmentation_overlay,
     save_visualization (synthetic PNG round-trip)
 10. compute_composite_score: normal case, HD95=NaN redistribution,
     sensitivity=NaN guard, early stopping counter simulation

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

    # Verify pos_weight parameter is wired through correctly.
    # A high pos_weight should increase the loss when positive voxels are
    # predicted as zero (i.e. when logits are very negative).
    criterion_pw = DiceBCELoss(dice_weight=1.0, bce_weight=1.0, pos_weight=10.0)
    neg_logits = torch.full((2, 1, 20, 64, 64), -5.0, device=DEVICE)  # all predict 0
    pos_targets = torch.ones(2, 1, 20, 64, 64, device=DEVICE)         # all actually 1
    loss_no_pw = criterion(neg_logits, pos_targets)
    loss_with_pw = criterion_pw(neg_logits, pos_targets)
    if loss_with_pw.item() > loss_no_pw.item():
        ok(
            f"pos_weight=10 raises loss on FN voxels: "
            f"{loss_no_pw.item():.4f} → {loss_with_pw.item():.4f}"
        )
    else:
        fail(
            f"pos_weight did not raise loss as expected: "
            f"no_pw={loss_no_pw.item():.4f}, with_pw={loss_with_pw.item():.4f}"
        )

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

    # --- Empty-target guard: all-zero target + all-zero prediction -----------
    # dice / iou / sensitivity must return nan (skipped — no positive voxels).
    # Before the fix these returned 1.0, which inflated validation metrics.
    zero_logits  = torch.full((1, 1, 20, 64, 64), -10.0, device=DEVICE)  # pred = 0
    zero_targets = torch.zeros(1, 1, 20, 64, 64, device=DEVICE)          # target = 0

    m_empty = compute_all_metrics(zero_logits, zero_targets)

    import math as _math
    for name in ("dice", "iou", "sensitivity"):
        if _math.isnan(m_empty[name]):
            ok(f"empty-target guard: {name} correctly returns nan")
        else:
            fail(
                f"empty-target guard: {name} should be nan for all-zero target "
                f"but got {m_empty[name]:.4f}"
            )
    # Specificity should still be finite (1.0 — no false positives)
    if not _math.isnan(m_empty["specificity"]):
        ok(f"empty-target guard: specificity={m_empty['specificity']:.4f} (finite, as expected)")
    else:
        fail("empty-target guard: specificity unexpectedly returned nan")

    # --- Empty-target guard: all-zero target + non-zero prediction -----------
    # dice / iou / sensitivity must still be nan (target has no positives).
    # specificity should be < 1.0 because the model produces false positives.
    pos_logits = torch.full((1, 1, 20, 64, 64), 10.0, device=DEVICE)   # pred = 1 everywhere

    m_fp = compute_all_metrics(pos_logits, zero_targets)

    for name in ("dice", "iou", "sensitivity"):
        if _math.isnan(m_fp[name]):
            ok(f"empty-target / FP guard: {name} correctly returns nan")
        else:
            fail(
                f"empty-target / FP guard: {name} should be nan but got {m_fp[name]:.4f}"
            )
    if m_fp["specificity"] < 0.01:
        ok(f"empty-target / FP guard: specificity={m_fp['specificity']:.4f} (near 0, as expected)")
    else:
        fail(
            f"empty-target / FP guard: specificity={m_fp['specificity']:.4f} "
            f"should be ~0 when model predicts all-positive on all-negative target"
        )

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
# 6c. stratified_train_val_split
# ---------------------------------------------------------------------------
section("6c. stratified_train_val_split")

if _sitk is None:
    skip("SimpleITK not installed — skipping stratified split test")
else:
    try:
        import tempfile

        from dataset import stratified_train_val_split

        # Build a synthetic fixture: 4 positive cases (non-zero .nii.gz via
        # nibabel) and 6 negative cases (0-byte touch-files, which
        # _case_has_lesion must handle gracefully by returning False).
        N_POS = 4
        N_NEG = 6

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_dir = tmp_path / "images"
            labels_dir = tmp_path / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            synth_cases: list[dict] = []
            for i in range(N_POS + N_NEG):
                case_id = f"case_{i:04d}"
                for suffix in ("_t2w.mha", "_adc.mha", "_hbv.mha"):
                    (images_dir / f"{case_id}{suffix}").touch()

                label_path = labels_dir / f"{case_id}.nii.gz"
                if i < N_POS:
                    # Write a minimal valid NIfTI with one positive voxel.
                    try:
                        import nibabel as nib
                        import numpy as _np_strat
                        arr = _np_strat.zeros((8, 8, 8), dtype=_np_strat.uint8)
                        arr[4, 4, 4] = 1
                        nib.save(nib.Nifti1Image(arr, _np_strat.eye(4)), str(label_path))
                    except ImportError:
                        # nibabel unavailable: fall back to touch (treated as negative)
                        label_path.touch()
                else:
                    label_path.touch()   # 0-byte → _case_has_lesion returns False

                synth_cases.append({
                    "case_id": case_id,
                    "t2w": images_dir / f"{case_id}_t2w.mha",
                    "adc": images_dir / f"{case_id}_adc.mha",
                    "hbv": images_dir / f"{case_id}_hbv.mha",
                    "label": label_path,
                })

            train_c, val_c = stratified_train_val_split(
                synth_cases, val_fraction=0.25, seed=0
            )

            # All cases should have been annotated with has_lesion
            annotated = all("has_lesion" in c for c in synth_cases)
            if annotated:
                ok("has_lesion key added to all case dicts in-place")
            else:
                fail("has_lesion key missing from one or more case dicts")

            # 0-byte files must be treated as negative
            n_detected = sum(1 for c in synth_cases if c.get("has_lesion"))
            try:
                import nibabel  # noqa: F401
                expected_pos = N_POS
            except ImportError:
                expected_pos = 0   # nibabel absent, all touch-files → False
            if n_detected == expected_pos:
                ok(f"_case_has_lesion: detected {n_detected}/{N_POS + N_NEG} positive cases correctly")
            else:
                fail(
                    f"_case_has_lesion: expected {expected_pos} positive, "
                    f"got {n_detected}"
                )

            total = len(train_c) + len(val_c)
            if total == N_POS + N_NEG:
                ok(f"stratified split: {len(train_c)} train / {len(val_c)} val (total={total})")
            else:
                fail(f"stratified split lost cases: {total} != {N_POS + N_NEG}")

            # Ratio preserved: both splits should have pos cases (if nibabel present)
            if expected_pos > 0:
                train_pos = sum(1 for c in train_c if c.get("has_lesion"))
                val_pos   = sum(1 for c in val_c if c.get("has_lesion"))
                if train_pos > 0 and val_pos > 0:
                    ok(
                        f"positive cases in both splits: "
                        f"train={train_pos}, val={val_pos}"
                    )
                else:
                    fail(
                        f"positives not in both splits: "
                        f"train_pos={train_pos}, val_pos={val_pos}"
                    )

    except Exception as exc:
        fail("stratified_train_val_split test failed", exc)

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
            best_composite_score=0.57,
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
        assert abs(ckpt.get("best_composite_score", -1) - 0.57) < 1e-6, "best_composite_score mismatch"
        assert "scheduler_state_dict" in ckpt, "scheduler_state_dict missing"
        ok(f"load_checkpoint restored epoch={ckpt['epoch']}, "
           f"best_val_dice={ckpt['best_val_dice']:.2f}, "
           f"best_composite_score={ckpt['best_composite_score']:.2f}, "
           f"scheduler state present")

except Exception as exc:
    fail("Checkpoint round-trip test failed", exc)

# ---------------------------------------------------------------------------
# 9. evaluate_checkpoint helpers
# ---------------------------------------------------------------------------
section("9. evaluate_checkpoint helpers (_normalize, _segmentation_overlay, save_visualization)")

try:
    import importlib
    import importlib.util
    import tempfile

    import numpy as np

    # Import the helpers directly from the script module so we don't rely on
    # argparse (which would call sys.exit on --help).
    _eval_spec = importlib.util.spec_from_file_location(
        "evaluate_checkpoint",
        Path(__file__).parent / "evaluate_checkpoint.py",
    )
    _eval_mod = importlib.util.module_from_spec(_eval_spec)  # type: ignore[arg-type]
    _eval_spec.loader.exec_module(_eval_mod)  # type: ignore[union-attr]

    _normalize = _eval_mod._normalize_vol_for_display
    _overlay   = _eval_mod._segmentation_overlay
    _save_vis  = _eval_mod.save_visualization

    # --- _normalize_vol_for_display -------------------------------------------
    rng_np = np.random.default_rng(0)
    vol    = rng_np.standard_normal((10, 32, 32)).astype(np.float32) * 500 + 1000
    norm   = _normalize(vol)
    assert norm.shape == vol.shape, "shape changed"
    assert float(norm.min()) >= 0.0 - 1e-6, f"min {norm.min()} < 0"
    assert float(norm.max()) <= 1.0 + 1e-6, f"max {norm.max()} > 1"
    ok(f"_normalize_vol_for_display OK — out range [{norm.min():.3f}, {norm.max():.3f}]")

    # --- _segmentation_overlay ------------------------------------------------
    gt   = (rng_np.random((32, 32)) > 0.7).astype(np.uint8)
    pred = (rng_np.random((32, 32)) > 0.7).astype(np.uint8)
    rgba = _overlay(gt, pred, alpha=0.5)
    assert rgba.shape == (32, 32, 4), f"unexpected shape {rgba.shape}"

    gt_only   = (gt > 0) & (pred == 0)
    pred_only = (pred > 0) & (gt == 0)
    both      = (gt > 0) & (pred > 0)

    if gt_only.any():
        assert rgba[gt_only, 1].mean() > 0.9, "GT-only pixels should be green"
    if pred_only.any():
        assert rgba[pred_only, 0].mean() > 0.9, "pred-only pixels should be red"
    if both.any():
        assert rgba[both, 0].mean() > 0.9 and rgba[both, 1].mean() > 0.9, \
            "overlap pixels should be yellow"
    ok("_segmentation_overlay colour logic OK (green/red/yellow)")

    # --- save_visualization (synthetic PNG round-trip) ------------------------
    D, H, W = 24, 48, 48
    fake_results = [
        {
            "case_id":  f"synth_{i:02d}",
            "t2w_vol":  rng_np.standard_normal((D, H, W)).astype(np.float32),
            "gt_vol":   (rng_np.random((D, H, W)) > 0.85).astype(np.float32),
            "pred_vol": (rng_np.random((D, H, W)) > 0.85).astype(np.float32),
        }
        for i in range(3)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "test_vis.png"
        _save_vis(fake_results, out_png, n_cols=5)
        assert out_png.exists(), "PNG file was not created"
        size_kb = out_png.stat().st_size / 1024
        ok(f"save_visualization wrote {out_png.name} ({size_kb:.0f} KB, 3 rows × 5 cols)")

except Exception as exc:
    fail("evaluate_checkpoint helpers test failed", exc)

# ---------------------------------------------------------------------------
# 10. compute_composite_score
# ---------------------------------------------------------------------------
section("10. compute_composite_score + early stopping counter logic")

try:
    import math as _math_cs

    from utils import compute_composite_score

    # --- 10a. Normal case: all metrics finite ---------------------------------
    metrics_full = {
        "sensitivity": 0.80,
        "dice":        0.70,
        "hd95":        5.0,
        "iou":         0.60,
        "specificity": 0.95,
    }
    score_full = compute_composite_score(
        metrics_full, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2
    )
    if not _math_cs.isnan(score_full) and 0.0 <= score_full <= 1.0:
        ok(f"all-finite metrics → composite_score={score_full:.4f} (in [0,1])")
    else:
        fail(f"all-finite metrics: unexpected score={score_full}")

    # Verify manual calculation matches
    hd95_term = 1.0 / (1.0 + 5.0)   # = 1/6 ≈ 0.1667
    total_w = 0.5 + 0.3 + 0.2       # = 1.0 (normalised)
    expected = (0.5 * 0.80 + 0.3 * 0.70 + 0.2 * hd95_term) / total_w
    if abs(score_full - expected) < 1e-6:
        ok(f"composite score matches manual calculation ({expected:.6f})")
    else:
        fail(f"composite score mismatch: got {score_full:.6f}, expected {expected:.6f}")

    # --- 10b. HD95 = NaN: weight redistributed to sensitivity + dice ----------
    metrics_no_hd95 = dict(metrics_full)
    metrics_no_hd95["hd95"] = float("nan")
    score_no_hd95 = compute_composite_score(
        metrics_no_hd95, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2
    )
    if not _math_cs.isnan(score_no_hd95) and 0.0 <= score_no_hd95 <= 1.0:
        ok(f"hd95=nan → weight redistributed, composite_score={score_no_hd95:.4f}")
    else:
        fail(f"hd95=nan: unexpected score={score_no_hd95}")

    # Manual: only sensitivity + dice, normalised by (0.5 + 0.3) = 0.8
    expected_no_hd95 = (0.5 * 0.80 + 0.3 * 0.70) / (0.5 + 0.3)
    if abs(score_no_hd95 - expected_no_hd95) < 1e-6:
        ok(f"redistributed score matches manual calculation ({expected_no_hd95:.6f})")
    else:
        fail(
            f"redistributed score mismatch: got {score_no_hd95:.6f}, "
            f"expected {expected_no_hd95:.6f}"
        )

    # --- 10c. Sensitivity = NaN: must return NaN (no positive cases) ----------
    metrics_no_sens = dict(metrics_full)
    metrics_no_sens["sensitivity"] = float("nan")
    metrics_no_sens["dice"] = float("nan")
    score_no_sens = compute_composite_score(
        metrics_no_sens, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2
    )
    if _math_cs.isnan(score_no_sens):
        ok("sensitivity=nan → composite_score=nan (no best.pt update)")
    else:
        fail(f"sensitivity=nan: expected nan but got {score_no_sens:.4f}")

    # --- 10d. Early stopping counter simulation --------------------------------
    # Simulate 3 epochs of improvement followed by 3 stagnant epochs, then
    # one improvement, then stagnation until patience is reached.
    patience = 3
    min_delta = 0.001
    best = 0.0
    counter = 0
    stopped_at: int | None = None

    sim_scores = [0.50, 0.55, 0.58,   # 3 improvements → counter stays 0
                  0.580, 0.581,        # delta < min_delta → counter 1, 2
                  0.600,               # improvement → counter resets to 0
                  0.600, 0.600, 0.600] # stagnant → counter 1, 2, 3 → STOP

    for ep, s in enumerate(sim_scores, start=1):
        if s > best + min_delta:
            best = s
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            stopped_at = ep
            break

    # Expect stop at epoch 9 (3rd stagnant epoch after the reset at epoch 6)
    if stopped_at == 9:
        ok(f"early stopping counter: triggered at epoch {stopped_at} (patience={patience})")
    else:
        fail(f"early stopping counter: expected stop at epoch 9, got {stopped_at}")

except Exception as exc:
    fail("compute_composite_score test failed", exc)

# ---------------------------------------------------------------------------
# 11. In-memory dataset cache (PiCaiDataset use_cache=True)
# ---------------------------------------------------------------------------
section("11. PiCaiDataset in-memory cache (use_cache=True)")

if _sitk is None:
    skip("SimpleITK not installed — skipping cache test")
else:
    try:
        import tempfile

        import numpy as np

        import SimpleITK as sitk  # noqa: N813
        from dataset import PiCaiDataset

        def _write_tiny_mha(path: Path) -> None:
            """Write a 4×4×4 MHA filled with ones."""
            arr = np.ones((4, 4, 4), dtype=np.float32)
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((1.0, 1.0, 3.0))
            sitk.WriteImage(img, str(path))

        def _write_tiny_nii(path: Path) -> None:
            """Write a tiny NIfTI label (one positive voxel)."""
            try:
                import nibabel as nib
                arr = np.zeros((4, 4, 4), dtype=np.uint8)
                arr[2, 2, 2] = 1
                nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))
            except ImportError:
                # nibabel absent: write a 0-byte file (treated as no-lesion)
                path.touch()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_dir = tmp_path / "images"
            lbl_dir = tmp_path / "labels"
            img_dir.mkdir()
            lbl_dir.mkdir()

            case_id = "test_0000_0000"
            for suffix in ("_t2w.mha", "_adc.mha", "_hbv.mha"):
                _write_tiny_mha(img_dir / f"{case_id}{suffix}")
            _write_tiny_nii(lbl_dir / f"{case_id}.nii.gz")

            synth_cases = [{
                "case_id": case_id,
                "t2w": img_dir / f"{case_id}_t2w.mha",
                "adc": img_dir / f"{case_id}_adc.mha",
                "hbv": img_dir / f"{case_id}_hbv.mha",
                "label": lbl_dir / f"{case_id}.nii.gz",
            }]

            ds = PiCaiDataset(
                images_dir=img_dir,
                labels_dir=lbl_dir,
                cases=synth_cases,
                use_cache=True,
                cache_rate=1.0,
            )

            # First access: populates cache.
            _ = ds[0]
            if 0 in ds._cache:
                ok("cache populated on first __getitem__ access")
            else:
                fail("cache was not populated after first access")

            # Second access: must be a cache hit (no disk I/O).
            # We verify this by temporarily removing the source files and
            # confirming the second call still succeeds.
            import shutil
            shutil.rmtree(str(img_dir))
            try:
                _ = ds[0]
                ok("second access succeeds from cache (source files removed)")
            except Exception as e:
                fail("second access failed despite cache being populated", e)

    except Exception as exc:
        fail("PiCaiDataset cache test failed", exc)

# ---------------------------------------------------------------------------
# 12. compute_all_metrics with compute_hd95=False
# ---------------------------------------------------------------------------
section("12. compute_all_metrics(compute_hd95=False)")

try:
    import math as _math_hd

    from metrics import compute_all_metrics

    logits_h = torch.randn(2, 1, 20, 32, 32, device=DEVICE)
    targets_h = (torch.rand(2, 1, 20, 32, 32, device=DEVICE) > 0.8).float()

    m_no_hd95 = compute_all_metrics(logits_h, targets_h, compute_hd95=False)

    if _math_hd.isnan(m_no_hd95["hd95"]):
        ok("compute_hd95=False → hd95 correctly returns nan")
    else:
        fail(
            f"compute_hd95=False: hd95 should be nan but got {m_no_hd95['hd95']:.4f}"
        )

    # Other metrics must still be finite
    for name in ("dice", "iou", "sensitivity", "specificity"):
        if not _math_hd.isnan(m_no_hd95[name]):
            ok(f"compute_hd95=False: {name}={m_no_hd95[name]:.4f} (finite, as expected)")
        else:
            fail(f"compute_hd95=False: {name} unexpectedly returned nan")

except Exception as exc:
    fail("compute_hd95=False test failed", exc)

# ---------------------------------------------------------------------------
# 13. BF16 autocast forward pass
# ---------------------------------------------------------------------------
section("13. BF16 autocast (AMP) forward pass")

if not cuda_ok:
    skip("CUDA not available — skipping BF16 autocast test")
else:
    try:
        amp_model = UNet3D(in_channels=3, out_channels=1, features=(16, 32, 64, 128))
        amp_model = amp_model.to(DEVICE)

        dummy = torch.randn(1, 3, 20, 64, 64, device=DEVICE)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_amp = amp_model(dummy)

        if out_amp.shape == (1, 1, 20, 64, 64):
            ok(
                f"BF16 autocast forward pass OK — output dtype={out_amp.dtype}, "
                f"shape={tuple(out_amp.shape)}"
            )
        else:
            fail(f"BF16 autocast: unexpected output shape {tuple(out_amp.shape)}")

        # Loss must also be finite under BF16 autocast.
        from losses import DiceBCELoss
        amp_criterion = DiceBCELoss().to(DEVICE)
        tgt = (torch.rand(1, 1, 20, 64, 64, device=DEVICE) > 0.8).float()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            amp_loss = amp_criterion(out_amp, tgt)

        if torch.isfinite(amp_loss):
            ok(f"DiceBCELoss under BF16 autocast: loss={amp_loss.item():.4f} (finite)")
        else:
            fail(f"DiceBCELoss under BF16 autocast returned non-finite value: {amp_loss.item()}")

    except Exception as exc:
        fail("BF16 autocast test failed", exc)

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
