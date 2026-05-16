"""
smoke_test.py — Sanity-check the full PI-CAI segmentation pipeline.

What this tests (no real data required):
  1. PyTorch + CUDA availability
  2. Optional dependency imports (SimpleITK, MONAI, nibabel, scipy)
  3. UNet3D: instantiation, parameter count, forward pass shape
   3b. AttentionUNet3D: instantiation, parameter count, forward pass, build_model factory
   3c. Modality flag selection: build_model derives in_channels from use_t2w/use_adc/use_hbv
   3d. Deep supervision: list output, auxiliary shapes, build_model DS, DeepSupervisionWrapper
   3e. Deconver: instantiation, forward pass, build_model factory, deep supervision (num_deep_supr),
       and stride wiring regression guard
   3f. FCT: instantiation, forward pass, build_model factory, deep supervision outputs
  4. DiceBCELoss: forward pass with random logits/targets
  4b. TverskyBCELoss: forward pass, FN-penalty bias, tversky_loss function
  4c. LR warmup: LinearLR warm-up + CosineAnnealingLR via SequentialLR
  4d. Loss robustness: negative-sample exclusion + FP16 overflow guard
  5. All metrics: dice, iou, sensitivity, precision, hd95
  6. Dataset helpers: discover_cases + train_val_split on a synthetic fixture
   6d. Prostate158 helpers: discover_unlabeled_cases + DWI-as-HBV preprocessing
  7. Transforms: get_train_transforms / get_val_transforms on a dummy batch
  8. Checkpoint save/load round-trip (save_checkpoint + load_checkpoint),
      including best_composite_score, last_hd95 persistence, and GradScaler state
   8b. SSL transfer helpers: get_encoder_state_dict, load_pretrained_encoder,
       set_encoder_trainable
   8c. current-config init helpers: model-weight-only load + resume/current-config
       conflict guard
   9b. visualize_3d helpers: seg auto-detection, downsampling, Plotly HTML output
  10. compute_composite_score: normal case, hd95_scale parameter, HD95=NaN
      redistribution, sensitivity=NaN guard, cached-HD95 consistency,
      early stopping counter simulation
 11. PiCaiDataset cache modes (ram/storage/none)
 12. compute_all_metrics(compute_hd95=False)
 13. AMP forward+backward: FP16+GradScaler (Volta/Turing) and BF16 (Ampere+/Blackwell)
 14. ntfy notifications: send_ntfy no-op, URL/header/body correctness, error handling

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
# 3b. AttentionUNet3D + build_model factory
# ---------------------------------------------------------------------------
section("3b. AttentionUNet3D + build_model factory")

try:
    from models import AttentionUNet3D, build_model

    # --- AttentionUNet3D instantiation + forward pass ----------------------
    attn_model = AttentionUNet3D(in_channels=3, out_channels=1, features=(16, 32, 64, 128))
    attn_model = attn_model.to(DEVICE)
    n_attn = sum(p.numel() for p in attn_model.parameters())
    ok(f"AttentionUNet3D instantiated — {n_attn:,} parameters")

    # Attention gates add parameters; model must be strictly larger than UNet3D
    try:
        _baseline = UNet3D(in_channels=3, out_channels=1, features=(16, 32, 64, 128))
        n_base = sum(p.numel() for p in _baseline.parameters())
        if n_attn > n_base:
            ok(
                f"AttentionUNet3D has more parameters than UNet3D "
                f"({n_attn:,} > {n_base:,})"
            )
        else:
            fail(
                f"AttentionUNet3D should have more params than UNet3D "
                f"(got {n_attn:,} vs {n_base:,})"
            )
    except Exception:
        pass  # UNet3D import may have failed in section 3; skip comparison

    B, C, D, H, W = 2, 3, 20, 64, 64
    dummy_attn = torch.randn(B, C, D, H, W, device=DEVICE)
    with torch.no_grad():
        out_attn = attn_model(dummy_attn)

    expected_attn = (B, 1, D, H, W)
    if out_attn.shape == expected_attn:
        ok(f"AttentionUNet3D forward pass OK — output shape {tuple(out_attn.shape)}")
    else:
        fail(f"AttentionUNet3D: output shape {tuple(out_attn.shape)} != {expected_attn}")

    # --- Odd spatial dimensions (off-by-one padding path) ------------------
    # Use a 2-level model so D=5 survives two MaxPool3d ops (5→2→1).
    # The upsampling path (1→2, 2→4) triggers the padding logic because
    # 4 < 5, ensuring spatial alignment with the skip connections.
    _odd_model = AttentionUNet3D(
        in_channels=3, out_channels=1, features=(16, 32)
    ).to(DEVICE)
    odd_input = torch.randn(1, 3, 5, 33, 33, device=DEVICE)
    with torch.no_grad():
        out_odd = _odd_model(odd_input)
    if out_odd.shape == (1, 1, 5, 33, 33):
        ok(f"AttentionUNet3D odd-dim forward pass OK — output shape {tuple(out_odd.shape)}")
    else:
        fail(f"AttentionUNet3D odd-dim: output shape {tuple(out_odd.shape)} != (1, 1, 5, 33, 33)")

    # --- build_model factory -----------------------------------------------
    _all_flags = {"use_t2w": True, "use_adc": True, "use_hbv": True}
    m_unet = build_model({"model": "unet3d", **_all_flags, "out_channels": 1,
                          "features": [16, 32]})
    ok(f"build_model('unet3d') → {type(m_unet).__name__}")

    m_attn = build_model({"model": "attention_unet3d", **_all_flags,
                          "out_channels": 1, "features": [16, 32]})
    ok(f"build_model('attention_unet3d') → {type(m_attn).__name__}")

    # Default (no 'model' key, no flags) must produce UNet3D with 3 channels
    m_default = build_model({"out_channels": 1})
    if type(m_default).__name__ == "UNet3D":
        ok("build_model with no 'model' key defaults to UNet3D")
    else:
        fail(f"Default model should be UNet3D, got {type(m_default).__name__}")

    # Unknown model name must raise ValueError
    try:
        build_model({"model": "nonexistent_model"})
        fail("build_model should raise ValueError for unknown model name")
    except ValueError as e:
        ok(f"build_model raises ValueError for unknown model: {e}")

except Exception as exc:
    fail("AttentionUNet3D / build_model test failed", exc)

# ---------------------------------------------------------------------------
# 3c. Modality flag selection: build_model derives in_channels from flags
# ---------------------------------------------------------------------------
section("3c. Modality flag selection (use_t2w / use_adc / use_hbv)")

try:
    from models import build_model as _bm

    # --- All three enabled → in_channels = 3 ---------------------------------
    m3 = _bm({"use_t2w": True, "use_adc": True, "use_hbv": True,
               "out_channels": 1, "features": [16, 32]})
    _enc0 = list(m3.encoders.children())[0] if hasattr(m3, "encoders") else None
    _c3 = _enc0[0].in_channels if _enc0 is not None else None
    if _c3 == 3:
        ok("all flags True  → model in_channels = 3")
    else:
        fail(f"all flags True: expected in_channels=3, first Conv3d has {_c3}")

    # --- T2w + ADC only (HBV disabled) → in_channels = 2 --------------------
    m2 = _bm({"use_t2w": True, "use_adc": True, "use_hbv": False,
               "out_channels": 1, "features": [16, 32]})
    _enc0_2 = list(m2.encoders.children())[0] if hasattr(m2, "encoders") else None
    _c2 = _enc0_2[0].in_channels if _enc0_2 is not None else None
    if _c2 == 2:
        ok("use_hbv=False   → model in_channels = 2")
    else:
        fail(f"use_hbv=False: expected in_channels=2, first Conv3d has {_c2}")

    # A 2-channel model must also produce the correct output shape
    B, C, D, H, W = 1, 2, 20, 32, 32
    dummy2 = torch.randn(B, C, D, H, W, device=DEVICE)
    m2 = m2.to(DEVICE)
    with torch.no_grad():
        out2 = m2(dummy2)
    if out2.shape == (B, 1, D, H, W):
        ok(f"2-channel forward pass OK — output shape {tuple(out2.shape)}")
    else:
        fail(f"2-channel forward: shape {tuple(out2.shape)} != {(B, 1, D, H, W)}")

    # --- T2w only → in_channels = 1 ------------------------------------------
    m1 = _bm({"use_t2w": True, "use_adc": False, "use_hbv": False,
               "out_channels": 1, "features": [16, 32]})
    _enc0_1 = list(m1.encoders.children())[0] if hasattr(m1, "encoders") else None
    _c1 = _enc0_1[0].in_channels if _enc0_1 is not None else None
    if _c1 == 1:
        ok("use_adc=False, use_hbv=False → model in_channels = 1")
    else:
        fail(f"t2w-only: expected in_channels=1, first Conv3d has {_c1}")

    # --- No flags set → defaults to all True (in_channels = 3) ---------------
    m_defaults = _bm({"out_channels": 1, "features": [16, 32]})
    _enc0_d = list(m_defaults.encoders.children())[0] if hasattr(m_defaults, "encoders") else None
    _cd = _enc0_d[0].in_channels if _enc0_d is not None else None
    if _cd == 3:
        ok("no flags in config → all default True → in_channels = 3")
    else:
        fail(f"no flags: expected in_channels=3, first Conv3d has {_cd}")

    # --- All flags False → must raise ValueError ------------------------------
    try:
        _bm({"use_t2w": False, "use_adc": False, "use_hbv": False})
        fail("build_model should raise ValueError when all modality flags are False")
    except ValueError as e:
        ok(f"all flags False → ValueError raised: {e}")

except Exception as exc:
    fail("Modality flag selection test failed", exc)

# ---------------------------------------------------------------------------
# 3d. Deep supervision: UNet3D, AttentionUNet3D, build_model, DeepSupervisionWrapper
# ---------------------------------------------------------------------------
section("3d. Deep supervision (UNet3D, AttentionUNet3D, build_model, DeepSupervisionWrapper)")

try:
    from models import AttentionUNet3D as _AttnUNet, UNet3D as _UNet, build_model as _bm_ds
    from losses import DeepSupervisionWrapper, DiceBCELoss as _DiceBCE

    _B, _C, _D, _H, _W = 2, 3, 20, 64, 64
    _FEATURES = (16, 32, 64, 128)                # 4 levels → 3 auxiliary heads
    _N_LEVELS = len(_FEATURES)                   # 4

    # --- 3d-i. UNet3D(deep_supervision=True) returns a list ------------------
    ds_unet = _UNet(in_channels=_C, out_channels=1, features=_FEATURES,
                    deep_supervision=True).to(DEVICE)
    _inp = torch.randn(_B, _C, _D, _H, _W, device=DEVICE)
    with torch.no_grad():
        ds_out = ds_unet(_inp)

    if isinstance(ds_out, list):
        ok(f"UNet3D DS=True → list of {len(ds_out)} tensors")
    else:
        fail(f"UNet3D DS=True should return list, got {type(ds_out).__name__}")

    # List length must equal number of feature levels
    if len(ds_out) == _N_LEVELS:
        ok(f"UNet3D DS list length == {_N_LEVELS} (== len(features))")
    else:
        fail(f"UNet3D DS list length {len(ds_out)} != {_N_LEVELS}")

    # output[0] must be full resolution
    if ds_out[0].shape == (_B, 1, _D, _H, _W):
        ok(f"UNet3D DS output[0] full-res shape {tuple(ds_out[0].shape)}")
    else:
        fail(f"UNet3D DS output[0] shape {tuple(ds_out[0].shape)} != {(_B, 1, _D, _H, _W)}")

    # Each subsequent output must be (approximately) half the spatial size
    _ds_shapes_ok = True
    for _lvl in range(1, len(ds_out)):
        _prev = ds_out[_lvl - 1].shape[2:]
        _curr = ds_out[_lvl].shape[2:]
        # Allow for floor division (e.g. 20 // 2 = 10)
        _expected = tuple(s // 2 for s in _prev)
        if _curr != _expected:
            fail(
                f"UNet3D DS scale {_lvl}: shape {tuple(_curr)} "
                f"!= expected half of {tuple(_prev)} = {_expected}"
            )
            _ds_shapes_ok = False
    if _ds_shapes_ok:
        ok("UNet3D DS: each auxiliary output is half the spatial resolution of the previous")

    # --- 3d-ii. UNet3D(deep_supervision=False) regression guard ---------------
    plain_unet = _UNet(in_channels=_C, out_channels=1, features=_FEATURES,
                       deep_supervision=False).to(DEVICE)
    with torch.no_grad():
        plain_out = plain_unet(_inp)

    if isinstance(plain_out, torch.Tensor) and not isinstance(plain_out, list):
        ok(f"UNet3D DS=False still returns plain Tensor (regression guard)")
    else:
        fail(f"UNet3D DS=False should return Tensor, got {type(plain_out).__name__}")

    # --- 3d-iii. AttentionUNet3D(deep_supervision=True) ----------------------
    ds_attn = _AttnUNet(in_channels=_C, out_channels=1, features=_FEATURES,
                        deep_supervision=True).to(DEVICE)
    with torch.no_grad():
        ds_attn_out = ds_attn(_inp)

    if isinstance(ds_attn_out, list) and len(ds_attn_out) == _N_LEVELS:
        ok(f"AttentionUNet3D DS=True → list of {len(ds_attn_out)} tensors")
    else:
        fail(
            f"AttentionUNet3D DS=True: expected list[{_N_LEVELS}], "
            f"got {type(ds_attn_out).__name__} len={len(ds_attn_out) if isinstance(ds_attn_out, list) else 'N/A'}"
        )

    if isinstance(ds_attn_out, list) and ds_attn_out[0].shape == (_B, 1, _D, _H, _W):
        ok(f"AttentionUNet3D DS output[0] full-res shape {tuple(ds_attn_out[0].shape)}")
    else:
        _shape = tuple(ds_attn_out[0].shape) if isinstance(ds_attn_out, list) else "N/A"
        fail(f"AttentionUNet3D DS output[0] shape {_shape} != {(_B, 1, _D, _H, _W)}")

    # --- 3d-iv. build_model with deep_supervision=True -----------------------
    _all_flags_ds = {"use_t2w": True, "use_adc": True, "use_hbv": True}
    m_ds = _bm_ds({
        "model": "unet3d", **_all_flags_ds,
        "out_channels": 1,
        "features": list(_FEATURES),
        "deep_supervision": True,
    }).to(DEVICE)
    with torch.no_grad():
        bm_out = m_ds(_inp)
    if isinstance(bm_out, list) and len(bm_out) == _N_LEVELS:
        ok(f"build_model(deep_supervision=True) → list of {len(bm_out)} tensors")
    else:
        fail(
            f"build_model DS=True: expected list[{_N_LEVELS}], "
            f"got {type(bm_out).__name__}"
        )

    # --- 3d-v. DeepSupervisionWrapper with list input -------------------------
    _base_crit = _DiceBCE(dice_weight=1.0, bce_weight=1.0)
    ds_crit = DeepSupervisionWrapper(_base_crit, num_levels=_N_LEVELS)

    ds_unet.train()
    _inp_t = torch.randn(_B, _C, _D, _H, _W, device=DEVICE, requires_grad=False)
    _tgt = (torch.rand(_B, 1, _D, _H, _W, device=DEVICE) > 0.8).float()
    _list_out = ds_unet(_inp_t)
    _ds_loss = ds_crit(_list_out, _tgt)

    if torch.isfinite(_ds_loss) and _ds_loss.shape == ():
        ok(f"DeepSupervisionWrapper list input → scalar loss={_ds_loss.item():.4f}")
    else:
        fail(f"DeepSupervisionWrapper list input: loss={_ds_loss}, shape={_ds_loss.shape}")

    # Backward must succeed
    try:
        _ds_loss.backward()
        ok("DeepSupervisionWrapper backward pass OK")
    except Exception as _bwd_exc:
        fail("DeepSupervisionWrapper backward pass failed", _bwd_exc)

    # --- 3d-vi. DeepSupervisionWrapper with plain Tensor (delegation) ---------
    plain_unet.train()
    _plain_out = plain_unet(_inp_t)
    _plain_loss = ds_crit(_plain_out, _tgt)
    _base_loss  = _base_crit(_plain_out.detach(), _tgt)

    if torch.isfinite(_plain_loss) and abs(_plain_loss.item() - _base_loss.item()) < 1e-5:
        ok(
            f"DeepSupervisionWrapper delegates plain Tensor to base criterion "
            f"(loss={_plain_loss.item():.4f})"
        )
    else:
        fail(
            f"DeepSupervisionWrapper Tensor delegation mismatch: "
            f"wrapper={_plain_loss.item():.6f}, base={_base_loss.item():.6f}"
        )

    # --- 3d-vii. Weight vector sums to 1.0 and is decreasing -----------------
    _w = ds_crit.weights
    if abs(sum(_w) - 1.0) < 1e-6:
        ok(f"DeepSupervisionWrapper weights sum to 1.0: {[f'{v:.3f}' for v in _w]}")
    else:
        fail(f"DeepSupervisionWrapper weights do not sum to 1: sum={sum(_w):.6f}")
    if all(_w[i] >= _w[i + 1] for i in range(len(_w) - 1)):
        ok("DeepSupervisionWrapper weights are decreasing (finest → coarsest)")
    else:
        fail(f"DeepSupervisionWrapper weights are not monotonically decreasing: {_w}")

except Exception as exc:
    fail("Deep supervision test failed", exc)

# ---------------------------------------------------------------------------
# 3e. Deconver: instantiation, forward pass, build_model factory,
#     deep supervision via num_deep_supr
# ---------------------------------------------------------------------------
section("3e. Deconver (NDC-based U-Net) + build_model factory")

try:
    from models import build_model as _bm_dconv

    _all_flags_dconv = {"use_t2w": True, "use_adc": True, "use_hbv": True}

    # --- 3e-i. build_model factory (no deep supervision) ----------------------
    _dconv_cfg_base: dict = {
        "model": "deconver",
        **_all_flags_dconv,
        "out_channels": 1,
        # Slim widths to keep memory low in the smoke test
        "deconver_encoder_depth": [1, 1, 1],
        "deconver_encoder_width": [16, 32, 64],
        "deconver_strides": [1, 2, 2],
        "deconver_kernel_size": [3, 3, 3],
        "deconver_groups": -1,
        "deconver_ndc_ratio": 2,
        "deep_supervision": False,
    }

    dconv_model = _bm_dconv(_dconv_cfg_base).to(DEVICE)
    n_dconv = sum(p.numel() for p in dconv_model.parameters())
    ok(f"build_model('deconver') → {type(dconv_model).__name__} — {n_dconv:,} parameters")

    # --- 3e-ii. Forward pass: output shape must match input spatial dims -------
    B, C, D, H, W = 1, 3, 20, 32, 32
    _dconv_inp = torch.randn(B, C, D, H, W, device=DEVICE)
    with torch.no_grad():
        _dconv_out = dconv_model(_dconv_inp)

    _dconv_expected = (B, 1, D, H, W)
    if isinstance(_dconv_out, torch.Tensor) and _dconv_out.shape == _dconv_expected:
        ok(f"Deconver forward pass OK — output shape {tuple(_dconv_out.shape)}")
    else:
        _got = tuple(_dconv_out.shape) if isinstance(_dconv_out, torch.Tensor) else type(_dconv_out).__name__
        fail(f"Deconver forward: got {_got}, expected {_dconv_expected}")

    # --- 3e-iii. Deep supervision (num_deep_supr) returns a list ---------------
    _dconv_cfg_ds = dict(_dconv_cfg_base)
    _dconv_cfg_ds["deep_supervision"] = True

    dconv_ds_model = _bm_dconv(_dconv_cfg_ds).to(DEVICE)
    with torch.no_grad():
        _dconv_ds_out = dconv_ds_model(_dconv_inp)

    if isinstance(_dconv_ds_out, (list, tuple)) and len(_dconv_ds_out) > 0:
        ok(
            f"Deconver DS=True → {type(_dconv_ds_out).__name__} of "
            f"{len(_dconv_ds_out)} tensors"
        )
        # The first (finest) output must match the input spatial resolution
        _dconv_ds_main = _dconv_ds_out[0]
        if _dconv_ds_main.shape == _dconv_expected:
            ok(f"Deconver DS output[0] full-res shape {tuple(_dconv_ds_main.shape)}")
        else:
            fail(
                f"Deconver DS output[0] shape {tuple(_dconv_ds_main.shape)} "
                f"!= expected {_dconv_expected}"
            )
    elif isinstance(_dconv_ds_out, torch.Tensor) and _dconv_ds_out.shape == _dconv_expected:
        # Some Deconver builds return a plain tensor even when num_deep_supr is set
        ok(
            f"Deconver DS=True → plain Tensor {tuple(_dconv_ds_out.shape)} "
            f"(built-in DS may require num_deep_supr > 1 stages)"
        )
    else:
        _got_ds = (
            tuple(_dconv_ds_out.shape)
            if isinstance(_dconv_ds_out, torch.Tensor)
            else f"{type(_dconv_ds_out).__name__}(len={len(_dconv_ds_out)})"
        )
        fail(f"Deconver DS=True unexpected output: {_got_ds}")

    # --- 3e-iv. Unknown model still raises ValueError -------------------------
    try:
        _bm_dconv({"model": "deconver_nonexistent"})
        fail("build_model should raise ValueError for unknown model name")
    except ValueError:
        ok("build_model raises ValueError for unknown model name (regression guard)")

    # --- 3e-v. Per-stage encoder stride wiring regression guard --------------
    # Ensure non-uniform stride tuples are passed through as configured
    # (and not hardcoded to stride=2).
    _dconv_cfg_stride = dict(_dconv_cfg_base)
    _dconv_cfg_stride["deconver_strides"] = [1, (2, 1, 1), 2]
    _dconv_stride_model = _bm_dconv(_dconv_cfg_stride).to(DEVICE)
    _stage1_downsample = _dconv_stride_model.encoder.blocks[1].downsample
    _actual_stride = getattr(_stage1_downsample, "stride", None)
    _expected_stride = (2, 1, 1)

    if _actual_stride == _expected_stride:
        ok(
            "Deconver stage stride wiring OK — "
            f"stage[1] downsample stride={_actual_stride}"
        )
    else:
        fail(
            "Deconver stage stride wiring mismatch: "
            f"got {_actual_stride}, expected {_expected_stride}"
        )

except Exception as exc:
    fail("Deconver test failed", exc)

# ---------------------------------------------------------------------------
# 3f. FCT: instantiation, forward pass, build_model factory, deep supervision
# ---------------------------------------------------------------------------
section("3f. FCT (slice-wise fully convolutional transformer) + build_model factory")

try:
    from models import build_model as _bm_fct

    _all_flags_fct = {"use_t2w": True, "use_adc": True, "use_hbv": True}

    _fct_cfg_base: dict = {
        "model": "fct",
        **_all_flags_fct,
        "out_channels": 1,
        # Keep this lightweight for CI/smoke runs.
        "features": [16, 32],
        "fct_heads": [1, 2],
        "fct_patch_kernel_size": 5,
        "fct_patch_strides": [2, 2],
        "fct_bottleneck_channels": 64,
        "fct_wide_focus_dilations": [1, 2, 3],
        "fct_dropout": 0.0,
        "deep_supervision": False,
    }

    fct_model = _bm_fct(_fct_cfg_base).to(DEVICE)
    n_fct = sum(p.numel() for p in fct_model.parameters())
    ok(f"build_model('fct') → {type(fct_model).__name__} — {n_fct:,} parameters")

    B, C, D, H, W = 1, 3, 8, 32, 32
    _fct_inp = torch.randn(B, C, D, H, W, device=DEVICE)
    with torch.no_grad():
        _fct_out = fct_model(_fct_inp)

    _fct_expected = (B, 1, D, H, W)
    if isinstance(_fct_out, torch.Tensor) and _fct_out.shape == _fct_expected:
        ok(f"FCT forward pass OK — output shape {tuple(_fct_out.shape)}")
    else:
        _got = tuple(_fct_out.shape) if isinstance(_fct_out, torch.Tensor) else type(_fct_out).__name__
        fail(f"FCT forward: got {_got}, expected {_fct_expected}")

    _fct_cfg_ds = dict(_fct_cfg_base)
    _fct_cfg_ds["deep_supervision"] = True
    fct_ds_model = _bm_fct(_fct_cfg_ds).to(DEVICE)
    with torch.no_grad():
        _fct_ds_out = fct_ds_model(_fct_inp)

    if isinstance(_fct_ds_out, list) and len(_fct_ds_out) == len(_fct_cfg_base["features"]):
        ok(f"FCT DS=True → list of {len(_fct_ds_out)} tensors")
        if _fct_ds_out[0].shape == _fct_expected:
            ok(f"FCT DS output[0] full-res shape {tuple(_fct_ds_out[0].shape)}")
        else:
            fail(
                f"FCT DS output[0] shape {tuple(_fct_ds_out[0].shape)} "
                f"!= expected {_fct_expected}"
            )
    else:
        _len = len(_fct_ds_out) if isinstance(_fct_ds_out, list) else "N/A"
        fail(
            "FCT DS=True should return list with len(features); "
            f"got {type(_fct_ds_out).__name__}(len={_len})"
        )

except Exception as exc:
    fail("FCT test failed", exc)

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
# 4b. TverskyBCELoss: forward pass, tversky_loss function, FN-penalty bias
# ---------------------------------------------------------------------------
section("4b. TverskyBCELoss + tversky_loss function")

try:
    from losses import TverskyBCELoss, tversky_loss

    # --- tversky_loss standalone function ------------------------------------
    tv_logits  = torch.randn(2, 1, 20, 64, 64, device=DEVICE)
    tv_targets = (torch.rand(2, 1, 20, 64, 64, device=DEVICE) > 0.8).float()

    tv_scalar = tversky_loss(tv_logits, tv_targets, alpha=0.3, beta=0.7)
    if torch.isfinite(tv_scalar) and 0.0 <= tv_scalar.item() <= 1.0:
        ok(f"tversky_loss (alpha=0.3, beta=0.7) → {tv_scalar.item():.4f} (in [0,1])")
    else:
        fail(f"tversky_loss returned unexpected value: {tv_scalar.item()}")

    # Verify alpha=beta=0.5 recovers a value close to Dice loss
    from losses import dice_loss
    dl_val = dice_loss(tv_logits, tv_targets).item()
    tl_sym = tversky_loss(tv_logits, tv_targets, alpha=0.5, beta=0.5).item()
    if abs(dl_val - tl_sym) < 1e-4:
        ok(f"tversky_loss(alpha=0.5, beta=0.5) == dice_loss ({tl_sym:.6f} ≈ {dl_val:.6f})")
    else:
        fail(
            f"tversky_loss(0.5,0.5) should equal dice_loss but got "
            f"{tl_sym:.6f} vs {dl_val:.6f}"
        )

    # --- TverskyBCELoss class: basic forward pass ----------------------------
    criterion_tv = TverskyBCELoss(
        tversky_weight=3.0, bce_weight=1.0,
        alpha=0.3, beta=0.7, pos_weight=50.0,
    )
    criterion_tv = criterion_tv.to(DEVICE)
    loss_tv = criterion_tv(tv_logits, tv_targets)
    if torch.isfinite(loss_tv):
        ok(f"TverskyBCELoss forward pass — loss={loss_tv.item():.4f}")
    else:
        fail(f"TverskyBCELoss returned non-finite value: {loss_tv.item()}")

    # --- FN-penalty bias: high-beta should penalise false negatives more ------
    # Setup: logits strongly predict background (all-negative prediction)
    # against an all-positive ground truth → all false negatives.
    # High beta (FN weight) must produce a *higher* loss than high alpha (FP weight).
    neg_logits_tv  = torch.full((2, 1, 20, 64, 64), -5.0, device=DEVICE)
    pos_targets_tv = torch.ones(2, 1, 20, 64, 64, device=DEVICE)

    # alpha=0.1, beta=0.9 → FN penalised 9× more than FP
    loss_fn_heavy = tversky_loss(neg_logits_tv, pos_targets_tv, alpha=0.1, beta=0.9)
    # alpha=0.9, beta=0.1 → FP penalised 9× more than FN
    loss_fp_heavy = tversky_loss(neg_logits_tv, pos_targets_tv, alpha=0.9, beta=0.1)

    if loss_fn_heavy.item() > loss_fp_heavy.item():
        ok(
            f"FN-penalty bias: beta=0.9 loss ({loss_fn_heavy.item():.4f}) "
            f"> alpha=0.9 loss ({loss_fp_heavy.item():.4f})"
        )
    else:
        fail(
            f"FN-penalty bias: expected beta=0.9 loss > alpha=0.9 loss but got "
            f"{loss_fn_heavy.item():.4f} vs {loss_fp_heavy.item():.4f}"
        )

    # --- pos_weight wired through BCE correctly --------------------------------
    # Higher pos_weight should increase total loss on all-FN scenario
    criterion_tv_pw_low  = TverskyBCELoss(
        tversky_weight=1.0, bce_weight=1.0, pos_weight=1.0
    ).to(DEVICE)
    criterion_tv_pw_high = TverskyBCELoss(
        tversky_weight=1.0, bce_weight=1.0, pos_weight=50.0
    ).to(DEVICE)
    loss_pw_low  = criterion_tv_pw_low(neg_logits_tv, pos_targets_tv)
    loss_pw_high = criterion_tv_pw_high(neg_logits_tv, pos_targets_tv)
    if loss_pw_high.item() > loss_pw_low.item():
        ok(
            f"TverskyBCELoss pos_weight=50 raises loss on FN voxels: "
            f"{loss_pw_low.item():.4f} → {loss_pw_high.item():.4f}"
        )
    else:
        fail(
            f"TverskyBCELoss pos_weight did not raise loss as expected: "
            f"pw=1 → {loss_pw_low.item():.4f}, pw=50 → {loss_pw_high.item():.4f}"
        )

except Exception as exc:
    fail("TverskyBCELoss test failed", exc)

# ---------------------------------------------------------------------------
# 4d. Loss robustness: negative-sample exclusion + FP16 overflow guard
# ---------------------------------------------------------------------------
section("4d. Loss robustness — negative-sample exclusion + FP16 overflow guard")

try:
    from losses import dice_loss, tversky_loss  # noqa: F811 (re-import is fine)

    # --- 4d-i. Mixed batch: negative samples must be excluded ----------------
    # Create a 4-sample batch: samples 0,1 have lesion labels,
    # samples 2,3 have all-zero labels (negative cases).
    # The loss on the mixed batch must equal the loss on the positive
    # subset alone — negative samples must not contribute any gradient.
    torch.manual_seed(42)
    _pos_logits = torch.randn(2, 1, 20, 32, 32, device=DEVICE)
    _pos_targets = (torch.rand(2, 1, 20, 32, 32, device=DEVICE) > 0.7).float()
    _pos_targets[:, :, 10, 16, 16] = 1.0   # guarantee at least 1 positive voxel
    _neg_logits  = torch.randn(2, 1, 20, 32, 32, device=DEVICE)
    _neg_targets = torch.zeros(2, 1, 20, 32, 32, device=DEVICE)

    _logits_all  = torch.cat([_pos_logits,  _neg_logits],  dim=0)
    _targets_all = torch.cat([_pos_targets, _neg_targets], dim=0)

    dl_mixed    = dice_loss(_logits_all,  _targets_all).item()
    dl_pos_only = dice_loss(_pos_logits, _pos_targets).item()
    if abs(dl_mixed - dl_pos_only) < 1e-5:
        ok(
            f"dice_loss: mixed batch == positive-only subset "
            f"({dl_mixed:.6f} ≈ {dl_pos_only:.6f})"
        )
    else:
        fail(
            f"dice_loss: mixed batch ({dl_mixed:.6f}) != positive-only "
            f"({dl_pos_only:.6f}); negative samples contaminate gradient"
        )

    tl_mixed    = tversky_loss(_logits_all,  _targets_all).item()
    tl_pos_only = tversky_loss(_pos_logits, _pos_targets).item()
    if abs(tl_mixed - tl_pos_only) < 1e-5:
        ok(
            f"tversky_loss: mixed batch == positive-only subset "
            f"({tl_mixed:.6f} ≈ {tl_pos_only:.6f})"
        )
    else:
        fail(
            f"tversky_loss: mixed batch ({tl_mixed:.6f}) != positive-only "
            f"({tl_pos_only:.6f}); negative samples contaminate gradient"
        )

    # --- 4d-ii. All-negative batch must return 0.0 (no spurious gradient) ---
    _all_neg_logits  = torch.randn(4, 1, 10, 16, 16, device=DEVICE)
    _all_neg_targets = torch.zeros(4, 1, 10, 16, 16, device=DEVICE)

    dl_all_neg = dice_loss(_all_neg_logits, _all_neg_targets)
    if dl_all_neg.item() == 0.0 and torch.isfinite(dl_all_neg):
        ok("dice_loss: all-negative batch → 0.0 (no spurious gradient)")
    else:
        fail(
            f"dice_loss: all-negative batch should return 0.0 "
            f"but got {dl_all_neg.item():.6f}"
        )

    tl_all_neg = tversky_loss(_all_neg_logits, _all_neg_targets)
    if tl_all_neg.item() == 0.0 and torch.isfinite(tl_all_neg):
        ok("tversky_loss: all-negative batch → 0.0 (no spurious gradient)")
    else:
        fail(
            f"tversky_loss: all-negative batch should return 0.0 "
            f"but got {tl_all_neg.item():.6f}"
        )

    # --- 4d-iii. FP16 overflow guard (full training patch size) --------------
    # At patch size 20×128×128 = 327,680 voxels, sigmoid(0) ≈ 0.5 per voxel.
    # Summing ~0.5 × 327,680 ≈ 163,840 overflows FP16 max (≈65,504), producing
    # inf and locking the loss at 1.0 for the entire training run.
    # The .float() cast inside dice_loss / tversky_loss must prevent this.
    if cuda_ok:
        _fp16_logits  = torch.zeros(2, 1, 20, 128, 128,
                                    device=DEVICE, dtype=torch.float16)
        _fp16_targets = torch.zeros(2, 1, 20, 128, 128,
                                    device=DEVICE, dtype=torch.float16)
        # Guarantee positive voxels so the mask passes
        _fp16_targets[:, :, 10, 64, 64] = 1.0

        dl_fp16 = dice_loss(_fp16_logits, _fp16_targets)
        if torch.isfinite(dl_fp16):
            ok(
                f"dice_loss FP16 patch 20×128×128: "
                f"finite ({dl_fp16.item():.4f}), no overflow"
            )
        else:
            fail(
                f"dice_loss FP16 large-patch: non-finite {dl_fp16.item()} "
                f"— FP16 overflow not prevented"
            )

        tl_fp16 = tversky_loss(_fp16_logits, _fp16_targets)
        if torch.isfinite(tl_fp16):
            ok(
                f"tversky_loss FP16 patch 20×128×128: "
                f"finite ({tl_fp16.item():.4f}), no overflow"
            )
        else:
            fail(
                f"tversky_loss FP16 large-patch: non-finite {tl_fp16.item()} "
                f"— FP16 overflow not prevented"
            )
    else:
        skip("CUDA not available — skipping FP16 overflow guard test")

except Exception as exc:
    fail("Loss robustness test failed", exc)

# ---------------------------------------------------------------------------
# 4c. LR warmup: LinearLR warm-up + CosineAnnealingLR via SequentialLR
# ---------------------------------------------------------------------------
section("4c. LR warmup — LinearLR → CosineAnnealingLR (SequentialLR)")

try:
    _lr_base      = 4e-4
    _warmup_eps   = 10
    _total_eps    = 50
    _cosine_t_max = _total_eps - _warmup_eps  # 40
    _eta_min      = _lr_base * 1e-2           # 4e-6

    # Build a minimal model + optimizer to attach the scheduler to
    _sched_model = torch.nn.Linear(8, 1)
    _sched_opt   = torch.optim.AdamW(_sched_model.parameters(), lr=_lr_base)

    _warmup_sched = torch.optim.lr_scheduler.LinearLR(
        _sched_opt, start_factor=0.1, end_factor=1.0, total_iters=_warmup_eps
    )
    _cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        _sched_opt, T_max=_cosine_t_max, eta_min=_eta_min
    )
    _seq_sched = torch.optim.lr_scheduler.SequentialLR(
        _sched_opt,
        schedulers=[_warmup_sched, _cosine_sched],
        milestones=[_warmup_eps],
    )

    # --- Epoch 1: LR should be at start_factor * lr_base = 0.1 * 4e-4 --------
    # SequentialLR calls step() after construction, so LR at epoch 0 (before
    # any step) is already set; we call get_last_lr() after the first step.
    _seq_sched.step()   # epoch 1 → LinearLR step 1
    lr_ep1 = _seq_sched.get_last_lr()[0]

    # LinearLR: after step 1, LR = start + (end-start)*(1/total_iters)
    #           = 0.1*lr + 0.9*lr*(1/10) = lr * (0.1 + 0.09) = lr * 0.19
    # (The exact value depends on the LinearLR interpolation formula.)
    # What we *do* know: it must be strictly less than lr_base (warmup not done)
    # and strictly greater than start_factor * lr_base (already stepped once).
    if _lr_base * 0.1 < lr_ep1 < _lr_base:
        ok(f"Epoch 1: LR={lr_ep1:.2e} in warmup range (0.1×base, base)")
    else:
        fail(f"Epoch 1: LR={lr_ep1:.2e} outside expected warmup range")

    # --- Step through remaining warmup epochs and verify LR reaches ~base -----
    for _ in range(_warmup_eps - 1):  # already called step() once above
        _seq_sched.step()
    lr_end_warmup = _seq_sched.get_last_lr()[0]

    # After total_iters steps, LinearLR should have reached end_factor*lr_base
    if abs(lr_end_warmup - _lr_base) < _lr_base * 0.01:   # within 1 % of base
        ok(f"End of warmup ({_warmup_eps} epochs): LR≈base ({lr_end_warmup:.2e} ≈ {_lr_base:.2e})")
    else:
        fail(f"End of warmup: LR={lr_end_warmup:.2e}, expected ≈{_lr_base:.2e}")

    # --- Cosine phase: LR must decrease monotonically toward eta_min ----------
    _prev_lr = lr_end_warmup
    _decay_ok = True
    for ep in range(1, _cosine_t_max + 1):
        _seq_sched.step()
        _cur_lr = _seq_sched.get_last_lr()[0]
        if _cur_lr > _prev_lr + 1e-12:  # allow tiny float noise
            _decay_ok = False
            fail(
                f"Cosine phase: LR increased at step {ep}: "
                f"{_prev_lr:.2e} → {_cur_lr:.2e}"
            )
            break
        _prev_lr = _cur_lr

    if _decay_ok:
        ok(f"Cosine phase: LR decreases monotonically ({_lr_base:.2e} → {_prev_lr:.2e})")

    # Final LR should be close to eta_min
    if abs(_prev_lr - _eta_min) < _eta_min * 0.5:
        ok(f"End of cosine phase: LR={_prev_lr:.2e} ≈ eta_min={_eta_min:.2e}")
    else:
        fail(f"End of cosine: LR={_prev_lr:.2e}, expected ≈eta_min={_eta_min:.2e}")

    # --- warmup_epochs=0 falls back to pure cosine (backward compatibility) ---
    _opt0 = torch.optim.AdamW(torch.nn.Linear(4, 1).parameters(), lr=_lr_base)
    _cosine_only = torch.optim.lr_scheduler.CosineAnnealingLR(
        _opt0, T_max=_total_eps, eta_min=_eta_min
    )
    _cosine_only.step()
    lr_cosine_ep1 = _cosine_only.get_last_lr()[0]
    # Pure cosine should start very close to lr_base (no warmup ramp)
    if lr_cosine_ep1 > _lr_base * 0.95:
        ok(f"warmup_epochs=0: pure cosine starts near base LR ({lr_cosine_ep1:.2e})")
    else:
        fail(
            f"warmup_epochs=0: pure cosine LR at ep1={lr_cosine_ep1:.2e} "
            f"is too low (expected > {_lr_base * 0.95:.2e})"
        )

except Exception as exc:
    fail("LR warmup / SequentialLR test failed", exc)

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
    for name in ("dice", "iou", "sensitivity", "precision"):
        if _math.isnan(m_empty[name]):
            ok(f"empty-target guard: {name} correctly returns nan")
        else:
            fail(
                f"empty-target guard: {name} should be nan for all-zero target "
                f"but got {m_empty[name]:.4f}"
            )

    # --- Empty-target guard: all-zero target + non-zero prediction -----------
    # dice / iou / sensitivity / precision must still be nan (target has no positives).
    pos_logits = torch.full((1, 1, 20, 64, 64), 10.0, device=DEVICE)   # pred = 1 everywhere

    m_fp = compute_all_metrics(pos_logits, zero_targets)

    for name in ("dice", "iou", "sensitivity", "precision"):
        if _math.isnan(m_fp[name]):
            ok(f"empty-target / FP guard: {name} correctly returns nan")
        else:
            fail(
                f"empty-target / FP guard: {name} should be nan but got {m_fp[name]:.4f}"
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

        # ---- 6b-partial. discover_cases with active_keys subset ----
        # Only T2w + ADC active; HBV files exist but must not be required.
        cases_partial = discover_cases(
            images_dir, labels_dir, active_keys=["t2w", "adc"]
        )
        if len(cases_partial) == N_CASES:
            ok(f"active_keys=['t2w','adc']: still found {len(cases_partial)} cases")
        else:
            fail(
                f"active_keys=['t2w','adc']: expected {N_CASES}, "
                f"got {len(cases_partial)}"
            )
        # Each case should have 't2w' and 'adc' but NOT 'hbv'
        has_no_hbv = all("hbv" not in c for c in cases_partial)
        if has_no_hbv:
            ok("partial active_keys: 'hbv' absent from case dicts")
        else:
            fail("partial active_keys: 'hbv' unexpectedly present in case dicts")

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
# 6d. Prostate158 helpers: discover_unlabeled_cases + DWI preprocess
# ---------------------------------------------------------------------------
section("6d. Prostate158 helpers (discover_unlabeled_cases + DWI-as-HBV preprocess)")

try:
    import tempfile

    import numpy as np

    from dataset import (
        _preprocess_dwi_as_hbv,
        discover_prostate158_cases,
        discover_unlabeled_cases,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        unlabeled_dir = root / "unlabeled_images"
        unlabeled_dir.mkdir(parents=True, exist_ok=True)

        case_complete = "00020"
        case_missing = "00021"

        # Complete triplet (should be discovered when HBV active).
        for suffix in ("_t2.nii.gz", "_adc.nii.gz", "_dwi.nii.gz"):
            (unlabeled_dir / f"{case_complete}{suffix}").touch()

        # Missing DWI (should be skipped when HBV active).
        for suffix in ("_t2.nii.gz", "_adc.nii.gz"):
            (unlabeled_dir / f"{case_missing}{suffix}").touch()

        cases_all = discover_unlabeled_cases(
            unlabeled_dir, active_keys=["t2w", "adc", "hbv"]
        )
        if len(cases_all) == 1 and cases_all[0]["case_id"] == case_complete:
            ok("discover_unlabeled_cases keeps only complete t2/adc/dwi triplets")
        else:
            fail(
                f"discover_unlabeled_cases expected 1 complete case, got {len(cases_all)}"
            )

        if cases_all and cases_all[0].get("hbv_source") == "dwi":
            ok("discover_unlabeled_cases maps dwi -> hbv and tags hbv_source='dwi'")
        else:
            fail("hbv_source='dwi' missing on discovered unlabeled case")

        cases_no_hbv = discover_unlabeled_cases(
            unlabeled_dir, active_keys=["t2w", "adc"]
        )
        if len(cases_no_hbv) == 2:
            ok("active_keys=['t2w','adc'] does not require DWI")
        else:
            fail(
                "active_keys=['t2w','adc'] should include both synthetic cases "
                f"(got {len(cases_no_hbv)})"
            )

        if all("hbv" not in c for c in cases_no_hbv):
            ok("inactive HBV key is absent from unlabeled case dicts")
        else:
            fail("HBV key unexpectedly present when active_keys excludes 'hbv'")

        prostate_root = root / "prostate158_train"
        case_dir = prostate_root / "train" / "020"
        case_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "t2.nii.gz",
            "adc.nii.gz",
            "dwi.nii.gz",
            "adc_tumor_reader1.nii.gz",
            "t2_prostate_reader1.nii.gz",
        ):
            (case_dir / filename).touch()
        (prostate_root / "train.csv").write_text(
            "t2,adc,dwi,adc_tumor_reader1,t2_prostate_reader1\n"
            "train/020/t2.nii.gz,train/020/adc.nii.gz,"
            "train/020/dwi.nii.gz,train/020/adc_tumor_reader1.nii.gz,"
            "train/020/t2_prostate_reader1.nii.gz\n"
        )

        prostate_cases = discover_prostate158_cases(
            prostate_root,
            split="train",
            active_keys=["t2w", "adc", "hbv"],
        )
        if len(prostate_cases) == 1 and prostate_cases[0]["case_id"] == "020":
            ok("discover_prostate158_cases reads upstream CSV rows")
        else:
            fail(
                "discover_prostate158_cases expected one CSV case "
                f"(got {len(prostate_cases)})"
            )

        if prostate_cases and prostate_cases[0].get("hbv_source") == "dwi":
            ok("discover_prostate158_cases maps dwi -> hbv")
        else:
            fail("discover_prostate158_cases did not tag DWI as HBV source")

        expected_label = case_dir / "adc_tumor_reader1.nii.gz"
        if prostate_cases and prostate_cases[0].get("label") == expected_label:
            ok("discover_prostate158_cases selects adc_tumor_reader1 by default")
        else:
            fail("discover_prostate158_cases selected the wrong default label")

        prostate_cases_with_gland = discover_prostate158_cases(
            prostate_root,
            split="train",
            active_keys=["t2w", "adc", "hbv"],
            prostate_label_col="t2_prostate_reader1",
        )
        expected_prostate = case_dir / "t2_prostate_reader1.nii.gz"
        if (
            prostate_cases_with_gland
            and prostate_cases_with_gland[0].get("prostate_label") == expected_prostate
        ):
            ok("discover_prostate158_cases attaches prostate_label when requested")
        else:
            fail("discover_prostate158_cases did not attach requested prostate_label")

    raw = np.array([-5.0, 0.0, 1.0, np.nan, np.inf], dtype=np.float32)
    proc = _preprocess_dwi_as_hbv(
        raw,
        clip_percentiles=(0.0, 100.0),
        use_log1p=True,
    )

    if np.isfinite(proc).all():
        ok("_preprocess_dwi_as_hbv removes NaN/inf")
    else:
        fail("_preprocess_dwi_as_hbv still contains non-finite values")

    if proc.min() >= 0.0:
        ok("_preprocess_dwi_as_hbv clamps negatives to non-negative range")
    else:
        fail(f"_preprocess_dwi_as_hbv produced negative values (min={proc.min():.4f})")

    expected_log1p_one = np.log1p(1.0)
    if np.any(np.isclose(proc, expected_log1p_one, atol=1e-6)):
        ok("_preprocess_dwi_as_hbv applies log1p when enabled")
    else:
        fail("_preprocess_dwi_as_hbv did not apply log1p as expected")

    proc_no_log = _preprocess_dwi_as_hbv(
        np.array([0.0, 1.0, 10.0], dtype=np.float32),
        clip_percentiles=(0.0, 100.0),
        use_log1p=False,
    )
    if np.isclose(proc_no_log[-1], 10.0, atol=1e-6):
        ok("_preprocess_dwi_as_hbv respects use_log1p=False")
    else:
        fail(
            "_preprocess_dwi_as_hbv use_log1p=False mismatch: "
            f"got {proc_no_log[-1]:.4f}, expected 10.0"
        )

except Exception as exc:
    fail("Prostate158 helper test failed", exc)

# ---------------------------------------------------------------------------
# 6e. ROI helpers + ROI dataset path
# ---------------------------------------------------------------------------
section("6e. ROI helpers and ROI dataset path")

if _sitk is None:
    skip("SimpleITK not installed — skipping ROI dataset tests")
else:
    try:
        import tempfile

        import numpy as np
        import SimpleITK as sitk  # noqa: N813

        from dataset import PiCaiDataset
        from postprocess import logits_to_binary_mask
        from roi import (
            compute_crop_bounds,
            restore_from_roi,
            resolve_roi_settings,
            validate_task_and_roi_config,
        )
        from transforms import get_train_transforms

        try:
            validate_task_and_roi_config(
                {
                    "task": "lesion_segmentation",
                    "roi": {"mode": "gt_mask"},
                    "prostate158_prostate_label_col": "t2_prostate_reader1",
                },
                "picai",
            )
            fail("gt_mask ROI should fail fast on dataset_type='picai'")
        except ValueError:
            ok("gt_mask ROI fails fast on non-prostate158 datasets")

        default_roi_settings = resolve_roi_settings(
            {
                "roi": {
                    "mode": "predicted_mask",
                    "localizer_run": "x",
                }
            }
        )
        if default_roi_settings.margin_mm == (3.0, 6.0, 6.0):
            ok("ROI settings use halved code default margin when margin_mm is omitted")
        else:
            fail(
                "ROI default margin mismatch: "
                f"got {default_roi_settings.margin_mm}, expected (3.0, 6.0, 6.0)"
            )

        explicit_roi_settings = resolve_roi_settings(
            {
                "roi": {
                    "mode": "predicted_mask",
                    "localizer_run": "x",
                    "margin_mm": [1.0, 2.0, 3.0],
                }
            }
        )
        if explicit_roi_settings.margin_mm == (1.0, 2.0, 3.0):
            ok("ROI settings preserve explicit config margin override")
        else:
            fail(
                "ROI explicit margin override mismatch: "
                f"got {explicit_roi_settings.margin_mm}, expected (1.0, 2.0, 3.0)"
            )

        mask = np.zeros((12, 40, 40), dtype=np.uint8)
        mask[3:6, 10:20, 15:25] = 1
        bounds = compute_crop_bounds(
            mask=mask,
            spacing_zyx=(3.0, 0.5, 0.5),
            margin_mm=(0.0, 0.0, 0.0),
            min_size_vox=(4, 16, 16),
            fallback_to_full_volume=True,
        )
        roi_logits = torch.ones(1, 1, 4, 16, 16)
        restored = restore_from_roi(roi_logits, bounds.as_dict())
        if restored.shape == (1, 1, 12, 40, 40) and float(restored.sum()) == float(roi_logits.sum()):
            ok("ROI restore pastes ROI-space tensor back into full-volume grid")
        else:
            fail(f"ROI restore shape/content mismatch: {tuple(restored.shape)}")

        restored_bin = logits_to_binary_mask(restored, threshold=0.5)
        if int(restored_bin.sum().item()) == int(roi_logits.numel()):
            ok("ROI-restored zero-padding stays background after thresholding")
        else:
            fail("ROI-restored zero-padding became foreground after thresholding")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            labels_dir = root / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            case_id = "roi_case"
            image_arr = np.zeros((20, 96, 96), dtype=np.float32)
            image_arr[:, 20:76, 20:76] = 100.0
            lesion_arr = np.zeros((20, 96, 96), dtype=np.uint8)
            lesion_arr[8:12, 42:54, 44:56] = 1
            prostate_arr = np.zeros((20, 96, 96), dtype=np.uint8)
            prostate_arr[6:14, 28:72, 30:74] = 1

            def _write_image(path: Path, arr: np.ndarray, spacing_xyz: tuple[float, float, float]) -> None:
                img = sitk.GetImageFromArray(arr)
                img.SetSpacing(spacing_xyz)
                sitk.WriteImage(img, str(path))

            _write_image(images_dir / f"{case_id}_t2w.mha", image_arr, (0.5, 0.5, 3.0))
            _write_image(labels_dir / f"{case_id}.nii.gz", lesion_arr, (0.5, 0.5, 3.0))
            prostate_path = labels_dir / f"{case_id}_prostate.nii.gz"
            _write_image(prostate_path, prostate_arr, (0.5, 0.5, 3.0))

            case = {
                "case_id": case_id,
                "t2w": images_dir / f"{case_id}_t2w.mha",
                "label": labels_dir / f"{case_id}.nii.gz",
                "prostate_label": prostate_path,
            }

            roi_cfg = {
                "task": "lesion_segmentation",
                "roi": {
                    "mode": "gt_mask",
                    "margin_mm": [0.0, 0.0, 0.0],
                    "min_size_vox": [4, 32, 32],
                    "fallback_to_full_volume": True,
                },
                "prostate158_prostate_label_col": "t2_prostate_reader1",
            }
            _, roi_settings = validate_task_and_roi_config(roi_cfg, "prostate158")

            ds_gt = PiCaiDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                target_spacing=(3.0, 0.5, 0.5),
                transform=None,
                cases=[case],
                active_modalities=["t2w"],
                task="lesion_segmentation",
                roi_settings=roi_settings,
                include_full_resampled=True,
            )
            sample_gt = ds_gt[0]
            if sample_gt["image"].shape[1:] == (8, 44, 44) and sample_gt["full_label"].shape[1:] == (20, 96, 96):
                ok("gt_mask ROI crops lesion sample and keeps full_label for restoration")
            else:
                fail(
                    "gt_mask ROI sample shapes mismatch: "
                    f"crop={tuple(sample_gt['image'].shape)}, full={tuple(sample_gt['full_label'].shape)}"
                )

            ds_loc = PiCaiDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                target_spacing=(3.0, 0.5, 0.5),
                transform=None,
                cases=[case],
                active_modalities=["t2w"],
                task="prostate_localization",
                roi_settings=roi_settings,
            )
            sample_loc = ds_loc[0]
            if sample_loc["label"].shape[1:] == (20, 96, 96) and int(sample_loc["label"].sum().item()) > 0:
                ok("prostate_localization task switches supervision target to prostate mask")
            else:
                fail("prostate_localization task did not expose prostate mask target")

            ds_pred = PiCaiDataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                target_spacing=(3.0, 0.5, 0.5),
                transform=get_train_transforms(patch_size=(4, 32, 32), num_samples=1),
                cases=[dict(case)],
                active_modalities=["t2w"],
                task="lesion_segmentation",
                roi_settings=validate_task_and_roi_config(
                    {
                        "task": "lesion_segmentation",
                        "roi": {
                            "mode": "predicted_mask",
                            "localizer_run": str(root / "fake_localizer"),
                            "margin_mm": [0.0, 0.0, 0.0],
                            "min_size_vox": [4, 32, 32],
                        },
                    },
                    "prostate158",
                )[1],
            )

            class _StubLocalizer:
                def predict_mask(self, _case: dict) -> np.ndarray:
                    return prostate_arr.astype(np.float32)

            ds_pred._roi_localizer = _StubLocalizer()  # type: ignore[assignment]
            sample_pred = ds_pred[0]
            if isinstance(sample_pred, list):
                sample_pred = sample_pred[0]
            if sample_pred["image"].shape == (1, 4, 32, 32):
                ok("predicted_mask ROI crops before train-time patch sampling")
            else:
                fail(
                    "predicted_mask ROI train sample shape mismatch: "
                    f"{tuple(sample_pred['image'].shape)}"
                )

    except Exception as exc:
        fail("ROI helper / dataset test failed", exc)

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

        # Run train transform on a dummy 3-channel volume (3 modalities: T2w, ADC, HBV).
        # Use a volume larger than the patch so RandCropByPosNegLabeld has room to crop.
        # The label has a small foreground region so positive-patch sampling can fire.
        dummy_vol: dict = {
            "image": torch.randn(3, 32, 96, 96),
            "label": torch.zeros(1, 32, 96, 96),
        }
        dummy_vol["label"][0, 10:20, 32:64, 32:64] = 1.0  # synthetic lesion region

        train_result = train_tfm(dummy_vol)

        # RandCropByPosNegLabeld with num_samples=1 returns a list[dict]
        if isinstance(train_result, list):
            train_result = train_result[0]

        assert isinstance(train_result, dict), (
            f"train_transforms returned unexpected type: {type(train_result)}"
        )
        assert "image" in train_result and "label" in train_result, (
            "train_transforms output missing 'image' or 'label' key"
        )
        img_out = train_result["image"]
        lbl_out = train_result["label"]
        assert img_out.shape == (3, 20, 64, 64), (
            f"train_transforms image shape mismatch: {tuple(img_out.shape)}"
        )
        assert lbl_out.shape == (1, 20, 64, 64), (
            f"train_transforms label shape mismatch: {tuple(lbl_out.shape)}"
        )
        ok(
            f"train_transforms forward pass OK — "
            f"image {tuple(img_out.shape)}, label {tuple(lbl_out.shape)}"
        )

        # Verify new MRI-specific augmentations are present in the pipeline
        transform_names = [type(t).__name__ for t in train_tfm.transforms]
        for expected in (
            "Rand3DElasticd",
            "RandAdjustContrastd",
            "RandBiasFieldd",
            "RandGibbsNoised",
            "RandCoarseDropoutd",
        ):
            if expected in transform_names:
                ok(f"{expected} present in train pipeline")
            else:
                fail(f"{expected} missing from train pipeline")

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

        # Build a GradScaler (enabled only when CUDA is available; FP16 mode)
        _ckpt_scaler = torch.amp.GradScaler("cuda", enabled=cuda_ok)  # type: ignore[attr-defined]

        # Save (with scaler and last_hd95)
        save_checkpoint(
            _ckpt_model, _ckpt_opt, epoch=1,
            path=str(ckpt_path),
            scheduler=_ckpt_sched,
            scaler=_ckpt_scaler,
            best_val_dice=0.42,
            best_composite_score=0.57,
            last_hd95=12.5,
        )
        ok(f"save_checkpoint wrote {ckpt_path.name}")

        # Verify scaler_state_dict is present in the raw checkpoint file
        _raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "scaler_state_dict" in _raw:
            ok("scaler_state_dict key present in checkpoint file")
        else:
            fail("scaler_state_dict missing from saved checkpoint")

        # Load into a fresh model/optimizer/scheduler/scaler
        _new_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16)).to(DEVICE)
        _new_opt = torch.optim.AdamW(_new_model.parameters(), lr=1e-4)
        _new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            _new_opt, T_max=10, eta_min=1e-6
        )
        _new_scaler = torch.amp.GradScaler("cuda", enabled=cuda_ok)  # type: ignore[attr-defined]

        ckpt = load_checkpoint(
            ckpt_path, _new_model, _new_opt, _new_sched,
            scaler=_new_scaler, device=DEVICE,
        )

        assert ckpt["epoch"] == 1, f"epoch mismatch: {ckpt['epoch']}"
        assert abs(ckpt.get("best_val_dice", -1) - 0.42) < 1e-6, "best_val_dice mismatch"
        assert abs(ckpt.get("best_composite_score", -1) - 0.57) < 1e-6, "best_composite_score mismatch"
        assert abs(ckpt.get("last_hd95", float("nan")) - 12.5) < 1e-6, "last_hd95 mismatch"
        assert "scheduler_state_dict" in ckpt, "scheduler_state_dict missing"
        ok(f"load_checkpoint restored epoch={ckpt['epoch']}, "
           f"best_val_dice={ckpt['best_val_dice']:.2f}, "
           f"best_composite_score={ckpt['best_composite_score']:.2f}, "
           f"last_hd95={ckpt['last_hd95']:.1f}mm, "
           f"scheduler+scaler state present")

except Exception as exc:
    fail("Checkpoint round-trip test failed", exc)

# ---------------------------------------------------------------------------
# 8b. SSL transfer helpers
# ---------------------------------------------------------------------------
section("8b. SSL transfer helpers (get_encoder_state_dict + load/freeze encoder)")

try:
    import tempfile

    from utils import (
        get_encoder_state_dict,
        load_pretrained_encoder,
        set_encoder_trainable,
    )

    enc_prefixes = ("encoders.", "bottleneck.")

    src_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16))
    enc_sd = get_encoder_state_dict(src_model)

    if len(enc_sd) > 0 and all(k.startswith(enc_prefixes) for k in enc_sd):
        ok(f"get_encoder_state_dict exported {len(enc_sd)} encoder tensors")
    else:
        fail("get_encoder_state_dict returned empty or non-encoder keys")

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "ssl_best.pt"
        torch.save({"encoder_state_dict": enc_sd}, ckpt_path)

        tgt_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16))
        stats = load_pretrained_encoder(tgt_model, ckpt_path, device=torch.device("cpu"))

        if int(stats["loaded"]) > 0:
            ok(
                "load_pretrained_encoder loaded encoder tensors "
                f"(loaded={stats['loaded']}, missing={len(stats['missing'])}, "
                f"shape_mismatch={len(stats['shape_mismatch'])})"
            )
        else:
            fail("load_pretrained_encoder loaded zero tensors")

        n_frozen = set_encoder_trainable(tgt_model, trainable=False)
        enc_flags = [
            p.requires_grad
            for n, p in tgt_model.named_parameters()
            if n.startswith(enc_prefixes)
        ]
        if n_frozen > 0 and enc_flags and not any(enc_flags):
            ok(f"set_encoder_trainable(False) froze encoder params ({n_frozen:,} parameters)")
        else:
            fail("set_encoder_trainable(False) did not freeze encoder parameters")

        n_unfrozen = set_encoder_trainable(tgt_model, trainable=True)
        enc_flags_after = [
            p.requires_grad
            for n, p in tgt_model.named_parameters()
            if n.startswith(enc_prefixes)
        ]
        if n_unfrozen == n_frozen and enc_flags_after and all(enc_flags_after):
            ok("set_encoder_trainable(True) restored encoder trainability")
        else:
            fail("set_encoder_trainable(True) did not fully restore trainability")

except Exception as exc:
    fail("SSL transfer helper test failed", exc)

# ---------------------------------------------------------------------------
# 8c. Current-config init helpers
# ---------------------------------------------------------------------------
section("8c. current-config init helpers (model-weight-only load + conflict guard)")

try:
    import tempfile

    from utils import (
        load_model_weights_for_current_config,
        resolve_checkpoint_init_paths,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # --- 8c-i. Successful model-weight-only load from checkpoint ----------
        src_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16))
        src_model_state = src_model.state_dict()
        ckpt_ok = tmp_path / "model_ok.pt"
        torch.save({"model_state_dict": src_model_state, "epoch": 123}, ckpt_ok)

        tgt_model = UNet3D(in_channels=3, out_channels=1, features=(8, 16))
        load_stats = load_model_weights_for_current_config(
            tgt_model, ckpt_ok, device=torch.device("cpu"), strict_shape=True
        )

        if int(load_stats["loaded"]) > 0:
            ok(
                "load_model_weights_for_current_config loaded tensors "
                f"(loaded={load_stats['loaded']}, missing={len(load_stats['missing'])}, "
                f"unexpected={len(load_stats['unexpected'])})"
            )
        else:
            fail("load_model_weights_for_current_config loaded zero tensors")

        rep_key = next(iter(src_model_state.keys()))
        if torch.allclose(tgt_model.state_dict()[rep_key], src_model_state[rep_key]):
            ok(f"current-config load updated representative tensor '{rep_key}'")
        else:
            fail(f"current-config load did not update tensor '{rep_key}'")

        # --- 8c-ii. Shape mismatch must raise (strict default) ----------------
        bad_shape_sd = dict(src_model_state)
        _shape_key = next(iter(bad_shape_sd.keys()))
        _bad_tensor = bad_shape_sd[_shape_key]
        if _bad_tensor.ndim == 0:
            _bad_tensor = _bad_tensor.view(1)
        bad_shape_sd[_shape_key] = (
            _bad_tensor[:-1] if _bad_tensor.numel() > 1 else _bad_tensor.repeat(2)
        )
        ckpt_bad_shape = tmp_path / "model_bad_shape.pt"
        torch.save({"model_state_dict": bad_shape_sd}, ckpt_bad_shape)

        try:
            load_model_weights_for_current_config(
                UNet3D(in_channels=3, out_channels=1, features=(8, 16)),
                ckpt_bad_shape,
                device=torch.device("cpu"),
                strict_shape=True,
            )
            fail("current-config loader should fail on shape mismatch")
        except ValueError:
            ok("current-config loader fails on shape mismatch (strict_shape=True)")

        # --- 8c-iii. No compatible keys must raise ----------------------------
        ckpt_no_match = tmp_path / "model_no_match.pt"
        torch.save({"model_state_dict": {"not_a_real_key.weight": torch.randn(1)}}, ckpt_no_match)

        try:
            load_model_weights_for_current_config(
                UNet3D(in_channels=3, out_channels=1, features=(8, 16)),
                ckpt_no_match,
                device=torch.device("cpu"),
                strict_shape=True,
            )
            fail("current-config loader should fail when no keys match")
        except ValueError:
            ok("current-config loader fails when no compatible keys match")

    # --- 8c-iv. Resume/current-config conflict guard ---------------------------
    _resume_only, _use_current_false = resolve_checkpoint_init_paths(
        resume_cli="/tmp/epoch_0010.pt",
        resume_cfg=None,
        use_current_config=False,
    )
    if _resume_only == "/tmp/epoch_0010.pt" and _use_current_false is False:
        ok("checkpoint init resolver returns resume path when only --resume is set")
    else:
        fail("checkpoint init resolver returned unexpected result for --resume-only")

    _resume_with_cfg, _use_current_true = resolve_checkpoint_init_paths(
        resume_cli="/tmp/best.pt",
        resume_cfg=None,
        use_current_config=True,
    )
    if _resume_with_cfg == "/tmp/best.pt" and _use_current_true is True:
        ok("checkpoint init resolver enables current-config mode with --resume path")
    else:
        fail("checkpoint init resolver returned unexpected result for --resume + --current-config")

    _resume_from_yaml, _use_current_yaml = resolve_checkpoint_init_paths(
        resume_cli=None,
        resume_cfg="/tmp/from_config.pt",
        use_current_config=True,
    )
    if _resume_from_yaml == "/tmp/from_config.pt" and _use_current_yaml is True:
        ok("checkpoint init resolver supports current-config mode with resume_checkpoint in YAML")
    else:
        fail("checkpoint init resolver returned unexpected result for YAML resume_checkpoint + --current-config")

    try:
        resolve_checkpoint_init_paths(
            resume_cli=None,
            resume_cfg=None,
            use_current_config=True,
        )
        fail("resolver should fail when --current-config is set without checkpoint source")
    except ValueError:
        ok("resolver rejects --current-config without --resume/resume_checkpoint source")

except Exception as exc:
    fail("current-config helper test failed", exc)

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
            "roi_bounds": (
                {
                    "start_zyx": (4, 10, 12),
                    "end_zyx": (18, 36, 34),
                    "full_shape_zyx": (D, H, W),
                    "used_fallback": False,
                }
                if i == 0 else None
            ),
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
# 9b. visualize_3d helpers
# ---------------------------------------------------------------------------
section("9b. visualize_3d helpers (_resolve_seg_path, _downsample_for_render, save_3d_visualization_html, save_3d_visualization_gif)")

try:
    import importlib.util
    import tempfile

    import numpy as np

    _vis_spec = importlib.util.spec_from_file_location(
        "visualize_3d",
        Path(__file__).parent / "visualize_3d.py",
    )
    _vis_mod = importlib.util.module_from_spec(_vis_spec)  # type: ignore[arg-type]
    _vis_spec.loader.exec_module(_vis_mod)  # type: ignore[union-attr]

    _downsample = _vis_mod._downsample_for_render
    _resolve_seg = _vis_mod._resolve_seg_path
    _save_html = _vis_mod.save_3d_visualization_html
    _save_gif = _vis_mod.save_3d_visualization_gif

    # Seg auto-detection smoke check from <root>/test_images/<case>_t2w.mha
    # to <root>/labels/<case>.nii.gz.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_images = root / "test_images"
        labels = root / "labels"
        test_images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)

        t2w_case = test_images / "demo_case_t2w.mha"
        seg_case = labels / "demo_case.nii.gz"
        t2w_case.write_bytes(b"dummy")
        seg_case.write_bytes(b"dummy")

        resolved_auto = _resolve_seg(t2w_case, None)
        if resolved_auto == seg_case.resolve():
            ok("_resolve_seg_path auto-detects <case>.nii.gz from T2w case id")
        else:
            fail(f"_resolve_seg_path auto path mismatch: {resolved_auto} != {seg_case.resolve()}")

        resolved_explicit = _resolve_seg(t2w_case, str(seg_case))
        if resolved_explicit == seg_case.resolve():
            ok("_resolve_seg_path honors explicit --seg path")
        else:
            fail(f"_resolve_seg_path explicit path mismatch: {resolved_explicit} != {seg_case.resolve()}")

    rng_np = np.random.default_rng(123)
    vol = rng_np.standard_normal((18, 64, 64)).astype(np.float32)
    gt = (rng_np.random((18, 64, 64)) > 0.92).astype(np.uint8)
    pred = (rng_np.random((18, 64, 64)) > 0.93).astype(np.uint8)

    # Downsample smoke check: verify output volume respects max_voxels cap.
    vol_ds, gt_ds, pred_ds, spacing_ds, stride = _downsample(
        t2w_vol=vol,
        gt_mask=gt,
        pred_mask=pred,
        spacing_zyx=(3.0, 0.5, 0.5),
        max_voxels=20_000,
    )
    if np.prod(vol_ds.shape) <= 20_000:
        ok(
            f"_downsample_for_render applied stride={stride} -> shape={vol_ds.shape} "
            f"({int(np.prod(vol_ds.shape)):,} voxels)"
        )
    else:
        fail(
            f"_downsample_for_render cap violation: got {int(np.prod(vol_ds.shape))} > 20000"
        )

    # HTML export smoke check.
    with tempfile.TemporaryDirectory() as tmp:
        out_html = Path(tmp) / "test_3d_vis.html"
        _save_html(
            t2w_vol=vol_ds,
            gt_mask=gt_ds,
            pred_mask=pred_ds,
            spacing_zyx=spacing_ds,
            output_path=out_html,
            case_id="synth_case",
            max_voxels=20_000,
        )
        assert out_html.exists(), "3-D HTML file was not created"
        html_text = out_html.read_text(encoding="utf-8", errors="ignore")
        assert "plotly" in html_text.lower(), "output HTML does not look like Plotly export"
        ok(f"save_3d_visualization_html wrote {out_html.name} ({out_html.stat().st_size / 1024:.0f} KB)")

    if callable(_save_gif):
        ok("save_3d_visualization_gif helper is available")
    else:
        fail("save_3d_visualization_gif helper is missing or not callable")

except ImportError as exc:
    skip(f"visualize_3d helper test skipped ({exc})")
except Exception as exc:
    fail("visualize_3d helper test failed", exc)

# ---------------------------------------------------------------------------
# 10. compute_composite_score
# ---------------------------------------------------------------------------
section("10. compute_composite_score + early stopping counter logic")

try:
    import math as _math_cs

    from utils import compute_composite_score

    # --- 10a. Normal case: all metrics finite, hd95_scale=1.0 (legacy scale) --
    metrics_full = {
        "sensitivity": 0.80,
        "dice":        0.70,
        "hd95":        5.0,
        "iou":         0.60,
        "precision":   0.95,
    }
    score_full = compute_composite_score(
        metrics_full, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2, hd95_scale=1.0
    )
    if not _math_cs.isnan(score_full) and 0.0 <= score_full <= 1.0:
        ok(f"all-finite metrics → composite_score={score_full:.4f} (in [0,1])")
    else:
        fail(f"all-finite metrics: unexpected score={score_full}")

    # Verify manual calculation matches (hd95_scale=1.0 → 1/(1+hd95/1.0))
    hd95_term = 1.0 / (1.0 + 5.0 / 1.0)   # = 1/6 ≈ 0.1667
    total_w = 0.5 + 0.3 + 0.2              # = 1.0 (normalised)
    expected = (0.5 * 0.80 + 0.3 * 0.70 + 0.2 * hd95_term) / total_w
    if abs(score_full - expected) < 1e-6:
        ok(f"composite score matches manual calculation ({expected:.6f})")
    else:
        fail(f"composite score mismatch: got {score_full:.6f}, expected {expected:.6f}")

    # --- 10a'. Default scale (hd95_scale=10.0): better discrimination at 5mm --
    score_scale10 = compute_composite_score(
        metrics_full, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2, hd95_scale=10.0
    )
    # hd95=5mm with scale=10mm → term = 1/(1 + 5/10) = 1/1.5 ≈ 0.6667
    hd95_term_10 = 1.0 / (1.0 + 5.0 / 10.0)
    expected_10 = (0.5 * 0.80 + 0.3 * 0.70 + 0.2 * hd95_term_10) / 1.0
    if abs(score_scale10 - expected_10) < 1e-6:
        ok(
            f"hd95_scale=10mm: hd95=5mm → term={hd95_term_10:.4f}, "
            f"composite_score={score_scale10:.4f} (expected {expected_10:.4f})"
        )
    else:
        fail(
            f"hd95_scale=10mm mismatch: got {score_scale10:.6f}, "
            f"expected {expected_10:.6f}"
        )
    # Sanity: scale=10 should give a higher score than scale=1 for the same HD95,
    # because a 5mm HD95 is rated as more favourable relative to the midpoint.
    if score_scale10 > score_full:
        ok(
            f"hd95_scale=10mm ({score_scale10:.4f}) > "
            f"hd95_scale=1mm ({score_full:.4f}) for hd95=5mm ✓"
        )
    else:
        fail(
            f"expected scale=10mm score > scale=1mm for hd95=5mm; "
            f"got {score_scale10:.4f} vs {score_full:.4f}"
        )

    # --- 10b. HD95 = NaN: weight redistributed to sensitivity + dice ----------
    metrics_no_hd95 = dict(metrics_full)
    metrics_no_hd95["hd95"] = float("nan")
    score_no_hd95 = compute_composite_score(
        metrics_no_hd95, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2, hd95_scale=1.0
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
        metrics_no_sens, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2, hd95_scale=1.0
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

    # --- 10e. Cached HD95: score is consistent across HD95 and non-HD95 epochs -
    # Simulates the train.py fix: last_hd95 is carried forward so the effective
    # weights stay constant epoch-to-epoch.
    cached_hd95 = float("nan")

    # Epoch A: HD95 is computed → cache it
    metrics_epoch_a = {"sensitivity": 0.75, "dice": 0.65, "hd95": 8.0,
                       "iou": 0.50, "precision": 0.92}
    cached_hd95 = metrics_epoch_a["hd95"]   # 8.0 mm
    score_metrics_a = dict(metrics_epoch_a)
    score_metrics_a["hd95"] = cached_hd95
    score_a = compute_composite_score(
        score_metrics_a, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2, hd95_scale=10.0
    )

    # Epoch B: HD95 is NOT re-computed; use cached value from epoch A
    metrics_epoch_b = {"sensitivity": 0.76, "dice": 0.66, "hd95": float("nan"),
                       "iou": 0.51, "precision": 0.93}
    score_metrics_b_cached = dict(metrics_epoch_b)
    score_metrics_b_cached["hd95"] = cached_hd95   # inject cached value
    score_b_cached = compute_composite_score(
        score_metrics_b_cached, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.2,
        hd95_scale=10.0,
    )

    # Old behaviour (zeroing out w_hd95 on non-HD95 epochs)
    score_b_old = compute_composite_score(
        metrics_epoch_b, w_sensitivity=0.5, w_dice=0.3, w_hd95=0.0, hd95_scale=10.0
    )

    # Both new scores must be in [0, 1]
    if 0.0 <= score_a <= 1.0 and 0.0 <= score_b_cached <= 1.0:
        ok(
            f"cached-HD95 path: epoch_a={score_a:.4f}, "
            f"epoch_b (cached)={score_b_cached:.4f}"
        )
    else:
        fail(
            f"cached-HD95 path: scores out of range "
            f"({score_a:.4f}, {score_b_cached:.4f})"
        )

    # Verify cached path and old path are NOT the same (they differ in formula)
    if abs(score_b_cached - score_b_old) > 1e-6:
        ok(
            f"cached score ({score_b_cached:.4f}) differs from old zeroed-weight "
            f"score ({score_b_old:.4f}) — formulas correctly diverge"
        )
    else:
        fail(
            f"cached and old scores are identical ({score_b_cached:.4f}); "
            "expected divergence due to different effective HD95 weighting"
        )

    # Verify cached score uses the same weights as epoch A (HD95 term active)
    # Manual: hd95=8mm, scale=10 → term=1/(1+0.8)=1/1.8≈0.5556
    hd95_term_b = 1.0 / (1.0 + cached_hd95 / 10.0)
    expected_b = (0.5 * 0.76 + 0.3 * 0.66 + 0.2 * hd95_term_b) / 1.0
    if abs(score_b_cached - expected_b) < 1e-6:
        ok(
            f"cached-HD95 manual check: hd95=8mm term={hd95_term_b:.4f}, "
            f"score={score_b_cached:.4f} (expected {expected_b:.4f})"
        )
    else:
        fail(
            f"cached-HD95 manual mismatch: got {score_b_cached:.6f}, "
            f"expected {expected_b:.6f}"
        )

except Exception as exc:
    fail("compute_composite_score test failed", exc)

# ---------------------------------------------------------------------------
# 11. Dataset cache modes (ram/storage/none)
# ---------------------------------------------------------------------------
section("11. PiCaiDataset cache modes (ram/storage/none)")

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

        def _make_fixture(root: Path, tag: str) -> tuple[Path, Path, list[dict]]:
            img_dir = root / f"images_{tag}"
            lbl_dir = root / f"labels_{tag}"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            case_id = f"test_{tag}_0000"
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
            return img_dir, lbl_dir, synth_cases

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # --- RAM mode -------------------------------------------------
            import shutil
            img_dir_ram, lbl_dir_ram, cases_ram = _make_fixture(tmp_path, "ram")
            ds_ram = PiCaiDataset(
                images_dir=img_dir_ram,
                labels_dir=lbl_dir_ram,
                cases=cases_ram,
                cache_mode="ram",
                cache_rate=1.0,
            )
            _ = ds_ram[0]
            if 0 in ds_ram._cache:
                ok("ram mode: cache populated on first __getitem__ access")
            else:
                fail("ram mode: cache was not populated after first access")
            shutil.rmtree(str(img_dir_ram))
            try:
                _ = ds_ram[0]
                ok("ram mode: second access succeeds after source removal")
            except Exception as e:
                fail("ram mode: second access failed despite cache", e)

            # --- Storage mode ---------------------------------------------
            img_dir_storage, lbl_dir_storage, cases_storage = _make_fixture(tmp_path, "storage")
            storage_cache_dir = tmp_path / "storage_cache"
            ds_storage = PiCaiDataset(
                images_dir=img_dir_storage,
                labels_dir=lbl_dir_storage,
                cases=cases_storage,
                cache_mode="storage",
                cache_rate=1.0,
                cache_dir=storage_cache_dir,
            )
            _ = ds_storage[0]
            storage_entries = list(storage_cache_dir.glob("*.pt"))
            if storage_entries:
                ok("storage mode: cache file created on first access")
            else:
                fail("storage mode: cache file was not created")
            shutil.rmtree(str(img_dir_storage))
            shutil.rmtree(str(lbl_dir_storage))
            try:
                _ = ds_storage[0]
                ok("storage mode: second access succeeds from persistent cache")
            except Exception as e:
                fail("storage mode: second access failed after source removal", e)

            # --- None mode ------------------------------------------------
            img_dir_none, lbl_dir_none, cases_none = _make_fixture(tmp_path, "none")
            ds_none = PiCaiDataset(
                images_dir=img_dir_none,
                labels_dir=lbl_dir_none,
                cases=cases_none,
                cache_mode="none",
                cache_rate=1.0,
            )
            _ = ds_none[0]
            shutil.rmtree(str(img_dir_none))
            shutil.rmtree(str(lbl_dir_none))
            try:
                _ = ds_none[0]
                fail("none mode: second access unexpectedly succeeded after source removal")
            except Exception:
                ok("none mode: second access fails after source removal (expected)")

    except Exception as exc:
        fail("PiCaiDataset cache-mode test failed", exc)

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
    for name in ("dice", "iou", "sensitivity", "precision"):
        if not _math_hd.isnan(m_no_hd95[name]):
            ok(f"compute_hd95=False: {name}={m_no_hd95[name]:.4f} (finite, as expected)")
        else:
            fail(f"compute_hd95=False: {name} unexpectedly returned nan")

except Exception as exc:
    fail("compute_hd95=False test failed", exc)

# ---------------------------------------------------------------------------
# 13. AMP forward pass: FP16 + GradScaler (Volta/Turing) and BF16 (Ampere+)
# ---------------------------------------------------------------------------
section("13. AMP forward pass — FP16+GradScaler and BF16")

if not cuda_ok:
    skip("CUDA not available — skipping AMP autocast tests")
else:
    try:
        amp_model = UNet3D(in_channels=3, out_channels=1, features=(16, 32, 64, 128))
        amp_model = amp_model.to(DEVICE)

        from losses import DiceBCELoss
        amp_criterion = DiceBCELoss().to(DEVICE)
        tgt = (torch.rand(1, 1, 20, 64, 64, device=DEVICE) > 0.8).float()
        dummy = torch.randn(1, 3, 20, 64, 64, device=DEVICE)

        # --- FP16 + GradScaler (server path: TITAN V / Volta) -----------------
        fp16_scaler = torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]
        amp_model.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):  # type: ignore[attr-defined]
            out_fp16 = amp_model(dummy)
            loss_fp16 = amp_criterion(out_fp16, tgt)

        fp16_scaler.scale(loss_fp16).backward()
        fp16_scaler.step(torch.optim.AdamW(amp_model.parameters(), lr=1e-4))
        fp16_scaler.update()

        if out_fp16.shape == (1, 1, 20, 64, 64) and torch.isfinite(loss_fp16):
            ok(
                f"FP16+GradScaler forward+backward OK — "
                f"output dtype={out_fp16.dtype}, loss={loss_fp16.item():.4f}"
            )
        else:
            fail(
                f"FP16+GradScaler: shape={tuple(out_fp16.shape)}, "
                f"loss={loss_fp16.item()}"
            )

        # --- BF16 (laptop path: RTX 5070 / Blackwell) -------------------------
        bf16_supported_fn = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_supported = bool(bf16_supported_fn()) if callable(bf16_supported_fn) else False
        if not bf16_supported:
            skip("BF16 autocast skipped — GPU does not report BF16 support")
        else:
            amp_model.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # type: ignore[attr-defined]
                out_bf16 = amp_model(dummy)
                loss_bf16 = amp_criterion(out_bf16, tgt)

            loss_bf16.backward()   # no scaler needed for BF16

            if out_bf16.shape == (1, 1, 20, 64, 64) and torch.isfinite(loss_bf16):
                ok(
                    f"BF16 forward+backward OK — "
                    f"output dtype={out_bf16.dtype}, loss={loss_bf16.item():.4f}"
                )
            else:
                fail(
                    f"BF16: shape={tuple(out_bf16.shape)}, loss={loss_bf16.item()}"
                )

    except Exception as exc:
        fail("AMP autocast test failed", exc)

# ---------------------------------------------------------------------------
# 13b. Logit clamp: FP16 overflow guard and NaN-skip resilience
# ---------------------------------------------------------------------------
section("13b. Logit clamp — FP16 overflow guard and NaN-skip resilience")

try:
    from losses import TverskyBCELoss

    clamp_criterion = TverskyBCELoss().to("cpu")
    tgt_clamp = (torch.rand(1, 1, 8, 16, 16) > 0.8).float()

    # --- 13b-i. Extreme logits (simulating FP16 overflow) are tamed by clamp --
    # Without clamp: sigmoid(1e4) = 1.0 exactly → log(1 - 1.0) = -inf → NaN BCE
    extreme_logits = torch.full((1, 1, 8, 16, 16), 1e4)
    clamped = extreme_logits.clamp(-10.0, 10.0)
    loss_clamped = clamp_criterion(clamped, tgt_clamp)
    if torch.isfinite(loss_clamped):
        ok(f"Clamped extreme logits (1e4 → 10.0) produce finite loss: {loss_clamped.item():.4f}")
    else:
        fail(f"Clamped logits still produce non-finite loss: {loss_clamped.item()}")

    # Verify unclamped logits would produce non-finite loss (confirms the guard is needed)
    loss_unclamped = clamp_criterion(extreme_logits.float(), tgt_clamp)
    if not torch.isfinite(loss_unclamped):
        ok("Unclamped extreme logits correctly produce non-finite loss (guard is necessary)")
    else:
        ok(
            "Unclamped extreme logits happen to be finite on this platform "
            f"(loss={loss_unclamped.item():.4f}) — clamp is still a safe no-op"
        )

    # --- 13b-ii. List-of-tensors (deep supervision) clamp path ----------------
    ds_logits = [torch.full((1, 1, 8, 16, 16), 1e4) for _ in range(4)]
    ds_clamped = [l.clamp(-10.0, 10.0) for l in ds_logits]
    if all(t.max().item() <= 10.0 and t.min().item() >= -10.0 for t in ds_clamped):
        ok("Deep-supervision list clamp: all 4 tensors correctly bounded to [-10, 10]")
    else:
        fail("Deep-supervision list clamp: some tensors out of [-10, 10] range")

    # --- 13b-iii. Finite logits are unaffected by clamp (no-op on normal range) -
    normal_logits = torch.randn(1, 1, 8, 16, 16)  # std≈1, well within ±10
    normal_clamped = normal_logits.clamp(-10.0, 10.0)
    if torch.allclose(normal_logits, normal_clamped):
        ok("Normal-range logits (std≈1) are unchanged by clamp — no-op confirmed")
    else:
        fail("Clamp unexpectedly modified normal-range logits")

except Exception as exc:
    fail("Logit clamp smoke test failed", exc)

# ---------------------------------------------------------------------------
# 14. ntfy notifications (send_ntfy)
# ---------------------------------------------------------------------------
section("14. ntfy notifications (send_ntfy)")

try:
    from unittest.mock import MagicMock, call, patch

    from notify import send_ntfy

    # --- 14a. No-op when ntfy_url / ntfy_topic are absent --------------------
    cfg_no_ntfy: dict = {"experiment_name": "test"}
    send_ntfy(cfg_no_ntfy, title="Test", message="should be silently ignored")
    ok("send_ntfy is a no-op when ntfy_url/ntfy_topic are absent")

    # --- 14b. No-op when values are empty strings ----------------------------
    cfg_empty: dict = {"ntfy_url": "", "ntfy_topic": "", "experiment_name": "test"}
    send_ntfy(cfg_empty, title="Test", message="should also be silently ignored")
    ok("send_ntfy is a no-op when ntfy_url/ntfy_topic are empty strings")

    # --- 14c. Correct URL, headers, and body when configured -----------------
    cfg_ntfy: dict = {
        "ntfy_url": "https://ntfy.sh",
        "ntfy_topic": "test-training-alerts",
        "experiment_name": "smoke_test_run",
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("notify.requests") as mock_requests:
        mock_requests.post.return_value = mock_response
        send_ntfy(
            cfg_ntfy,
            title="Training started: smoke_test_run",
            message="Epochs: 5\nTrain cases: 40 | Val cases: 10",
            tags=["rocket"],
            priority="default",
        )
        mock_requests.post.assert_called_once()
        _call_args = mock_requests.post.call_args
        _url = _call_args[0][0]
        _headers = _call_args[1]["headers"]
        _body = _call_args[1]["data"]

    if _url == "https://ntfy.sh/test-training-alerts":
        ok(f"send_ntfy builds correct endpoint URL: {_url}")
    else:
        fail(f"URL mismatch: got '{_url}', expected 'https://ntfy.sh/test-training-alerts'")

    if _headers.get("Title") == "Training started: smoke_test_run":
        ok("send_ntfy sets Title header correctly")
    else:
        fail(f"Title header mismatch: {_headers.get('Title')!r}")

    if _headers.get("Tags") == "rocket":
        ok("send_ntfy sets Tags header correctly")
    else:
        fail(f"Tags header mismatch: {_headers.get('Tags')!r}")

    if _headers.get("Priority") == "default":
        ok("send_ntfy sets Priority header correctly")
    else:
        fail(f"Priority header mismatch: {_headers.get('Priority')!r}")

    if b"Epochs: 5" in _body:
        ok("send_ntfy encodes message body as UTF-8 bytes")
    else:
        fail(f"Body encoding unexpected: {_body!r}")

    # --- 14d. Multiple tags are comma-joined ---------------------------------
    with patch("notify.requests") as mock_requests:
        mock_requests.post.return_value = mock_response
        send_ntfy(cfg_ntfy, title="T", message="m", tags=["x", "rotating_light"])
        _tags_hdr = mock_requests.post.call_args[1]["headers"].get("Tags", "")
    if _tags_hdr == "x,rotating_light":
        ok("send_ntfy joins multiple tags with comma")
    else:
        fail(f"Multi-tag header mismatch: {_tags_hdr!r}")

    # --- 14e. URL trailing slash is stripped properly ------------------------
    cfg_slash = dict(cfg_ntfy)
    cfg_slash["ntfy_url"] = "https://ntfy.sh/"
    with patch("notify.requests") as mock_requests:
        mock_requests.post.return_value = mock_response
        send_ntfy(cfg_slash, title="T", message="m")
        _slashed_url = mock_requests.post.call_args[0][0]
    if _slashed_url == "https://ntfy.sh/test-training-alerts":
        ok("send_ntfy strips trailing slash from ntfy_url")
    else:
        fail(f"Trailing-slash URL mismatch: {_slashed_url!r}")

    # --- 14f. Network errors are swallowed, training continues ---------------
    with patch("notify.requests") as mock_requests:
        mock_requests.post.side_effect = Exception("Connection refused")
        try:
            send_ntfy(cfg_ntfy, title="T", message="m")
            ok("send_ntfy swallows network errors (training continues)")
        except Exception as e:
            fail(f"send_ntfy propagated a network error: {e}")

except Exception as exc:
    fail("ntfy notification test failed", exc)

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
