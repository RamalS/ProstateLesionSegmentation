"""
Resource estimation script for ProstateLesionSegmentation.

When a CUDA GPU is available the script measures *actual* peak VRAM
by running a real forward+backward pass (training) and a real forward
pass (validation/inference) on the GPU.  This is far more accurate than
any analytical estimate.

When no GPU is available (or the model does not fit), the script falls
back to a torchinfo-based analytical estimate with a per-model backprop
overhead factor derived from empirical measurements:

    unet3d / attention_unet3d : measured ratio ≈ 3.9× forward peak
    fct                       : measured ratio ≈ 4.1× forward peak
    deconver                  : measured ratio ≈ 3.8× forward peak

A 15% CUDA fragmentation overhead is applied to all VRAM totals.
MONAI's sliding-window accumulation buffers (importance map + count map)
are included in the validation estimate.

Usage
-----
    PYTHONPATH=. python scripts/estimate_resources.py
    PYTHONPATH=. python scripts/estimate_resources.py --config configs/default.yaml
    PYTHONPATH=. python scripts/estimate_resources.py --gpu-vram 12
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.config import (
    load_config,
    resolve_active_modalities,
    resolve_dataset_cache_config,
)
from src.models import build_model

# ---------------------------------------------------------------------------
# Optional torchinfo import
# ---------------------------------------------------------------------------

try:
    import torchinfo as _torchinfo
    _TORCHINFO_AVAILABLE = True
except ImportError:
    _torchinfo = None  # type: ignore[assignment]
    _TORCHINFO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Typical resampled PI-CAI full volume after target_spacing = [3.0, 0.5, 0.5]
_TYPICAL_VOLUME_SHAPE: tuple[int, int, int] = (20, 400, 400)

# Empirical backprop overhead factors (fwd+bwd peak / fwd-only peak)
# measured on actual models via torch.cuda.max_memory_allocated().
_BACKPROP_FACTOR: dict[str, float] = {
    "unet3d":           3.9,
    "attention_unet3d": 3.9,
    "fct":              4.1,
    "deconver":         3.8,
}
_BACKPROP_FACTOR_DEFAULT = 3.9  # used for unknown models

# CUDA caching-allocator fragmentation: effective usable VRAM ≈ physical / 1.15
_CUDA_FRAG: float = 1.15

# OOM risk thresholds (fraction of usable VRAM)
_RISK_OK   = 0.70
_RISK_WARN = 0.90

# Bytes per dtype string
_DTYPE_BYTES: dict[str, int] = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: float) -> str:
    """Return a human-readable memory string (MB or GB)."""
    mb = n / (1024 ** 2)
    if mb >= 1024:
        return f"{mb / 1024:>8.2f} GB"
    return f"{mb:>8.2f} MB"


def _fmt_flops(n: float) -> str:
    """Return a human-readable FLOPs string."""
    g = n / 1e9
    if g >= 1000:
        return f"{g / 1000:.2f} TFLOPs"
    return f"{g:.2f} GFLOPs"


def _risk_tag(used: float, usable: float) -> str:
    """Return an inline risk label."""
    if usable <= 0:
        return ""
    frac = used / usable
    pct  = frac * 100
    if frac <= _RISK_OK:
        label = "OK"
    elif frac <= _RISK_WARN:
        label = "WARNING"
    else:
        label = "HIGH OOM RISK"
    return f"  [{pct:.0f}% of usable — {label}]"


# ---------------------------------------------------------------------------
# GPU peak-memory measurement (most accurate)
# ---------------------------------------------------------------------------

def _measure_train_vram(
    model: nn.Module,
    patch_size: tuple[int, int, int],
    in_channels: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Optional[int], Optional[int], str]:
    """
    Measure actual peak VRAM for one forward+backward step on *device*.

    Returns
    -------
    (peak_bytes_b1, peak_bytes_batch, source_note)
        peak_bytes_b1    : peak VRAM for batch=1 (None on failure)
        peak_bytes_batch : peak VRAM for batch=batch_size (None on failure or OOM)
        source_note      : human-readable description of the measurement result
    """
    peak_b1: Optional[int]   = None
    peak_bs: Optional[int]   = None
    source_note: str

    def _run(bs: int) -> Optional[int]:
        """Run one fwd+bwd, return peak bytes above baseline, or None on OOM."""
        model.train()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        try:
            x = torch.zeros(bs, in_channels, *patch_size,
                            device=device, requires_grad=False)
            with torch.no_grad():
                # Quick check without grad first to see if it fits at all
                out = model(x)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception:
            torch.cuda.empty_cache()
            return None

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        try:
            x2 = torch.zeros(bs, in_channels, *patch_size,
                             device=device, requires_grad=True)
            out2 = model(x2)
            loss = (sum(o.mean() for o in out2)
                    if isinstance(out2, list) else out2.mean())
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception:
            torch.cuda.empty_cache()
            return None

        peak = torch.cuda.max_memory_allocated(device) - baseline
        torch.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        return int(peak)

    peak_b1 = _run(1)
    if peak_b1 is not None:
        peak_bs = _run(batch_size)
        if peak_bs is not None:
            source_note = f"measured on {device} (fwd+bwd, batch={batch_size})"
        else:
            source_note = (
                f"measured on {device} (batch=1); "
                f"batch={batch_size} caused OOM — extrapolated"
            )
    else:
        source_note = f"GPU measurement failed (OOM even at batch=1); using analytical estimate"

    return peak_b1, peak_bs, source_note


def _measure_val_vram(
    model: nn.Module,
    patch_size: tuple[int, int, int],
    in_channels: int,
    sw_batch_size: int,
    device: torch.device,
) -> tuple[Optional[int], Optional[int], str]:
    """
    Measure peak VRAM for inference (no grad) with up to sw_batch_size windows.

    If sw_batch_size OOMs, retries with sw_batch=1 and scales linearly.

    Returns
    -------
    (peak_bytes_at_configured, peak_bytes_per_window, source_note)
        peak_bytes_at_configured : peak VRAM for the configured sw_batch_size
                                   (None if even sw_batch=1 OOMs)
        peak_bytes_per_window    : peak VRAM for a single window (None on failure)
        source_note              : description
    """
    def _run(bs: int) -> Optional[int]:
        model.eval()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        try:
            x = torch.zeros(bs, in_channels, *patch_size, device=device)
            with torch.inference_mode():
                out = model(x)
                _ = out[0] if isinstance(out, list) else out
            peak = int(torch.cuda.max_memory_allocated(device) - baseline)
            torch.cuda.empty_cache()
            return peak
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None
        except Exception:
            torch.cuda.empty_cache()
            return None

    # Try configured sw_batch_size first
    peak_cfg = _run(sw_batch_size)
    if peak_cfg is not None:
        # Also measure b=1 so we can scale for the suggestion
        peak_b1 = _run(1)
        per_win = peak_b1 if peak_b1 is not None else peak_cfg // sw_batch_size
        return (
            peak_cfg,
            per_win,
            f"measured on {device} (inference, sw_batch={sw_batch_size})",
        )

    # Configured OOMs — measure at b=1 and scale
    peak_b1 = _run(1)
    if peak_b1 is not None:
        scaled = peak_b1 * sw_batch_size
        return (
            scaled,
            peak_b1,
            f"measured on {device} (sw_batch=1, ×{sw_batch_size} extrapolated — "
            f"configured sw_batch OOM'd)",
        )

    return None, None, f"OOM even at sw_batch=1; using analytical estimate"


# ---------------------------------------------------------------------------
# FLOPs: manual hook-based count
# ---------------------------------------------------------------------------

def _count_flops_manual(
    model: nn.Module,
    patch_size: tuple[int, int, int],
    in_channels: int,
) -> tuple[int, int]:
    """
    Count MACs × 2 = FLOPs via forward hooks (batch=1, no grad).

    Returns
    -------
    (total_flops, activation_output_bytes_fp32)
        activation_output_bytes_fp32 : sum of all layer output tensor sizes in
        fp32 bytes (batch=1).  Used as fallback activation estimate.
    """
    spatial_map: dict[int, tuple] = {}
    hooks: list = []

    def _make_hook(mid: int):
        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            in_s  = tuple(inp[0].shape) if inp else ()
            out_s = tuple(out.shape) if isinstance(out, torch.Tensor) else ()
            spatial_map[mid] = (in_s, out_s)
        return hook

    tracked = (nn.Conv3d, nn.ConvTranspose3d, nn.Linear,
               nn.BatchNorm3d, nn.InstanceNorm3d)
    for mid, m in enumerate(model.modules()):
        if isinstance(m, tracked):
            hooks.append(m.register_forward_hook(_make_hook(mid)))

    model.eval()
    dummy = torch.zeros(1, in_channels, *patch_size)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    total_flops  = 0
    total_act_b  = 0  # fp32 bytes

    for mid, m in enumerate(model.modules()):
        if mid not in spatial_map:
            continue
        in_s, out_s = spatial_map[mid]

        if out_s:
            n = 1
            for d in out_s:
                n *= d
            total_act_b += n * 4

        if isinstance(m, nn.Conv3d) and len(out_s) >= 5:
            c_in  = in_s[1] if in_s else m.in_channels
            c_out = out_s[1]
            kD, kH, kW = m.kernel_size  # type: ignore[misc]
            total_flops += 2 * (c_in // m.groups) * kD * kH * kW * c_out \
                           * out_s[2] * out_s[3] * out_s[4]

        elif isinstance(m, nn.ConvTranspose3d) and len(in_s) >= 5:
            c_in  = in_s[1]
            c_out = out_s[1] if out_s else m.out_channels
            kD, kH, kW = m.kernel_size  # type: ignore[misc]
            total_flops += 2 * (c_in // m.groups) * kD * kH * kW * c_out \
                           * in_s[2] * in_s[3] * in_s[4]

        elif isinstance(m, nn.Linear) and len(in_s) >= 2:
            n = 1
            for d in in_s[:-1]:
                n *= d
            total_flops += 2 * n * m.in_features * m.out_features

        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)) and len(in_s) >= 2:
            n = 1
            for d in in_s[1:]:
                n *= d
            total_flops += 2 * n

    return total_flops, total_act_b


# ---------------------------------------------------------------------------
# FLOPs via torchinfo
# ---------------------------------------------------------------------------

def _count_flops_torchinfo(
    model: nn.Module,
    patch_size: tuple[int, int, int],
    in_channels: int,
) -> tuple[int, int]:
    """
    Returns (total_flops, activation_output_bytes_fp32) via torchinfo.

    Raises RuntimeError if torchinfo's internal forward pass fails.
    """
    s = _torchinfo.summary(
        model,
        input_size=(1, in_channels, *patch_size),
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=10,
    )
    return int(s.total_mult_adds) * 2, int(s.total_output_bytes)


# ---------------------------------------------------------------------------
# Sliding-window window count
# ---------------------------------------------------------------------------

def _count_sw_windows(
    vol: tuple[int, int, int],
    patch: tuple[int, int, int],
    overlap: float,
) -> int:
    """Total number of SW patches MONAI will generate."""
    total = 1
    for v, p in zip(vol, patch):
        stride = max(1, math.ceil(p * (1.0 - overlap)))
        total *= (math.ceil((v - p) / stride) + 1) if v > p else 1
    return total


# ---------------------------------------------------------------------------
# Safe sw_batch_size suggestion (analytical, used when GPU measurement unavailable)
# ---------------------------------------------------------------------------

def _suggest_sw_batch_size(
    gpu_vram_bytes: float,
    model_vram: float,
    optimizer_vram: float,
    vol_input_vram: float,
    vol_accum_vram: float,
    patch_size: tuple[int, int, int],
    in_channels: int,
    param_bytes: int,
    act_val_b1: float,
    safety: float = 0.85,
) -> int:
    """Return the largest sw_batch_size that fits within safety fraction of usable VRAM."""
    usable    = (gpu_vram_bytes / _CUDA_FRAG) * safety
    fixed     = model_vram + optimizer_vram + vol_input_vram + vol_accum_vram
    remaining = usable - fixed
    pD, pH, pW = patch_size
    per_win   = (in_channels * pD * pH * pW * param_bytes
                 + act_val_b1
                 + 1 * pD * pH * pW * param_bytes)
    if remaining <= 0 or per_win <= 0:
        return 1
    return max(1, int(remaining // per_win))


# ---------------------------------------------------------------------------
# Dataset case counting
# ---------------------------------------------------------------------------

def _count_cases(images_dir: Path) -> Optional[int]:
    """Count *_t2w.mha files (flat or nested layout). Returns None if not found."""
    if not images_dir.exists():
        return None
    count = len(list(images_dir.rglob("*_t2w.mha")))
    return count if count > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def estimate(config_path: str, gpu_vram_gb: float = 12.0) -> None:
    """Run the full resource estimation and print a report."""
    cfg = load_config(config_path)
    cache_mode, cache_rate, cache_dir = resolve_dataset_cache_config(cfg)

    # ── Config values ──────────────────────────────────────────────────
    model_name:       str       = cfg.get("model", "unet3d").lower()
    batch_size:       int       = cfg.get("batch_size", 4)
    num_workers:      int       = cfg.get("num_workers", 2)
    patch_size:       list[int] = cfg.get("patch_size", [20, 128, 128])
    sw_batch_size:    int       = cfg.get("sw_batch_size", 2)
    sw_overlap:       float     = cfg.get("sw_overlap", 0.5)
    amp_dtype_str:    str       = cfg.get("amp_dtype", "fp32").lower()
    use_amp:          bool      = cfg.get("use_amp", False)
    num_samples:      int       = cfg.get("num_samples", 1)
    images_dir:       Path      = Path(cfg.get("images_dir", "data/images"))
    deep_supervision: bool      = cfg.get("deep_supervision", False)
    val_fraction:     float     = cfg.get("val_fraction", 0.2)

    active_modalities = resolve_active_modalities(cfg)
    in_channels = len(active_modalities)
    modality_str = " + ".join(active_modalities)

    param_bytes: int
    if use_amp:
        param_bytes = _DTYPE_BYTES.get(amp_dtype_str, 2)
        amp_label   = amp_dtype_str
    else:
        param_bytes = 4
        amp_label   = "fp32 (AMP disabled)"

    pD, pH, pW = patch_size[0], patch_size[1], patch_size[2]
    vD, vH, vW = _TYPICAL_VOLUME_SHAPE
    gpu_vram_bytes = gpu_vram_gb * 1024 ** 3
    usable_vram    = gpu_vram_bytes / _CUDA_FRAG

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = torch.device("cuda")
        actual_vram_bytes = torch.cuda.get_device_properties(device).total_memory
        actual_vram_gb    = actual_vram_bytes / 1024 ** 3
        usable_vram       = actual_vram_bytes / _CUDA_FRAG
        gpu_vram_gb       = actual_vram_gb
    else:
        device            = torch.device("cpu")
        actual_vram_bytes = gpu_vram_bytes

    # ── Build model ────────────────────────────────────────────────────
    print(f"\nBuilding model '{model_name}' …")
    try:
        model = build_model(cfg)
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── FLOPs ──────────────────────────────────────────────────────────
    print(f"Counting FLOPs ({'torchinfo' if _TORCHINFO_AVAILABLE else 'manual'}) …")
    forward_flops: int
    act_raw_fp32_b1: int
    flop_source: str
    torchinfo_ok = False

    if _TORCHINFO_AVAILABLE:
        try:
            forward_flops, act_raw_fp32_b1 = _count_flops_torchinfo(
                model, (pD, pH, pW), in_channels
            )
            flop_source  = "torchinfo"
            torchinfo_ok = True
        except Exception as exc:
            print(f"  torchinfo failed ({type(exc).__name__}); using manual fallback.",
                  file=sys.stderr)
            forward_flops, act_raw_fp32_b1 = _count_flops_manual(
                model, (pD, pH, pW), in_channels
            )
            flop_source = "manual (torchinfo failed)"
    else:
        forward_flops, act_raw_fp32_b1 = _count_flops_manual(
            model, (pD, pH, pW), in_channels
        )
        flop_source = "manual (torchinfo not installed)"

    step_flops = forward_flops * 3

    # ── GPU measurement (best accuracy) ────────────────────────────────
    train_peak_b1:  Optional[int] = None
    train_peak_bs:  Optional[int] = None
    val_peak_sw:    Optional[int] = None
    train_source:   str           = "analytical"
    val_source:     str           = "analytical"

    val_peak_per_win: Optional[int] = None

    if cuda_available:
        print("Measuring actual GPU peak memory …")
        model_gpu = build_model(cfg).to(device)
        (train_peak_b1, train_peak_bs, train_source) = _measure_train_vram(
            model_gpu, (pD, pH, pW), in_channels, batch_size, device
        )
        (val_peak_sw, val_peak_per_win, val_source) = _measure_val_vram(
            model_gpu, (pD, pH, pW), in_channels, sw_batch_size, device
        )
        del model_gpu
        torch.cuda.empty_cache()

    # ── Analytical activation estimates (fallback) ─────────────────────
    bp_factor   = _BACKPROP_FACTOR.get(model_name, _BACKPROP_FACTOR_DEFAULT)
    act_fwd_b1  = int(act_raw_fp32_b1 * (param_bytes / 4))   # fwd-only, training dtype
    act_train_b1 = int(act_fwd_b1 * bp_factor)               # + backprop graph
    act_train_bs = act_train_b1 * batch_size

    # ── Fixed VRAM components (always analytical) ──────────────────────
    model_vram     = total_params * param_bytes
    grad_vram      = total_params * 4
    optimizer_vram = total_params * 2 * 4
    gradscaler_vram = (total_params * 2) if use_amp else 0

    # ── Training VRAM ──────────────────────────────────────────────────
    if train_peak_bs is not None:
        # GPU-measured: peak already includes model weights + activations.
        # Add optimizer + gradscaler on top (they exist separately from the
        # forward/backward graph and are not counted in the raw peak delta).
        train_vram_raw  = train_peak_bs + optimizer_vram + gradscaler_vram
        train_vram_real = train_vram_raw * _CUDA_FRAG
        train_note      = train_source
    elif train_peak_b1 is not None:
        # Measured b=1, scale linearly for batch_size (activations scale linearly;
        # model+grad+opt are constant).
        act_b1_meas = train_peak_b1  # peak at b=1 ≈ act + model weights
        fixed_non_act = grad_vram + optimizer_vram + gradscaler_vram
        train_vram_raw  = act_b1_meas * batch_size + fixed_non_act
        train_vram_real = train_vram_raw * _CUDA_FRAG
        train_note      = train_source + f" (extrapolated to batch={batch_size})"
    else:
        train_vram_raw  = model_vram + act_train_bs + grad_vram + optimizer_vram + gradscaler_vram
        train_vram_real = train_vram_raw * _CUDA_FRAG
        train_note      = f"analytical (backprop ×{bp_factor})"

    # ── Validation VRAM ────────────────────────────────────────────────
    vol_input_vram = in_channels * vD * vH * vW * param_bytes
    vol_accum_vram = 2 * 1 * vD * vH * vW * 4  # importance map + count map, fp32

    if val_peak_sw is not None:
        # GPU-measured: peak includes model + sw_batch activations.
        # Add optimizer + vol buffers on top.
        val_vram_raw  = val_peak_sw + optimizer_vram + vol_input_vram + vol_accum_vram
        val_vram_real = val_vram_raw * _CUDA_FRAG
        val_note      = val_source
    elif val_peak_per_win is not None:
        # Extrapolated from b=1 measurement
        val_vram_raw  = (val_peak_per_win * sw_batch_size
                         + optimizer_vram + vol_input_vram + vol_accum_vram)
        val_vram_real = val_vram_raw * _CUDA_FRAG
        val_note      = val_source
    else:
        sw_input_vram  = sw_batch_size * in_channels * pD * pH * pW * param_bytes
        sw_act_vram    = sw_batch_size * act_fwd_b1
        sw_output_vram = sw_batch_size * 1 * pD * pH * pW * param_bytes
        val_vram_raw   = (model_vram + optimizer_vram + vol_input_vram
                          + vol_accum_vram + sw_input_vram + sw_act_vram + sw_output_vram)
        val_vram_real  = val_vram_raw * _CUDA_FRAG
        val_note       = f"analytical (forward-only, no backprop)"

    # ── Peak overall ───────────────────────────────────────────────────
    peak_vram_real = max(train_vram_real, val_vram_real)

    # ── SW window count ────────────────────────────────────────────────
    total_windows = _count_sw_windows((vD, vH, vW), (pD, pH, pW), sw_overlap)
    sw_batches    = math.ceil(total_windows / sw_batch_size)

    # ── sw_batch_size suggestion ───────────────────────────────────────
    # Use per-window cost from measurement if available, else analytical
    per_win_cost = val_peak_per_win if val_peak_per_win is not None else None
    if per_win_cost is not None:
        fixed_val = optimizer_vram + vol_input_vram + vol_accum_vram
        avail_for_windows = (actual_vram_bytes / _CUDA_FRAG) * 0.85 - fixed_val - model_vram
        suggested_sw = max(1, int(avail_for_windows // per_win_cost))
    else:
        suggested_sw = _suggest_sw_batch_size(
            gpu_vram_bytes = actual_vram_bytes,
            model_vram     = model_vram,
            optimizer_vram = optimizer_vram,
            vol_input_vram = vol_input_vram,
            vol_accum_vram = vol_accum_vram,
            patch_size     = (pD, pH, pW),
            in_channels    = in_channels,
            param_bytes    = param_bytes,
            act_val_b1     = act_fwd_b1,
        )

    # ── RAM ────────────────────────────────────────────────────────────
    worker_ram   = num_workers * batch_size * in_channels * pD * pH * pW * 4
    n_cases      = _count_cases(images_dir)
    per_case_ram = in_channels * vD * vH * vW * 4
    cache_uses_ram = cache_mode == "ram"
    cached_frac = cache_rate if cache_uses_ram else 0.0

    cache_ram: Optional[int]
    cache_note: str
    if cache_uses_ram and n_cases is not None:
        cached_cases = int(n_cases * cached_frac)
        cache_ram    = cached_cases * per_case_ram
        cache_note   = f"{cached_cases}/{n_cases} cases × {_fmt_bytes(per_case_ram).strip()}/case"
    elif cache_uses_ram:
        cache_ram  = None
        cache_note = f"data dir not found — N_cases × {_fmt_bytes(per_case_ram).strip()}/case"
    else:
        cache_ram = 0
        cache_note = "n/a"

    total_ram = worker_ram + (cache_ram if cache_ram is not None else 0)

    # ── Per-epoch FLOPs ────────────────────────────────────────────────
    epoch_flops:     Optional[int] = None
    steps_per_epoch: Optional[int] = None
    if n_cases is not None:
        train_cases     = int(n_cases * (1.0 - val_fraction))
        steps_per_epoch = max(1, (train_cases * num_samples) // batch_size)
        epoch_flops     = step_flops * steps_per_epoch

    # ── Max safe batch_size suggestion ─────────────────────────────────
    if train_peak_b1 is not None:
        # Use measured b=1 cost to extrapolate
        fixed_non_act = grad_vram + optimizer_vram + gradscaler_vram
        avail_act     = (actual_vram_bytes / _CUDA_FRAG * 0.85) - fixed_non_act
        max_train_batch = max(1, int(avail_act // train_peak_b1))
    else:
        max_train_batch = 1
        for b in range(1, 128):
            total_b = (model_vram + act_train_b1 * b
                       + grad_vram + optimizer_vram
                       + gradscaler_vram) * _CUDA_FRAG
            if total_b > actual_vram_bytes * 0.85:
                max_train_batch = b - 1
                break
        else:
            max_train_batch = 127
        max_train_batch = max(1, max_train_batch)

    # ── Print report ───────────────────────────────────────────────────
    W  = 68
    LW = 42

    def _sep() -> None:
        print(f"    {'─' * (W - 4)}")

    def _hdr(title: str) -> None:
        print(f"\n  {'─' * (W - 4)}")
        print(f"  {title}")
        print()

    def _row(label: str, value: str, note: str = "") -> None:
        ns = f"  ← {note}" if note else ""
        print(f"    {label:<{LW}}{value}{ns}")

    def _risk_row(label: str, used: float, note: str = "") -> None:
        tag = _risk_tag(used, usable_vram)
        ns  = f"  ← {note}" if note else ""
        print(f"    {label:<{LW}}{_fmt_bytes(used)}{tag}{ns}")

    print()
    print("=" * W)
    print(f"  Resource Estimation  —  {config_path}")
    print("=" * W)
    print()
    print(f"  {'Model':<26}: {model_name}")
    print(f"  {'In channels':<26}: {in_channels}  ({modality_str})")
    print(f"  {'Patch size (D, H, W)':<26}: {patch_size}")
    print(f"  {'Batch size':<26}: {batch_size}")
    print(f"  {'AMP dtype':<26}: {amp_label}  ({param_bytes} bytes/elem)")
    print(f"  {'Deep supervision':<26}: {deep_supervision}")
    print(f"  {'num_workers':<26}: {num_workers}")
    print(f"  {'cache_mode':<26}: {cache_mode} (rate={cache_rate})")
    if cache_mode == "storage":
        print(f"  {'dataset_cache_dir':<26}: {cache_dir}")
    print(f"  {'sw_batch_size':<26}: {sw_batch_size}  "
          f"({total_windows} windows → {sw_batches} batch(es) per volume)")
    print(f"  {'sw_overlap':<26}: {sw_overlap}")
    if cuda_available:
        print(f"  {'GPU':<26}: {torch.cuda.get_device_name(device)}")
    print(f"  {'GPU VRAM (physical)':<26}: {gpu_vram_gb:.1f} GB  "
          f"(usable ≈ {usable_vram / 1024**3:.1f} GB after fragmentation)")
    if cuda_available:
        print(f"  {'Measurement mode':<26}: GPU (most accurate)")
    else:
        print(f"  {'Measurement mode':<26}: analytical (no CUDA GPU detected)")

    # ── Parameters ────────────────────────────────────────────────────
    _hdr("PARAMETERS")
    _row("Total parameters",   f"{total_params:>14,}")
    _row("  Trainable",        f"{trainable_params:>14,}")
    _row("Model weights VRAM", f"{_fmt_bytes(model_vram).strip():>14}")

    # ── FLOPs ─────────────────────────────────────────────────────────
    _hdr(f"FLOPs  (source: {flop_source})")
    _row("Forward pass (batch=1)",    _fmt_flops(forward_flops))
    _row("Training step (≈ 3× fwd)", _fmt_flops(step_flops))
    if epoch_flops is not None and steps_per_epoch is not None:
        _row(f"Per epoch ({steps_per_epoch} steps)", _fmt_flops(epoch_flops))
    else:
        _row("Per epoch", "N/A  (data dir not found)")
    if model_name == "deconver" and not torchinfo_ok:
        print()
        print("    NOTE: Deconver NDC layers may be undercounted. "
              "Install torchinfo for accuracy.")

    # ── VRAM: Training ────────────────────────────────────────────────
    _hdr(f"VRAM — TRAINING  ({train_note})")
    if train_peak_bs is not None:
        _row("  fwd+bwd peak (measured, no opt)",  _fmt_bytes(train_peak_bs))
        _row("  AdamW state  m₁+m₂  (fp32)",      _fmt_bytes(optimizer_vram))
        if use_amp:
            _row("  GradScaler fp16 grad copy",    _fmt_bytes(gradscaler_vram))
    elif train_peak_b1 is not None:
        _row(f"  fwd+bwd peak b=1 (measured)",     _fmt_bytes(train_peak_b1))
        _row(f"  × {batch_size} (linear extrap.)",
             _fmt_bytes(train_peak_b1 * batch_size))
        _row("  AdamW state  m₁+m₂  (fp32)",      _fmt_bytes(optimizer_vram))
        if use_amp:
            _row("  GradScaler fp16 grad copy",    _fmt_bytes(gradscaler_vram))
    else:
        _row("  Model weights",                    _fmt_bytes(model_vram))
        _row(f"  Activations  batch={batch_size}  "
             f"(×{bp_factor} backprop)",           _fmt_bytes(act_train_bs))
        _row("  Gradients (fp32 master)",          _fmt_bytes(grad_vram))
        _row("  AdamW state  m₁+m₂  (fp32)",      _fmt_bytes(optimizer_vram))
        if use_amp:
            _row("  GradScaler fp16 grad copy",    _fmt_bytes(gradscaler_vram))
    _sep()
    _row("  Subtotal (no fragmentation)",          _fmt_bytes(train_vram_raw))
    _risk_row("  Realistic total (+15% frag)",     train_vram_real)

    # ── VRAM: Validation ──────────────────────────────────────────────
    _hdr(f"VRAM — VALIDATION  ({val_note})")
    if val_peak_sw is not None:
        _row(f"  fwd peak sw_batch={sw_batch_size} (measured)",
             _fmt_bytes(val_peak_sw))
    elif val_peak_per_win is not None:
        _row(f"  fwd peak sw_batch=1 (measured)",
             _fmt_bytes(val_peak_per_win))
        _row(f"  × {sw_batch_size} (linear extrap.)",
             _fmt_bytes(val_peak_per_win * sw_batch_size))
    _row("  AdamW state stays resident",           _fmt_bytes(optimizer_vram))
    _row(f"  Full volume input  {list(_TYPICAL_VOLUME_SHAPE)}",
         _fmt_bytes(vol_input_vram))
    _row("  MONAI accum buffers (2×, fp32)",       _fmt_bytes(vol_accum_vram))
    if val_peak_sw is None:
        sw_in  = sw_batch_size * in_channels * pD * pH * pW * param_bytes
        sw_act = sw_batch_size * act_fwd_b1
        sw_out = sw_batch_size * 1 * pD * pH * pW * param_bytes
        _row(f"  SW input patches  ×{sw_batch_size}",  _fmt_bytes(sw_in))
        _row(f"  SW activations    ×{sw_batch_size}",  _fmt_bytes(sw_act))
        _row(f"  SW output logits  ×{sw_batch_size}",  _fmt_bytes(sw_out))
    _sep()
    _row("  Subtotal (no fragmentation)",          _fmt_bytes(val_vram_raw))
    _risk_row("  Realistic total (+15% frag)",     val_vram_real)

    print()
    _risk_row("  OVERALL PEAK (max of train/val)", peak_vram_real)

    # ── RAM ───────────────────────────────────────────────────────────
    _hdr("RAM")
    _row("DataLoader worker buffers", _fmt_bytes(worker_ram))
    if cache_mode == "ram":
        _row(f"Dataset cache  (rate={cache_rate})",
             _fmt_bytes(cache_ram) if cache_ram is not None else "  see below")
        print(f"    {'':>{LW}}  {cache_note}")
    elif cache_mode == "storage":
        _row("Dataset cache", "persistent storage")
        _row("Storage cache directory", str(cache_dir))
        _row("Dataset cache RAM impact", "none (beyond worker buffers)")
    else:
        _row("Dataset cache", "disabled")
    _sep()
    if cache_mode != "ram":
        _row("RAM total estimate", _fmt_bytes(total_ram))
    elif cache_ram is not None:
        _row("RAM total estimate", _fmt_bytes(total_ram))
    else:
        print(f"    {'RAM total':<{LW}}worker_ram + (N_cases × {_fmt_bytes(per_case_ram).strip()})")
        print(f"    {'':>{LW}}  (data dir not found; N_cases unknown)")

    # ── Recommendations ───────────────────────────────────────────────
    _hdr("RECOMMENDATIONS")

    train_fits = train_vram_real <= usable_vram
    val_fits   = val_vram_real   <= usable_vram

    _row("Configured batch_size",
         f"{batch_size}  {'✓ fits' if train_fits else '✗ may OOM'}")
    _row("Max safe batch_size",
         f"~{max_train_batch}  (targets 85% of usable VRAM)")

    print()
    _row("Configured sw_batch_size",
         f"{sw_batch_size}  ({total_windows} windows → {sw_batches} batch(es))"
         f"  {'✓ fits' if val_fits else '✗ may OOM'}")
    _row("Suggested  sw_batch_size",
         f"{suggested_sw}  (targets 85% of usable VRAM during validation)")

    if not val_fits and suggested_sw < sw_batch_size:
        print()
        print(f"    *** sw_batch_size={sw_batch_size} will likely cause OOM during validation.")
        print(f"        Reduce it to {suggested_sw} in your config.")
        safe_batches = math.ceil(total_windows / suggested_sw)
        print(f"        ({total_windows} windows would run in {safe_batches} batch(es) per volume.)")
    elif suggested_sw > sw_batch_size:
        print()
        print(f"    sw_batch_size could be increased to ~{suggested_sw} for faster validation.")

    if not train_fits and max_train_batch < batch_size:
        print()
        print(f"    *** batch_size={batch_size} will likely OOM during training.")
        print(f"        Reduce it to {max_train_batch} in your config.")

    # ── Notes ─────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print()
    print("  NOTES")
    print("  ─────")
    if cuda_available:
        print("  • VRAM figures are measured directly on your GPU and are accurate")
        print("    to within ~5–10% (CUDA allocator granularity).")
        print("  • Training peak = measured fwd+bwd + AdamW state + GradScaler.")
        print("  • Validation peak = measured sw inference + AdamW (stays resident)")
        print("    + full-volume input + MONAI importance-map + count-map buffers.")
    else:
        print(f"  • No CUDA GPU detected.  Activation estimates use a ×{bp_factor}")
        print("    backprop overhead factor (empirically measured on real hardware).")
        print("  • Run on a machine with CUDA for accurate GPU measurements.")
    print(f"  • Usable VRAM ≈ physical / {_CUDA_FRAG} (CUDA allocator fragmentation).")
    print("  • AdamW m₁/m₂ and master gradients are always fp32 even with AMP.")
    print(f"  • Full-volume shape assumed: {list(_TYPICAL_VOLUME_SHAPE)} at target_spacing.")
    print("    Actual volumes may differ; larger volumes → more validation VRAM.")
    if not _TORCHINFO_AVAILABLE:
        print("  • Install torchinfo for more accurate FLOPs: pip install torchinfo")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run resource estimation."""
    parser = argparse.ArgumentParser(
        description="Estimate VRAM, RAM, and FLOPs for a training configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/local_default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--gpu-vram",
        type=float,
        default=12.0,
        metavar="GB",
        help="Physical GPU VRAM in GB — used only when no CUDA GPU is detected. "
             "When a GPU is present, its actual VRAM is queried automatically.",
    )
    args = parser.parse_args()
    estimate(args.config, gpu_vram_gb=args.gpu_vram)


if __name__ == "__main__":
    main()
