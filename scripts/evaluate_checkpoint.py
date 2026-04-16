"""
evaluate_checkpoint.py — Load a trained checkpoint and evaluate on the
fixed hold-out test set (data/test_images/).

The test set is a permanently reserved group of 10 PI-CAI cases (5 positive,
5 negative) that are segregated from the training pool by download_dataset.sh.
Using a fixed set makes all evaluation runs directly comparable across
checkpoints and training runs.

Usage
-----
python scripts/evaluate_checkpoint.py \\
    --checkpoint outputs/runs/<run>/checkpoints/best.pt \\
    [--images-dir data/test_images] \\
    [--labels-dir data/labels]

Output
------
- Per-case metrics table printed to stdout.
- Aggregate metric summary printed to stdout.
- eval_visualization.png (5 rows × 20 axial slices) saved next to the
  checkpoint file, with semi-transparent colour overlays:
      Green  = ground truth only
      Red    = prediction only
      Yellow = overlap (both masks active)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve src/ imports (scripts/ are not on PYTHONPATH by default)
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from config import load_config  # noqa: E402
from dataset import PiCaiDataset, discover_cases, stratified_train_val_split  # noqa: E402
from metrics import compute_all_metrics  # noqa: E402
from models import UNet3D  # noqa: E402
from transforms import get_val_transforms  # noqa: E402
from utils import load_checkpoint  # noqa: E402

# ---------------------------------------------------------------------------
# Logging — configured once here (entry-point only)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress MONAI class-balanced sampler warnings that fire during
# sliding-window inference on volumes that have no lesion voxels.
warnings.filterwarnings(
    "ignore",
    message=".*unable to generate class balanced samples.*",
    category=UserWarning,
    module="monai",
)

# ---------------------------------------------------------------------------
# Visualization constants
# ---------------------------------------------------------------------------

_N_VIS_COLS: int = 20                                        # axial slices per row
_GT_COLOR:   tuple[float, float, float] = (0.0, 1.0, 0.0)   # green — ground truth
_PRED_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)   # red   — prediction
_OVL_COLOR:  tuple[float, float, float] = (1.0, 1.0, 0.0)   # yellow — overlap
_OVERLAY_ALPHA: float = 0.50


# ---------------------------------------------------------------------------
# Console formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    """Format a metric value to 4 d.p.; return 'n/a' for NaN."""
    return f"{v:.4f}" if not math.isnan(v) else "n/a"


def _section(title: str) -> None:
    """Print a labelled section divider to stdout."""
    bar = "─" * 68
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _normalize_vol_for_display(vol: np.ndarray) -> np.ndarray:
    """
    Clip a 3-D float volume to its [p1, p99] percentile range then
    scale linearly to [0, 1] for display.

    Parameters
    ----------
    vol : (D, H, W) float array

    Returns
    -------
    (D, H, W) float32 in [0, 1]
    """
    p1  = float(np.percentile(vol, 1))
    p99 = float(np.percentile(vol, 99))
    clipped = np.clip(vol, p1, p99)
    return ((clipped - p1) / max(p99 - p1, 1e-8)).astype(np.float32)


def _segmentation_overlay(
    gt: np.ndarray,
    pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Build an RGBA (H, W, 4) overlay from binary GT and prediction masks.

    Colour coding
    -------------
    - **Green**  where only GT is 1
    - **Red**    where only Pred is 1
    - **Yellow** where both are 1  (explicit, not an alpha blend artefact)
    - Transparent (alpha = 0) where neither mask is active

    Parameters
    ----------
    gt   : (H, W) binary array {0, 1}
    pred : (H, W) binary array {0, 1}
    alpha : opacity for coloured regions in [0, 1]

    Returns
    -------
    (H, W, 4) float32 RGBA array
    """
    h, w = gt.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    gt_only   = (gt > 0) & (pred == 0)
    pred_only = (pred > 0) & (gt == 0)
    both      = (gt > 0) & (pred > 0)

    rgba[gt_only,   :3] = _GT_COLOR    # green
    rgba[pred_only, :3] = _PRED_COLOR  # red
    rgba[both,      :3] = _OVL_COLOR   # yellow

    active = gt_only | pred_only | both
    rgba[active, 3] = alpha

    return rgba


def save_visualization(
    pos_results: list[dict],
    output_path: Path,
    n_cols: int = _N_VIS_COLS,
) -> None:
    """
    Save a grid of axial overlay images to *output_path*.

    Layout
    ------
    - Rows    : one per positive case (at most 5)
    - Columns : *n_cols* (default 20) evenly-spaced axial slices
    - Per cell:
        - Greyscale T2w channel as background
        - Semi-transparent colour overlay (green / red / yellow)

    Colour key
    ----------
    Green  = ground truth lesion  (GT only)
    Red    = model prediction     (Pred only)
    Yellow = overlap of both masks

    Parameters
    ----------
    pos_results : list of dicts; each must contain:
                  "case_id"  str
                  "t2w_vol"  np.ndarray (D, H, W)  z-scored T2w channel
                  "gt_vol"   np.ndarray (D, H, W)  binary GT mask
                  "pred_vol" np.ndarray (D, H, W)  binary prediction mask
    output_path : destination PNG file path
    n_cols      : number of axial slices per row
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend; safe for scripts
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )
        return

    n_rows = len(pos_results)
    if n_rows == 0:
        logger.warning("No positive cases available — skipping visualization.")
        return

    cell_w, cell_h = 1.15, 1.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_w, n_rows * cell_h + 0.5),
        squeeze=False,
        gridspec_kw={"wspace": 0.03, "hspace": 0.08},
    )
    fig.patch.set_facecolor("#111111")

    for row_idx, res in enumerate(pos_results):
        case_id  = res["case_id"]
        t2w_norm = _normalize_vol_for_display(res["t2w_vol"])  # (D, H, W) in [0,1]
        gt_vol   = res["gt_vol"]                               # (D, H, W) binary
        pred_vol = res["pred_vol"]                             # (D, H, W) binary

        D = t2w_norm.shape[0]
        slice_indices = np.linspace(0, D - 1, n_cols, dtype=int)

        for col_idx, s in enumerate(slice_indices):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("black")

            # Layer 1: T2w greyscale background
            ax.imshow(
                t2w_norm[s],
                cmap="gray", vmin=0.0, vmax=1.0,
                aspect="equal", interpolation="nearest",
            )

            # Layer 2: segmentation overlay (green / red / yellow)
            ax.imshow(
                _segmentation_overlay(gt_vol[s], pred_vol[s], _OVERLAY_ALPHA),
                aspect="equal", interpolation="nearest",
            )

            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Column header: slice index (top row only)
            if row_idx == 0:
                ax.set_title(str(s), fontsize=5, color="#aaaaaa", pad=1.5)

        # Case-ID label in the top-left corner of the first column
        axes[row_idx, 0].text(
            0.03, 0.97,
            case_id,
            transform=axes[row_idx, 0].transAxes,
            fontsize=5,
            color="white",
            va="top",
            ha="left",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.65,
                edgecolor="none",
            ),
        )

    # Colour legend centred at the bottom of the figure
    legend_patches = [
        mpatches.Patch(facecolor=(*_GT_COLOR,   0.9), label="Ground truth (GT)"),
        mpatches.Patch(facecolor=(*_PRED_COLOR, 0.9), label="Prediction"),
        mpatches.Patch(facecolor=(*_OVL_COLOR,  0.9), label="Overlap (GT ∩ Pred)"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=8,
        framealpha=0.3,
        facecolor="#333333",
        edgecolor="none",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.003),
    )

    plt.subplots_adjust(bottom=0.06, left=0.01, right=0.99, top=0.97)
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\n  Visualization saved  →  {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, run evaluation on the fixed test set, print results, save visualization."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained checkpoint on the fixed hold-out test set "
            "(data/test_images/) and produce an axial overlay visualization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        metavar="PATH",
        help="Path to a .pt checkpoint file (e.g. outputs/runs/.../best.pt)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/test_images",
        metavar="DIR",
        help="Root directory of PI-CAI .mha image files for the test set",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        metavar="DIR",
        help="Directory containing .nii.gz label masks",
    )
    args = parser.parse_args()

    # ---- Validate checkpoint path -----------------------------------------------
    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # ---- Auto-detect config ------------------------------------------------------
    # Training saves config.yaml at <run_dir>/config.yaml.
    # The checkpoint is at <run_dir>/checkpoints/<name>.pt, so run_dir is two levels up.
    run_dir  = ckpt_path.parent.parent
    cfg_path = run_dir / "config.yaml"
    fallback = Path(__file__).parent.parent / "configs" / "local_default.yaml"

    if cfg_path.exists():
        cfg = load_config(str(cfg_path))
        logger.info("Config loaded from %s", cfg_path)
    else:
        cfg = load_config(str(fallback))
        logger.warning(
            "config.yaml not found at %s; falling back to %s",
            cfg_path, fallback,
        )

    # ---- Device -----------------------------------------------------------------
    use_cuda = torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
    device   = torch.device("cuda" if use_cuda else "cpu")

    # ---- Model + checkpoint -----------------------------------------------------
    model = UNet3D(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 1),
        features=tuple(cfg.get("features", [32, 64, 128, 256])),
    ).to(device)

    ckpt          = load_checkpoint(ckpt_path, model, device=device)
    ckpt_epoch    = ckpt.get("epoch", "?")
    best_val_dice = float(ckpt.get("best_val_dice", float("nan")))
    model.eval()

    # ---- Discover + annotate cases ----------------------------------------------
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    logger.info("Discovering cases in %s ...", images_dir)
    all_cases = discover_cases(images_dir, labels_dir)

    if not all_cases:
        logger.error(
            "No cases found in %s. Check that the data directory is correct.",
            images_dir,
        )
        sys.exit(1)

    # stratified_train_val_split annotates every case dict with has_lesion
    # in-place (reads all label files once).  We call it purely for that
    # side-effect; the actual split result is not used here.
    logger.info(
        "Annotating %d test cases with has_lesion ...",
        len(all_cases),
    )
    stratified_train_val_split(all_cases, val_fraction=0.2, seed=42)

    test_cases = all_cases
    pos_count  = sum(1 for c in test_cases if c.get("has_lesion", False))
    neg_count  = len(test_cases) - pos_count

    # ---- Dataset + loader -------------------------------------------------------
    target_spacing: tuple[float, ...] = tuple(
        float(v) for v in cfg.get("target_spacing", [3.0, 0.5, 0.5])
    )
    patch_size: tuple[int, ...] = tuple(
        int(v) for v in cfg.get("patch_size", [20, 128, 128])
    )
    sw_overlap    = float(cfg.get("sw_overlap", 0.5))
    sw_batch_size = int(cfg.get("sw_batch_size", 4))

    ds = PiCaiDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        target_spacing=target_spacing,
        transform=get_val_transforms(),
        cases=test_cases,
    )

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,           # fixed small set — no worker overhead needed
        pin_memory=(device.type == "cuda"),
    )

    # ---- Header -----------------------------------------------------------------
    _section("Checkpoint Evaluation  (fixed test set)")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"  Epoch       : {ckpt_epoch}   |   Best val Dice (training): {_fmt(best_val_dice)}")
    print(f"  Device      : {device}")
    print(f"  Test cases  : {len(test_cases)}  ({pos_count} positive, {neg_count} negative)")
    print(f"  Images dir  : {images_dir}")
    print(f"  Patch size  : {patch_size}   SW overlap: {sw_overlap}")

    # ---- Inference loop ---------------------------------------------------------
    per_case: list[dict] = []
    vis_data: list[dict] = []          # volumetric arrays for up to 5 positive cases

    # Fast case_id → has_lesion lookup (avoids re-searching test_cases per batch)
    lesion_map: dict[str, bool] = {
        c["case_id"]: c.get("has_lesion", False) for c in test_cases
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="vol"):
            images   = batch["image"].to(device)    # (1, 3, D, H, W)
            labels   = batch["label"].to(device)    # (1, 1, D, H, W)
            case_id: str = batch["case_id"][0]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=sw_overlap,
            )   # (1, 1, D, H, W) — raw logits

            m          = compute_all_metrics(logits, labels)
            has_lesion = lesion_map.get(case_id, False)

            per_case.append({
                "case_id":    case_id,
                "has_lesion": has_lesion,
                **m,
            })

            # Store volumetric data for the first 5 positive cases (visualization)
            if has_lesion and len(vis_data) < 5:
                pred_bin = (torch.sigmoid(logits) >= 0.5).float()
                vis_data.append({
                    "case_id":  case_id,
                    "t2w_vol":  images[0, 0].cpu().numpy(),      # (D, H, W)
                    "gt_vol":   labels[0, 0].cpu().numpy(),      # (D, H, W)
                    "pred_vol": pred_bin[0, 0].cpu().numpy(),    # (D, H, W)
                })

    # ---- Per-case table ---------------------------------------------------------
    _section("Per-Case Results")
    cid_w  = max(len(r["case_id"]) for r in per_case)
    header = (
        f"  {'case_id':<{cid_w}}  {'lesion':<7}"
        f"  {'dice':>7}  {'iou':>7}  {'sens':>7}  {'prec':>7}  {'hd95':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in per_case:
        tag = "yes" if r["has_lesion"] else "no"
        print(
            f"  {r['case_id']:<{cid_w}}  {tag:<7}"
            f"  {_fmt(r['dice']):>7}  {_fmt(r['iou']):>7}"
            f"  {_fmt(r['sensitivity']):>7}  {_fmt(r['precision']):>7}"
            f"  {_fmt(r['hd95']):>8}"
        )

    # ---- Aggregate metrics ------------------------------------------------------
    pos_rows = [r for r in per_case if r["has_lesion"]]
    neg_rows = [r for r in per_case if not r["has_lesion"]]

    # dice / iou / sensitivity: positive cases only (nan-guard)
    dice_vals = [r["dice"]        for r in pos_rows if not math.isnan(r["dice"])]
    iou_vals  = [r["iou"]         for r in pos_rows if not math.isnan(r["iou"])]
    sens_vals = [r["sensitivity"] for r in pos_rows if not math.isnan(r["sensitivity"])]

    # precision: positive cases only (nan when target is empty)
    prec_vals = [r["precision"] for r in pos_rows if not math.isnan(r["precision"])]

    # hd95: non-empty pairs only (nan when either mask is empty)
    hd95_vals = [r["hd95"] for r in per_case if not math.isnan(r["hd95"])]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    _section(
        f"Aggregate Metrics  "
        f"({len(pos_rows)} positive | {len(neg_rows)} negative | {len(per_case)} total)"
    )
    print(f"  Dice         (positive cases only) : {_fmt(_mean(dice_vals))}")
    print(f"  IoU          (positive cases only) : {_fmt(_mean(iou_vals))}")
    print(f"  Sensitivity  (positive cases only) : {_fmt(_mean(sens_vals))}")
    print(f"  Precision    (positive cases only) : {_fmt(_mean(prec_vals))}")
    print(f"  HD95         (non-empty pairs)     : {_fmt(_mean(hd95_vals))} voxels")

    # ---- Visualization ----------------------------------------------------------
    _section("Visualization  (5 rows × 20 axial slices)")
    vis_path = ckpt_path.parent / "eval_visualization.png"
    save_visualization(vis_data, vis_path, n_cols=_N_VIS_COLS)
    print(
        "\n  Colour key:\n"
        "    Green  = ground truth only\n"
        "    Red    = prediction only\n"
        "    Yellow = overlap (GT ∩ Pred)\n"
    )


if __name__ == "__main__":
    main()
