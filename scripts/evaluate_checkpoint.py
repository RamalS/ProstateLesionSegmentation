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
    --run outputs/runs/<run> \\
    [--images-dir data/test_images] \\
    [--labels-dir data/labels]

The run directory must contain:
  config.yaml       — the YAML config the model was trained with
  checkpoints/      — one or more .pt checkpoint files

An interactive arrow-key menu lets you select which checkpoint to evaluate.
In non-interactive environments (no TTY) best.pt is selected automatically;
if best.pt is absent the newest checkpoint by filename is used.

Output
------
- Per-case metrics table printed to stdout.
- Aggregate metric summary printed to stdout.
- eval_visualization.png (5 rows × 20 axial slices) saved in the run dir next to the
  selected checkpoint file, with semi-transparent colour overlays:
      Green  = ground truth only
      Red    = prediction only
      Yellow = overlap (both masks active)
"""

from __future__ import annotations

import argparse
import curses
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
from models import build_model  # noqa: E402
from postprocess import postprocess_logits  # noqa: E402
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
# Checkpoint selection
# ---------------------------------------------------------------------------

def _load_epoch(path: Path) -> str:
    """
    Read only the ``epoch`` scalar from a checkpoint file without loading
    the full tensor state.  Returns the epoch as a string, or ``"?"`` on
    any failure.

    Parameters
    ----------
    path : Path
        Absolute path to a ``.pt`` checkpoint file.

    Returns
    -------
    str
        Epoch number as a string (e.g. ``"139"``), or ``"?"`` if the key
        is absent or the file cannot be read.
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", "?")
        return str(epoch)
    except Exception:  # noqa: BLE001
        return "?"


def _build_checkpoint_list(ckpts_dir: Path) -> list[Path]:
    """
    Return all ``.pt`` files in *ckpts_dir* sorted for display:
    ``best.pt`` pinned first, then remaining files in descending
    lexicographic order (newest epoch filename first).

    Parameters
    ----------
    ckpts_dir : Path
        Directory that contains ``.pt`` checkpoint files.

    Returns
    -------
    list[Path]
        Sorted list of absolute checkpoint paths.  Empty list when no
        ``.pt`` files are found.
    """
    all_pts = list(ckpts_dir.glob("*.pt"))
    best    = [p for p in all_pts if p.name == "best.pt"]
    others  = sorted(
        [p for p in all_pts if p.name != "best.pt"],
        key=lambda p: p.name,
        reverse=True,   # descending: epoch_0139 before epoch_0138
    )
    return best + others


def _format_size(n_bytes: int) -> str:
    """
    Format a byte count as a human-readable string (KB / MB).

    Parameters
    ----------
    n_bytes : int
        File size in bytes.

    Returns
    -------
    str
        Formatted string such as ``"123.4 MB"`` or ``"456.7 KB"``.
    """
    mb = n_bytes / (1024 * 1024)
    if mb >= 1.0:
        return f"{mb:.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def select_checkpoint(ckpts_dir: Path) -> Path:
    """
    Interactively select a ``.pt`` checkpoint from *ckpts_dir* using a
    ``curses`` arrow-key menu.

    Each row in the menu shows:
    ``filename     <size>   [epoch <n>]``

    Controls
    --------
    ↑ / ↓          move selection up / down
    Page Up / Down  jump 5 rows
    Enter           confirm selection
    q / Ctrl-C      abort (exits the process)

    Non-TTY fallback
    ----------------
    When ``sys.stdin`` is not a TTY (e.g. piped input or Docker without
    ``-it``), the menu is skipped: ``best.pt`` is returned automatically
    if present, otherwise the first entry in the sorted list (newest
    epoch).  A warning is logged in this case.

    Parameters
    ----------
    ckpts_dir : Path
        Directory containing ``.pt`` checkpoint files.

    Returns
    -------
    Path
        Absolute path to the selected checkpoint file.

    Raises
    ------
    SystemExit
        If no ``.pt`` files are found, or if the user quits the menu.
    """
    checkpoints = _build_checkpoint_list(ckpts_dir)
    if not checkpoints:
        logger.error("No .pt checkpoint files found in %s", ckpts_dir)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Non-interactive fallback
    # ------------------------------------------------------------------ #
    if not sys.stdin.isatty():
        best_pts = [p for p in checkpoints if p.name == "best.pt"]
        chosen   = best_pts[0] if best_pts else checkpoints[0]
        logger.warning(
            "Non-interactive environment detected — auto-selecting: %s",
            chosen.name,
        )
        return chosen

    # ------------------------------------------------------------------ #
    # Pre-load epoch labels (done once before entering curses)
    # ------------------------------------------------------------------ #
    logger.info("Reading checkpoint metadata (%d files) …", len(checkpoints))
    epochs: list[str] = [_load_epoch(p) for p in checkpoints]
    sizes:  list[str] = [_format_size(p.stat().st_size) for p in checkpoints]

    # Build display rows: pad filename and size columns for alignment
    names      = [p.name for p in checkpoints]
    name_w     = max(len(n) for n in names)
    size_w     = max(len(s) for s in sizes)
    rows: list[str] = [
        f"{n:<{name_w}}   {s:>{size_w}}   [epoch {e}]"
        for n, s, e in zip(names, sizes, epochs)
    ]

    # ------------------------------------------------------------------ #
    # curses UI
    # ------------------------------------------------------------------ #
    selected_idx: int = 0

    def _draw(stdscr: curses.window, idx: int) -> None:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        title  = "Select checkpoint  (↑/↓  PgUp/PgDn  Enter to confirm  q to quit)"
        border = "─" * min(len(title), max_x - 1)

        stdscr.addstr(0, 0, title[:max_x - 1],  curses.A_BOLD)
        stdscr.addstr(1, 0, border[:max_x - 1])

        # Scrolling: keep selected row visible
        visible = max_y - 3   # rows available for the list
        start   = max(0, min(idx - visible // 2, len(rows) - visible))

        for offset, (row_text, row_idx) in enumerate(
            zip(rows[start : start + visible], range(start, start + visible))
        ):
            y = 2 + offset
            if y >= max_y - 1:
                break
            prefix = "▶ " if row_idx == idx else "  "
            line   = (prefix + row_text)[: max_x - 1]
            attr   = curses.A_REVERSE if row_idx == idx else curses.A_NORMAL
            stdscr.addstr(y, 0, line, attr)

        stdscr.refresh()

    def _run(stdscr: curses.window) -> int:
        nonlocal selected_idx
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            _draw(stdscr, selected_idx)
            key = stdscr.getch()

            if key in (curses.KEY_UP, ord("k")):
                selected_idx = max(0, selected_idx - 1)
            elif key in (curses.KEY_DOWN, ord("j")):
                selected_idx = min(len(rows) - 1, selected_idx + 1)
            elif key == curses.KEY_PPAGE:       # Page Up
                selected_idx = max(0, selected_idx - 5)
            elif key == curses.KEY_NPAGE:       # Page Down
                selected_idx = min(len(rows) - 1, selected_idx + 5)
            elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
                return selected_idx
            elif key in (ord("q"), ord("Q"), 27):   # q / Esc
                return -1

        return selected_idx  # unreachable

    try:
        chosen_idx = curses.wrapper(_run)
    except KeyboardInterrupt:
        chosen_idx = -1

    if chosen_idx == -1:
        print("Aborted.")
        sys.exit(0)

    return checkpoints[chosen_idx]


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
        "--run",
        required=True,
        metavar="DIR",
        help=(
            "Path to a training run directory "
            "(e.g. /outputs/20260418_224246_deconver). "
            "Must contain config.yaml and a checkpoints/ subdirectory."
        ),
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
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="DEVICE",
        help=(
            "Override the compute device from config "
            "(e.g. 'cpu', 'cuda', 'cuda:0'). "
            "Useful when the GPU is not compatible with the installed PyTorch build."
        ),
    )
    args = parser.parse_args()

    # ---- Validate run directory -------------------------------------------------
    run_dir   = Path(args.run).resolve()
    ckpts_dir = run_dir / "checkpoints"
    cfg_path  = run_dir / "config.yaml"

    if not run_dir.is_dir():
        logger.error("Run directory not found: %s", run_dir)
        sys.exit(1)
    if not cfg_path.exists():
        logger.error("config.yaml not found in run directory: %s", cfg_path)
        sys.exit(1)
    if not ckpts_dir.is_dir():
        logger.error("checkpoints/ subdirectory not found in: %s", run_dir)
        sys.exit(1)

    # ---- Load config ------------------------------------------------------------
    cfg = load_config(str(cfg_path))
    logger.info("Config loaded from %s", cfg_path)
    pred_threshold: float = float(cfg.get("pred_threshold", 0.5))
    postprocess_enabled: bool = bool(cfg.get("postprocess_enabled", False))
    postprocess_min_component_volume_mm3: float = float(
        cfg.get("postprocess_min_component_volume_mm3", 30.0)
    )
    postprocess_connectivity: int = int(cfg.get("postprocess_connectivity", 26))

    if not (0.0 <= pred_threshold <= 1.0):
        logger.error("pred_threshold must be in [0,1], got %s", pred_threshold)
        sys.exit(1)
    if postprocess_min_component_volume_mm3 < 0.0:
        logger.error(
            "postprocess_min_component_volume_mm3 must be >= 0, got %s",
            postprocess_min_component_volume_mm3,
        )
        sys.exit(1)
    if postprocess_connectivity not in (6, 18, 26):
        logger.error(
            "postprocess_connectivity must be one of {6,18,26}, got %s",
            postprocess_connectivity,
        )
        sys.exit(1)

    # ---- Select checkpoint interactively ----------------------------------------
    ckpt_path = select_checkpoint(ckpts_dir)
    logger.info("Selected checkpoint: %s", ckpt_path.name)

    # ---- Device -----------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
        logger.info("Device overridden via --device flag: %s", device)
    else:
        use_cuda = torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda"
        device   = torch.device("cuda" if use_cuda else "cpu")

    # ---- Model + checkpoint -----------------------------------------------------
    model = build_model(cfg).to(device)

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
    print(f"  Threshold   : {pred_threshold:.3f}")
    print(
        "  Postprocess : "
        f"{'on' if postprocess_enabled else 'off'} "
        f"(min_component_volume_mm3={postprocess_min_component_volume_mm3:.1f}, "
        f"connectivity={postprocess_connectivity})"
    )

    # ---- Inference loop ---------------------------------------------------------
    per_case: list[dict] = []
    vis_data: list[dict] = []          # volumetric arrays for up to 5 positive cases

    # Fast case_id → has_lesion lookup (avoids re-searching test_cases per batch)
    lesion_map: dict[str, bool] = {
        c["case_id"]: c.get("has_lesion", False) for c in test_cases
    }

    # When deep supervision is active the model returns a list of tensors
    # (finest → coarsest).  sliding_window_inference requires a callable
    # that returns a single tensor, so we wrap accordingly.
    _predictor = (lambda x: model(x)[0]) if cfg.get("deep_supervision") else model

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="vol"):
            images   = batch["image"].to(device)    # (1, 3, D, H, W)
            labels   = batch["label"].to(device)    # (1, 1, D, H, W)
            case_id: str = batch["case_id"][0]

            logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=_predictor,
                overlap=sw_overlap,
            )   # (1, 1, D, H, W) — raw logits

            metric_logits, pred_bin = postprocess_logits(
                logits=logits.float(),
                threshold=pred_threshold,
                enabled=postprocess_enabled,
                spacing_zyx=target_spacing,
                min_component_volume_mm3=postprocess_min_component_volume_mm3,
                connectivity=postprocess_connectivity,
            )
            m = compute_all_metrics(
                metric_logits,
                labels,
                threshold=pred_threshold,
            )
            has_lesion = lesion_map.get(case_id, False)

            per_case.append({
                "case_id":    case_id,
                "has_lesion": has_lesion,
                **m,
            })

            # Store volumetric data for the first 5 positive cases (visualization)
            if has_lesion and len(vis_data) < 5:
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
    vis_path = run_dir / "eval_visualization.png"
    save_visualization(vis_data, vis_path, n_cols=_N_VIS_COLS)
    print(
        "\n  Colour key:\n"
        "    Green  = ground truth only\n"
        "    Red    = prediction only\n"
        "    Yellow = overlap (GT ∩ Pred)\n"
    )


if __name__ == "__main__":
    main()
