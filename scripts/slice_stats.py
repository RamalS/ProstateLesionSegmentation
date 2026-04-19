"""
slice_stats.py - Report slice-count summary statistics for a PI-CAI dataset.

Usage
-----
    # Uses paths from config (default: configs/local_default.yaml)
    PYTHONPATH=. python scripts/slice_stats.py

    # Explicit config
    PYTHONPATH=. python scripts/slice_stats.py --config configs/default.yaml

    # Override paths from CLI
    PYTHONPATH=. python scripts/slice_stats.py \
        --images-dir data/images \
        --labels-dir data/labels

Output
------
Prints slice-count summary statistics across all discovered cases:

    Cases scanned : 1500
    Min           : 16
    Q1 (25%)      : 24
    Median (50%)  : 28
    Q3 (75%)      : 32
    Max           : 48
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from dataset import discover_cases  # noqa: E402


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report min/Q1/median/Q3/max slice counts across a PI-CAI dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="configs/local_default.yaml",
        metavar="FILE",
        help="YAML config file containing images_dir and labels_dir.",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        metavar="DIR",
        help="Override images_dir from config.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        metavar="DIR",
        help="Override labels_dir from config.",
    )
    return parser.parse_args()


def _resolve_config_path(raw_path: str) -> Path:
    """
    Resolve a path loaded from YAML config.

    Relative paths follow the same convention as train.py and are interpreted
    relative to the current working directory.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def _resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve dataset image/label directories from config and CLI overrides."""
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config file does not exist: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(str(config_path))
    if "images_dir" not in cfg or "labels_dir" not in cfg:
        print(
            "ERROR: config must define both 'images_dir' and 'labels_dir'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.images_dir is not None:
        images_dir = Path(args.images_dir).resolve()
    else:
        images_dir = _resolve_config_path(str(cfg["images_dir"]))

    if args.labels_dir is not None:
        labels_dir = Path(args.labels_dir).resolve()
    else:
        labels_dir = _resolve_config_path(str(cfg["labels_dir"]))

    return images_dir, labels_dir


# ---------------------------------------------------------------------------
# Slice counting
# ---------------------------------------------------------------------------

def _read_num_slices(t2w_path: Path) -> int:
    """
    Return number of slices (z-axis) from a raw T2w file.

    Uses SimpleITK metadata-only read (`ReadImageInformation`) so pixel
    intensities are not loaded into memory.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(t2w_path))
    reader.ReadImageInformation()

    size = reader.GetSize()  # SimpleITK order: (x, y, z)
    if len(size) != 3:
        raise RuntimeError(f"Expected 3D volume, got size={tuple(size)} for {t2w_path}")

    return int(size[2])


def _fmt_stat(value: float) -> str:
    """Format a statistic value for terminal output."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}"


def main() -> None:
    """Entry point."""
    args = _parse_args()
    images_dir, labels_dir = _resolve_dirs(args)

    if not images_dir.exists():
        print(f"ERROR: images-dir does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: labels-dir does not exist: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning images : {images_dir}")
    print(f"Scanning labels : {labels_dir}")
    print()

    cases = discover_cases(images_dir, labels_dir, active_keys=("t2w",))
    if not cases:
        print("No cases found. Check dataset paths and layout.", file=sys.stderr)
        sys.exit(1)

    slice_counts: list[int] = []
    for case in cases:
        case_id = case["case_id"]
        try:
            slice_counts.append(_read_num_slices(case["t2w"]))
        except Exception as exc:
            print(
                f"ERROR: failed to read slices for case {case_id} ({case['t2w']}): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    values = np.asarray(slice_counts, dtype=np.float64)
    q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])

    print("=" * 34)
    print(f"  Cases scanned : {len(values):>6d}")
    print("-" * 34)
    print(f"  Min           : {_fmt_stat(float(values.min())):>6}")
    print(f"  Q1 (25%)      : {_fmt_stat(float(q1)):>6}")
    print(f"  Median (50%)  : {_fmt_stat(float(median)):>6}")
    print(f"  Q3 (75%)      : {_fmt_stat(float(q3)):>6}")
    print(f"  Max           : {_fmt_stat(float(values.max())):>6}")
    print("=" * 34)


if __name__ == "__main__":
    main()
