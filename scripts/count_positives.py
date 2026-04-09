"""
count_positives.py — Report positive/negative case statistics for a PI-CAI dataset.

Usage
-----
    PYTHONPATH=. python scripts/count_positives.py \\
        --images-dir data/images \\
        --labels-dir data/labels

    # Docker
    docker compose run --rm trainer shell
    python scripts/count_positives.py --images-dir /data/images --labels-dir /data/labels

Output
------
Prints a summary table:

    Total cases     : 1500
    Positive (csPCa): 425   (28.3 %)
    Negative        : 1075  (71.7 %)
    No label file   : 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dataset import _case_has_lesion, discover_cases  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,   # suppress INFO-level discovery noise
    format="%(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count positive / negative cases in a PI-CAI dataset directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        metavar="DIR",
        help="Root directory of PI-CAI images (flat or nested layout).",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        metavar="DIR",
        help="Directory containing .nii.gz label masks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-case positivity status.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    args = _parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        print(f"ERROR: images-dir does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.exists():
        print(f"ERROR: labels-dir does not exist: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning images : {images_dir}")
    print(f"Scanning labels : {labels_dir}")
    print()

    cases = discover_cases(images_dir, labels_dir)

    if not cases:
        print("No cases found. Check --images-dir and --labels-dir paths.")
        sys.exit(1)

    n_total = len(cases)
    n_no_label = sum(1 for c in cases if c["label"] is None)
    n_labelled = n_total - n_no_label

    print(f"Discovering lesion presence in {n_labelled} labelled cases "
          f"({n_no_label} unlabelled)...")

    n_pos = 0
    n_neg = 0
    for c in cases:
        if c["label"] is None:
            continue
        if _case_has_lesion(c):
            n_pos += 1
            if args.verbose:
                print(f"  [POS] {c['case_id']}")
        else:
            n_neg += 1
            if args.verbose:
                print(f"  [NEG] {c['case_id']}")

    pct_pos = 100.0 * n_pos / n_labelled if n_labelled > 0 else 0.0
    pct_neg = 100.0 * n_neg / n_labelled if n_labelled > 0 else 0.0

    print()
    print("=" * 42)
    print(f"  Total cases          : {n_total:>6d}")
    print(f"  Labelled             : {n_labelled:>6d}")
    print(f"  No label file        : {n_no_label:>6d}")
    print("-" * 42)
    print(f"  Positive (csPCa ≥1)  : {n_pos:>6d}  ({pct_pos:.1f} %)")
    print(f"  Negative (all-zero)  : {n_neg:>6d}  ({pct_neg:.1f} %)")
    print("=" * 42)


if __name__ == "__main__":
    main()
