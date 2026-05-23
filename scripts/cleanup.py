#!/usr/bin/env python3
"""
Delete checkpoint files for runs that completed fewer than N epochs.

By default this script performs a dry run and prints what would be deleted.
Use ``--apply`` to actually remove checkpoint files.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


RUN_NAME_RE = re.compile(r"^\d{8}_\d{6}_.+$")
EPOCH_FILE_RE = re.compile(r"^epoch_(\d{4})\.pt$")


@dataclass
class CleanupCandidate:
    run_dir: Path
    max_epoch: int
    checkpoint_files: list[Path]


def _iter_run_dirs(runs_dir: Path) -> list[Path]:
    return sorted(
        [p for p in runs_dir.iterdir() if p.is_dir() and RUN_NAME_RE.match(p.name)],
        key=lambda p: p.name,
    )


def _max_epoch_from_checkpoints(ckpt_dir: Path) -> int | None:
    max_epoch: int | None = None
    for path in ckpt_dir.iterdir():
        match = EPOCH_FILE_RE.match(path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        if max_epoch is None or epoch > max_epoch:
            max_epoch = epoch
    return max_epoch


def _collect_candidates(runs_dir: Path, min_epochs: int) -> list[CleanupCandidate]:
    candidates: list[CleanupCandidate] = []
    for run_dir in _iter_run_dirs(runs_dir):
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            continue

        max_epoch = _max_epoch_from_checkpoints(ckpt_dir)
        if max_epoch is None or max_epoch >= min_epochs:
            continue

        checkpoint_files = sorted(ckpt_dir.glob("*.pt"))
        if not checkpoint_files:
            continue

        candidates.append(
            CleanupCandidate(
                run_dir=run_dir,
                max_epoch=max_epoch,
                checkpoint_files=checkpoint_files,
            )
        )
    return candidates


def _format_size(n_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(n_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n_bytes}B"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete checkpoint files for runs with fewer than N completed epochs. "
            "Dry-run by default."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/runs"),
        help="Root directory containing run folders (default: outputs/runs).",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
        help="Delete checkpoints when max epoch is below this threshold (default: 10).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, only print what would be deleted.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    runs_dir: Path = args.runs_dir
    min_epochs: int = args.min_epochs

    if min_epochs <= 0:
        raise ValueError("--min-epochs must be greater than 0.")
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not runs_dir.is_dir():
        raise NotADirectoryError(f"--runs-dir is not a directory: {runs_dir}")

    candidates = _collect_candidates(runs_dir=runs_dir, min_epochs=min_epochs)

    if not candidates:
        print(f"No checkpoint cleanup candidates found (min_epochs={min_epochs}).")
        return 0

    total_files = sum(len(c.checkpoint_files) for c in candidates)
    total_bytes = sum(p.stat().st_size for c in candidates for p in c.checkpoint_files)

    mode_label = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode_label}] Found {len(candidates)} run(s) below {min_epochs} epochs.")
    for candidate in candidates:
        run_name = candidate.run_dir.name
        print(
            f"- {run_name}: max_epoch={candidate.max_epoch}, "
            f"files={len(candidate.checkpoint_files)}"
        )
        for ckpt in candidate.checkpoint_files:
            print(f"  {ckpt}")

    print(f"Total files: {total_files} ({_format_size(total_bytes)})")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete these files.")
        return 0

    deleted = 0
    for candidate in candidates:
        for ckpt in candidate.checkpoint_files:
            ckpt.unlink()
            deleted += 1

    print(f"Deleted {deleted} checkpoint file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
