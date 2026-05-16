#!/usr/bin/env python3
"""
Extract the first frame from GIF files as PNG images.

By default, the script scans `visualizations/` recursively and extracts the
first frame from every `.gif` it finds.

You can also pass one or more explicit GIF paths with `--gif`.
Both modes can be used together in one command.

Examples:
    PYTHONPATH=. python scripts/extract_first_gif_frame.py

    PYTHONPATH=. python scripts/extract_first_gif_frame.py \
        --gif visualizations/20260514_153746_deconver_tuned_a_10726_1000742_t2w_orbit.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first frame from GIF(s) into PNG files."
    )
    parser.add_argument(
        "--gif",
        nargs="+",
        default=[],
        help="One or more GIF files to process explicitly.",
    )
    parser.add_argument(
        "--visualizations-dir",
        default="visualizations",
        help="Directory to scan recursively for GIF files (default: visualizations).",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip scanning --visualizations-dir and only process --gif inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output PNG files.",
    )
    return parser.parse_args()


def output_path_for_gif(gif_path: Path) -> Path:
    return gif_path.with_name(f"{gif_path.stem}_first_frame.png")


def collect_gifs(args: argparse.Namespace) -> list[Path]:
    collected: list[Path] = []
    seen: set[Path] = set()

    for raw in args.gif:
        p = Path(raw).expanduser().resolve()
        if p.suffix.lower() != ".gif":
            raise ValueError(f"Not a GIF file: {p}")
        if not p.exists():
            raise FileNotFoundError(f"GIF file not found: {p}")
        if p not in seen:
            collected.append(p)
            seen.add(p)

    if not args.skip_visualizations:
        visualizations_dir = Path(args.visualizations_dir).expanduser().resolve()
        if not visualizations_dir.exists():
            raise FileNotFoundError(
                f"Visualizations directory not found: {visualizations_dir}"
            )
        if not visualizations_dir.is_dir():
            raise NotADirectoryError(
                f"Visualizations path is not a directory: {visualizations_dir}"
            )

        for p in sorted(visualizations_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() != ".gif":
                continue
            resolved = p.resolve()
            if resolved in seen:
                continue
            collected.append(resolved)
            seen.add(resolved)

    return collected


def extract_first_frame(gif_path: Path, overwrite: bool) -> tuple[bool, Path]:
    output_path = output_path_for_gif(gif_path)
    if output_path.exists() and not overwrite:
        return False, output_path

    with Image.open(gif_path) as image:
        image.seek(0)
        # Convert explicitly to avoid mode issues when saving as PNG.
        frame = image.convert("RGBA")
        frame.save(output_path, format="PNG")

    return True, output_path


def main() -> int:
    args = parse_args()
    gifs = collect_gifs(args)
    if not gifs:
        print("No GIF files found to process.")
        return 0

    created_count = 0
    skipped_count = 0
    print(f"Processing {len(gifs)} GIF file(s)...")

    for gif_path in gifs:
        created, output_path = extract_first_frame(gif_path, overwrite=args.overwrite)
        if created:
            created_count += 1
            print(f"[created] {output_path}")
        else:
            skipped_count += 1
            print(f"[skipped] {output_path} (already exists; use --overwrite)")

    print(
        "Done. "
        f"Created: {created_count}, "
        f"Skipped: {skipped_count}, "
        f"Total: {len(gifs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
