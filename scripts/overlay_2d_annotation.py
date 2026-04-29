#!/usr/bin/env python3
"""
overlay_2d_annotation.py - Overlay a 2D annotation mask on a 2D image.

Usage
-----
  PYTHONPATH=. python scripts/overlay_2d_annotation.py \
      --image data/test_images_2d/<case_id>.jpg \
      --annotation data/test_labels_2d/<case_id>.png \
      --output outputs/visualizations/<case_id>_overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_CLASS_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 0, 128),
)


def _parse_alpha(raw: str) -> float:
    try:
        alpha = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"alpha must be a float in [0, 1], got {raw!r}"
        ) from exc

    if not (0.0 <= alpha <= 1.0):
        raise argparse.ArgumentTypeError(
            f"alpha must be in [0, 1], got {alpha}"
        )
    return alpha


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"color must be 'R,G,B' with 3 values, got {raw!r}"
        )

    vals: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"color channel must be integer in [0, 255], got {p!r}"
            ) from exc
        if not (0 <= v <= 255):
            raise argparse.ArgumentTypeError(
                f"color channel must be in [0, 255], got {v}"
            )
        vals.append(v)

    return int(vals[0]), int(vals[1]), int(vals[2])


def _parse_class_colors(raw: str) -> dict[int, tuple[int, int, int]]:
    mappings: dict[int, tuple[int, int, int]] = {}
    entries = [entry.strip() for entry in raw.split(";") if entry.strip()]
    if not entries:
        raise argparse.ArgumentTypeError(
            "class-colors must contain at least one mapping like '1:255,0,0'"
        )

    for entry in entries:
        if ":" not in entry:
            raise argparse.ArgumentTypeError(
                f"invalid class-colors entry {entry!r}; expected '<class_id>:R,G,B'"
            )
        class_id_raw, color_raw = entry.split(":", 1)
        try:
            class_id = int(class_id_raw.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"class id must be an integer, got {class_id_raw!r}"
            ) from exc

        if not (0 <= class_id <= 255):
            raise argparse.ArgumentTypeError(
                f"class id must be in [0, 255], got {class_id}"
            )

        mappings[class_id] = _parse_rgb(color_raw.strip())

    return mappings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay a multi-class annotation mask onto a 2D image and save a blended PNG.",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Input image path (JPG expected, any PIL-readable format accepted).",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        type=str,
        help="Annotation/mask path (PNG expected). Pixel values are treated as class IDs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--alpha",
        type=_parse_alpha,
        default=0.5,
        help="Overlay opacity in [0,1]. Default: 0.5.",
    )
    parser.add_argument(
        "--color",
        type=_parse_rgb,
        default=(255, 0, 0),
        help="Primary overlay color as 'R,G,B'. Default: 255,0,0.",
    )
    parser.add_argument(
        "--class-colors",
        type=_parse_class_colors,
        default=None,
        help=(
            "Optional explicit per-class colors as "
            "'1:255,0,0;2:0,255,0;3:0,0,255'. "
            "Unspecified classes use an automatic palette."
        ),
    )
    parser.add_argument(
        "--background-class",
        type=int,
        default=0,
        help="Class ID treated as background and not overlaid. Default: 0.",
    )
    parser.add_argument(
        "--legend-font-size",
        type=int,
        default=0,
        help=(
            "Legend font size in px. Default 0 = auto-size. "
            "Use e.g. 24 or 32 for larger labels."
        ),
    )
    parser.add_argument(
        "--legend-font",
        type=str,
        default=None,
        help="Optional path to a .ttf/.otf font file for legend labels.",
    )
    return parser.parse_args()


def _load_image_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image file not found: {path}")
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    return arr


def _load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"annotation file not found: {path}")
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
    return arr


def _build_class_color_map(
    class_ids: list[int],
    primary_color: tuple[int, int, int],
    explicit_map: dict[int, tuple[int, int, int]] | None,
) -> dict[int, tuple[int, int, int]]:
    palette: list[tuple[int, int, int]] = [primary_color]
    for c in _DEFAULT_CLASS_PALETTE:
        if c != primary_color:
            palette.append(c)

    class_map: dict[int, tuple[int, int, int]] = {}
    palette_idx = 0
    for class_id in class_ids:
        if explicit_map is not None and class_id in explicit_map:
            class_map[class_id] = explicit_map[class_id]
            continue
        class_map[class_id] = palette[palette_idx % len(palette)]
        palette_idx += 1

    return class_map


def _blend_multiclass_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    background_class: int,
    alpha: float,
) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(
            f"image must have shape (H, W, 3), got {tuple(image_rgb.shape)}"
        )
    if mask.ndim != 2:
        raise ValueError(f"annotation must have shape (H, W), got {tuple(mask.shape)}")
    if image_rgb.shape[:2] != mask.shape:
        raise ValueError(
            "image and annotation size mismatch: "
            f"image={tuple(image_rgb.shape[:2])}, annotation={tuple(mask.shape)}"
        )
    if not (0 <= background_class <= 255):
        raise ValueError(
            f"background-class must be in [0, 255], got {background_class}"
        )

    out = image_rgb.astype(np.float32, copy=True)
    for class_id, color in class_colors.items():
        class_region = mask == class_id
        if not np.any(class_region):
            continue
        overlay_color = np.asarray(color, dtype=np.float32)
        out[class_region] = ((1.0 - alpha) * out[class_region]) + (alpha * overlay_color)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _load_legend_font(font_size: int, legend_font: str | None) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if legend_font:
        candidates.append(str(Path(legend_font).expanduser()))
    candidates.extend(
        [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
    )

    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size=font_size)
        except OSError:
            continue

    # Pillow >=10.1 may still provide a scalable default if FreeType is available.
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(1, right - left), max(1, bottom - top)


def _render_text_mask(text: str, font: ImageFont.ImageFont) -> Image.Image:
    probe = Image.new("L", (1, 1), 0)
    probe_draw = ImageDraw.Draw(probe)
    left, top, right, bottom = probe_draw.textbbox((0, 0), text, font=font)
    width = max(1, right - left)
    height = max(1, bottom - top)

    text_mask = Image.new("L", (width, height), 0)
    text_draw = ImageDraw.Draw(text_mask)
    text_draw.text((-left, -top), text, fill=255, font=font)
    return text_mask


def _draw_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    scale: int,
) -> None:
    if scale <= 1:
        draw.text(xy, text, fill=fill, font=font)
        return

    text_mask = _render_text_mask(text, font)
    if hasattr(Image, "Resampling"):
        resample_mode = Image.Resampling.NEAREST  # type: ignore[attr-defined]
    else:
        resample_mode = Image.NEAREST  # type: ignore[attr-defined]
    scaled_mask = text_mask.resize(
        (text_mask.width * scale, text_mask.height * scale),
        resample=resample_mode,
    )
    image.paste(fill, box=(xy[0], xy[1]), mask=scaled_mask)


def _draw_class_legend(
    image_rgb: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    legend_font_size: int = 0,
    legend_font: str | None = None,
) -> np.ndarray:
    if not class_colors:
        return image_rgb

    image = Image.fromarray(image_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)

    if legend_font_size > 0:
        base_font_size = legend_font_size
    else:
        base_font_size = max(22, min(image.width, image.height) // 24)
    font = _load_legend_font(base_font_size, legend_font)

    entries = [(class_id, class_colors[class_id]) for class_id in sorted(class_colors)]
    labels = [
        f"class {class_id} ({color[0]},{color[1]},{color[2]})"
        for class_id, color in entries
    ]

    # If we ended up with a non-scalable bitmap font, emulate larger text
    # by upscaling rendered text masks.
    probe_h = _text_size(draw, labels[0], font)[1]
    text_scale = max(1, int(round(base_font_size / max(1, probe_h))))

    pad = max(10, min(image.width, image.height) // 80)
    swatch = max(18, int(base_font_size * 0.9))

    text_w_max = 0
    text_h_max = 0
    for label in labels:
        w, h = _text_size(draw, label, font)
        text_w_max = max(text_w_max, w * text_scale)
        text_h_max = max(text_h_max, h * text_scale)
    line_h = max(swatch, text_h_max) + pad

    legend_w = (pad * 3) + swatch + text_w_max
    legend_h = (pad * 2) + (line_h * len(entries))

    x0 = pad
    y0 = pad
    x1 = min(image.width - pad, x0 + legend_w)
    y1 = min(image.height - pad, y0 + legend_h)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0), outline=(255, 255, 255), width=1)

    for idx, (class_id, color) in enumerate(entries):
        row_y = y0 + pad + (idx * line_h)
        swatch_y = row_y + max(0, (line_h - pad - swatch) // 2)
        sx0 = x0 + pad
        sx1 = sx0 + swatch
        sy0 = swatch_y
        sy1 = swatch_y + swatch
        draw.rectangle([sx0, sy0, sx1, sy1], fill=color, outline=(255, 255, 255), width=1)
        _draw_text(
            image=image,
            draw=draw,
            xy=(sx1 + pad, row_y),
            text=labels[idx],
            font=font,
            fill=(255, 255, 255),
            scale=text_scale,
        )

    return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = _parse_args()

    image_path = Path(args.image).expanduser().resolve()
    annotation_path = Path(args.annotation).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        image_rgb = _load_image_rgb(image_path)
        mask = _load_mask(annotation_path)

        class_ids = sorted(int(v) for v in np.unique(mask) if int(v) != int(args.background_class))
        class_color_map = _build_class_color_map(
            class_ids=class_ids,
            primary_color=args.color,
            explicit_map=args.class_colors,
        )

        blended = _blend_multiclass_overlay(
            image_rgb=image_rgb,
            mask=mask,
            class_colors=class_color_map,
            background_class=int(args.background_class),
            alpha=float(args.alpha),
        )
        blended = _draw_class_legend(
            blended,
            class_color_map,
            legend_font_size=int(args.legend_font_size),
            legend_font=args.legend_font,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended, mode="RGB").save(output_path, format="PNG")
    print(f"Saved overlay PNG: {output_path}")


if __name__ == "__main__":
    main()
