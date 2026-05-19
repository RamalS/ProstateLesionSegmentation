"""
Convert an external Deconver checkpoint into the repo's model-state format.

This script wraps a checkpoint containing top-level ``model`` weights into a
dictionary with ``model_state_dict`` so it can be consumed by
``load_pretrained_encoder`` in ``src/utils.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_INPUT = Path(
    "/home/ramals/School/deconver_isles22/logs/"
    "train_fold0_250225_082642565889/model_fold=0_checkpoint_epoch=500.pt"
)
DEFAULT_OUTPUT = Path(
    "/outputs/pretrained_external/"
    "deconver_isles22_fold0_epoch500_model_state_dict.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an external Deconver checkpoint with top-level 'model' "
            "weights into {'model_state_dict': ...} format."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to source checkpoint (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to converted checkpoint (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def _validate_and_extract_model_state(ckpt: Any, input_path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Expected top-level checkpoint object to be a dict, got {type(ckpt)!r} "
            f"for {input_path}."
        )
    if "model" not in ckpt:
        keys = sorted(str(k) for k in ckpt.keys())
        raise ValueError(
            f"Checkpoint {input_path} is missing top-level 'model' key. "
            f"Available keys: {keys}"
        )

    model_state = ckpt["model"]
    if not isinstance(model_state, dict):
        raise ValueError(
            f"Checkpoint {input_path} has non-dict 'model' payload of type "
            f"{type(model_state)!r}."
        )

    if not model_state:
        raise ValueError(f"Checkpoint {input_path} has an empty 'model' state dict.")

    non_tensor_keys = [k for k, v in model_state.items() if not torch.is_tensor(v)]
    if non_tensor_keys:
        preview = ", ".join(str(k) for k in non_tensor_keys[:5])
        raise ValueError(
            f"Checkpoint {input_path} has non-tensor model entries (e.g. {preview})."
        )

    return model_state


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model_state_dict = _validate_and_extract_model_state(checkpoint, input_path)

    converted = {
        "model_state_dict": model_state_dict,
        "source_path": str(input_path),
        "source_top_level_keys": (
            sorted(str(k) for k in checkpoint.keys())
            if isinstance(checkpoint, dict)
            else []
        ),
        "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)

    print(f"Converted checkpoint written to: {output_path}")
    print(f"Loaded tensors: {len(model_state_dict)}")


if __name__ == "__main__":
    main()
