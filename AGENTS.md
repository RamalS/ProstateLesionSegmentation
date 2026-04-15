# AGENTS.md — ProstateLesionSegmentation

Coding-agent reference for the ProstateLesionSegmentation repository.
This is a Python/PyTorch project for 3-D volumetric binary segmentation of
prostate lesions from multi-parametric MRI (PI-CAI dataset).

---

## Project Layout

```
src/              # All importable source code (PYTHONPATH=.)
  config.py       # YAML config loader
  dataset.py      # PiCaiDataset, discover_cases, stratified_train_val_split
  losses.py       # DiceBCELoss, TverskyBCELoss, DeepSupervisionWrapper
  metrics.py      # dice, iou, sensitivity, specificity, hd95, compute_all_metrics
  models/
    __init__.py   # build_model factory (selects unet3d or attention_unet3d)
    unet3d.py     # UNet3D (3-D encoder-decoder with skip connections)
    attention_unet3d.py  # AttentionUNet3D (UNet3D + attention gates on skip connections)
  notify.py       # ntfy push notification helper
  train.py        # Training + validation loop (entry point)
  transforms.py   # MONAI augmentation pipelines
  utils.py        # Shared helpers (checkpointing, run dirs, composite score)
scripts/
  smoke_test.py          # Manual integration smoke test (see below)
  start.sh               # Docker entrypoint dispatcher (train|tensorboard|smoke-test|evaluate|shell)
  evaluate_checkpoint.py # Evaluate a saved checkpoint on the hold-out test set
  count_positives.py     # Print dataset statistics (positive/negative case counts)
  download_dataset.sh    # Download PI-CAI image folds from zenodo
  download_labels.sh     # Download PI-CAI annotation labels
  list_checkpoints.sh    # List saved checkpoints for a run
  select_checkpoint.py   # Interactive checkpoint selection helper
  select_checkpoint.sh   # Shell wrapper for select_checkpoint.py
  train-sync.sh          # Sync code to the remote training machine
  update_repo.sh         # Pull latest code on the remote machine
configs/
  default.yaml          # Production / Docker paths (300 epochs)
  local_default.yaml    # Local dev paths (5 epochs, relative paths)
```

---

## Environment Setup

### Local (Python venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### Docker (preferred for training)

```bash
docker compose build
docker compose run --rm trainer shell   # interactive shell
docker compose run --rm trainer train   # run training
```

The Docker image is `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` with
PyTorch 2.6.0 + CUDA 12.6 wheels pinned in the Dockerfile.

---

## Build / Run Commands

| Task | Command |
|---|---|
| Train (Docker) | `docker compose run --rm trainer train` |
| Train (local) | `PYTHONPATH=. python src/train.py --config configs/local_default.yaml` |
| TensorBoard | `tensorboard --logdir outputs/runs --port 6006` |
| TensorBoard (Docker) | `docker compose run --rm --service-ports trainer tensorboard` |
| Evaluate checkpoint | `docker compose run --rm trainer evaluate --checkpoint <path>` |
| Interactive shell | `docker compose run --rm trainer shell` |

---

## Test / Smoke-Test Commands

There is **no pytest suite**. Testing is done via a single manual smoke test:

```bash
# Docker (canonical)
docker compose run --rm trainer smoke-test

# Local
PYTHONPATH=. python scripts/smoke_test.py
```

The smoke test verifies:
- PyTorch + CUDA availability
- Optional imports (SimpleITK, MONAI, nibabel, scipy)
- `UNet3D` instantiation, parameter count, forward-pass shape
- `AttentionUNet3D` instantiation, forward pass, `build_model` factory
- Modality flag selection: `build_model` derives `in_channels` from `use_t2w/use_adc/use_hbv`
- Deep supervision: list output, auxiliary shapes, `DeepSupervisionWrapper`
- `DiceBCELoss` and `TverskyBCELoss` forward passes
- LR warmup: `LinearLR` warm-up + `CosineAnnealingLR` via `SequentialLR`
- Loss robustness: negative-sample exclusion + FP16 overflow guard
- All five metrics functions (dice, iou, sensitivity, specificity, hd95)
- `discover_cases` / `stratified_train_val_split` with synthetic tempfile fixtures
- `get_train_transforms` / `get_val_transforms` on a dummy batch
- Checkpoint save/load round-trip (including `best_composite_score` and `GradScaler` state)
- `evaluate_checkpoint` helpers: visualization, overlay, PNG round-trip
- `compute_composite_score`: HD95=NaN redistribution, early stopping counter simulation
- `PiCaiDataset` in-memory cache (`use_cache=True`)
- AMP forward+backward: FP16+GradScaler (Volta/Turing) and BF16 (Ampere+/Blackwell)
- `send_ntfy`: no-op, URL/header/body correctness, error handling

When adding new functionality, add a corresponding block to `scripts/smoke_test.py`
and ensure `python scripts/smoke_test.py` exits with code 0.

---

## Linting and Formatting

No linter or formatter is currently enforced. Follow the conventions below
manually. If you add tooling, prefer **ruff** (lint + format) and **mypy**.

---

## Code Style Guidelines

### Imports

```python
from __future__ import annotations   # always first — enables PEP 563

# 1. Standard library
import logging
from pathlib import Path
from typing import Optional, Sequence

# 2. Third-party (torch, monai, numpy, …)
import torch
from monai.inferers import sliding_window_inference

# 3. Local (src.*)
from src.config import load_config
from src.dataset import PiCaiDataset
```

- Always include `from __future__ import annotations` as the very first line
  of every source file in `src/`.
- Follow stdlib → third-party → local ordering; separate groups with a blank line.

### Type Annotations

- Annotate **all** function signatures (parameters and return types).
- Use modern built-in generics (`list[...]`, `dict[...]`, `tuple[...]`) — they
  work because of `from __future__ import annotations`.
- Use `str | Path` union syntax instead of `Union[str, Path]`.
- `Optional[X]` (from `typing`) is acceptable but `X | None` is preferred.
- Use `# type: ignore[<code>]` only when a third-party stub is wrong; document why.

```python
def train_val_split(
    cases: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
```

### Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Functions / variables | `snake_case` | `discover_cases`, `val_fraction` |
| Classes | `PascalCase` | `UNet3D`, `PiCaiDataset`, `DiceBCELoss` |
| Module-level constants | `UPPER_SNAKE_CASE` | `MODALITY_KEYS`, `LABEL_SUFFIX` |
| Private helpers | `_leading_underscore` | `_resample`, `_zscore_normalize` |

### Docstrings

Every public function, method, and class must have a docstring.
Use NumPy-style with dashed section underlines:

```python
def dice_coefficient(preds: Tensor, targets: Tensor, smooth: float = 1e-6) -> float:
    """
    Volumetric Dice Similarity Coefficient (DSC).

    DSC = 2|P ∩ T| / (|P| + |T|)

    Parameters
    ----------
    preds   : raw logits or probabilities, shape (N, 1, D, H, W)
    targets : binary ground-truth mask, same shape
    smooth  : Laplace smoothing term to avoid division by zero

    Returns
    -------
    float in [0, 1]; higher is better.
    """
```

### Section Dividers

Use horizontal banners to separate logical sections within a file:

```python
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
```

### Logging

- Use `logging`, never `print`, for runtime messages.
- Declare a module-level logger: `logger = logging.getLogger(__name__)`
- Use `%`-style format strings (defers string formatting until the log record
  is actually emitted):

```python
logger = logging.getLogger(__name__)
logger.info("Split: %d train | %d val", len(train_cases), len(val_cases))
logger.warning("Case %s: missing modality '%s'", case_id, key)
```

- Root logging is configured once in `train.py`; do not call
  `logging.basicConfig` elsewhere.

### Error Handling

- Raise specific, descriptive exceptions:

```python
raise FileNotFoundError(f"Label not found for case {case_id}: {label_path}")
raise RuntimeError(
    f"No cases found in {cfg['images_dir']}. "
    "Check that your data is mounted correctly (./data -> /data)."
)
```

- Wrap optional dependencies in `try/except ImportError` with a boolean flag:

```python
try:
    from scipy.ndimage import distance_transform_edt as _edt
    _SCIPY_AVAILABLE = True
except ImportError:
    _edt = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False
```

### Configuration

- All hyperparameters and paths come from a YAML config dict (`cfg`), never
  hard-coded in `src/` modules.
- Use `configs/local_default.yaml` for local development (low epoch count,
  relative paths). Use `configs/default.yaml` for Docker/production.

---

## CI / CD

- The `train` branch triggers a **GitHub Actions self-hosted runner** that
  syncs code to the remote training machine (see `.github/workflows/train-sync.yml`).
- Do **not** push directly to `train` unless you intend to start a training run.
- Main development happens on `main` (or feature branches merged to `main`).
