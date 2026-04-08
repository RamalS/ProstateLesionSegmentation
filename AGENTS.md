# AGENTS.md — ProstateLesionSegmentation

Coding-agent reference for the ProstateLesionSegmentation repository.
This is a Python/PyTorch project for 3-D volumetric binary segmentation of
prostate lesions from multi-parametric MRI (PI-CAI dataset).

---

## Project Layout

```
src/              # All importable source code (PYTHONPATH=/workspace)
  config.py       # YAML config loader
  dataset.py      # PiCaiDataset, discover_cases, train_val_split
  losses.py       # DiceBCELoss
  metrics.py      # dice, iou, sensitivity, specificity, hd95
  models/
    unet3d.py     # UNet3D (3-D encoder-decoder with skip connections)
  train.py        # Training + validation loop (entry point)
  transforms.py   # MONAI augmentation pipelines
  utils.py        # Shared helpers
scripts/
  smoke_test.py   # Manual integration smoke test (see below)
  start.sh        # Docker entrypoint dispatcher
configs/
  default.yaml          # Production / Docker paths
  local_default.yaml    # Local dev paths (epochs: 5)
```

---

## Environment Setup

### Local (Python 3.14 venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
- `DiceBCELoss` forward pass
- All five metrics functions (dice, iou, sensitivity, specificity, hd95)
- `discover_cases` / `train_val_split` with synthetic tempfile fixtures
- `get_train_transforms` / `get_val_transforms` on a dummy batch

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
