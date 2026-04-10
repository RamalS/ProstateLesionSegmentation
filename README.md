# Prostate Lesion Segmentation

## Description

Systems that enable fast and accurate detection and quantification of prostate tumor lesions are important in clinical practice, as they improve both the speed and quality of diagnosis. This project is part of a master's thesis focused on developing and evaluating a method for the automatic segmentation of suspicious prostate lesions using deep neural networks.

The work includes analysis of prostate MRI modalities (T2-weighted, ADC, DWI), investigation of PI-RADS criteria, and exploration of modern segmentation approaches such as CNN- and transformer-based architectures. A multimodal model is implemented using one or more MRI sequences, including preprocessing steps like normalization and registration, followed by training for lesion segmentation.

Evaluation is performed using standard metrics such as Dice, IoU, Sensitivity, Specificity, and 95th-percentile Hausdorff Distance (HD95), with comparisons across different model variants.

---

## Project Layout

```
src/                        # All importable source code (PYTHONPATH=.)
  config.py                 # YAML config loader
  dataset.py                # PiCaiDataset, discover_cases, train_val_split
  losses.py                 # DiceBCELoss (Dice + BCE combined loss)
  metrics.py                # dice, iou, sensitivity, specificity, hd95
  models/
    unet3d.py               # UNet3D (3D encoder-decoder with skip connections)
  train.py                  # Training + validation loop (entry point)
  transforms.py             # MONAI augmentation pipelines
  utils.py                  # Shared helpers (checkpointing, run directories)
scripts/
  smoke_test.py             # Manual integration smoke test
  start.sh                  # Docker entrypoint dispatcher
configs/
  default.yaml              # Production / Docker paths (100 epochs)
  local_default.yaml        # Local dev paths (5 epochs, relative paths)
```

---

## Model Architecture

**UNet3D** — a symmetric 3D U-Net (Çiçek et al., MICCAI 2016) with:

- **Encoder**: N levels of 2× (Conv3d → BatchNorm3d → LeakyReLU) + MaxPool3d
- **Bottleneck**: 2× conv block at the coarsest resolution
- **Decoder**: ConvTranspose3d upsampling + skip-connection concatenation + 2× conv block
- **Output**: 1×1×1 Conv3d producing raw logits (sigmoid applied externally)

Default configuration: feature sizes `[32, 64, 128, 256]` with a 512-channel bottleneck, 3 input channels (T2w + ADC + HBV), 1 output channel (binary segmentation).

---

## Environment Setup

### Docker (preferred)

Requires Docker Engine + Compose v2 and the NVIDIA Container Toolkit.

```bash
docker compose build
```

The image is based on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` with PyTorch 2.7.0 + CUDA 12.8 wheels.

### Local (Python venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

---

## Hardware

### Laptop (local dev)

| | |
|---|---|
| **OS** | Arch Linux (kernel 6.19.10) |
| **CPU** | Intel Core Ultra 7 255HX — 20 cores, 30 MB L3 cache |
| **RAM** | 32 GB |
| **Storage** | 1 TB NVMe SSD (SK Hynix PVC10) |
| **GPU** | NVIDIA GeForce RTX 5070 Laptop — Blackwell (sm_120), 8 GB VRAM |
| **CUDA** | 13.0, driver 595.58.03 |
| **AMP** | `local_default.yaml` — `amp_dtype: bf16` |

### Server (training)

| | |
|---|---|
| **OS** | KVM virtual machine |
| **CPU** | 6 vCPU (Skylake) |
| **RAM** | 80 GB |
| **Storage** | 128 GB QEMU virtual disk |
| **GPU** | NVIDIA TITAN V — Volta (sm_70), 12 GB VRAM |
| **AMP** | `default.yaml` — `amp_dtype: fp16` |

---

## Data Layout

Place the PI-CAI dataset under `./data/`:

```
data/
  images/          # Multi-channel MRI volumes (T2w, ADC, HBV)
  labels/          # Binary lesion segmentation masks
```

The Docker Compose file mounts `./data` → `/data` inside the container.

---

## Dataset

This project uses the **[PI-CAI (Prostate Imaging: Cancer AI)](https://pi-cai.grand-challenge.org/)** public training and development dataset.

### Overview

| Property | Value |
|---|---|
| Total cases | 1,500 |
| Positive (csPCa, ISUP ≥ 2) | 425 (28.3 %) |
| Negative (benign / indolent) | 1,075 (71.7 %) |
| Imaging modalities | Axial T2w, ADC, HBV (high b-value DWI) |
| Label format | Voxel-level lesion mask (grades 2–5); binarised to {0, 1} for training |
| Scanner vendors | Siemens Healthineers, Philips Medical Systems |
| Acquisition centers | RUMC (NL), ZGT (NL), PCNN (NL), STOH (NO) |

### Patient Population

All patients are men referred for prostate MRI due to clinical suspicion of prostate cancer (elevated PSA or abnormal DRE findings), without prior ISUP ≥ 2 findings or treatment. The 28 % positive rate reflects real-world clinical prevalence in this screening population.

### Labels

Labels use the **combined human-expert + AI-derived** annotation set from [`DIAGNijmegen/picai_labels`](https://github.com/DIAGNijmegen/picai_labels):

- **220 positive cases** carry human-expert voxel delineations.
- **205 positive cases** carry AI-derived delineations (semi-supervised; Bosma et al., 2022).
- **1,075 negative cases** have confirmed all-zero masks (histopathology ISUP ≤ 1 or PI-RADS ≤ 2).

Labels are binarised at load time: any voxel value > 0 is treated as lesion.

### Class Imbalance Strategy

The ~1:2.5 case-level and ~1:50+ voxel-level imbalance is addressed at four stacked layers:

| Layer | Mechanism |
|---|---|
| Data split | `stratified_train_val_split` preserves the 28/72 % ratio in both train and val subsets |
| Case sampling | `WeightedRandomSampler` (weight = 1 / class\_count) equalises positive/negative case frequency per epoch |
| Patch sampling | `RandCropByPosNegLabeld` with `pos_fraction=0.75` ensures 75 % of training patches are centred on a lesion voxel |
| Loss | `DiceBCELoss` with `bce_pos_weight=10.0` up-weights lesion-voxel gradients in the BCE term |

### Downloading the Data

```bash
# Images — download all 5 folds (~25 GB total, ~5 GB each)
bash scripts/download_dataset.sh 5

# Labels — follow the instructions at:
# https://github.com/DIAGNijmegen/picai_labels

# Verify dataset statistics after download
PYTHONPATH=. python scripts/count_positives.py \
    --images-dir data/images \
    --labels-dir data/labels
```

---

## Commands

### Docker

| Task | Command |
|---|---|
| Build image | `docker compose build` |
| Train | `docker compose run --rm trainer train` |
| Smoke test | `docker compose run --rm trainer smoke-test` |
| TensorBoard | `docker compose run --rm --service-ports trainer tensorboard` |
| Interactive shell | `docker compose run --rm trainer shell` |

### Local

| Task | Command |
|---|---|
| Train | `PYTHONPATH=. python src/train.py --config configs/local_default.yaml` |
| Smoke test | `PYTHONPATH=. python scripts/smoke_test.py` |
| TensorBoard | `tensorboard --logdir outputs/runs --port 6006` |

---

## Configuration

All hyperparameters and paths are defined in YAML config files. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 100 (5 local) | Number of training epochs |
| `batch_size` | 2 | Training batch size |
| `learning_rate` | 1e-4 | Initial AdamW learning rate |
| `weight_decay` | 1e-5 | AdamW weight decay |
| `patch_size` | `[20, 128, 128]` | (D, H, W) random crop size for training |
| `pos_fraction` | 0.75 | Fraction of patches containing a lesion |
| `target_spacing` | `[3.0, 0.5, 0.5]` | Voxel spacing (z, y, x) in mm after resampling |
| `val_fraction` | 0.2 | Fraction of data held out for validation |
| `in_channels` | 3 | Input MRI channels (T2w + ADC + HBV) |
| `features` | `[32, 64, 128, 256]` | Encoder feature map sizes |
| `dice_weight` | 1.0 | Weight on the Dice term in DiceBCELoss |
| `bce_weight` | 1.0 | Weight on the BCE term in DiceBCELoss |
| `sw_overlap` | 0.5 | Sliding-window overlap fraction during validation |

---

## Training Pipeline

1. Load YAML config and set up output directories + TensorBoard.
2. Discover PI-CAI cases and split into train/validation sets (stratified by `val_fraction`).
3. Build `PiCaiDataset` with MONAI augmentation transforms.
4. Instantiate `UNet3D`, `DiceBCELoss`, AdamW + CosineAnnealingLR.
5. **Train**: random-patch forward pass → Dice+BCE loss → backward.
6. **Validate**: sliding-window inference over full volumes → Dice, IoU, Sensitivity, Specificity, HD95.
7. Save per-epoch checkpoints and a `best.pt` checkpoint by validation Dice.

Training artifacts are written to `./outputs/runs/<experiment_name>_<timestamp>/`:

```
outputs/runs/<run>/
  checkpoints/
    epoch_0001.pt … epoch_NNNN.pt
    best.pt
  tensorboard/
  logs/
  config.yaml
  metadata.json
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Dice (DSC) | `2|P∩T| / (|P| + |T|)` — volumetric overlap; higher is better |
| IoU | `|P∩T| / |P∪T|` — Jaccard index; higher is better |
| Sensitivity | `TP / (TP + FN)` — true positive rate; higher is better |
| Specificity | `TN / (TN + FP)` — true negative rate; higher is better |
| HD95 | 95th-percentile Hausdorff Distance in voxels; lower is better |

---

## Smoke Test

The smoke test verifies the full stack without real data:

```bash
# Docker (canonical)
docker compose run --rm trainer smoke-test

# Local
PYTHONPATH=. python scripts/smoke_test.py
```

Checks: PyTorch + CUDA availability, optional imports, `UNet3D` forward pass, `DiceBCELoss`, all five metrics, `discover_cases` / `train_val_split` with synthetic fixtures, and MONAI transforms on a dummy batch.

---

## Notes

- The `train` branch triggers a GitHub Actions self-hosted runner that syncs code to the remote training machine. Do **not** push directly to `train` unless you intend to start a training run.
- Main development happens on `main` (or feature branches merged to `main`).
- Persistent Docker volumes: `./data` → `/data`, `./outputs` → `/outputs`, `./cache` → `/cache`.
