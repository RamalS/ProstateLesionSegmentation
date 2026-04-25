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
  dataset.py                # PiCaiDataset, discover_cases, stratified_train_val_split
  losses.py                 # DiceBCELoss, TverskyBCELoss, DeepSupervisionWrapper
  metrics.py                # dice, iou, sensitivity, specificity, hd95, compute_all_metrics
  models/
    __init__.py             # build_model factory (selects unet3d or attention_unet3d)
    unet3d.py               # UNet3D (3D encoder-decoder with skip connections)
    attention_unet3d.py     # AttentionUNet3D (UNet3D + attention gates on skip connections)
  notify.py                 # ntfy push notification helper
  train.py                  # Training + validation loop (entry point)
  transforms.py             # MONAI augmentation pipelines
  utils.py                  # Shared helpers (checkpointing, run directories, composite score)
scripts/
  smoke_test.py             # Manual integration smoke test
  start.sh                  # Docker entrypoint dispatcher (train|tensorboard|smoke-test|evaluate|visualize-3d|visualize-3d-app|learnability|download|shell)
  evaluate_checkpoint.py    # Evaluate a saved checkpoint on the hold-out test set
  visualize_3d.py           # Interactive 3-D HTML visualizer (GT only or GT vs model prediction)
  visualize_3d_app.py       # Streamlit localhost app wrapper for visualize_3d.py
  count_positives.py        # Print dataset statistics (positive/negative case counts)
  download_dataset.sh       # Download PI-CAI + Prostate158 unlabeled images (single command)
  list_checkpoints.sh       # List saved checkpoints for a run
  select_checkpoint.py      # Interactive checkpoint selection helper
  select_checkpoint.sh      # Shell wrapper for select_checkpoint.py
  train-sync.sh             # Sync code to the remote training machine
  update_repo.sh            # Pull latest code on the remote machine
configs/
  default.yaml              # Production / Docker paths (300 epochs)
  local_default.yaml        # Local dev paths (5 epochs, relative paths)
```

---

## Model Architecture

**UNet3D** — a symmetric 3D U-Net (Çiçek et al., MICCAI 2016) with:

- **Encoder**: N levels of 2× (Conv3d → BatchNorm3d → LeakyReLU) + MaxPool3d
- **Bottleneck**: 2× conv block at the coarsest resolution
- **Decoder**: ConvTranspose3d upsampling + skip-connection concatenation + 2× conv block
- **Output**: 1×1×1 Conv3d producing raw logits (sigmoid applied externally)

**AttentionUNet3D** — identical to UNet3D with Attention Gates (Oktay et al., MIDL 2018) on every skip connection:

- Each gate reweights the encoder feature map by a spatially learned alpha mask before concatenation in the decoder
- Supports **deep supervision**: when `deep_supervision: true`, auxiliary output heads are attached at each decoder level (nnU-Net style); `forward` returns a `list[Tensor]` ordered finest → coarsest
- Selected at runtime via the `model` config key (`unet3d` or `attention_unet3d`)

Default configuration: feature sizes `[32, 64, 128, 256]` with a 512-channel bottleneck, up to 3 input channels (T2w + ADC + HBV, controlled by `use_t2w/use_adc/use_hbv` flags), 1 output channel (binary segmentation).

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
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
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
| **RAM** | 150 GB |
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
# Full dataset (default): PI-CAI (all 5 folds + labels) + Prostate158 unlabeled MRI
bash scripts/download_dataset.sh
# Prostate158 files are flattened into data/unlabeled_images as <case>_{t2,adc,dwi}.nii.gz

# Docker equivalent
docker compose run --rm trainer download

# Skip Prostate158 unlabeled download when needed
bash scripts/download_dataset.sh --no-unlabeled

# Optional PI-CAI subsets
bash scripts/download_dataset.sh 5 --no-labels --no-unlabeled
bash scripts/download_dataset.sh --labels-only

# Prostate158 unlabeled only (no PI-CAI images/labels)
bash scripts/download_dataset.sh --no-images --no-labels

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
| Download data | `docker compose run --rm trainer download` |
| Train | `docker compose run --rm trainer train` |
| Smoke test | `docker compose run --rm trainer smoke-test` |
| TensorBoard | `docker compose run --rm --service-ports trainer tensorboard` |
| 3-D visualizer (GT only) | `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha` |
| 3-D visualizer (GT vs model) | `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha --run /outputs/runs/<run_name>` |
| 3-D visualizer app (localhost) | `docker compose run --rm --service-ports trainer visualize-3d-app` then open `http://localhost:8501` |
| Evaluate checkpoint | `docker compose run --rm trainer evaluate --run /outputs/runs/<run_name>` |
| Interactive shell | `docker compose run --rm trainer shell` |

### Local

| Task | Command |
|---|---|
| Train | `PYTHONPATH=. python src/train.py --config configs/local_default.yaml` |
| Smoke test | `PYTHONPATH=. python scripts/smoke_test.py` |
| 3-D visualizer (GT only) | `PYTHONPATH=. python scripts/visualize_3d.py --t2w data/test_images/<case>_t2w.mha` |
| 3-D visualizer (GT vs model) | `PYTHONPATH=. python scripts/visualize_3d.py --t2w data/test_images/<case>_t2w.mha --run outputs/runs/<run_name>` |
| 3-D visualizer app (localhost) | `PYTHONPATH=. streamlit run scripts/visualize_3d_app.py --server.address 0.0.0.0 --server.port 8501` |
| Estimate resources | `PYTHONPATH=. python scripts/estimate_resources.py --config configs/default.yaml` |
| TensorBoard | `tensorboard --logdir outputs/runs --port 6006` |

---

## Configuration

All hyperparameters and paths are defined in YAML config files. Key parameters:

| Parameter | Default (Docker) | Local dev | Description |
|---|---|---|---|
| `epochs` | 300 | 5 | Number of training epochs |
| `batch_size` | 16 | 4 | Training batch size |
| `learning_rate` | 4e-4 | 2e-4 | Initial AdamW learning rate |
| `weight_decay` | 1e-5 | 1e-5 | AdamW weight decay |
| `patch_size` | `[20, 128, 128]` | `[20, 128, 128]` | (D, H, W) random crop size for training |
| `pos_fraction` | 0.85 | 0.75 | Fraction of patches containing a lesion |
| `target_spacing` | `[3.0, 0.5, 0.5]` | `[3.0, 0.5, 0.5]` | Voxel spacing (z, y, x) in mm after resampling |
| `val_fraction` | 0.2 | 0.2 | Fraction of data held out for validation |
| `model` | `attention_unet3d` | `unet3d` | Model architecture (`unet3d` or `attention_unet3d`) |
| `use_t2w / use_adc / use_hbv` | `true` | `true` | Modality flags (control number of input channels) |
| `features` | `[32, 64, 128, 256]` | `[32, 64, 128, 256]` | Encoder feature map sizes |
| `deep_supervision` | `false` | `false` | Auxiliary losses at each decoder level |
| `loss_fn` | `tversky_bce` | `dice_bce` | Loss function (`dice_bce` or `tversky_bce`) |
| `tversky_alpha / beta` | 0.3 / 0.7 | — | FP / FN penalty weights for TverskyBCELoss |
| `dice_weight` | 3.0 | 1.0 | Weight on the Dice/Tversky term |
| `bce_weight` | 1.0 | 1.0 | Weight on the BCE term |
| `bce_pos_weight` | 50.0 | 10.0 | BCE positive-voxel weight (class imbalance) |
| `warmup_epochs` | 10 | — | Linear LR warm-up epochs before cosine annealing |
| `sw_overlap` | 0.5 | 0.5 | Sliding-window overlap fraction during validation |
| `val_every` | 2 | 1 | Run validation every N epochs |
| `keep_last_checkpoints` | 3 | 3 | Recent `epoch_NNNN.pt` files to keep (0 = keep all) |
| `early_stopping_patience` | 30 | 0 | Consecutive val epochs without improvement before stopping (0 = disabled) |
| `amp_dtype` | `fp16` | `bf16` | AMP dtype (`fp16` for Volta/Turing, `bf16` for Ampere+/Blackwell) |
| `ntfy_url / ntfy_topic` | set | `""` | ntfy push notification server URL and topic (empty = disabled) |

---

## Training Pipeline

1. Load YAML config and set up output directories + TensorBoard.
2. Discover PI-CAI cases and split into train/validation sets (`stratified_train_val_split` preserves positive/negative ratio).
3. Build `PiCaiDataset` with MONAI augmentation transforms; optionally cache preprocessed volumes in RAM.
4. Instantiate model via `build_model(cfg)` (UNet3D or AttentionUNet3D), loss (`DiceBCELoss` or `TverskyBCELoss`), AdamW + linear warm-up + CosineAnnealingLR.
5. **Train**: random-patch forward pass → loss → backward; `WeightedRandomSampler` equalises positive/negative case frequency per epoch.
6. **Validate** (every `val_every` epochs): sliding-window inference over full volumes → Dice, IoU, Sensitivity, Specificity, HD95.
7. Save per-epoch checkpoints (rotating, keep last N) and a `best.pt` checkpoint by composite score (weighted sensitivity + Dice + HD95).
8. Send ntfy push notifications on training events when `ntfy_url` / `ntfy_topic` are set.

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

Checks (14 test groups): PyTorch + CUDA availability, optional imports, `UNet3D` and `AttentionUNet3D` forward passes, `build_model` factory and modality flags, deep supervision, `DiceBCELoss` and `TverskyBCELoss`, LR warm-up schedule, loss robustness (negative-sample exclusion + FP16 overflow guard), all five metrics, `discover_cases` / `stratified_train_val_split` with synthetic fixtures, MONAI transforms on a dummy batch, checkpoint save/load round-trip, `evaluate_checkpoint` helpers, `compute_composite_score`, `PiCaiDataset` in-memory cache, AMP (FP16 + BF16), and `send_ntfy` no-op / error handling.

---

## Notes

- The `train` branch triggers a GitHub Actions self-hosted runner that syncs code to the remote training machine. Do **not** push directly to `train` unless you intend to start a training run.
- Main development happens on `main` (or feature branches merged to `main`).
- Docker volumes: `./` → `/workspace` (full project), `./data` → `/data`, `./outputs` → `/outputs`, `./cache` → `/cache`.
