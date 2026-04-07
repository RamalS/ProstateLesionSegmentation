# Prostate Lesion Segmentation Environment

## Description

Systems that enable fast and accurate detection and quantification of prostate tumor lesions are important in clinical practice, as they improve both the speed and quality of diagnosis. This project is part of a master’s thesis focused on developing and evaluating a method for the automatic segmentation of suspicious prostate lesions using deep neural networks.

The work includes analysis of prostate MRI modalities (T2-weighted, ADC, DWI), investigation of PI-RADS criteria, and exploration of modern segmentation approaches such as CNN- and transformer-based architectures. A multimodal model is implemented using one or more MRI sequences, including preprocessing steps like normalization and registration, followed by training for lesion segmentation.

Evaluation is performed using standard metrics such as Dice, IoU, Sensitivity, and Hausdorff distance, with comparisons across different model variants.

## Build

```bash
docker compose build
```

## Docker Commands

This repository is set up to run via Docker Compose using `compose.yml` and the `trainer` service.

Prereqs:
- Docker Engine + `docker compose` (Compose v2)
- NVIDIA Container Toolkit (required for GPU runs)

### Build

```bash
docker compose build
```

### Smoke Test (CUDA/GPU visibility)

```bash
docker compose run --rm trainer smoke-test
```

### Training

Runs `python -m src.train --config /workspace/configs/default.yaml` inside the container.

```bash
docker compose run --rm trainer train
```

Artifacts:
- Host: `./outputs/runs/...`
- Container: `/outputs/runs/...`

### TensorBoard

Serves TensorBoard on `http://localhost:6006`.

```bash
docker compose run --rm --service-ports trainer tensorboard
```

### Shell

Interactive shell inside the container:

```bash
docker compose run --rm trainer shell
```

The default CMD is also `shell`, so this works too:

```bash
docker compose run --rm trainer
```

### Run Arbitrary Commands

Because the image has an entrypoint (`/workspace/scripts/start.sh`), the easiest way to run ad-hoc commands is overriding the entrypoint:

```bash
docker compose run --rm --entrypoint bash trainer
```

Notes:
- Repo is mounted at `/workspace`.
- Persistent directories are mounted:
  - `./data` -> `/data`
  - `./outputs` -> `/outputs`
  - `./cache` -> `/cache`
