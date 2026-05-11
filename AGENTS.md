# AGENTS.md - ProstateLesionSegmentation

## Fast commands (repo root)

- Build image (modern default, cu128): `docker compose build` (uses `compose.yml`, service `trainer`).
- Build image (Volta/TITAN V, cu126): `docker compose -f compose.yml -f compose.volta.yml build`
- Train in Docker: `docker compose run --rm trainer train`
- Train Deconver tuned A: `docker compose run --rm trainer train --config /workspace/configs/deconver_conf.yaml`
- Train Deconver tuned B (num_samples=2): `docker compose run --rm trainer train --config /workspace/configs/deconver_tuned_b.yaml`
- Train Deconver tuned C (bce_pos_weight=20): `docker compose run --rm trainer train --config /workspace/configs/deconver_tuned_c.yaml`
- Train with current config from resumed checkpoint weights only: `docker compose run --rm trainer train --config /workspace/configs/deconver_conf.yaml --resume /outputs/runs/<run_name>/checkpoints/best.pt --current-config`
- Pretrain encoder (SSL) in Docker: `docker compose run --rm trainer pretrain`
- Smoke test (primary regression check): `docker compose run --rm trainer smoke-test`
- Train in Docker (Volta/TITAN V): `docker compose -f compose.yml -f compose.volta.yml run --rm trainer train`
- Pretrain in Docker (Volta/TITAN V): `docker compose -f compose.yml -f compose.volta.yml run --rm trainer pretrain`
- TensorBoard: `docker compose run --rm --service-ports trainer tensorboard`
- 3-D visualizer (GT only): `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha`
- 3-D visualizer (GT vs model): `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha --run /outputs/runs/<run_name>`
- 3-D visualizer GIF export: `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha --gif`
- 3-D visualizer app (localhost): `docker compose run --rm --service-ports trainer visualize-3d-app` then open `http://localhost:8501`
- Run report regeneration: `docker compose run --rm trainer report-runs --visualizations-dir /workspace/visualizations --output /workspace/report.md`
- Full reporting pipeline (missing-only, Docker-first): `PYTHONPATH=. python scripts/report_pipeline.py`
- Full reporting pipeline (force all runs): `PYTHONPATH=. python scripts/report_pipeline.py --all`
- Learnability sanity run: `docker compose run --rm trainer learnability [N]`
- Shell in container: `docker compose run --rm trainer shell`
- Local train: `PYTHONPATH=. python -m src.train --config configs/local_default.yaml`
- Local train Deconver tuned A: `PYTHONPATH=. python -m src.train --config configs/deconver_conf.yaml`
- Local train Deconver tuned B (num_samples=2): `PYTHONPATH=. python -m src.train --config configs/deconver_tuned_b.yaml`
- Local train Deconver tuned C (bce_pos_weight=20): `PYTHONPATH=. python -m src.train --config configs/deconver_tuned_c.yaml`
- Local train with current config from resumed checkpoint weights only: `PYTHONPATH=. python -m src.train --config configs/deconver_conf.yaml --resume outputs/runs/<run_name>/checkpoints/best.pt --current-config`
- Local pretrain: `PYTHONPATH=. python -m src.pretrain --config configs/pretrain_local.yaml`
- Local smoke test: `PYTHONPATH=. python scripts/smoke_test.py`
- Local 3-D visualizer: `PYTHONPATH=. python scripts/visualize_3d.py --t2w data/test_images/<case>_t2w.mha [--run outputs/runs/<run_name>]`
- Local 3-D visualizer GIF export: `PYTHONPATH=. python scripts/visualize_3d.py --t2w data/test_images/<case>_t2w.mha --gif`
- Local 3-D visualizer app: `PYTHONPATH=. streamlit run scripts/visualize_3d_app.py --server.address 0.0.0.0 --server.port 8501`

## Evaluation is easy to run wrong

- `scripts/evaluate_checkpoint.py` requires `--run <run_dir>`, not `--checkpoint`.
- Docker: `docker compose run --rm -it trainer evaluate --run /outputs/runs/<run_name>`
- Local: `PYTHONPATH=. python scripts/evaluate_checkpoint.py --run outputs/runs/<run_name>`
- Without a TTY, checkpoint selection auto-picks `best.pt` (or newest epoch file).
- Defaults are `--images-dir data/test_images` and `--labels-dir data/labels`.

## Repo wiring

- Source is imported from `src/` via `PYTHONPATH`; prefer `python -m src.train`.
- Main supervised entrypoint: `src/train.py`; SSL pretraining entrypoint: `src/pretrain.py`.
- Model factory: `src/models/__init__.py`.
- Supported model names: `unet3d`, `attention_unet3d`, `deconver`.
- `deconver` is vendored under `src/models/deconver/` and added to `sys.path` in `src/models/__init__.py`.
- If editing vendored Deconver internals, read `src/models/deconver/AGENTS.md` too.

## Paths and outputs (Docker defaults)

- `compose.yml` mounts: `./ -> /workspace`, `./data -> /data`, `./outputs -> /outputs`, `./cache -> /cache`.
- Docker train mode uses `configs/default.yaml`; local runs use `configs/local_default.yaml`.
- Docker pretrain mode uses `configs/pretrain_default.yaml`; local pretrain uses `configs/pretrain_local.yaml`.
- Run artifacts go to `<base_output_dir>/<timestamp>_<experiment_name>/`.
- Each run stores `checkpoints/`, `tensorboard/`, `config.yaml`, and `metadata.json`.
- `scripts/start.sh --local ...` switches container commands to local config and repo-relative paths.

## Data assumptions (`src/dataset.py`)

- Both PI-CAI layouts are supported: flat files or nested per-patient/per-case folders.
- T2w (`*_t2w.mha`) is always required for case discovery.
- ADC/HBV files are required only when `use_adc` / `use_hbv` are enabled.
- Unlabeled Prostate158 discovery expects flattened files in `data/unlabeled_images/` as `<case>_{t2,adc,dwi}.nii.gz`.
- For SSL pretraining, Prostate158 DWI is mapped to the HBV channel (`hbv_source="dwi"`), with optional DWI preprocessing via `dwi_hbv_preprocess`.
- Labels are `<case_id>.nii.gz`; labels are binarized at load time (`>0` means lesion).

## Training and metric quirks for debugging

- Validation can reduce `sw_batch_size` automatically on CUDA OOM (`validate_with_oom_retry`).
- Dice/IoU/Sensitivity/Precision are averaged over positive-label cases only and may be `nan`.
- `best.pt` is selected by composite score (`sensitivity`, `dice`, optional `hd95`), not dice alone.
- `keep_last_checkpoints` rotates only `epoch_*.pt`; `best.pt` is not removed.
- Supervised training can warm-start from SSL weights via `pretrained_encoder_checkpoint`; optional staged unfreezing is controlled by `freeze_encoder_epochs`.
- `--resume` restores full state (model + optimizer + scheduler + scaler + epoch); add `--current-config` to switch resume into model-weights-only init under current config (fresh optimizer/scheduler/scaler, epoch 1). `--current-config` requires `--resume` or `resume_checkpoint`.

## Dependency and CI gotchas

- Docker pins `torch==2.7.0`, `torchvision==0.22.0`, `torchaudio==2.7.0` in `compose.yml` build args (passed into `Dockerfile`).
- Default Docker stack is `cu128` (modern GPUs); Volta/TITAN V uses `compose.volta.yml` override (`cu126`).
- Docker filters torch packages out of `requirements.txt`; update `Dockerfile` too when changing torch versions.
- Docker GIF export path relies on `plotly==6.7.0` + `kaleido==0.2.1` (pinned in `requirements.txt`) to avoid known hangs with newer Kaleido + headless Chrome combos.
- `setuptools<80` is intentional (TensorBoard still depends on `pkg_resources`).
- Pushing to branch `train` triggers `.github/workflows/train-sync.yml` on a self-hosted runner.
- `scripts/train-sync.sh` hard-resets and cleans the remote clone before checkout; treat pushes to `train` as destructive sync/deploy triggers. It defaults to the Volta stack (`TORCH_STACK=volta`).
