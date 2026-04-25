# AGENTS.md - ProstateLesionSegmentation

## Fast commands (repo root)

- Build image: `docker compose build` (uses `compose.yml`, service `trainer`).
- Train in Docker: `docker compose run --rm trainer train`
- Smoke test (primary regression check): `docker compose run --rm trainer smoke-test`
- TensorBoard: `docker compose run --rm --service-ports trainer tensorboard`
- 3-D visualizer (GT only): `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha`
- 3-D visualizer (GT vs model): `docker compose run --rm trainer visualize-3d --t2w /data/test_images/<case>_t2w.mha --run /outputs/runs/<run_name>`
- 3-D visualizer app (localhost): `docker compose run --rm --service-ports trainer visualize-3d-app` then open `http://localhost:8501`
- Learnability sanity run: `docker compose run --rm trainer learnability [N]`
- Shell in container: `docker compose run --rm trainer shell`
- Local train: `PYTHONPATH=. python -m src.train --config configs/local_default.yaml`
- Local smoke test: `PYTHONPATH=. python scripts/smoke_test.py`
- Local 3-D visualizer: `PYTHONPATH=. python scripts/visualize_3d.py --t2w data/test_images/<case>_t2w.mha [--run outputs/runs/<run_name>]`
- Local 3-D visualizer app: `PYTHONPATH=. streamlit run scripts/visualize_3d_app.py --server.address 0.0.0.0 --server.port 8501`

## Evaluation is easy to run wrong

- `scripts/evaluate_checkpoint.py` requires `--run <run_dir>`, not `--checkpoint`.
- Docker: `docker compose run --rm -it trainer evaluate --run /outputs/runs/<run_name>`
- Local: `PYTHONPATH=. python scripts/evaluate_checkpoint.py --run outputs/runs/<run_name>`
- Without a TTY, checkpoint selection auto-picks `best.pt` (or newest epoch file).
- Defaults are `--images-dir data/test_images` and `--labels-dir data/labels`.

## Repo wiring

- Source is imported from `src/` via `PYTHONPATH`; prefer `python -m src.train`.
- Main entrypoint: `src/train.py`; model factory: `src/models/__init__.py`.
- Supported model names: `unet3d`, `attention_unet3d`, `deconver`.
- `deconver` is vendored under `src/models/deconver/` and added to `sys.path` in `src/models/__init__.py`.
- If editing vendored Deconver internals, read `src/models/deconver/AGENTS.md` too.

## Paths and outputs (Docker defaults)

- `compose.yml` mounts: `./ -> /workspace`, `./data -> /data`, `./outputs -> /outputs`, `./cache -> /cache`.
- Docker train mode uses `configs/default.yaml`; local runs use `configs/local_default.yaml`.
- Run artifacts go to `<base_output_dir>/<timestamp>_<experiment_name>/`.
- Each run stores `checkpoints/`, `tensorboard/`, `config.yaml`, and `metadata.json`.
- `scripts/start.sh --local ...` switches container commands to local config and repo-relative paths.

## Data assumptions (`src/dataset.py`)

- Both PI-CAI layouts are supported: flat files or nested per-patient/per-case folders.
- T2w (`*_t2w.mha`) is always required for case discovery.
- ADC/HBV files are required only when `use_adc` / `use_hbv` are enabled.
- Labels are `<case_id>.nii.gz`; labels are binarized at load time (`>0` means lesion).

## Training and metric quirks for debugging

- Validation can reduce `sw_batch_size` automatically on CUDA OOM (`validate_with_oom_retry`).
- Dice/IoU/Sensitivity/Precision are averaged over positive-label cases only and may be `nan`.
- `best.pt` is selected by composite score (`sensitivity`, `dice`, optional `hd95`), not dice alone.
- `keep_last_checkpoints` rotates only `epoch_*.pt`; `best.pt` is not removed.

## Dependency and CI gotchas

- Docker pins `torch==2.7.0`, `torchvision==0.22.0`, `torchaudio==2.7.0` in `Dockerfile`.
- Docker filters torch packages out of `requirements.txt`; update `Dockerfile` too when changing torch versions.
- `setuptools<80` is intentional (TensorBoard still depends on `pkg_resources`).
- Pushing to branch `train` triggers `.github/workflows/train-sync.yml` on a self-hosted runner.
- `scripts/train-sync.sh` hard-resets and cleans the remote clone before checkout; treat pushes to `train` as destructive sync/deploy triggers.
