#!/usr/bin/env bash
set -e

MODE="shell"
USE_LOCAL=false
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    --local)
      USE_LOCAL=true
      ;;
    train|train-2d|pretrain|tensorboard|smoke-test|shell|evaluate|evaluate-2d|learnability|download|visualize-3d|visualize-3d-app|report-runs)
      MODE="$arg"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

if [ "$USE_LOCAL" = true ]; then
  echo "Using LOCAL paths"
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  CONFIG_PATH="$PROJECT_ROOT/configs/local_default.yaml"
  CONFIG_2D_PATH="$PROJECT_ROOT/configs/deconver_2d_local.yaml"
  PRETRAIN_CONFIG_PATH="$PROJECT_ROOT/configs/pretrain_local.yaml"
  OUTPUTS_PATH="$PROJECT_ROOT/outputs"
else
  echo "Using DEFAULT container paths"
  PROJECT_ROOT="/workspace"
  CONFIG_PATH="/workspace/configs/default.yaml"
  CONFIG_2D_PATH="/workspace/configs/deconver_2d.yaml"
  PRETRAIN_CONFIG_PATH="/workspace/configs/pretrain_default.yaml"
  OUTPUTS_PATH="/outputs"
fi

case "$MODE" in
  train)
    echo "Starting training..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python -m src.train --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
    ;;
  train-2d)
    echo "Starting 2D Deconver training..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python -m src.train_deconver_2d --config "$CONFIG_2D_PATH" "${EXTRA_ARGS[@]}"
    ;;
  pretrain)
    echo "Starting SSL encoder pretraining..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python -m src.pretrain --config "$PRETRAIN_CONFIG_PATH" "${EXTRA_ARGS[@]}"
    ;;
  tensorboard)
    echo "Starting TensorBoard on 0.0.0.0:6006..."
    tensorboard --logdir "$OUTPUTS_PATH/runs" --host 0.0.0.0 --port 6006
    ;;
  smoke-test)
    echo "Running smoke test..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/smoke_test.py"
    ;;
  evaluate)
    echo "Running checkpoint evaluation..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/evaluate_checkpoint.py" "${EXTRA_ARGS[@]}"
    ;;
  evaluate-2d)
    echo "Running 2D checkpoint evaluation..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/evaluate_checkpoint_2d.py" "${EXTRA_ARGS[@]}"
    ;;
  visualize-3d)
    echo "Running 3D visualizer..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/visualize_3d.py" "${EXTRA_ARGS[@]}"
    ;;
  visualize-3d-app)
    echo "Starting 3D visualizer app on 0.0.0.0:8501..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" streamlit run "$PROJECT_ROOT/scripts/visualize_3d_app.py" --server.address 0.0.0.0 --server.port 8501 --server.headless true "${EXTRA_ARGS[@]}"
    ;;
  report-runs)
    echo "Generating run report..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/report_runs.py" "${EXTRA_ARGS[@]}"
    ;;
  learnability)
    echo "Running learnability test..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python -m src.train --config "$CONFIG_PATH" --learnability "${EXTRA_ARGS[@]}"
    ;;
  download)
    echo "Downloading dataset..."
    cd "$PROJECT_ROOT"
    bash "$PROJECT_ROOT/scripts/download_dataset.sh" "${EXTRA_ARGS[@]}"
    ;;
  shell)
    echo "Opening shell..."
    cd "$PROJECT_ROOT"
    exec bash
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Available modes: train | train-2d | pretrain | tensorboard | smoke-test | evaluate | evaluate-2d | visualize-3d | visualize-3d-app | report-runs | learnability | download | shell"
    exit 1
    ;;
esac
