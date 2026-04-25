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
    train|tensorboard|smoke-test|shell|evaluate|learnability|download|visualize-3d|visualize-3d-app)
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
  OUTPUTS_PATH="$PROJECT_ROOT/outputs"
else
  echo "Using DEFAULT container paths"
  PROJECT_ROOT="/workspace"
  CONFIG_PATH="/workspace/configs/default.yaml"
  OUTPUTS_PATH="/outputs"
fi

case "$MODE" in
  train)
    echo "Starting training..."
    cd "$PROJECT_ROOT"
    PYTHONPATH="$PROJECT_ROOT" python -m src.train --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}"
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
    echo "Available modes: train | tensorboard | smoke-test | evaluate | visualize-3d | visualize-3d-app | learnability | download | shell"
    exit 1
    ;;
esac
