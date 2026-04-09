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
    train|tensorboard|smoke-test|shell)
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
  shell)
    echo "Opening shell..."
    cd "$PROJECT_ROOT"
    exec bash
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Available modes: train | tensorboard | smoke-test | shell"
    exit 1
    ;;
esac
