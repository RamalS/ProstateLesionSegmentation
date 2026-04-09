#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="$PROJECT_ROOT" python "$PROJECT_ROOT/scripts/select_checkpoint.py" "$@"
