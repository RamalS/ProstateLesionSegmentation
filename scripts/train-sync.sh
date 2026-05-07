#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/spajic/Projects/ProstteLesionSegmentation"
BRANCH="train"
TORCH_STACK_OVERRIDE="${TORCH_STACK:-}"

detect_torch_stack() {
  local caps raw_cap
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "modern|nvidia-smi not found; defaulting to modern stack"
    return 0
  fi

  if ! caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null); then
    echo "modern|failed to query GPU compute capability; defaulting to modern stack"
    return 0
  fi

  if [ -z "${caps//[[:space:]]/}" ]; then
    echo "modern|no GPU compute capability reported; defaulting to modern stack"
    return 0
  fi

  while IFS= read -r raw_cap; do
    raw_cap="${raw_cap%%,*}"
    raw_cap="${raw_cap//[[:space:]]/}"
    case "${raw_cap}" in
      7.0|7.2)
        echo "volta|detected Volta-class GPU compute capability (${raw_cap})"
        return 0
        ;;
    esac
  done <<< "${caps}"

  echo "modern|detected non-Volta GPU compute capability (${caps//$'\n'/, })"
}

resolve_torch_stack() {
  if [ -n "${TORCH_STACK_OVERRIDE}" ]; then
    case "${TORCH_STACK_OVERRIDE}" in
      modern|volta)
        echo "${TORCH_STACK_OVERRIDE}|TORCH_STACK override provided"
        return 0
        ;;
      *)
        echo "invalid|unsupported TORCH_STACK='${TORCH_STACK_OVERRIDE}' (use 'modern' or 'volta')"
        return 0
        ;;
    esac
  fi

  detect_torch_stack
}

STACK_RESOLUTION="$(resolve_torch_stack)"
TORCH_STACK="${STACK_RESOLUTION%%|*}"
STACK_REASON="${STACK_RESOLUTION#*|}"

case "${TORCH_STACK}" in
  modern)
    COMPOSE_FILES=(-f compose.yml)
    STACK_LABEL="modern-cu128"
    ;;
  volta)
    COMPOSE_FILES=(-f compose.yml -f compose.volta.yml)
    STACK_LABEL="volta-cu126"
    ;;
  invalid)
    echo "Error: ${STACK_REASON}"
    exit 1
    ;;
  *)
    echo "Error: failed to resolve TORCH_STACK (resolved='${TORCH_STACK}')"
    exit 1
    ;;
esac

echo "Starting sync on $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Branch: ${BRANCH}"
echo "Torch stack: ${STACK_LABEL}"
echo "Stack selection: ${STACK_REASON}"

if [ ! -d "${PROJECT_DIR}/.git" ]; then
  echo "Error: ${PROJECT_DIR} is not a git repository"
  exit 1
fi

cd "${PROJECT_DIR}"

echo "Discarding local repository changes..."

# Abort in-progress git operations so reset/checkout can proceed cleanly.
if [ -f .git/MERGE_HEAD ]; then
  git merge --abort || true
fi

if [ -d .git/rebase-apply ] || [ -d .git/rebase-merge ]; then
  git rebase --abort || true
fi

if [ -f .git/CHERRY_PICK_HEAD ]; then
  git cherry-pick --abort || true
fi

if [ -f .git/REVERT_HEAD ]; then
  git revert --abort || true
fi

if [ -f .git/BISECT_LOG ]; then
  git bisect reset || true
fi

git reset --hard
git clean -fd

echo "Fetching latest code..."
git fetch --prune origin
git checkout -B "${BRANCH}" "origin/${BRANCH}"
git reset --hard "origin/${BRANCH}"
git clean -fd

echo "Checking if Docker image needs rebuilding..."
# Hash the files that affect the built image.
# src/, scripts/, configs/ are volume-mounted at runtime so they never need a rebuild.
HASH_INPUTS=(Dockerfile requirements.txt compose.yml)
if [ "${TORCH_STACK}" = "volta" ]; then
  HASH_INPUTS+=(compose.volta.yml)
fi
HASH=$(sha256sum "${HASH_INPUTS[@]}" | sha256sum | cut -d' ' -f1)
HASH_FILE="${PROJECT_DIR}/.docker_build_hash_${TORCH_STACK}"

if [ ! -f "${HASH_FILE}" ] || [ "$(cat "${HASH_FILE}")" != "${HASH}" ]; then
    echo "Build inputs changed for ${STACK_LABEL} — rebuilding Docker image..."
    docker compose "${COMPOSE_FILES[@]}" build && echo "${HASH}" > "${HASH_FILE}"
    echo "Docker image rebuilt successfully."
else
    echo "Docker image is up to date for ${STACK_LABEL} — skipping rebuild."
fi

echo "Finished successfully."
