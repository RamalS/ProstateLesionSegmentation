#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/spajic/Projects/ProstteLesionSegmentation"
BRANCH="train"

echo "Starting sync on $(hostname)"
echo "Project directory: ${PROJECT_DIR}"
echo "Branch: ${BRANCH}"

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
# Hash the two files that affect the built image (Dockerfile and requirements.txt).
# src/, scripts/, configs/ are volume-mounted at runtime so they never need a rebuild.
HASH=$(sha256sum Dockerfile requirements.txt | sha256sum | cut -d' ' -f1)
HASH_FILE="${PROJECT_DIR}/.docker_build_hash"

if [ ! -f "${HASH_FILE}" ] || [ "$(cat "${HASH_FILE}")" != "${HASH}" ]; then
    echo "Dockerfile or requirements.txt changed — rebuilding Docker image..."
    docker compose build && echo "${HASH}" > "${HASH_FILE}"
    echo "Docker image rebuilt successfully."
else
    echo "Docker image is up to date — skipping rebuild."
fi

echo "Finished successfully."
