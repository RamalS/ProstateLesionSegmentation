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

echo "Fetching latest code..."
git fetch origin
git checkout "${BRANCH}"
git pull origin "${BRANCH}"

echo "Running your custom steps..."
# Put your own commands here, for example:
# source venv/bin/activate
# pip install -r requirements.txt
# docker compose up -d --build
# systemctl restart my-service

echo "Finished successfully."
