#!/usr/bin/env bash
set -e

BRANCH="${1:-${BRANCH:-main}}"

cd /workspace

if [ ! -d .git ]; then
  echo "No git repository found in /workspace"
  exit 1
fi

echo "Fetching latest changes from origin..."
git fetch origin

echo "Checking out branch: $BRANCH"
git checkout "$BRANCH"

echo "Pulling latest commit..."
git pull origin "$BRANCH"

echo "Current commit hash:"
git rev-parse HEAD
