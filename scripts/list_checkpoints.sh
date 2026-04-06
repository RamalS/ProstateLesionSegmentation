#!/usr/bin/env bash
set -e

echo "Available checkpoints:"
ls -lah /outputs/checkpoints || echo "No checkpoints found."
