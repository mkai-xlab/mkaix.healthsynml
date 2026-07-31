#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="knee-roi-augmentation-preview:latest"
CONTAINER_NAME="knee-roi-augmentation-preview"
PORT="${PORT:-8090}"

docker build \
  --tag "${IMAGE_NAME}" \
  "${REPOSITORY_ROOT}/scripts/roi_augmentation_preview"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm --force "${CONTAINER_NAME}" >/dev/null
fi

docker run \
  --rm \
  --detach \
  --name "${CONTAINER_NAME}" \
  --publish "${PORT}:8090" \
  "${IMAGE_NAME}"

echo "Knee ROI augmentation preview: http://127.0.0.1:${PORT}"
