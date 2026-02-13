#!/usr/bin/env bash
set -euo pipefail

PY_VERSIONS=${PY_VERSIONS:-"3.10 3.11 3.13"}
IMAGE_TAG_PREFIX=${IMAGE_TAG_PREFIX:-ztb-ci}

for ver in $PY_VERSIONS; do
  tag="$IMAGE_TAG_PREFIX:$ver"
  echo "Building Docker image for Python $ver -> $tag"
  docker build --build-arg PYTHON_VERSION=$ver -t $tag -f docker/ci.Dockerfile .

  echo "Running validation in container ($tag)"
  docker run --rm -v "$(pwd)":/workspace -w /workspace $tag
done

echo "Local validation runs finished."
