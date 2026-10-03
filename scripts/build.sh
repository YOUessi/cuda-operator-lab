#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLKIT="${CUDA_TOOLKIT_DIR:-$ROOT/.cuda-toolkit}"

"$ROOT/scripts/bootstrap_cuda_toolkit.sh" >/dev/null

cmake -S "$ROOT" -B "$ROOT/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$TOOLKIT/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$ROOT/build" --parallel