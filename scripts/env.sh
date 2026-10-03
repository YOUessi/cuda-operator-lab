#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="${CUDA_TOOLKIT_DIR:-$ROOT/.cuda-toolkit}"

"$ROOT/scripts/bootstrap_cuda_toolkit.sh" >/dev/null

export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"

echo "CUDA_HOME=$CUDA_HOME"
"$CUDACXX" --version | tail -n 4