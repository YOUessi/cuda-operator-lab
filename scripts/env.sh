#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/home/you/anaconda3/pkgs/cuda-nvcc-tools-12.8.93-hbdd6827_3}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "CUDA_HOME=${CUDA_HOME}"
"${CUDACXX}" --version | tail -n 4
