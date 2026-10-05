#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKGS_ROOT="${CONDA_PKGS_ROOT:-$HOME/anaconda3/pkgs}"
OUT="${CUDA_TOOLKIT_DIR:-$ROOT/.cuda-toolkit}"

pick_dir() {
  local pattern="$1"
  local match
  match="$(find "$PKGS_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$pattern" | sort -V | tail -n 1)"
  if [[ -z "$match" ]]; then
    echo "missing CUDA package: $pattern" >&2
    exit 2
  fi
  printf '%s' "$match"
}

NVCC_TOOLS="$(pick_dir 'cuda-nvcc-tools-12.8.*')"
NVCC_DEV="$(pick_dir 'cuda-nvcc-dev_linux-64-12.8.*')"
NVVM_TOOLS="$(pick_dir 'cuda-nvvm-tools-12.8.*')"
CUDART_DEV="$(pick_dir 'cuda-cudart-dev_linux-64-12.8.*')"
CRT_DEV="$(pick_dir 'cuda-crt-dev_linux-64-12.8.*')"
CRT_TOOLS="$(pick_dir 'cuda-crt-tools-12.8.*')"
CCCL_DEV="$(pick_dir 'cuda-cccl_linux-64-12.8.*')"

rm -rf "$OUT"
mkdir -p "$OUT/bin/crt" \
  "$OUT/targets/x86_64-linux/include" \
  "$OUT/targets/x86_64-linux/lib" \
  "$OUT/targets/x86_64-linux/nvvm/bin" \
  "$OUT/targets/x86_64-linux/nvvm/libdevice"

for tool in nvcc ptxas fatbinary nvlink; do
  ln -s "$NVCC_TOOLS/bin/$tool" "$OUT/bin/$tool"
done
ln -s "$NVCC_TOOLS/bin/nvcc.profile" "$OUT/bin/nvcc.profile"
ln -s "$CRT_TOOLS/bin/crt/link.stub" "$OUT/bin/crt/link.stub"
ln -s "$NVVM_TOOLS/nvvm/bin/cicc" "$OUT/targets/x86_64-linux/nvvm/bin/cicc"
ln -s "$NVVM_TOOLS/nvvm/libdevice/libdevice.10.bc" \
  "$OUT/targets/x86_64-linux/nvvm/libdevice/libdevice.10.bc"

cp -as "$NVCC_DEV/targets/x86_64-linux/include/." \
  "$OUT/targets/x86_64-linux/include/"
cp -as "$CUDART_DEV/targets/x86_64-linux/include/." \
  "$OUT/targets/x86_64-linux/include/"
cp -as "$CRT_DEV/targets/x86_64-linux/include/." \
  "$OUT/targets/x86_64-linux/include/"
cp -as "$CUDART_DEV/targets/x86_64-linux/lib/." \
  "$OUT/targets/x86_64-linux/lib/"

echo "$OUT"
"$OUT/bin/nvcc" --version | tail -n 4