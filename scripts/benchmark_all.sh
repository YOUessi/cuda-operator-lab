#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f build/lib/libcuda_operator_lab.so ]]; then
  echo "CUDA operator library is missing; building first..."
  ./scripts/build.sh
fi

export PYTHONPATH="$ROOT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
python3 benchmarks/run_all.py "$@"
