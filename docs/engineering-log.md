# Engineering Log

## Project bootstrap

- Target machine: RTX 4090 Laptop GPU, compute capability 8.9.
- Driver: 580.178.04.
- System PyTorch: 2.10.0+cu128.
- System `/usr/bin/nvcc`: CUDA 11.5, too old to be the primary compiler for Ada SM 8.9.
- Selected local CUDA compiler: CUDA 12.8 nvcc from the existing Anaconda package cache.
- Development strategy: local-first execution on Tang, Git-backed from the first commit, and push small verified milestones to GitHub rather than waiting until the project is finished.


## Reduction V0 baseline

### Toolchain integration

The first CUDA 12.8 smoke build exposed that the cached CUDA packages are split across multiple Conda packages rather than installed as one monolithic toolkit. The compiler initially failed because its relative NVVM and CUDA CRT helper paths did not exist under the compiler package root.

Resolution:

- added `scripts/bootstrap_cuda_toolkit.sh`;
- assemble a local `.cuda-toolkit/` using symlinks into the existing CUDA 12.8 package cache;
- combine nvcc tools, NVVM/libdevice, CUDA CRT, CUDA runtime headers, and cudart libraries;
- keep the assembled toolkit untracked;
- confirmed the resulting compiler supports `compute_89` and successfully launched an SM 8.9 smoke kernel.

This is environment glue only; the operator source does not depend on Tang-specific Python ABI or PyTorch C++ symbols.

### Reduction V0 architecture

The initial reduction path deliberately separates four layers:

1. CUDA kernel: a one-thread serial sum, used only as a lower-bound baseline.
2. C ABI: `cuda_operator_reduction_v0(...)`, taking raw device pointers and a CUDA stream.
3. Python binding: `ctypes` passes PyTorch tensor pointers without a PyTorch C++ extension dependency.
4. Benchmark/correctness harness: PyTorch supplies the trusted reference and CUDA-event timing.

This keeps kernel code independently compilable while still allowing exact same-input comparisons from Python.

### Validation

- CUDA 12.8 smoke kernel on RTX 4090 Laptop: passed.
- CMake Release build for `sm_89`: passed.
- Reduction correctness suite: 11 passed.
- Baseline benchmark: completed over N = 1K through 16M.
- Largest V0 case measured about 608 ms versus about 30.7 us for `torch.sum`, which is intentionally catastrophic and establishes the parallelism gap V1 must close.
- Canonical baseline data is tracked at `reports/data/reduction_v0_rtx4090_laptop.csv`.