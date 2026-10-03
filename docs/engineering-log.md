# Engineering Log

## Project bootstrap

- Target machine: RTX 4090 Laptop GPU, compute capability 8.9.
- Driver: 580.178.04.
- System PyTorch: 2.10.0+cu128.
- System `/usr/bin/nvcc`: CUDA 11.5, too old to be the primary compiler for Ada SM 8.9.
- Selected local CUDA compiler: CUDA 12.8 nvcc from the existing Anaconda package cache.

### Development workflow

The source of truth is GitHub.

- Source code, tests, build files, reports and documentation are authored on GitHub feature branches.
- Tang is used only when real CUDA hardware is required: nvcc/CMake builds, GPU correctness tests, benchmarks, Compute Sanitizer, Nsight, and environment diagnosis.
- Measured results and any fixes discovered by hardware validation are committed back to GitHub before merge.

This keeps ordinary code editing independent from remote-desktop access while preserving real RTX 4090 validation.

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

### V0 validation

- CUDA 12.8 smoke kernel on RTX 4090 Laptop: passed.
- CMake Release build for `sm_89`: passed.
- Reduction correctness suite: 11 passed.
- Baseline benchmark: completed over N = 1K through 16M.
- Largest V0 case measured about 608 ms versus about 30.7 us for `torch.sum`, which establishes the parallelism gap V1 must close.

## Reduction V1 parallel atomic

### Design

V1 adds 256-thread blocks and grid-stride loops. Each thread first creates a register-local partial sum and then performs one global `atomicAdd` into the output scalar. The launch is capped at 1,024 blocks, so at most 262,144 threads issue atomics.

The output is reset with `cudaMemsetAsync` on the active PyTorch CUDA stream.

### Hardware-validation issues

The first V1 compile exposed two toolchain integration details:

1. the local assembled CUDA 12.8 toolkit did not include the CUDA nvcc development headers, so generated host-stub compilation could fall back to system CUDA headers; `cuda-nvcc-dev_linux-64-12.8.*` is now included in the assembled toolkit;
2. an unnecessary C++ `<algorithm>` include caused the nvcc/GCC 11 host pass to enter `<functional>` and fail in `std_function.h`; V1 no longer depends on that header and uses a simple capped-block expression instead.

Both fixes were made on GitHub and then revalidated on Tang.

### V1 validation

- Clean CUDA 12.8 / `sm_89` build: passed.
- Full test suite: **26 passed**.
- Non-default CUDA stream test: passed.
- Reused/pre-filled output reset test: passed.
- V0/V1/PyTorch benchmark: completed over N = 1K through 16M.
- At N = 16,777,216, V1 measured 393.216 us vs V0 608,578.979 us: about **1,547.70x faster than V0**.
- V1 remains 13.24x slower than `torch.sum`, and its 384–393 us latency plateau points to global atomic contention as the next bottleneck.

Canonical results are tracked at `reports/data/reduction_v0_v1_rtx4090_laptop.csv`.
