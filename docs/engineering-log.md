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

## Reduction V0 baseline

### Toolchain integration

The cached CUDA packages are split across multiple Conda packages rather than installed as one monolithic toolkit.

Resolution:

- added `scripts/bootstrap_cuda_toolkit.sh`;
- assemble a local `.cuda-toolkit/`;
- combine nvcc tools, NVVM/libdevice, CUDA CRT, CUDA runtime headers, and cudart libraries;
- keep the assembled toolkit untracked;
- confirm `compute_89` support with a real RTX 4090 smoke kernel.

### V0 validation

- clean CUDA 12.8 / `sm_89` build: passed;
- reduction correctness suite: 11 passed;
- largest V0 case: about 608 ms.

## Reduction V1 parallel atomic

### Design

V1 adds 256-thread blocks and grid-stride loops. Each thread accumulates a private partial sum and performs one global `atomicAdd`.

The launch is capped at 1,024 blocks, so at most 262,144 threads issue atomics.

### Hardware-validation issues

The first V1 compile exposed two toolchain integration details:

1. the local assembled CUDA 12.8 toolkit needed `cuda-nvcc-dev_linux-64-12.8.*` headers to avoid falling back to system CUDA headers;
2. an unnecessary C++ `<algorithm>` include triggered an nvcc/GCC 11 host-pass failure in `std_function.h`, so the dependency was removed.

### V1 validation

- full test suite: 26 passed;
- at N = 16,777,216, V1 measured about 393–450 us depending on run conditions;
- the large-input latency plateau identified same-address global atomic contention as the next bottleneck.

## Reduction V2 shared-memory block reduction

### Design

V2 keeps the per-thread grid-stride local sum but writes one value per thread into a 256-float shared-memory array. A power-of-two tree reduction collapses the block to one sum, so only thread 0 performs the global atomic.

Maximum global atomic count falls from 262,144 to 1,024.

### Validation

- clean CUDA 12.8 / `sm_89` build: passed;
- complete test suite: **45 passed**;
- signed random data: passed;
- pre-filled output reset: passed;
- non-default CUDA stream: passed.

### ptxas

V2 uses:

- 12 registers per thread;
- 1,024 bytes shared memory per block;
- zero spills;
- zero stack frame.

### Benchmark methodology correction

The first V2 hot-cache result reported logical input throughput above the GPU's physical DRAM peak. That is not a kernel bug: the RTX 4090 Laptop reports a **64 MiB L2 cache**, and the largest benchmark input is also 64 MiB. Repeated runs can therefore be substantially served by L2.

To avoid mislabeling cache throughput as DRAM throughput:

- the CSV metric was renamed to `logical_input_gbps`;
- the benchmark now records `cache_mode`, `l2_bytes` and `flush_bytes`;
- a `cold` mode touches a 128 MiB buffer before every timed launch, evicting a working set twice the L2 capacity;
- hot and L2-evicted results are reported separately.

The cache-preparation kernel also sustains GPU clocks, so small-input hot/cold timings are not mixed into one claim.

### V2 measured result

Under L2-evicted conditions at N = 16,777,216:

- V1: 448.144 us;
- V2: 177.152 us;
- PyTorch: 172.032 us;
- V1 → V2: **2.53x**;
- V2 / PyTorch: **1.03x**;
- logical input throughput: 378.821 GB/s.

At N = 262,144, removing per-thread global atomics produces a **46.44x** V1 → V2 speedup.

The V1 atomic-contention plateau disappears, validating the shared-memory reduction hypothesis.

### Next bottleneck

V2 still synchronizes at every shared-memory tree level. V3 will isolate warp-level optimization by using warp shuffle for the final reduction stage before any vectorized-load work is introduced.
