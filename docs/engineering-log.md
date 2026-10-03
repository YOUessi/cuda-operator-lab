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
- Tang is used only for real CUDA execution: nvcc/CMake builds, GPU tests, benchmarks, Compute Sanitizer, Nsight/profiling and environment diagnosis.
- Measured results and hardware-discovered fixes are committed back to GitHub before merge.

## Reduction V0

V0 uses one CUDA thread for the entire sum.

Validation:

- CUDA 12.8 / `sm_89` build: passed;
- correctness suite: 11 passed;
- largest recorded case: roughly 608 ms.

## Reduction V1

V1 adds 256-thread blocks and grid-stride loops, but each participating thread performs a same-address global atomic.

Hardware-validation findings:

- the assembled CUDA 12.8 toolkit needed nvcc development headers to avoid system-header fallback;
- an unnecessary `<algorithm>` dependency triggered a host-pass nvcc/GCC 11 issue and was removed.

Validation:

- full test suite: 26 passed;
- V1 large-shape latency plateaus near 400–450 us;
- atomic contention identified as the next bottleneck.

## Reduction V2

V2 reduces per-thread values inside a 256-float shared-memory tree and performs one global atomic per block.

Validation:

- full test suite: 45 passed;
- V2 ptxas: 12 registers/thread, 1,024 B shared/block, 0 spills;
- L2-evicted 16M case: ~177 us vs ~172 us for PyTorch.

### Benchmark methodology correction

The RTX 4090 Laptop reports 64 MiB L2, equal to the largest 64 MiB benchmark input. Repeated hot runs can therefore report logical throughput above physical DRAM bandwidth.

The benchmark now:

- separates hot and L2-evicted regimes;
- records cache mode, L2 size and flush size;
- uses a 128 MiB cache-preparation buffer in cold mode;
- labels the derived metric `logical_input_gbps`, not DRAM bandwidth.

## Reduction V3

### Design

V3 replaces the full shared-memory tree with two levels of warp shuffle:

1. each of the eight warps reduces 32 thread-local sums in registers with `__shfl_down_sync`;
2. lane 0 of each warp stores one value, producing only eight shared-memory values;
3. after one block synchronization, the first warp reduces those eight values with a second shuffle tree;
4. lane 0 performs one global atomic.

The launch geometry is unchanged from V2, so the experiment isolates the reduction primitive.

### Validation

- clean CUDA 12.8 / `sm_89` build: passed;
- complete pytest suite: **67 passed**;
- V3 signed input, output reset, and non-default stream checks: passed;
- CUDA 12.8 Compute Sanitizer memcheck: **0 errors**;
- CUDA 12.8 Compute Sanitizer synccheck: **0 errors**.

The system-wide Compute Sanitizer is an older 2021.3.1 install and failed to locate its injection library. Hardware validation therefore uses the CUDA 12.8-matched sanitizer from the existing `cuda-sanitizer-api-12.8.93` package.

### ptxas

V3:

- 13 registers/thread;
- 32 B shared memory/block;
- one barrier resource;
- 0 spills;
- 0-byte stack frame.

V2 used 1,024 B shared memory/block, so V3 reduces shared-memory footprint by 32x at the cost of one extra register.

### Performance finding

The focused L2-evicted run uses 20 warmups + 100 repeats:

- 262K: 8.192 us → 7.168 us, about **1.14x** faster;
- 4M: 47.840 us → 47.104 us, about **1.02x** faster;
- 16M: 177.152 us → 177.200 us, effectively unchanged.

This is an important non-monotonic optimization result. Warp shuffle helps when block-reduction coordination is still visible in total latency, but it does not improve the largest memory-dominated case.

The next isolated bottleneck is therefore the input/load path rather than the final reduction tree.


## Reduction V4 vectorized input

### Design

V4 keeps V3's warp-shuffle reduction and one-atomic-per-block structure, but changes the aligned input path to `float4` loads. The scalar tail handles `N % 4`.

Because a contiguous PyTorch tensor can have a non-zero storage offset, V4 checks 16-byte pointer alignment at launch time. Misaligned contiguous inputs fall back to V3.

### Validation

- clean CUDA 12.8 / `sm_89` build: passed;
- full pytest suite: **89 passed**;
- intentionally unaligned contiguous input (`base[1:]`, pointer mod 16 = 4): passed through fallback;
- Compute Sanitizer memcheck fast path: 0 errors;
- Compute Sanitizer racecheck fast path: 0 hazards / 0 errors;
- Compute Sanitizer synccheck fast path: 0 errors;
- Compute Sanitizer memcheck unaligned fallback: 0 errors.

### Generated-code evidence

SM 8.9 SASS contains `LDG.E.128` in the V4 fast path, confirming a 128-bit global load. The scalar tail remains `LDG.E`.

ptxas reports:

- V3: 13 registers/thread, 32 B shared memory/block, 0 spills;
- V4: 16 registers/thread, 32 B shared memory/block, 0 spills.

### Performance finding

Stable L2-evicted comparison, 20 warmups + 100 repeats:

- 262K: 7.168 us → 8.192 us (**regression**);
- 4M: 46.896 us → 46.080 us (~1.02x);
- 16M: 177.152 us → 168.960 us (~1.05x).

V4 therefore validates two things at once:

1. wider loads can help a genuinely memory-dominated large reduction;
2. vectorization without shape-aware launch geometry is not universally beneficial.

At 262K, the unchanged scalar-based block count launches many threads that have no float4 work. The next isolated experiment should correct vector-path launch geometry before changing any other kernel mechanism.
