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


## Reduction V5 shape-aware dispatch

### Design

V5 keeps the V4 float4 kernel body and V3 fallback unchanged. It corrects the launch policy:

- vector grid size is computed from `N/4` work items rather than scalar `N`;
- the final measured vector crossover is `N >= 524,288`;
- smaller inputs use V3;
- unaligned contiguous inputs use V3.

The crossover was not guessed. An initial vector-work-only candidate was benchmarked, then hot/cold sweeps were used to choose a conservative threshold that avoids relying on a cache-state-sensitive small-shape win.

### Validation

- clean build: PASS;
- full test suite: **116 passed**;
- aligned vector path sanitizer: memcheck 0, racecheck 0, synccheck 0;
- below-crossover scalar path memcheck: 0;
- unaligned fallback memcheck: 0.

### Final performance

L2-evicted, 20 warmups + 100 repeats:

- 512K: 10.256 us -> 10.240 us;
- 1M: 17.008 us -> 15.376 us;
- 4M: 47.104 us -> 45.152 us;
- 16M: 176.128 us -> 167.936 us.

The full operation-by-operation record is maintained in `docs/experiment-log.md`, including the preliminary 32K threshold, crossover sweeps, raw CSVs, final threshold revision, and sanitizer evidence.

### Decision

Stop deepening Reduction after V5. The next implementation target is row-wise Softmax so the project demonstrates the same profile-guided workflow on a multi-stage normalization operator.


## Softmax V0 serial row baseline

### Design

Softmax V0 introduces the project's first 2-D normalization operator. Each CUDA thread owns one row and performs three serial passes:

1. row maximum;
2. exponentiation plus denominator accumulation;
3. normalization.

The implementation subtracts the row maximum before `expf` for numerical stability and uses the caller's active PyTorch CUDA stream.

### Hardware-validation issue

The first clean build failed because `CUDART_INF_F` was not defined by the assembled CUDA header view. The fix was made on GitHub by replacing it with the standard `-FLT_MAX` initialization from `<cfloat>`.

After the fix:

- clean build: PASS;
- complete repository suite: **131 passed**;
- Compute Sanitizer memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- ptxas: 24 registers/thread, 0 spills, no barriers.

### Performance finding

V0 scales primarily with row width because intra-row work is serial.

Representative results:

- 128 × 128: 60.416 us vs PyTorch 9.216 us;
- 128 × 1024: 416.768 us vs PyTorch 8.192 us;
- 128 × 4096: 1,595.392 us vs PyTorch 9.328 us.

The next isolated change is one-block-per-row intra-row parallelism with shared-memory max and sum reductions.


## Softmax V1 block-parallel shared-memory reduction

### Design

V1 moves from one thread per row to one 256-thread block per row. Threads use strided column access and shared-memory tree reductions for both row max and denominator sum.

### Concurrency bug caught only by Racecheck

The first clean build and ordinary pytest suite passed (153 tests), but Compute Sanitizer Racecheck reported a shared-memory race: `shared[0]` was reused for denominator partial sums before all warps were guaranteed to have read the row maximum.

Fix: add a block barrier immediately after loading `row_max = shared[0]`.

After the GitHub-side fix and a clean Tang rebuild:

- full suite: **153 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

This is a concrete example where deterministic unit tests were insufficient for GPU synchronization correctness.

### Resources and performance

ptxas:

- 21 registers/thread;
- 1,024 B shared memory/block;
- 0 spills.

Representative speedups:

- 128 x 512: 224.256 us -> 11.264 us;
- 128 x 1024: 405.632 us -> 11.264 us;
- 128 x 4096: 1,614.752 us -> 14.336 us;
- 1024 x 4096: 1,353.728 us -> 37.888 us, versus PyTorch 36.960 us.

Next: replace the two full shared-memory trees with warp-shuffle reductions while holding the rest of Softmax constant.
