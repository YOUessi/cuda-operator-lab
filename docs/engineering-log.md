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


## Softmax V2 warp-shuffle reductions

### Design

V2 keeps one 256-thread block per row and replaces V1's two 256-entry shared-memory trees with two-level warp-shuffle reductions. Eight warp partials are stored in shared memory for both max and denominator sum reductions.

The shared partial buffer is not reused until all threads have consumed the final row maximum, preserving the synchronization fix learned from V1 Racecheck.

### Validation

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full repository suite: **180 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- ptxas: 23 registers/thread, 32 B shared/block, 0 spills.

### Performance finding

Stable 100-repeat benchmark:

- 1024 x 128: 14.336 us -> 10.240 us (~1.40x);
- 1024 x 512: 16.192 us -> 12.256 us (~1.32x);
- 1024 x 4096: 41.792 us -> 39.600 us (~1.06x);
- 128 x 512: 10.240 us -> 10.592 us (small regression).

Conclusion: warp-level reduction meaningfully reduces coordination cost for high-row-count workloads, but fixed 256-thread blocks are not efficient for all widths. Next isolate width-aware block sizing.


## Softmax V3 width-aware block sizing

V3 tested whether reducing threads/block for narrow rows improves Softmax while holding the V2 warp-shuffle math constant.

Validation:

- clean build: PASS;
- full suite: **209 passed**;
- memcheck/racecheck/synccheck clean for 128-thread and 256-thread representative paths;
- ptxas unchanged from V2: 23 registers/thread, 32 B shared/block, 0 spills.

Performance was not robustly positive. Stable runs showed neutral behavior at several widths, ~2% improvement at 1024 x 128, but regressions at 128 x 32 and 1024 x 64.

Decision: record V3 as a valid negative experiment. Small-row optimization should preserve a full block and pack multiple rows across warps instead of shrinking each row's block.


## Softmax V4 warp-per-row packing

V4 changes the narrow-row execution layout:

- 256 threads/block;
- 8 warps/block;
- one warp owns one row;
- up to 8 rows/block;
- no shared memory and no block-wide barriers on the packed path;
- wider rows fall back to V3.

Validation:

- full suite: 229 passed;
- memcheck/racecheck/synccheck clean;
- ptxas: 30 registers/thread, 0 B shared memory, 0 spills.

At high row counts V4 is materially better than V3, e.g. 16384 x 128 improves from about 32.75 us to 19.14 us, but 128 x 128 regresses. This motivates a measured dispatcher.

## Softmax V5 empirical shape dispatch

V5 is dispatch-only: no new device kernel.

Crossover sweep:

- widths 32 / 64 / 128;
- rows 128 through 16384;
- 20 warmups + 100 repeats.

Final policy:

- cols <= 64 and rows >= 4096 -> V4 packed;
- 65..128 cols and rows >= 2048 -> V4 packed;
- otherwise -> V3 width-aware.

Validation:

- clean build: PASS;
- full suite: **250 passed**;
- Python compileall: PASS;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

Final representative timings:

- 2048 x 128: V3 11.264 us -> V5 10.592 us;
- 4096 x 64: V3 11.264 us -> V5 10.496 us;
- 4096 x 128: V3 14.336 us -> V5 10.912 us;
- 16384 x 128: V3 32.704 us -> V5 18.528 us.

A benchmark-reporting bug was caught during validation: V5 fallback calls were executing the correct V3 kernels but reporting 256 threads/block in CSV metadata. The reporting logic was fixed and the final benchmark was regenerated.

Decision: stop deepening Softmax after V5 and move to RMSNorm.


## RMSNorm V0 serial-row baseline

RMSNorm V0 starts the third operator case study after Reduction and Softmax.

Design:

- one CUDA thread owns one row;
- serial sum of squares;
- `rsqrt(mean_square + eps)`;
- serial normalize and multiply by the learned weight vector.

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full repository suite: **268 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- ptxas: 20 registers/thread, 0 B shared memory, 0 spills.

Performance exposes a serial hidden-width bottleneck:

- 128 x 128: 40.960 us;
- 128 x 1024: 270.336 us;
- 128 x 4096: 1,035.216 us;
- 128 x 8192: 1,691.936 us.

Next: one-block-per-row cooperative sum-of-squares reduction.


## RMSNorm V1 block-parallel reduction

V1 moves from one serial thread per row to one 256-thread block per row.

Each thread accumulates a strided local sum of squares, the block combines those partials in a 256-float shared-memory tree, thread 0 computes inverse RMS, and all threads write normalized weighted output in parallel.

Validation:

- clean build: PASS;
- full suite: **290 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- ptxas: 18 registers/thread, 1,024 B shared memory/block, 0 spills.

Representative V0 -> V1 speedups:

- 128 x 512: 144.560 us -> 10.272 us (~14.1x);
- 128 x 1024: 279.552 us -> 10.464 us (~26.7x);
- 128 x 4096: 1,037.120 us -> 14.080 us (~73.7x);
- 128 x 8192: 1,692.320 us -> 17.408 us (~97.2x).

The PyTorch comparison in this phase is a multi-op correctness expression, not an optimized fused RMSNorm kernel, so its latency is retained for context only.

Next: replace the shared-memory tree with warp shuffle while holding the rest of the kernel constant.


## RMSNorm V2 warp-shuffle reduction

V2 keeps one 256-thread block per row and replaces V1's 256-float shared-memory tree with two-level warp shuffle.

Validation:

- full suite: **313 passed**;
- memcheck/racecheck/synccheck clean;
- ptxas: 17 registers/thread, 32 B shared memory/block, 0 spills.

Selected V1 -> V2:

- 128 x 4096: 14.160 us -> 13.312 us (~1.06x);
- 1024 x 512: 11.904 us -> 11.200 us (~1.06x);
- 2048 x 4096: 58.368 us -> 52.688 us (~1.11x);
- 128 x 512: 10.560 us -> 11.056 us (small regression).

Conclusion: reduction coordination is reduced, but the result is already moving toward input/output-path limits. Next target is vectorized IO rather than deeper reduction-tree tuning.
