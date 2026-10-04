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

## RMSNorm V3 float4 IO

V3 keeps the V2 warp-shuffle reduction but vectorizes aligned input, weight, and output traffic with `float4`. Non-multiple-of-four or unaligned inputs safely fall back to V2.

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full repository suite after V3: **337 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 22 registers/thread, 32 B shared memory/block, 0 spills.

Warm-cache benchmark showed large wins on several wide rows, e.g.:

- 128 x 4096: 13.312 us -> 10.240 us (~1.30x);
- 128 x 8192: 17.200 us -> 12.032 us (~1.43x);
- 1024 x 4096: 26.512 us -> 21.472 us (~1.23x).

However, repeated and L2-evicted measurements showed strong shape dependence and materially smaller gains. Some shapes regress. For example, under L2 eviction, 1024 x 512 measured about 13.35 us for V2 versus 16.38 us for V3, while 128 x 8192 still favored V3 strongly.

Conclusion: vectorized IO is a valid optimization path but not a universal default. Warm-cache benchmark results alone are insufficient to choose a dispatcher.

## RMSNorm V4 empirical dispatcher candidate

V4 adds no new device kernel. It dispatches between V2 and V3 from an empirically measured profile table.

The first warm-cache policy was rejected after L2-evicted measurements exposed unstable crossover points. A more conservative cold-cache profile was implemented and validated for correctness.

Validation:

- clean build: PASS;
- full suite: **355 passed**;
- memcheck/racecheck/synccheck: clean on dispatcher boundaries;
- non-profiled shapes fall back to V2.

Decision: **do not merge V4 yet**.

Reason: microsecond-level timing around several crossover shapes remains sensitive to cache state, execution order, and GPU operating state. In one final cold-cache run, identical underlying kernels reached noticeably different timings depending on whether they were invoked directly as V3 or indirectly through the V4 dispatch path, which is too large to attribute to host dispatch alone.

Next profiling step should control GPU clocks/power state where possible, use alternating randomized variant order, L2 eviction, repeated independent rounds, and report confidence intervals before freezing the final dispatcher.

## RMSNorm V4 stable profiling harness and conservative profile

The first dispatcher benchmark was rejected because variants were timed in long contiguous groups, which allowed GPU DVFS and thermal state to bias microsecond-scale comparisons.

The profiling harness was upgraded to:

- touch a 64 MiB CUDA buffer before every timed launch to perturb L2 state;
- interleave V2, V3, and V4 at the **individual sample** level;
- vary execution order across samples;
- run multiple independent rounds;
- repeat the entire profile with two independent seeds;
- report median, mean, standard deviation, P10/P90, and coefficient of variation.

Dispatcher acceptance uses **direct V2 vs direct V3 timings only**. V4 timings are not used to decide the policy because V4 can execute the exact same device kernel as V2 or V3, yet microsecond timing still reflects GPU operating-state noise.

Acceptance gate:

```text
V3 is allowed only if:
  speedup >= 1.05x
  in every independent profiling run
```

Two independent interleaved runs produced only four accepted measured shapes:

- 1024 x 512: V3 speedup 1.231x / 1.271x;
- 512 x 4096: 1.170x / 1.181x;
- 128 x 8192: 1.435x / 1.435x;
- 512 x 8192: 1.051x / 1.051x.

All other measured shapes fall back to V2.

The current V4 policy is therefore intentionally a **strict measured whitelist** for the RTX 4090 Laptop profile, not a claimed universal rule.

Validation after policy update:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **355 passed**;
- third independent whitelist-validation seed preserved the accepted/rejected direction;
- non-profiled shapes safely use V2.

Artifacts:

- `reports/data/rmsnorm_stable_profile_interleaved_rtx4090.csv`
- `reports/data/rmsnorm_stable_profile_interleaved_seed2_rtx4090.csv`
- `reports/data/rmsnorm_dispatch_profile_rtx4090.csv`
- `reports/data/rmsnorm_v4_whitelist_validation_rtx4090.csv`

Engineering conclusion: a profile-guided static dispatcher is defensible only when its policy is tied to measured hardware evidence. Unmeasured shapes should not inherit guessed thresholds.

## LayerNorm V0 serial-row baseline

LayerNorm starts the fourth operator case study after Reduction, Softmax, and RMSNorm.

V0 intentionally uses one CUDA thread per row:

- serial mean over hidden width;
- serial variance over hidden width;
- inverse standard deviation via `rsqrtf`;
- serial affine output `(x - mean) * inv_std * weight + bias`.

The purpose is to expose the cost of two row-wise reductions plus affine output before introducing cooperative block reduction or Welford.

### LayerNorm V0 validation

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **373 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 24 registers/thread, 0 B shared memory, 0 spills.

Representative performance versus PyTorch `layer_norm`:

- 128 x 512: 185.344 us vs 10.464 us;
- 128 x 1024: 356.096 us vs 10.592 us;
- 128 x 4096: 1317.088 us vs 11.264 us (~116.9x slower);
- 128 x 8192: 2161.056 us vs 15.360 us (~140.7x slower);
- 1024 x 4096: 1086.416 us vs 24.576 us (~44.2x slower).

Conclusion: the serial mean + serial variance + serial affine passes are the dominant bottleneck. V1 should parallelize hidden-width work inside each row while keeping scalar IO and a simple shared-memory reduction.

## LayerNorm V1 block-parallel shared-memory reductions

V1 changes the execution layout from one serial thread per row to one 256-thread block per row.

Each row performs:

1. thread-local strided sum;
2. 256-entry shared-memory tree reduction for mean;
3. all threads load the final mean;
4. a synchronization barrier protects shared-buffer reuse;
5. thread-local squared-deviation accumulation;
6. second shared-memory tree reduction for variance;
7. parallel affine output.

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **396 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 21 registers/thread, 1,024 B shared memory/block, 0 spills.

Representative V0 -> V1:

- 128 x 512: 189.408 us -> 10.560 us (~17.9x);
- 128 x 1024: 355.088 us -> 11.264 us (~31.5x);
- 128 x 4096: 1316.896 us -> 14.384 us (~91.6x);
- 128 x 8192: 2162.784 us -> 20.144 us (~107.4x);
- 1024 x 4096: 1084.416 us -> 34.736 us (~31.2x).

The 128 x 512 result is effectively equal to the PyTorch LayerNorm reference in this run. Wider/high-row cases still leave 10–40% headroom.

Next: replace both full shared-memory trees with warp-shuffle reductions while keeping scalar IO and the two-pass mean/variance structure unchanged.

## LayerNorm V2 warp-shuffle reductions

V2 preserves the V1 two-pass LayerNorm structure and scalar IO, but replaces both 256-entry shared-memory trees with two-level warp-shuffle reductions.

Validation:

- clean build: PASS;
- full suite: **419 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- V2 ptxas: 23 registers/thread, 32 B shared memory/block, 0 spills.

Selected V1 -> V2:

- 1024 x 128: 14.336 us -> 11.264 us (~1.27x);
- 1024 x 512: 15.360 us -> 12.288 us (~1.25x);
- 1024 x 4096: 34.816 us -> 31.856 us (~1.09x);
- 128 x 512: 10.528 us -> 10.272 us;
- 128 x 4096: 14.336 us -> 14.272 us (neutral).

Shared reduction scratch falls from 1024 B to 32 B, but wide low-row workloads are already close to memory/math limits.

Next: evaluate Welford online mean/variance to combine statistics into one numerically stable pass before vectorized IO.

## LayerNorm V3 Welford statistics

V3 evaluates whether online Welford statistics can replace the separate V2 mean and variance passes.

Final retained implementation:

- float32 Welford state;
- one input traversal for `count / mean / M2`;
- warp-level Welford combination;
- one block per row;
- scalar IO;
- affine output unchanged.

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **443 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 30 registers/thread, 96 B shared memory/block, 0 spills.

Performance result: V3 is not a general performance win over V2.

Representative V2 -> V3:

- 128 x 512: 11.264 us -> 11.264 us;
- 128 x 4096: 14.336 us -> 14.336 us;
- 1024 x 512: 12.768 us -> 14.336 us (regression);
- 1024 x 4096: 31.744 us -> 31.552 us (small win);
- 128 x 8192: 19.456 us -> 19.744 us (small regression).

Numerical stability experiment on inputs near `1000 + N(0, 0.1)`, using float64 LayerNorm as truth:

- PyTorch float32 max abs error: ~2.39e-3;
- V2 two-pass float32: ~2.54e-3;
- V3 float Welford: ~2.35e-3;
- experimental double-Welford: ~1.01e-3.

The double-Welford experiment was not retained: it used 48 registers/thread, 160 B shared memory, and caused roughly 5–12x performance regressions on representative shapes.

Conclusion: Welford is useful as a numerical-method study, but on this GPU the float32 version does not provide a robust speedup and the double version is too expensive. The next performance target is vectorized IO while retaining the faster V2 two-pass statistics.

## LayerNorm V4 aligned float4 IO

V4 returns to the faster V2 two-pass statistics path and changes only memory access.

Aligned fast path:

- `float4` input loads for mean;
- `float4` input loads for variance;
- `float4` weight and bias loads;
- `float4` output stores;
- unchanged V2 warp-shuffle reductions.

Unaligned pointers or `cols % 4 != 0` fall back to V2.

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **467 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 26 registers/thread, 32 B shared memory/block, 0 spills.

Warm-cache representative V2 -> V4:

- 128 x 4096: 14.304 us -> 11.264 us (~1.27x);
- 1024 x 4096: 31.840 us -> 25.456 us (~1.25x);
- 128 x 8192: 19.456 us -> 13.312 us (~1.46x);
- 1024 x 512: 12.512 us -> 11.888 us (~1.05x).

Stable L2-evicted interleaved profiling showed that the benefit is strongly shape-dependent.

Robust gains across two independent seeds:

- 128 x 4096: ~1.41x / ~1.38x;
- 256 x 4096: ~1.25x / ~1.26x;
- 512 x 4096: ~1.18x / ~1.18x;
- 128 x 8192: ~1.42x / ~1.42x;
- 256 x 8192: ~1.18x / ~1.18x;
- 512 x 8192: ~1.16x / ~1.16x;
- 1024 x 512: ~1.12x / ~1.11x;
- 1536 x 512: ~1.06x / ~1.07x;
- 1536 x 1024: ~1.07x / ~1.05x.

Neutral or negative examples:

- 2048 x 4096: ~1.00x;
- 1024 x 4096: ~1.02x;
- 2048 x 512: ~1.01–1.04x;
- 2048 x 1024: V4 regresses to ~0.93x V2.

Conclusion: float4 is a strong memory-path optimization for selected shapes, but should not be enabled universally. Next: V5 profile-guided dispatcher between V2 and V4.

## LayerNorm V5 profile-guided dispatcher

V5 adds no new device kernel. It selects between:

- V2: scalar IO + warp-shuffle two-pass statistics;
- V4: aligned float4 IO + the same V2 reductions.

The policy is generated from repeated L2-evicted interleaved benchmark CSVs with this acceptance gate:

```text
minimum independent runs: 2
minimum speedup in every run: 1.05x
```

Accepted RTX 4090 Laptop profile:

- 128 x 4096;
- 256 x 4096;
- 512 x 4096;
- 128 x 8192;
- 256 x 8192;
- 512 x 8192;
- 1024 x 512;
- 1536 x 512;
- 1536 x 1024.

All other measured and unmeasured shapes fall back to V2.

Validation:

- clean build: PASS;
- full suite: **484 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors.

Final cold-cache examples:

- 128 x 4096: V2 24.384 us -> V5 17.344 us;
- 512 x 4096: 51.200 us -> 44.288 us;
- 128 x 8192: 37.888 us -> 25.904 us;
- 512 x 8192: 95.232 us -> 82.912 us;
- 1536 x 1024: 34.032 us -> 31.728 us.

Fallback validation:

- 1024 x 4096: V2 79.968 us, V5 79.872 us;
- 2048 x 1024: V2 42.304 us, V5 41.920 us;
- 2048 x 512: V2 24.576 us, V5 24.592 us.

Conclusion: LayerNorm now has a complete optimization path from serial baseline through block parallelism, warp reductions, Welford study, vectorized IO, and evidence-driven dispatch.

## Fused Residual + LayerNorm V0 baseline

The fifth operator case study fuses residual addition directly into LayerNorm:

```text
z = x + residual
y = LayerNorm(z)
```

V0 intentionally keeps one CUDA thread per row and serial mean/variance/affine passes. The only architectural change versus an unfused graph is eliminating the materialized intermediate residual-add tensor.

### Fused Residual + LayerNorm V0 validation

Validation:

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full suite: **499 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors;
- ptxas: 26 registers/thread, 0 B shared memory, 0 spills.

Representative latency versus unfused PyTorch `x + residual` followed by `layer_norm`:

- 128 x 512: 331.616 us vs 13.120 us;
- 128 x 1024: 580.608 us vs 13.264 us;
- 128 x 4096: 1874.432 us vs 14.336 us;
- 128 x 8192: 3895.296 us vs 19.136 us;
- 1024 x 4096: 1882.112 us vs 40.944 us.

Conclusion: eliminating the intermediate tensor is insufficient when the fused kernel remains serial within each row. The next experiment must combine fusion with intra-row block parallelism.
