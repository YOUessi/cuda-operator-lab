# CUDA Operator Experiment Ledger

This file is the chronological experiment ledger for the repository. It is intentionally more operational than `reports/reduction.md`: every optimization round records the hypothesis, code change, hardware validation, benchmark protocol, measured result, interpretation, and resulting next action.

## Fixed environment

- Development source of truth: GitHub.
- Real GPU executor: Tang.
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU.
- Compute capability: 8.9 (Ada).
- Driver: 580.178.04.
- PyTorch: 2.10.0+cu128.
- CUDA compiler used by this repository: CUDA 12.8.93.
- Benchmark dtype so far: float32.
- Reference implementation: `torch.sum(..., dtype=torch.float32)`.

## Workflow rule

Ordinary source edits happen on GitHub feature branches. Tang is used only when a step requires real CUDA hardware:

1. fetch / checkout the GitHub branch;
2. clean CUDA build;
3. GPU correctness tests;
4. benchmark;
5. optional ptxas / SASS / Compute Sanitizer / profiler inspection;
6. write measured artifacts and conclusions back to GitHub;
7. merge only after the hardware evidence is recorded.

## Reduction experiment summary

| Version | Main change | GPU tests | Key measured conclusion | Merge commit |
|---|---|---:|---|---|
| V0 | one-thread serial baseline | 11 passed | 16M ≈ 608 ms; establishes lower bound | `8307f78` |
| V1 | grid-stride parallelism + per-thread global atomic | 26 passed | serial gap closes, but ~400–450 us atomic-contention plateau appears | `a9c20da` |
| V2 | shared-memory block reduction + one atomic/block | 45 passed | 16M cold ≈ 177 us, near PyTorch; V1 plateau disappears | `58baea1` |
| V3 | warp shuffle for intra-block reduction | 67 passed | +14% at 262K, ~0% at 16M; large input becomes memory dominated | `83e3820` |
| V4 | aligned `float4` / 128-bit input loads | 89 passed | +4.8% at 16M, but regression at 262K due scalar-based launch geometry | `93355b2` |
| V5 | vector-work-based grid + empirical shape dispatch | 116 passed | fixes V4 medium-shape overlaunch; 16M cold 167.936 us | `a472636` |

---

## E00 — Toolchain bootstrap

### Observation

Tang's system `/usr/bin/nvcc` is CUDA 11.5, while the RTX 4090 Laptop requires SM 8.9 support. CUDA 12.8 components already existed in the Anaconda package cache, but were split across packages.

### Action

Added `scripts/bootstrap_cuda_toolkit.sh` to assemble an ignored local `.cuda-toolkit/` from existing CUDA 12.8 package contents:

- nvcc / ptxas / nvlink / fatbinary;
- NVVM / libdevice;
- CUDA CRT;
- CUDA runtime headers and cudart;
- nvcc development headers.

### Validation

- CUDA 12.8 compiler detected.
- `compute_89` supported.
- real SM 8.9 smoke kernel launched successfully.

### Engineering issue discovered

The first V1 compilation exposed fallback to system headers; adding the CUDA 12.8 nvcc development headers removed that mismatch. An unnecessary `<algorithm>` include also triggered an nvcc/GCC 11 host-pass failure in `std_function.h`; the include was removed.

---

## E01 — Reduction V0: serial baseline

### Hypothesis

A deliberately serial GPU implementation gives a clean lower bound against which every later optimization can be measured.

### Kernel change

One active CUDA thread loops over all N values and writes one scalar result.

### Validation

- build: PASS;
- full test suite: 11 passed.

### Recorded benchmark

5 warmups + 20 timed repeats, CUDA events.

Largest shape:

```text
N = 16,777,216
V0        ≈ 608,579 us
torch.sum ≈      29.7 us
```

### Conclusion

The dominant bottleneck is insufficient parallelism.

### Artifact

- `reports/data/reduction_v0_rtx4090_laptop.csv`

---

## E02 — Reduction V1: grid-stride parallelism

### Hypothesis

Parallel grid-stride input processing should eliminate the catastrophic serial bottleneck.

### Kernel change

- 256 threads/block;
- up to 1,024 blocks;
- each thread accumulates a register-local grid-stride sum;
- each participating thread performs one global `atomicAdd`.

### Validation

- clean build: PASS;
- full test suite: 26 passed;
- non-default CUDA stream: PASS;
- output reset: PASS.

### Result

At 16M elements:

```text
V0 ≈ 608,579 us
V1 ≈     393–450 us
```

Large-shape latency plateaus near 400–450 us.

### Conclusion

Parallelism succeeds, but up to 262,144 threads contend on the same global atomic target. The next experiment must reduce global atomic count.

### Artifact

- `reports/data/reduction_v0_v1_rtx4090_laptop.csv`

---

## E03 — Reduction V2: shared-memory block reduction

### Hypothesis

Reducing thread-local values inside each block before the global atomic should remove the V1 contention plateau.

### Kernel change

- same grid-stride local accumulation;
- 256-float shared-memory tree;
- one `atomicAdd` per block instead of per thread.

Worst-case atomics:

```text
V1: 262,144
V2:   1,024
```

### Validation

- full test suite: 45 passed;
- ptxas: 12 registers/thread, 1,024 B shared/block, 0 spills.

### Benchmark methodology correction

The GPU reports 64 MiB L2 and the largest input is also 64 MiB. Hot repeated runs can therefore be cache served and cannot be called DRAM bandwidth.

The benchmark was changed to record:

- `cache_mode`;
- `l2_bytes`;
- `flush_bytes`;
- `logical_input_gbps`.

Cold mode touches a 128 MiB buffer before each timed launch.

### L2-evicted result

```text
N = 16,777,216
V1      448.144 us
V2      177.152 us
PyTorch 172.032 us
```

### Conclusion

The atomic-contention hypothesis is validated. Large-shape reduction is now close to memory-throughput limited.

### Artifacts

- `reports/data/reduction_v1_v2_hot_rtx4090_laptop.csv`
- `reports/data/reduction_v1_v2_cold_rtx4090_laptop.csv`
- `reports/data/reduction_v2_ptxas_sm89.txt`

---

## E04 — Reduction V3: warp shuffle

### Hypothesis

Once global atomic contention is removed, replacing the full shared-memory tree with warp-level register exchange should reduce coordination overhead.

### Kernel change

- each warp reduces via `__shfl_down_sync`;
- lane 0 writes one value per warp;
- only 8 floats stored in shared memory;
- first warp performs the final block reduction;
- one global atomic/block remains.

### Validation

- full test suite: 67 passed;
- Compute Sanitizer 12.8 memcheck: 0 errors;
- synccheck: 0 errors;
- ptxas: 13 registers/thread, 32 B shared/block, 0 spills.

### Stable L2-evicted result

20 warmups + 100 repeats:

```text
262K: V2 8.192 us -> V3 7.168 us  (~1.14x)
4M:   V2 47.840 us -> V3 47.104 us (~1.02x)
16M:  V2 177.152 us -> V3 177.200 us (~1.00x)
```

### Conclusion

Warp shuffle helps medium shapes but not the largest shape. The dominant large-shape bottleneck has shifted from coordination to input movement.

### Artifacts

- `reports/data/reduction_v2_v3_hot_rtx4090_laptop.csv`
- `reports/data/reduction_v2_v3_cold_rtx4090_laptop.csv`
- `reports/data/reduction_v2_v3_cold100_rtx4090_laptop.csv`
- `reports/data/reduction_v3_ptxas_sm89.txt`
- `reports/data/reduction_v3_compute_sanitizer.txt`

---

## E05 — Reduction V4: aligned float4 loads

### Hypothesis

For large memory-dominated reductions, fewer/wider load instructions should improve the input path.

### Kernel change

- keep V3 warp-shuffle reduction;
- aligned path interprets input as `float4`;
- scalar tail handles N % 4;
- 16-byte-unaligned contiguous input falls back to V3.

### Generated-code check

SM 8.9 SASS contains:

```text
LDG.E.128
```

confirming the aligned fast path becomes a real 128-bit global load.

### Validation

- full test suite: 89 passed;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- unaligned fallback memcheck: 0 errors;
- ptxas: 16 registers/thread, 32 B shared/block, 0 spills.

### Stable L2-evicted result

20 warmups + 100 repeats:

```text
262K: V3 7.168 us   -> V4 8.192 us   (regression)
4M:   V3 46.896 us  -> V4 46.080 us  (~1.02x)
16M:  V3 177.152 us -> V4 168.960 us (~1.05x)
```

### Conclusion

Vectorization helps large memory-dominated input but is not universally beneficial. At 262K, V4 still launches blocks from scalar N while the vector loop contains only N/4 work items; many launched threads receive no vector work but still participate in reduction.

### Artifacts

- `reports/data/reduction_v3_v4_hot_rtx4090_laptop.csv`
- `reports/data/reduction_v3_v4_cold_rtx4090_laptop.csv`
- `reports/data/reduction_v3_v4_cold100_rtx4090_laptop.csv`
- `reports/data/reduction_v4_ptxas_sm89.txt`
- `reports/data/reduction_v4_sass_sm89.txt`
- `reports/data/reduction_v4_compute_sanitizer.txt`

---

## E06 — Reduction V5: shape-aware vector launch

Status: **validated; ready to merge**

### Initial hypothesis

The V4 medium-shape regression was caused by grid geometry being computed from scalar `N` even though the vector kernel has only `N/4` `float4` work items.

### Operation 1 — change only vector-path grid sizing

The V4 kernel body was kept unchanged.

```text
V4 blocks = ceil(N / 256)
V5 blocks = ceil((N / 4) / 256)
```

The 1,024-block cap and V3 fallback were preserved.

### Operation 2 — first hardware validation

Tang pulled the GitHub branch and performed a clean CUDA 12.8 / SM 8.9 build.

Result:

```text
build: PASS
full pytest: 110 passed
```

### Operation 3 — preliminary V4 vs V5 benchmark

Hot and L2-evicted 50-repeat runs plus a focused 100-repeat cold run were recorded.

The important initial result at 262K:

```text
L2-evicted:
V4  8.192 us
V5  7.392 us
```

So sizing the grid from vector work fixed the specific V4 overlaunch regression.

Raw artifacts:

- `reports/data/reduction_v4_v5_prethreshold_hot50_rtx4090_laptop.csv`
- `reports/data/reduction_v4_v5_prethreshold_cold50_rtx4090_laptop.csv`
- `reports/data/reduction_v4_v5_prethreshold_cold100_rtx4090_laptop.csv`

### Operation 4 — crossover sweep

A wider V3 vs V5 sweep was run before hard-coding a dispatch threshold.

Cold, 20 warmups + 100 repeats:

```text
16K:  V3 4.160 us   V5 4.096 us
32K:  V3 4.336 us   V5 4.096 us
64K:  V3 5.120 us   V5 5.120 us
128K: V3 6.144 us   V5 6.144 us
262K: V3 7.408 us   V5 7.168 us
512K: V3 10.272 us  V5 10.240 us
1M:   V3 16.384 us  V5 15.392 us
2M:   V3 28.672 us  V5 25.600 us
4M:   V3 47.104 us  V5 45.056 us
8M:   V3 91.136 us  V5 86.576 us
16M:  V3 176.128 us V5 167.936 us
```

Hot-cache sweep showed that very small shapes could be neutral or regress depending on run state, while 512K and larger were consistently neutral-to-positive.

Raw artifacts:

- `reports/data/reduction_v3_v5_prethreshold_crossover_hot100_rtx4090_laptop.csv`
- `reports/data/reduction_v3_v5_prethreshold_crossover_cold100_rtx4090_laptop.csv`

### Operation 5 — empirical dispatch threshold

An initial 32K threshold was tried, then the cross-regime results were reviewed again.

To avoid promoting a cache-state-sensitive small-shape result into the default policy, the final vector threshold was made more conservative:

```text
N < 524,288
    -> V3 scalar warp-shuffle path

N >= 524,288 and pointer is 16-byte aligned
    -> float4 vector path with blocks computed from N/4

unaligned
    -> V3 scalar fallback
```

Threshold-boundary correctness tests were added around both the initial and final crossover.

### Operation 6 — final clean regression validation

After the final 512K crossover was committed:

```text
clean build: PASS
full pytest: 116 passed
```

### Operation 7 — final benchmark

Final comparison uses 20 warmups + 100 timed repeats.

L2-evicted:

| N | V3 scalar | V5 dispatch | V3 → V5 | torch.sum |
|---:|---:|---:|---:|---:|
| 262,144 | 8.176 us | 7.424 us* | 1.10x* | 11.120 us |
| 524,288 | 10.256 us | 10.240 us | ~1.00x | 12.288 us |
| 1,048,576 | 17.008 us | 15.376 us | 1.11x | 17.408 us |
| 4,194,304 | 47.104 us | 45.152 us | 1.04x | 48.128 us |
| 16,777,216 | 176.128 us | 167.936 us | 1.05x | 170.896 us |

`*` At 262K the final V5 dispatcher selects the V3 scalar kernel. The timing difference is measurement/run-state noise between two calls to the same underlying kernel, not a different kernel optimization.

Hot-cache:

| N | V3 scalar | V5 dispatch | V3 → V5 |
|---:|---:|---:|---:|
| 262,144 | 12.288 us | 12.448 us* | ~0.99x* |
| 524,288 | 13.312 us | 12.608 us | 1.06x |
| 1,048,576 | 12.976 us | 13.056 us | ~0.99x |
| 4,194,304 | 18.080 us | 15.360 us | 1.18x |
| 16,777,216 | 40.960 us | 33.792 us | 1.21x |

Again, below the crossover V5 is intentionally the scalar V3 path.

Final raw artifacts:

- `reports/data/reduction_v3_v5_final_hot100_rtx4090_laptop.csv`
- `reports/data/reduction_v3_v5_final_cold100_rtx4090_laptop.csv`

### Operation 8 — final Compute Sanitizer

CUDA 12.8 Compute Sanitizer:

```text
aligned vector path, N=1,048,576:
  memcheck:  0 errors
  racecheck: 0 hazards / 0 errors
  synccheck: 0 errors

below-crossover scalar path, N=262,144:
  memcheck: 0 errors

unaligned contiguous fallback:
  memcheck: 0 errors
```

Artifact:

- `reports/data/reduction_v5_compute_sanitizer.txt`

### Conclusion

The V4 regression was not caused by `float4` itself; it was caused by applying scalar launch geometry to vector work.

V5 fixes that mismatch and then adds a conservative empirical dispatcher so the vector path is used only where the measured benefit is robust enough to justify it.

The Reduction line now has a complete profile-guided story:

```text
V0 serial
 -> V1 parallel
 -> V2 block reduction
 -> V3 warp shuffle
 -> V4 vector loads
 -> V5 shape-aware dispatch
```

### Next action

Reduction is now sufficiently deep for the project. The next operator should reuse the same benchmark / correctness / profiling discipline rather than continue adding increasingly marginal Reduction-only optimizations.

Next target: **row-wise Softmax baseline and optimization ladder**.

---

## E07 — Softmax V0: one serial CUDA thread per row

Status: **validated; ready to merge**

### Hypothesis

A deliberately simple row-wise baseline should expose the cost of serial max / exp-sum / normalize work inside each row while still allowing different rows to execute in parallel.

### Operation 1 — GitHub implementation

Added a new 2-D float32 Softmax path:

```text
one CUDA thread
  -> one row
  -> serial max pass
  -> serial exp + denominator pass
  -> serial normalization pass
```

The kernel subtracts the row maximum before exponentiation for numerical stability.

Added:

- `csrc/softmax/softmax.cu` / `.cuh`;
- CMake integration;
- ctypes binding on the active PyTorch CUDA stream;
- `torch.softmax(..., dim=-1)` reference;
- correctness tests for row/column boundaries, extreme logits, preallocated output, non-default stream, and non-contiguous rejection;
- `benchmarks/softmax_baseline.py`.

### Operation 2 — first Tang build and compile failure

The first clean CUDA 12.8 / SM 8.9 build failed because `CUDART_INF_F` was not defined in the assembled toolkit header set.

Observed compiler error:

```text
softmax.cu: identifier "CUDART_INF_F" is undefined
```

The fix was made on GitHub, not locally:

- include `<cfloat>`;
- initialize the row maximum with `-FLT_MAX`.

Commit: `c1ff891`.

### Operation 3 — clean hardware regression

Tang fetched the fixed GitHub branch and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 131 passed
```

The 131 tests include the complete Reduction regression suite plus the new Softmax tests.

### Operation 4 — Softmax V0 benchmark

Protocol:

- CUDA events;
- 5 warmups;
- 20 timed repeats;
- float32;
- same CUDA input used for our kernel and `torch.softmax`.

| Shape | V0 serial-row | torch.softmax | Slowdown |
|---:|---:|---:|---:|
| 1 × 128 | 18.432 us | 9.216 us | 2.00x |
| 32 × 128 | 40.800 us | 9.216 us | 4.43x |
| 128 × 128 | 60.416 us | 9.216 us | 6.56x |
| 128 × 512 | 214.736 us | 9.376 us | 22.90x |
| 128 × 1024 | 416.768 us | 8.192 us | 50.88x |
| 128 × 4096 | 1,595.392 us | 9.328 us | **171.03x** |
| 1024 × 128 | 57.344 us | 9.328 us | 6.15x |
| 1024 × 512 | 209.440 us | 10.160 us | 20.61x |
| 1024 × 4096 | 1,607.408 us | 37.600 us | 42.75x |

Numerical behavior remained close to PyTorch:

- max absolute element error: on the order of `1e-8`;
- largest recorded row-sum error: about `3.10e-6`.

Raw artifact:

- `reports/data/softmax_v0_rtx4090_laptop.csv`

### Operation 5 — ptxas resource capture

CUDA 12.8 / `sm_89`:

```text
registers/thread: 24
shared memory/block: 0 B
spill stores: 0
spill loads: 0
barriers: 0
stack frame: 0 B
```

Artifact:

- `reports/data/softmax_v0_ptxas_sm89.txt`

### Operation 6 — Compute Sanitizer

Representative non-power-of-two shape: `[17, 513]`.

CUDA 12.8 Compute Sanitizer:

```text
memcheck:  0 errors
racecheck: 0 hazards / 0 errors
synccheck: 0 errors
```

Observed row-sum error during sanitizer runs stayed below approximately `8.35e-7`.

Artifact:

- `reports/data/softmax_v0_compute_sanitizer.txt`

### Conclusion

The first Softmax bottleneck is clear and differs from Reduction V0.

Rows already execute in parallel, but **every individual row is completely serial**:

```text
serial max over cols
  -> serial exp + sum over cols
  -> serial normalize over cols
```

The width dimension therefore dominates. At fixed 128 rows, increasing width from 128 to 4096 increases V0 latency from about 60 us to about 1.6 ms, while PyTorch stays around 8–9 us.

The 128 × 4096 case is roughly **171x slower than PyTorch**.

### Next action

Softmax V1 should parallelize work **inside each row** while keeping the algorithm structure otherwise unchanged:

```text
one block per row
  -> per-thread strided max
  -> shared-memory max reduction
  -> parallel exp + local sum
  -> shared-memory sum reduction
  -> parallel normalize
```

Warp shuffle is intentionally deferred to a later version so the value of basic intra-row parallelism and shared-memory reduction can be measured independently.

---

## E08 — Softmax V1: one block per row + shared-memory reductions

Status: **validated; ready to merge**

### Hypothesis

Softmax V0 is dominated by serial work inside each row. Giving one CUDA block to each row should expose 256-way intra-row parallelism while preserving the same max / exp-sum / normalize structure.

### Operation 1 — GitHub implementation

V1 uses one 256-thread block per row.

```text
thread-local max over strided columns
  -> shared[256]
  -> shared-memory max tree
  -> row max

parallel exp(x - row_max)
  -> thread-local denominator sum
  -> shared[256]
  -> shared-memory sum tree
  -> denominator

parallel normalize
```

The implementation intentionally keeps a full shared-memory tree and `__syncthreads()` at every level. Warp shuffle is deferred to V2.

Added:

- `cuda_operator_softmax_v1` C ABI;
- Python binding;
- correctness tests around 255/256/257 and 511/512/513 widths;
- extreme-logit stability;
- output-reuse and active-stream tests;
- `benchmarks/softmax_benchmark.py` for V0 / V1 / PyTorch.

### Operation 2 — first clean hardware regression

Tang fetched the GitHub branch and performed a clean CUDA 12.8 / SM 8.9 build.

```text
build: PASS
full repository pytest: 153 passed
```

Ordinary correctness tests did not expose a synchronization bug that was found next by Racecheck.

### Operation 3 — first Compute Sanitizer run found a real shared-memory race

Representative shape: `[17, 513]`.

`memcheck` was clean, but `racecheck` reported:

```text
1 displayed race error
72 hazards
```

Root cause:

1. all threads completed the max-reduction tree;
2. each thread executed `row_max = shared[0]`;
3. there was no block barrier after that read;
4. faster warps could begin the denominator stage and overwrite `shared[0]` while slower warps were still reading the row max.

This is exactly the type of bug that normal numerical tests can miss.

### Operation 4 — GitHub-side race fix

The fix was committed on GitHub:

```cpp
const float row_max = shared[0];
__syncthreads();  // all warps finish reading shared[0]

... reuse shared[] for denominator partial sums ...
```

Commit: `5c294da`.

No local source edit was used.

### Operation 5 — clean regression after the race fix

Tang fetched the fixed branch and rebuilt from scratch.

```text
build: PASS
full repository pytest: 153 passed
```

### Operation 6 — post-fix sanitizer validation

CUDA 12.8 Compute Sanitizer, shape `[17, 513]`:

```text
memcheck:  0 errors
racecheck: 0 hazards / 0 errors
synccheck: 0 errors
```

Representative post-fix numerical observations:

- max absolute error <= approximately `7.45e-9`;
- max row-sum error <= approximately `1.79e-7`.

Artifact:

- `reports/data/softmax_v1_compute_sanitizer.txt`

### Operation 7 — ptxas resource capture

CUDA 12.8 / `sm_89`:

```text
V0:
  registers/thread: 24
  shared memory/block: 0 B
  barriers: 0
  spills: 0

V1:
  registers/thread: 21
  shared memory/block: 1,024 B
  barrier resource: used
  spills: 0
```

Artifact:

- `reports/data/softmax_v1_ptxas_sm89.txt`

### Operation 8 — final V0 vs V1 benchmark

Protocol:

- CUDA events;
- 10 warmups;
- 50 timed repeats;
- float32;
- same input per shape.

| Shape | V0 | V1 | V0 -> V1 | torch.softmax | V1 / torch |
|---:|---:|---:|---:|---:|---:|
| 1 x 128 | 17.408 us | 10.240 us | 1.70x | 9.536 us | 1.07x |
| 32 x 128 | 38.912 us | 15.360 us | 2.53x | 8.320 us | 1.85x |
| 128 x 128 | 56.352 us | **9.216 us** | 6.11x | 10.496 us | 0.88x |
| 128 x 512 | 224.256 us | **11.264 us** | 19.91x | 10.240 us | 1.10x |
| 128 x 1024 | 405.632 us | **11.264 us** | 36.01x | 8.832 us | 1.28x |
| 128 x 4096 | 1,614.752 us | **14.336 us** | **112.64x** | 9.216 us | 1.56x |
| 1024 x 128 | 57.344 us | 15.360 us | 3.73x | 9.216 us | 1.67x |
| 1024 x 512 | 208.192 us | 17.296 us | 12.04x | 8.880 us | 1.95x |
| 1024 x 4096 | 1,353.728 us | **37.888 us** | 35.73x | 36.960 us | **1.03x** |

Raw artifact:

- `reports/data/softmax_v0_v1_rtx4090_laptop.csv`

### Interpretation

The V0 hypothesis is decisively validated: intra-row parallelism removes almost all of the catastrophic width scaling.

The `128 x 4096` case improves by more than **112x**, and `1024 x 4096` reaches within about **3%** of PyTorch.

The remaining overhead is now concentrated in the block-level reduction machinery:

- two full 256-thread shared-memory trees per row;
- repeated `__syncthreads()` at each max-reduction level;
- repeated `__syncthreads()` at each sum-reduction level;
- 1,024 B shared memory per block.

### Next action

Softmax V2 should keep one block per row and keep the input/output passes unchanged, but replace the shared-memory trees with warp-shuffle reductions:

```text
thread-local max / sum
  -> warp __shfl_down_sync
  -> 8 warp partials in shared memory
  -> first warp final reduction
```

This isolates synchronization/shared-memory reduction overhead before any vectorized input or shape dispatch work.

---

## Template for future experiments

### Hypothesis

What bottleneck or profiler observation is being targeted?

### Isolated change

Exactly what mechanism changes, and what is intentionally held constant?

### Validation

- build;
- correctness;
- sanitizer;
- generated-code/resource checks if relevant.

### Benchmark protocol

- shapes;
- cache mode;
- warmups;
- repeats;
- timing mechanism.

### Results

Record raw CSV and the smallest table needed to support the conclusion.

### Interpretation

State what the data supports. Do not promote a shape-specific win into a universal claim.

### Next action

Derive the next change from the measured bottleneck, not from a predetermined optimization checklist.


---

## E09 — Softmax V2: warp-shuffle block reductions

Status: **validated; ready to merge**

### Hypothesis

Softmax V1 removed the serial-width bottleneck, but every row still performs two complete 256-thread shared-memory trees:

1. max reduction;
2. denominator-sum reduction.

Each tree uses repeated `__syncthreads()` stages and 1,024 B shared memory per block.

If the row traversal and one-block-per-row launch policy are held constant, replacing only those two trees with two-level warp-shuffle reductions should isolate the cost of block-level reduction coordination.

### Operation 1 — experiment branch and scope lock

Created branch:

```text
feat/softmax-v2-warp-reduce
```

Scope is intentionally limited to:

- keep 256 threads per row;
- keep the same three Softmax passes;
- keep block-stride column traversal;
- replace max/sum reduction machinery only;
- do not add float4, shape dispatch, or change block size in this experiment.

Planned hardware validation:

- clean CUDA 12.8 / SM 8.9 build;
- full repository pytest;
- Compute Sanitizer memcheck / racecheck / synccheck;
- ptxas resource capture;
- V1 vs V2 benchmark on the same Softmax shape matrix.


### Operation 2 — GitHub implementation completed

Implemented Softmax V2 directly on the GitHub feature branch without editing Tang locally.

Changed:

- `csrc/softmax/softmax.cu`: add two-level warp-shuffle max/sum reductions;
- `csrc/softmax/softmax.cuh`: expose `cuda_operator_softmax_v2`;
- `python/cuda_operator_lab/bindings.py`: add V2 ctypes binding;
- `benchmarks/softmax_benchmark.py`: add `v2_warp_shuffle`;
- `tests/test_softmax_v2.py`: add warp/block boundary, wide-row, stability, stream, output-reuse, and row-normalization coverage.

The V2 reduction structure is:

```text
thread-local max
  -> warp max via __shfl_down_sync
  -> 8 warp maxima in shared memory
  -> first warp final max
  -> row_max

thread-local exp sum
  -> warp sum via __shfl_down_sync
  -> 8 warp sums in shared memory
  -> first warp final sum
  -> denominator
```

Only 8 float partials are stored in shared memory. The one-block-per-row launch, 256 threads/block, column traversal, exponentiation pass, and normalization pass are intentionally unchanged from V1.

A safety barrier is kept after all threads load the final row maximum before the shared partial buffer is reused for the denominator reduction. This directly preserves the Racecheck lesson from V1.

GitHub commits in this operation:

- `9667d1f`: expose V2 C ABI;
- `4b3b919`: add warp-shuffle Softmax kernel;
- `1fe3cc8`: bind V2 in Python;
- `a1304c5`: add V2 benchmark variant;
- `a8167d1`: add V2 correctness tests.

### Operation 3 — real-GPU validation gate attempted

A Tang hardware-validation run was requested after the GitHub implementation.

Result:

```text
Tang device status: OFFLINE
last seen: approximately 3 hours before validation attempt
```

Therefore no CUDA build, pytest, benchmark, sanitizer, or ptxas result is claimed yet for V2.

Decision:

- keep the branch unmerged;
- do not invent performance numbers;
- do not start Softmax V3 before V2 receives real RTX 4090 validation;
- resume at the clean-build step when Tang is online.


### Operation 4 — Tang returned online; clean hardware regression

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA compiler: 12.8.93
target: sm_89
build: PASS
full repository pytest: 180 passed
```

No local source edits were used.

### Operation 5 — Compute Sanitizer

Representative shape: `[17, 513]`.

CUDA 12.8 Compute Sanitizer:

```text
memcheck:
  0 errors

racecheck:
  0 hazards / 0 errors

synccheck:
  0 errors
```

Observed numerical values in all three runs:

```text
max_abs_error      = 3.725290298461914e-09
max_row_sum_error  = 1.1920928955078125e-07
```

Artifact:

- `reports/data/softmax_v2_compute_sanitizer.txt`

### Operation 6 — ptxas resource capture

CUDA 12.8 / `sm_89`:

```text
V1 shared-tree:
  registers/thread: 21
  shared memory/block: 1024 B
  spills: 0

V2 warp-shuffle:
  registers/thread: 23
  shared memory/block: 32 B
  spills: 0
```

V2 trades two additional registers per thread for a **32x reduction in reduction scratch shared memory**.

Artifact:

- `reports/data/softmax_v2_ptxas_sm89.txt`

### Operation 7 — V1 vs V2 benchmark

First matrix:

- 10 warmups;
- 50 timed repeats;
- CUDA events;
- float32;
- same input per shape.

Representative results:

```text
128 x 128:
  V1 11.264 us
  V2 10.992 us

128 x 4096:
  V1 13.616 us
  V2 13.248 us

1024 x 128:
  V1 14.336 us
  V2 10.256 us

1024 x 512:
  V1 16.352 us
  V2 12.288 us

1024 x 4096:
  V1 41.984 us
  V2 39.744 us
```

Artifact:

- `reports/data/softmax_v1_v2_50_rtx4090_laptop.csv`

### Operation 8 — stable focused benchmark

A second comparison used 20 warmups + 100 timed repeats on the main shape set.

| Shape | V1 shared tree | V2 warp shuffle | V1 -> V2 | torch.softmax | V2 / torch |
|---:|---:|---:|---:|---:|---:|
| 128 x 128 | 10.368 us | 10.336 us | ~1.00x | 8.192 us | 1.26x |
| 128 x 512 | **10.240 us** | 10.592 us | 0.97x | 8.176 us | 1.30x |
| 128 x 1024 | 10.512 us | 10.400 us | 1.01x | 8.256 us | 1.26x |
| 128 x 4096 | 13.664 us | **13.216 us** | 1.03x | 9.216 us | 1.43x |
| 1024 x 128 | 14.336 us | **10.240 us** | **1.40x** | 8.096 us | 1.26x |
| 1024 x 512 | 16.192 us | **12.256 us** | **1.32x** | 8.192 us | 1.50x |
| 1024 x 4096 | 41.792 us | **39.600 us** | 1.06x | 36.608 us | 1.08x |

Artifact:

- `reports/data/softmax_v1_v2_100_rtx4090_laptop.csv`

### Interpretation

The hypothesis is partly validated and, importantly, the benefit is shape-dependent.

V2 removes almost all reduction scratch storage:

```text
1024 B -> 32 B shared memory/block
```

and materially helps high-row-count cases. The largest observed V1 -> V2 gains in the stable run are:

```text
1024 x 128:  ~1.40x
1024 x 512:  ~1.32x
1024 x 4096: ~1.06x
```

For 128-row shapes the benefit is small, neutral, or slightly negative. The 128 x 512 case regresses by about 3.4%, so V2 is not promoted as a universal "warp shuffle is always faster" result.

The new shape clue is the **row width versus fixed 256-thread block**. At cols=128, half of a 256-thread block has no column work but still participates in the block-level control flow. V2 made this waste easier to see because the reduction tree is no longer the dominant cost.

### Next action

Softmax V3 should keep the V2 warp-shuffle algorithm but make the block width depend on row width:

```text
cols <= 32   -> 32 threads
cols <= 64   -> 64 threads
cols <= 128  -> 128 threads
otherwise    -> 256 threads
```

This isolates launch/work efficiency before adding vectorized loads or changing the three-pass Softmax algorithm.


---

## E10 — Softmax V3: width-aware block sizing

Status: **validated; experimental result**

### Hypothesis

Softmax V2 removes most shared-memory reduction overhead, but still launches 256 threads for every row width.

For narrow rows this creates inactive column workers:

```text
cols = 128
block = 256 threads
-> 128 threads have column work
-> 128 threads contribute only reduction identities
```

The stable V2 benchmark still shows `1024 x 128 = 10.240 us` versus PyTorch `8.096 us`.

### Operation 1 — experiment branch and scope lock

Created branch:

```text
feat/softmax-v3-width-aware-blocks
```

This experiment will change only block width:

```text
cols <= 32   -> 32 threads
cols <= 64   -> 64 threads
cols <= 128  -> 128 threads
otherwise    -> 256 threads
```

Held constant:

- one block per row;
- V2 warp-shuffle max/sum reduction;
- three Softmax passes;
- block-stride column traversal;
- float32;
- no float4/vectorized IO yet.

Planned evidence:

- clean CUDA build and full pytest;
- memcheck/racecheck/synccheck on both narrow and wide rows;
- ptxas resources;
- boundary benchmark around 32/64/128/256 columns;
- stable V2 vs V3 benchmark on 128-row and 1024-row shapes.


### Operation 2 — GitHub implementation

Implemented V3 entirely on GitHub.

Changed:

- `csrc/softmax/softmax.cu`: add dynamic-warp-count reduction helpers and width-aware launch;
- `csrc/softmax/softmax.cuh`: expose V3 C ABI;
- `python/cuda_operator_lab/bindings.py`: bind V3;
- `benchmarks/softmax_benchmark.py`: add V3 and record threads/block;
- `tests/test_softmax_v3.py`: add dispatch-boundary correctness coverage.

Dispatch policy:

```text
1..32 cols    -> 32 threads
33..64 cols   -> 64 threads
65..128 cols  -> 128 threads
>128 cols     -> 256 threads
```

The dynamic reduction helpers compute `warp_count = blockDim.x / 32` so only launched warps contribute to the first-warp final reduction.

V2 remains unchanged and fixed at 256 threads/block for direct A/B comparison.


### Operation 3 — clean Tang hardware regression

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 209 passed
```

### Operation 4 — sanitizer validation across two dispatch regimes

Narrow dispatch representative:

```text
shape: 17 x 65
threads/block: 128
memcheck: 0 errors
racecheck: 0 hazards / 0 errors
synccheck: 0 errors
max_abs <= 1.49e-8
max_row_sum_error <= 1.79e-7
```

Wide dispatch representative:

```text
shape: 17 x 513
threads/block: 256
memcheck: 0 errors
racecheck: 0 hazards / 0 errors
synccheck: 0 errors
max_abs <= 5.59e-9
max_row_sum_error <= 1.19e-7
```

Artifact:

- `reports/data/softmax_v3_compute_sanitizer.txt`

### Operation 5 — ptxas resource capture

V2 and V3 compile to the same static resource footprint:

```text
registers/thread: 23
shared memory/block: 32 B
spills: 0
barrier resource: used
```

Artifact:

- `reports/data/softmax_v3_ptxas_sm89.txt`

### Operation 6 — dispatch-boundary sweep

10 warmups + 50 timed repeats, 1024 rows.

Selected results:

| Width | V2 threads | V2 | V3 threads | V3 | Result |
|---:|---:|---:|---:|---:|---|
| 31 | 256 | 10.240 us | 32 | 10.256 us | neutral |
| 33 | 256 | 10.944 us | 64 | 10.480 us | V3 +4.4% |
| 63 | 256 | 11.200 us | 64 | 10.448 us | V3 +7.2% |
| 64 | 256 | 11.008 us | 64 | 10.608 us | V3 +3.8% |
| 127 | 256 | 11.008 us | 128 | 10.464 us | V3 +5.2% |
| 128 | 256 | 10.240 us | 128 | 10.208 us | neutral |
| 129 | 256 | 10.448 us | 256 | 10.496 us | same kernel / noise |

Artifact:

- `reports/data/softmax_v2_v3_boundaries_50_rtx4090_laptop.csv`

### Operation 7 — stable 100-repeat comparison

20 warmups + 100 timed repeats.

| Shape | V2 | V3 | V2 -> V3 | PyTorch |
|---:|---:|---:|---:|---:|
| 128 x 32 | **10.240 us** | 10.864 us | 0.94x | 8.192 us |
| 128 x 64 | 10.240 us | 10.240 us | 1.00x | 8.192 us |
| 128 x 128 | 10.240 us | 10.240 us | 1.00x | 8.192 us |
| 1024 x 32 | 10.240 us | 10.240 us | 1.00x | 7.920 us |
| 1024 x 64 | **10.240 us** | 10.992 us | 0.93x | 8.048 us |
| 1024 x 128 | 10.464 us | **10.272 us** | 1.02x | 8.192 us |

For widths above 128 the V3 dispatcher intentionally uses 256 threads, so V2/V3 differences there are timing noise between equivalent launch geometry.

Artifact:

- `reports/data/softmax_v2_v3_100_rtx4090_laptop.csv`

### Interpretation

The simple hypothesis “fewer inactive threads must be faster” is **not supported as a robust default policy**.

Boundary sweeps show some local wins, but the stable run also shows regressions:

- 128 x 32: ~6% slower;
- 1024 x 64: ~7% slower;
- 1024 x 128: only ~2% faster.

Shrinking the block reduces unused column workers, but it also reduces warps per block. With one block assigned to one row, that can reduce useful warp-level parallelism and make scheduler / resident-block limits matter. The experiment therefore exposes a structural problem in one-block-per-row dispatch rather than a simple thread-count tuning problem.

### Decision

Keep V3 as an explicit, validated **negative/shape-sensitive experiment**, but do not treat it as the preferred Softmax path.

### Next action

Softmax V4 should keep a full 256-thread block but assign **one warp per small row**, packing up to 8 rows into one block:

```text
256-thread block
  -> 8 warps
  -> each warp owns one row
  -> no cross-warp reduction for that row
```

For small widths this preserves 8 useful warps per block instead of shrinking the block to 1–4 warps. Wider rows will fall back to the validated V2 one-block-per-row path.


### E10 addendum — high-row-count revalidation changes the V3 conclusion

The original V3 sweep covered 128-row and 1024-row shapes and concluded that width-aware block shrinkage was too shape-sensitive to use as a default policy.

After Tang reconnected, the same already-merged V3 implementation was revalidated on larger row counts before moving on.

Protocol:

- current `main` Softmax V3 implementation;
- 20 warmups;
- 100 timed repeats;
- V2 fixed-256 versus V3 width-aware;
- no code changes between the two variants other than the existing V3 launch policy.

Results:

| Shape | V2 fixed 256 | V3 width-aware | V2 -> V3 | PyTorch |
|---:|---:|---:|---:|---:|
| 4096 x 32 | 17.408 us | **10.256 us** | **1.70x** | 8.160 us |
| 4096 x 64 | 17.408 us | **11.264 us** | **1.55x** | 7.456 us |
| 4096 x 128 | 18.080 us | **13.968 us** | **1.29x** | 8.000 us |
| 16384 x 32 | 46.128 us | **18.432 us** | **2.50x** | 7.168 us |
| 16384 x 64 | 47.104 us | **21.520 us** | **2.19x** | 8.864 us |
| 16384 x 128 | 49.152 us | **32.448 us** | **1.51x** | 11.136 us |

Artifact:

- `reports/data/softmax_v2_v3_highrows_revalidation100_rtx4090.csv`

Revised interpretation:

- V3 is near-neutral or shape-sensitive at low/moderate row counts;
- V3 is clearly beneficial for high-row-count narrow matrices;
- the previous “negative experiment” label was therefore incomplete rather than wrong;
- the useful condition is **narrow width + enough rows for oversized 256-thread blocks to become a scheduling cost**.

This addendum is kept instead of rewriting the earlier E10 section so the project preserves how the conclusion evolved when a missing workload regime was added.

---

## E11 — Softmax V4: one warp per row, multiple rows per block

Status: **in progress**

### Hypothesis

V3 proves that 256 threads per narrow row wastes scheduling capacity when row count is high. However, simply shrinking each block to one or two warps can reduce the number of useful warps resident in a block and gives limited benefit at lower row counts.

V4 will test a different layout for narrow rows:

```text
one 256-thread block
  -> 8 warps
  -> each warp owns one row
  -> up to 8 rows processed per block
```

This keeps a full 8-warps-per-block launch while eliminating cross-warp reduction for each row.

### Operation 1 — scope lock

V4 will be isolated to `cols <= 128`.

For each warp-owned row:

- lane-stride max over the row;
- warp-only `__shfl_down_sync` max;
- warp-only exp + denominator sum;
- warp-only `__shfl_down_sync` sum;
- warp-only normalize.

No shared memory and no block-wide synchronization should be required for the packed path.

For `cols > 128`, V4 will fall back to the already validated V2/V3 path instead of changing wide-row behavior.

The first implementation will use 8 rows per 256-thread block for all `cols <= 128`; if this regresses 128-column rows, a later dispatch threshold can be derived from measured data rather than guessed.

Planned evidence:

- clean CUDA build and full pytest;
- boundary correctness at 31/32/33, 63/64/65, 127/128/129;
- Compute Sanitizer memcheck / racecheck / synccheck;
- ptxas resources;
- V3 versus V4 benchmark at 128 / 1024 / 4096 / 16384 rows.


### Operation 2 — GitHub implementation

Implemented V4 directly on GitHub.

Packed path for `cols <= 128`:

```text
256-thread block
  -> 8 warps
  -> warp 0 handles row 0
  -> warp 1 handles row 1
  -> ...
  -> warp 7 handles row 7
```

Each warp performs the entire row Softmax independently:

```text
lane-stride row max
  -> warp_reduce_max
  -> lane-0 broadcast
  -> lane-stride exp + local sum
  -> warp_reduce_sum
  -> lane-0 denominator broadcast
  -> lane-stride normalize
```

No shared memory or block-wide synchronization is used by the packed kernel.

Rows are mapped as:

```text
row = blockIdx.x * 8 + warp_id
blocks = ceil(rows / 8)
```

For `cols > 128`, V4 falls back to V3 so wide-row behavior is held constant.

Added:

- `cuda_operator_softmax_v4`;
- Python V4 binding;
- benchmark variant with `rows_per_block`;
- correctness coverage for row-count boundaries 7 / 8 / 9 and width boundaries 31 / 32 / 33, 63 / 64 / 65, 127 / 128 / 129.

### Operation 3 — clean hardware regression

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 229 passed
```

No local source edits were used.

### Operation 4 — Compute Sanitizer

Representative shapes exercise packed and fallback paths:

```text
[17,31]
[17,65]
[17,127]
[17,129]
[9,32]
```

Final CUDA 12.8 Compute Sanitizer:

```text
memcheck:  0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

Observed packed-path max absolute error stayed at or below approximately `1.49e-8`; row-sum error stayed below approximately `1.79e-7`.

Artifact:

- `reports/data/softmax_v4_compute_sanitizer.txt`

### Operation 5 — ptxas resource capture

CUDA 12.8 / SM 8.9:

```text
V4 warp-per-row:
  registers/thread: 30
  shared memory/block: 0 B
  barriers: 0
  spills: 0

V3 width-aware:
  registers/thread: 23
  shared memory/block: 32 B
  barriers: used
  spills: 0
```

The packed path removes all block-level scratch and barriers, at the cost of seven additional registers per thread.

Artifact:

- `reports/data/softmax_v4_ptxas_sm89.txt`

### Operation 6 — stable V3 vs V4 benchmark

Protocol:

- CUDA events;
- 20 warmups;
- 100 timed repeats;
- identical input per shape.

Results:

| Shape | V3 width-aware | V4 warp rows | V3 -> V4 | PyTorch |
|---:|---:|---:|---:|---:|
| 128 x 32 | 10.416 us | 10.368 us | ~1.00x | 8.576 us |
| 128 x 64 | 11.056 us | **10.624 us** | 1.04x | 8.192 us |
| 128 x 128 | **10.480 us** | 11.152 us | **0.94x** | 8.304 us |
| 1024 x 32 | 10.560 us | **10.240 us** | 1.03x | 8.192 us |
| 1024 x 64 | 10.496 us | **10.368 us** | 1.01x | 8.192 us |
| 1024 x 128 | 10.288 us | **10.240 us** | ~1.00x | 8.192 us |
| 4096 x 32 | 11.024 us | **10.240 us** | 1.08x | 8.016 us |
| 4096 x 64 | 11.264 us | **10.576 us** | 1.07x | 8.160 us |
| 4096 x 128 | 14.336 us | **11.136 us** | **1.29x** | 8.688 us |
| 16384 x 32 | 19.392 us | **13.952 us** | **1.39x** | 8.192 us |
| 16384 x 64 | 21.776 us | **15.104 us** | **1.44x** | 9.216 us |
| 16384 x 128 | 32.752 us | **19.136 us** | **1.71x** | 11.424 us |
| 1024 x 512 | 12.288 us | 12.288 us | 1.00x | 8.192 us |
| 1024 x 4096 | 38.912 us | 38.912 us | 1.00x | 37.328 us |

Artifact:

- `reports/data/softmax_v3_v4_warp_rows100_rtx4090.csv`

### Interpretation

The packing hypothesis is validated for high-row-count narrow Softmax.

The gain grows with row count because the packed kernel converts eight separate small row blocks into one 8-warp block while keeping all warps useful.

At `16384 x 128`, V4 improves V3 by about **1.71x**.

However, V4 is not universal:

```text
128 x 128:
V3 10.480 us
V4 11.152 us
```

That regression is important. Packing eight rows reduces the number of blocks and changes scheduling granularity; when row count is small, the saved block overhead does not always compensate.

### V4 conclusion

Warp-per-row packing is a strong execution layout for sufficiently many narrow rows, but needs a row-count/width-aware dispatcher.

### Next action

Softmax V5 should not invent a threshold.

Before coding the final dispatcher:

1. sweep row count around the crossover separately for widths 32 / 64 / 128;
2. compare V3 and V4 under the same timing protocol;
3. choose conservative row thresholds from stable data;
4. dispatch to V4 only above measured crossover;
5. keep V3 below threshold and keep the wide-row path unchanged.


---

## E12 — Softmax V5: empirical V3/V4 shape dispatcher

Status: **in progress**

### Hypothesis

V4 warp-per-row packing is clearly superior for high-row-count narrow matrices, but the 128 x 128 case regresses. Therefore the final default policy should dispatch between V3 and V4 from measured row-count/width crossover data rather than assuming all `cols <= 128` should use the packed path.

### Operation 1 — experiment branch and crossover plan

Created branch:

```text
feat/softmax-v5-shape-dispatch
```

Before changing code, sweep V3 versus V4 at widths:

```text
32, 64, 128
```

and row counts:

```text
128, 256, 512, 1024, 2048, 4096, 8192, 16384
```

Protocol:

- same RTX 4090 Laptop;
- float32;
- 20 warmups;
- 100 CUDA-event timed repeats;
- same input per shape;
- no source change during crossover measurement.

Only after the sweep will thresholds be encoded.


### Operation 2 — candidate dispatcher implementation

A candidate V5 dispatcher was implemented on GitHub before hardware acceptance:

```text
cols <= 64 and rows >= 4096
  -> V4 packed rows

65 <= cols <= 128 and rows >= 2048
  -> V4 packed rows

otherwise
  -> V3 width-aware path
```

The V3/V4 kernels themselves were not changed. V5 is a host-side dispatch policy over already validated device kernels.

Added:

- `cuda_operator_softmax_v5`;
- Python V5 binding;
- benchmark variant;
- threshold-boundary correctness tests.

The thresholds remained provisional until the crossover sweep below.

### Operation 3 — clean Tang validation

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 250 passed
```

No local source edits were used.

### Operation 4 — V3/V4 crossover sweep

Protocol:

- 20 warmups;
- 100 timed repeats;
- CUDA events;
- widths: 32 / 64 / 128;
- rows: 128 / 256 / 512 / 1024 / 2048 / 4096 / 8192 / 16384.

Selected results:

| Shape | V3 width-aware | V4 packed | Interpretation |
|---:|---:|---:|---|
| 128 x 128 | **10.240 us** | 10.432 us | packing can regress small row counts |
| 2048 x 32 | 10.240 us | 10.096 us | small gain, still near timer floor |
| 2048 x 64 | 10.240 us | 10.128 us | small gain, still near timer floor |
| 2048 x 128 | 11.104 us | **10.240 us** | clear packed-row win |
| 4096 x 32 | 10.240 us | 10.240 us | neutral |
| 4096 x 64 | 11.264 us | **10.368 us** | clear packed-row win |
| 4096 x 128 | 14.336 us | **10.960 us** | strong packed-row win |
| 8192 x 32 | 13.312 us | **10.240 us** | strong packed-row win |
| 8192 x 64 | 14.672 us | **11.264 us** | strong packed-row win |
| 8192 x 128 | 20.480 us | **13.312 us** | strong packed-row win |
| 16384 x 128 | 32.672 us | **18.432 us** | ~1.77x V3 -> V4 |

The sweep supports the candidate thresholds as conservative crossover points:

```text
32 / 64 cols:
  use packed rows from 4096 rows

65..128 cols:
  use packed rows from 2048 rows
```

For smaller shapes V4 can sometimes be slightly faster, but the differences are close to the low-microsecond timing floor and are not robust enough to justify more aggressive default dispatch.

Artifact:

- `reports/data/softmax_v3_v4_crossover100_revalidate_rtx4090.csv`

### Operation 5 — benchmark metadata bug discovered and fixed

The first final V5 benchmark exposed a reporting bug in `benchmarks/softmax_benchmark.py`.

The V5 fallback path correctly executed V3, but the CSV metadata reported:

```text
threads_per_block = 256
```

instead of the actual V3 width-aware 32 / 64 / 128-thread launch.

This did **not** affect kernel execution or timing; it only made the benchmark metadata inaccurate.

GitHub fix:

- make V5 fallback inherit V3's width-aware `threads_per_block`;
- remove a duplicate `v5_uses_packed` metadata assignment.

After the fix:

```text
2047 x 128 V5 -> threads=128, rows/block=1
2048 x 128 V5 -> threads=256, rows/block=8

4095 x 32 V5 -> threads=32, rows/block=1
4096 x 32 V5 -> threads=256, rows/block=8

4095 x 64 V5 -> threads=64, rows/block=1
4096 x 64 V5 -> threads=256, rows/block=8
```

### Operation 6 — final dispatcher-boundary benchmark

Protocol:

- 20 warmups;
- 100 timed repeats;
- V3 / V4 / V5 on identical inputs.

Selected boundary results:

| Shape | V3 | V4 | V5 | V5 path |
|---:|---:|---:|---:|---|
| 2047 x 128 | 11.232 us | 10.880 us | 11.216 us | V3 |
| 2048 x 128 | 11.264 us | 10.528 us | **10.592 us** | V4 |
| 2049 x 128 | 11.200 us | 10.384 us | **10.304 us** | V4 |
| 4095 x 32 | 10.464 us | 10.240 us | 10.432 us | V3 |
| 4096 x 32 | 10.368 us | 10.240 us | **10.240 us** | V4 |
| 4097 x 32 | 10.384 us | 10.240 us | **10.240 us** | V4 |
| 4095 x 64 | 11.264 us | 10.656 us | 11.264 us | V3 |
| 4096 x 64 | 11.264 us | 10.432 us | **10.496 us** | V4 |
| 4097 x 64 | 11.264 us | 10.528 us | **10.576 us** | V4 |
| 4096 x 128 | 14.336 us | 10.912 us | **10.912 us** | V4 |
| 16384 x 128 | 32.704 us | 18.432 us | **18.528 us** | V4 |

Wide rows remain unchanged:

```text
1024 x 512:
  V3 12.288 us
  V5 12.288 us
```

Artifact:

- `reports/data/softmax_v3_v4_v5_final100_revalidate_rtx4090.csv`

### Operation 7 — dispatcher sanitizer validation

Representative shapes intentionally cover both sides of both crossover thresholds and the wide fallback:

```text
4095 x 32   -> V3
4096 x 32   -> V4
2047 x 128  -> V3
2048 x 128  -> V4
1024 x 512  -> wide V3
```

CUDA 12.8 Compute Sanitizer:

```text
memcheck:
  0 errors

racecheck:
  0 hazards / 0 errors / 0 warnings

synccheck:
  0 errors
```

Maximum observed element error was at most approximately `5.96e-8`; maximum row-sum error was at most approximately `2.38e-7`.

Artifact:

- `reports/data/softmax_v5_compute_sanitizer.txt`

### Operation 8 — final regression check

After the benchmark metadata fix:

```text
full repository pytest: 250 passed
Python compileall for python/ + benchmarks/: PASS
working tree: clean
```

### V5 conclusion

The final Softmax narrow-row dispatcher is:

```text
cols <= 64 and rows >= 4096
  -> V4: 8 warp-owned rows per 256-thread block

65 <= cols <= 128 and rows >= 2048
  -> V4 packed path

otherwise
  -> V3 width-aware one-row-per-block path
```

For `cols > 128`, V5 intentionally stays on the V3 wide-row path.

V5 adds no new device kernel, so there is no new ptxas resource footprint: it dispatches between the already recorded V3 and V4 kernels.

The important engineering outcome is not just a faster path but a measured **execution-layout dispatcher**: row packing is enabled only when row count is high enough for the scheduling benefit to be robust.

Status: **validated; ready to merge**.

### Next action

Softmax now has a complete optimization story from serial row work to block parallelism, warp reduction, negative thread-count tuning, row packing, and empirical shape dispatch.

The next operator should be **RMSNorm**, starting from a simple correct baseline and reusing the same operation ledger / sanitizer / benchmark discipline.


---

## E13 — RMSNorm V0: serial row baseline

Status: **in progress**

### Hypothesis

RMSNorm introduces a different normalization pattern from Softmax:

```text
mean_square = mean(x_i^2)
inverse_rms = rsqrt(mean_square + eps)
y_i = x_i * inverse_rms * weight_i
```

A deliberately simple one-thread-per-row implementation should provide a clean correctness and performance baseline before adding cooperative row reductions.

### Operation 1 — GitHub implementation

Created branch:

```text
feat/rmsnorm-v0-baseline
```

Implemented V0 entirely on GitHub:

- `csrc/rmsnorm/rmsnorm.cu` / `.cuh`;
- CMake integration;
- PyTorch reference implementation;
- ctypes binding on the active PyTorch CUDA stream;
- correctness tests;
- CUDA-event benchmark harness.

V0 execution:

```text
one CUDA thread owns one row
  -> serial sum of squares
  -> mean square
  -> rsqrt(mean_square + eps)
  -> serial x * inverse_rms * weight
```

The initial scope is float32, 2-D contiguous input `[rows, cols]`, 1-D contiguous weight `[cols]`, and positive epsilon.

No shared-memory reduction, warp shuffle, vectorized load, or shape dispatch is included in V0.

### Planned hardware validation

Tang is currently online. V0 will not be merged until completing:

- clean CUDA 12.8 / SM 8.9 build;
- full repository pytest;
- RMSNorm benchmark across narrow and LLM-style hidden sizes;
- Compute Sanitizer memcheck / racecheck / synccheck;
- ptxas resource capture;
- raw artifacts and conclusions written back to this ledger.


### Operation 2 — clean Tang build and full regression

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA compiler: 12.8.93
target: sm_89
clean build: PASS
full repository pytest: 268 passed
```

No source files were edited locally on Tang.

### Operation 3 — RMSNorm V0 baseline benchmark

Protocol:

- CUDA events;
- 10 warmups;
- 50 timed repeats;
- float32;
- eps = 1e-5;
- same input / weight used for V0 and PyTorch reference.

Representative results:

| Shape | V0 serial row | PyTorch reference | Slowdown |
|---:|---:|---:|---:|
| 1 x 128 | 15.680 us | 23.552 us | 0.67x |
| 128 x 128 | 40.960 us | 24.576 us | 1.67x |
| 128 x 512 | 145.104 us | 24.576 us | 5.90x |
| 128 x 1024 | 270.336 us | 24.576 us | 11.00x |
| 128 x 4096 | 1,035.216 us | 27.648 us | **37.44x** |
| 1024 x 4096 | 1,046.528 us | 61.488 us | 17.02x |
| 128 x 8192 | 1,691.936 us | 26.624 us | **63.55x** |

Maximum recorded absolute error in the benchmark matrix was approximately `1.72e-5`.

The width trend is clear: rows are already parallel, but the sum-of-squares and output passes inside each row are completely serial.

Artifact:

- `reports/data/rmsnorm_v0_rtx4090_laptop.csv`

### Operation 4 — ptxas resource capture

CUDA 12.8 / SM 8.9:

```text
registers/thread: 20
shared memory/block: 0 B
spills: 0
barriers: 0
stack frame: 0 B
```

Artifact:

- `reports/data/rmsnorm_v0_ptxas_sm89.txt`

### Operation 5 — Compute Sanitizer

Representative non-power-of-two hidden size:

```text
shape: [17, 4097]
dtype: float32
eps: 1e-5
```

CUDA 12.8 Compute Sanitizer:

```text
memcheck:  0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

Representative max absolute error: `7.15e-6`.

Artifact:

- `reports/data/rmsnorm_v0_compute_sanitizer.txt`

### V0 conclusion

The baseline exposes the intended bottleneck:

```text
one thread per row
  -> serial sum(x^2) over hidden width
  -> serial normalize * weight over hidden width
```

The next isolated experiment should parallelize work **inside each row** without adding warp shuffle or vectorized loads yet.

### Next action

RMSNorm V1:

```text
one block per row
  -> thread-local sum of squares over strided columns
  -> shared-memory sum reduction
  -> inverse RMS
  -> parallel normalize * weight
```

This mirrors the project's profile-guided discipline: isolate intra-row parallelism first, then measure the remaining coordination cost.

Status: **validated; ready to merge**.


---

## E14 — RMSNorm V1: one block per row + shared-memory sum-of-squares reduction

Status: **in progress**

### Hypothesis

RMSNorm V0 is row-parallel but hidden-width serial. The first isolated optimization is to parallelize only the per-row hidden dimension.

### Scope lock

V1 will keep:

- float32 input / weight / output;
- same RMSNorm math and epsilon;
- one row as the unit of normalization;
- scalar loads/stores;
- no warp shuffle;
- no vectorized IO;
- no shape dispatch.

V1 changes only the row execution layout:

```text
one 256-thread block per row
  -> thread-local sum(x^2) over strided columns
  -> shared-memory tree sum
  -> inverse RMS
  -> parallel x * inverse_rms * weight
```

Planned evidence:

- clean build and full pytest;
- V0/V1/PyTorch benchmark;
- memcheck / racecheck / synccheck;
- ptxas resource capture;
- raw results and all issues recorded before merge.


### Operation 2 — clean Tang build and regression

Tang fetched the GitHub implementation and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 290 passed
```

### Operation 3 — V0/V1 benchmark

Protocol:

- CUDA events;
- 10 warmups;
- 50 timed repeats;
- float32;
- eps = 1e-5;
- same input / weight for both variants and the PyTorch expression reference.

Representative results:

| Shape | V0 | V1 | V0 -> V1 | PyTorch expression |
|---:|---:|---:|---:|---:|
| 128 x 128 | 41.424 us | **11.264 us** | 3.68x | 24.912 us |
| 128 x 512 | 144.560 us | **10.272 us** | 14.07x | 24.576 us |
| 128 x 1024 | 279.552 us | **10.464 us** | 26.72x | 24.480 us |
| 128 x 4096 | 1,037.120 us | **14.080 us** | **73.66x** | 27.456 us |
| 1024 x 4096 | 868.352 us | **27.440 us** | 31.65x | 62.144 us |
| 128 x 8192 | 1,692.320 us | **17.408 us** | **97.22x** | 26.624 us |

Important interpretation note:

The repository's PyTorch RMSNorm reference is intentionally expressed as separate PyTorch operations:

```python
x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight
```

It is a trusted correctness reference, **not an optimized fused vendor RMSNorm kernel**. Therefore V1 being faster than that reference must not be presented as outperforming an optimized framework/vendor RMSNorm implementation.

Artifact:

- `reports/data/rmsnorm_v0_v1_rtx4090_laptop.csv`

### Operation 4 — ptxas resource capture

CUDA 12.8 / SM 8.9:

```text
V0:
  registers/thread: 20
  shared memory/block: 0 B
  barriers: 0
  spills: 0

V1:
  registers/thread: 18
  shared memory/block: 1,024 B
  barrier resource: used
  spills: 0
```

Artifact:

- `reports/data/rmsnorm_v1_ptxas_sm89.txt`

### Operation 5 — Compute Sanitizer

Representative non-power-of-two hidden size: `17 x 4097`.

```text
memcheck:  0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

Artifact:

- `reports/data/rmsnorm_v1_compute_sanitizer.txt`

### V1 conclusion

The V0 hidden-width bottleneck is decisively removed by intra-row block parallelism.

The remaining explicit coordination cost is now the 256-entry shared-memory tree:

```text
256 partial square sums
  -> 128
  -> 64
  -> 32
  -> ...
  -> 1
```

with a block barrier at each reduction stage.

### Next action

RMSNorm V2 should keep one block per row and all scalar input/output passes unchanged, but replace the full shared-memory reduction tree with a two-level warp-shuffle sum reduction.

This will isolate reduction coordination overhead before any float4/vectorized IO experiment.

Status: **validated; ready to merge**.


---

## E15 — RMSNorm V2: warp-shuffle sum-of-squares reduction

Status: **in progress**

### Hypothesis

RMSNorm V1 removes the serial hidden-width bottleneck, leaving the full 256-thread shared-memory tree as the next isolated coordination cost.

### Scope lock

V2 keeps:

- one 256-thread block per row;
- scalar input / weight / output access;
- thread-local strided sum of squares;
- identical RMSNorm math and epsilon;
- no vectorized IO;
- no shape dispatch.

Only the block reduction changes:

```text
V1:
256 shared partials + repeated __syncthreads()

V2:
warp shuffle within 8 warps
  -> 8 shared warp sums
  -> first warp final shuffle
```

The experiment will measure whether the coordination reduction matters after V1.


### Operation 2 — clean Tang validation

```text
clean CUDA 12.8 / sm_89 build: PASS
full repository pytest: 313 passed
```

### Operation 3 — stable V0/V1/V2 benchmark

Protocol:

- 20 warmups;
- 100 timed repeats;
- CUDA events;
- float32;
- eps = 1e-5.

Selected V1 -> V2 results:

| Shape | V1 shared tree | V2 warp shuffle | V1 -> V2 |
|---:|---:|---:|---:|
| 128 x 128 | 11.248 us | 11.152 us | 1.01x |
| 128 x 512 | **10.560 us** | 11.056 us | 0.96x |
| 128 x 1024 | 10.784 us | **10.400 us** | 1.04x |
| 128 x 4096 | 14.160 us | **13.312 us** | 1.06x |
| 1024 x 512 | 11.904 us | **11.200 us** | 1.06x |
| 1024 x 4096 | 27.232 us | **26.512 us** | 1.03x |
| 2048 x 4096 | 58.368 us | **52.688 us** | **1.11x** |
| 128 x 8192 | 17.408 us | **17.248 us** | 1.01x |

The result is shape-dependent rather than universally positive. V2 helps when enough row blocks / reduction work are present, but the 128 x 512 case slightly regresses.

Artifact:

- `reports/data/rmsnorm_v0_v1_v2_rtx4090_laptop.csv`

### Operation 4 — ptxas

```text
V1:
  registers/thread: 18
  shared memory/block: 1024 B
  spills: 0

V2:
  registers/thread: 17
  shared memory/block: 32 B
  spills: 0
```

V2 reduces shared-memory footprint by 32x and one register/thread.

Artifact:

- `reports/data/rmsnorm_v2_ptxas_sm89.txt`

### Operation 5 — Compute Sanitizer

Representative shape: `17 x 4097`.

```text
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
max_abs_error: 9.54e-7
```

Artifact:

- `reports/data/rmsnorm_v2_compute_sanitizer.txt`

### V2 conclusion

Warp shuffle removes almost all block scratch, but after V1 the reduction tree is not the only remaining cost. The largest gain in this matrix is about 11% at `2048 x 4096`, while some smaller shapes are neutral or slightly worse.

The next experiment should target the input/output path, not further reduction-tree tuning.

### Next action

RMSNorm V3 should keep the V2 warp reduction and add aligned vectorized input/weight/output handling, with safe scalar fallback and generated-code verification.

Status: **validated; ready to merge**.

## E16 — RMSNorm V3: aligned float4 IO

Status: **validated experimental result; not a universal default**

### Hypothesis

After V2 reduced coordination overhead, the next target was global memory traffic. V3 keeps the V2 reduction structure and replaces scalar aligned IO with `float4` loads/stores when safe.

### Implementation

Branch:

`feat/rmsnorm-v3-float4-io`

Changes:

- aligned `float4` input loads for sum-of-squares;
- aligned `float4` weight loads and output stores;
- V2 scalar fallback for non-multiple-of-four or unaligned tensors;
- Python binding, benchmark variant, and V3 tests.

### Validation

- clean CUDA 12.8 / SM 8.9 build: PASS;
- **337 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors.

ptxas:

```text
V2:
  17 registers/thread
  32 B shared memory/block
  0 spills

V3:
  22 registers/thread
  32 B shared memory/block
  0 spills
```

Artifacts:

- `reports/data/rmsnorm_v2_v3_float4_rtx4090.csv`
- `reports/data/rmsnorm_v2_v3_crossover100_rtx4090.csv`
- `reports/data/rmsnorm_v3_ptxas_sm89.txt`

### Warm-cache result

Representative wins:

```text
128 x 4096:  13.312 -> 10.240 us
128 x 8192:  17.200 -> 12.032 us
1024 x 4096: 26.512 -> 21.472 us
```

But repeated tests revealed non-stable crossover behavior. In particular, 2048 x 4096 alternated between apparent V3 wins and regressions depending on benchmark context.

### L2-evicted follow-up

A 64 MiB CUDA buffer was touched before each timed kernel launch to evict L2 state. Variants were measured in alternating order.

Selected results:

```text
1024 x 512:
  V2 13.352 us
  V3 16.384 us
  V3 regresses

1024 x 1024:
  V2 25.576 us
  V3 23.552 us
  V3 ~1.09x

128 x 4096:
  V2 17.408 us
  V3 16.168 us
  V3 ~1.08x

512 x 4096:
  V2 47.104 us
  V3 43.712 us
  V3 ~1.08x

128 x 8192:
  V2 34.344 us
  V3 23.416 us
  V3 ~1.47x

1024 x 8192:
  V2 151.520 us
  V3 147.752 us
  V3 ~1.03x
```

The important result is that float4 benefits are shape- and cache-state-dependent. The larger warm-cache gains cannot be treated as universal.

---

## E17 — RMSNorm V4: profile-guided V2/V3 dispatcher candidate

Status: **correctness validated; performance policy not accepted yet**

Branch:

`feat/rmsnorm-v4-shape-dispatch`

V4 introduces no new device kernel. It selects V2 or V3 from a measured shape profile.

The initial warm-cache policy was:

```text
512/1024 cols: vectorize at larger row counts
4096 cols: vectorize below a measured crossover
8192 cols: vectorize below a measured crossover
```

After L2-evicted measurements, this was replaced with a conservative cold-cache profile.

Validation after the policy revision:

- clean build: PASS;
- **355 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors / 0 warnings;
- synccheck: 0 errors.

Artifact:

- `reports/data/rmsnorm_v2_v3_v4_final100_rtx4090.csv`

### Why V4 is not merged

The final benchmark still showed timing instability around crossover shapes. The same underlying device kernel could show materially different microsecond timings depending on benchmark order and whether it was called directly or through the dispatcher symbol.

Therefore the current branch is intentionally kept as a candidate rather than merged into the default path.

This is recorded as a positive engineering result: the project rejects a benchmark-derived optimization when its performance conclusion is not yet robust enough.

### Next action

Before accepting V4:

1. stabilize GPU power/clock state where possible;
2. use L2 eviction for memory-path comparisons;
3. randomize or alternate V2/V3/V4 measurement order;
4. run multiple independent rounds per shape;
5. report median plus dispersion/confidence interval;
6. only encode profile entries that remain clearly separated across runs.

### E17 Operation — stable interleaved profiling

The earlier V4 threshold experiment exposed a measurement flaw: each variant was executed in a contiguous timing group. GPU dynamic clocks and thermal/power state could therefore correlate with variant order.

A new harness was added:

`benchmarks/rmsnorm_stable_profile.py`

Method:

1. allocate a 64 MiB CUDA flush buffer;
2. touch the buffer before every timed kernel launch;
3. interleave V2/V3/V4 on every sample;
4. alter variant order across samples;
5. collect 7 independent round medians, 80 launches per variant per round;
6. repeat with a second independent seed.

A profile generator was also added:

`benchmarks/generate_rmsnorm_dispatch_profile.py`

Acceptance rule:

```text
choose V3 only when direct V2/V3 speedup >= 1.05x
in every independent profiling run
```

Accepted shapes:

| Shape | Seed 1 V2/V3 | Seed 2 V2/V3 | Decision |
|---:|---:|---:|---|
| 1024 x 512 | 1.231x | 1.271x | V3 |
| 512 x 4096 | 1.170x | 1.181x | V3 |
| 128 x 8192 | 1.435x | 1.435x | V3 |
| 512 x 8192 | 1.051x | 1.051x | V3 |

Examples rejected by the 5% gate:

| Shape | Seed 1 | Seed 2 | Decision |
|---:|---:|---:|---|
| 1024 x 1024 | 1.043x | 1.029x | V2 |
| 1024 x 4096 | 1.042x | 1.039x | V2 |
| 1024 x 8192 | 0.999x | 0.997x | V2 |
| 1536 x 4096 | 0.975x | 0.972x | V2 |
| 1280 x 8192 | 0.960x | 0.957x | V2 |

The V4 implementation was updated to exactly match the generated conservative profile. It no longer extrapolates row-count thresholds beyond measured evidence.

A third independent run on the whitelist/fallback set preserved the same direct V2/V3 decision direction.

Current status:

- implementation correctness: PASS;
- full repository suite: **355 passed**;
- profiling methodology: stabilized enough for a hardware-specific static profile;
- policy scope: RTX 4090 Laptop measured profile only;
- unlisted shapes: V2 fallback.

## E18 — LayerNorm V0: serial row baseline

Status: **implementation complete; awaiting Tang hardware validation**

Branch:

`feat/layernorm-v0-baseline`

V0 scope:

- float32;
- contiguous 2-D input `[rows, cols]`;
- 1-D weight and bias `[cols]`;
- positive epsilon;
- one CUDA thread per row;
- three serial passes: mean, variance, affine output;
- active PyTorch CUDA stream propagated through the C ABI.

Reference: PyTorch `torch.nn.functional.layer_norm`.

Next validation gate:

- clean CUDA 12.8 / SM 8.9 build;
- full pytest;
- benchmark versus PyTorch LayerNorm;
- ptxas resource capture;
- Compute Sanitizer memcheck/racecheck/synccheck.

### E18 Operation 2 — Tang hardware validation

Tang fetched the GitHub branch and rebuilt from scratch.

```text
CUDA 12.8 / sm_89 build: PASS
full repository pytest: 373 passed
```

### Operation 3 — baseline benchmark

Protocol:

- 20 warmups;
- 100 timed repeats;
- CUDA events;
- PyTorch `torch.nn.functional.layer_norm` reference.

Selected results:

| Shape | V0 serial row | PyTorch | Slowdown |
|---:|---:|---:|---:|
| 128 x 128 | 51.376 us | 10.496 us | 4.89x |
| 128 x 512 | 185.344 us | 10.464 us | 17.71x |
| 128 x 1024 | 356.096 us | 10.592 us | 33.62x |
| 128 x 4096 | 1317.088 us | 11.264 us | 116.93x |
| 1024 x 4096 | 1086.416 us | 24.576 us | 44.21x |
| 128 x 8192 | 2161.056 us | 15.360 us | 140.69x |

Artifact:

- `reports/data/layernorm_v0_rtx4090.csv`

### Operation 4 — ptxas

```text
registers/thread: 24
shared memory/block: 0 B
spills: 0
barriers: 0
```

Artifact:

- `reports/data/layernorm_v0_ptxas_sm89.txt`

### Operation 5 — Compute Sanitizer

Representative shape: `17 x 4097`.

```text
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
max_abs_error: 1.14e-5
```

### V0 conclusion

The baseline intentionally exposes the cost of three serial row passes. The next isolated experiment is V1: one block per row with shared-memory reductions for mean and variance, followed by parallel affine output.

Status: **validated; ready to merge**.

## E19 — LayerNorm V1: block-parallel mean/variance reductions

Status: **validated; ready to merge**

### Hypothesis

LayerNorm V0 is dominated by serial hidden-width work. V1 parallelizes only the row dimension internals while holding the math and scalar IO constant.

### Design

```text
one 256-thread block per row
  -> strided local sum
  -> shared-memory tree -> mean
  -> barrier before shared-buffer reuse
  -> strided local squared deviation
  -> shared-memory tree -> variance
  -> inverse std
  -> parallel affine output
```

The explicit barrier after all threads load the mean preserves the same shared-buffer reuse safety lesson learned from Softmax V1.

### Validation

```text
clean CUDA 12.8 / sm_89 build: PASS
full pytest: 396 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

ptxas:

```text
V0:
  24 registers/thread
  0 B shared memory

V1:
  21 registers/thread
  1024 B shared memory/block
  0 spills
```

Selected timings:

| Shape | V0 | V1 | PyTorch |
|---:|---:|---:|---:|
| 128 x 512 | 189.408 us | 10.560 us | 10.528 us |
| 128 x 1024 | 355.088 us | 11.264 us | 10.624 us |
| 128 x 4096 | 1316.896 us | 14.384 us | 13.056 us |
| 1024 x 4096 | 1084.416 us | 34.736 us | 24.576 us |
| 128 x 8192 | 2162.784 us | 20.144 us | 15.360 us |

Artifacts:

- `reports/data/layernorm_v0_v1_rtx4090.csv`
- `reports/data/layernorm_v1_ptxas_sm89.txt`

### Next action

V2 keeps the two-pass mean/variance algorithm and scalar IO, but replaces both 256-entry shared-memory trees with two-level warp-shuffle reductions. This isolates synchronization/reduction overhead before introducing Welford or vectorized IO.

## E20 — LayerNorm V2: warp-shuffle mean/variance reductions

Status: **validated; ready to merge**

### Scope lock

V2 keeps:

- one 256-thread block per row;
- scalar input / weight / bias / output access;
- separate mean and variance passes;
- identical LayerNorm math and epsilon.

Only reduction coordination changes:

```text
V1:
256 shared partials + repeated block barriers

V2:
warp-local shuffle
  -> 8 shared warp partials
  -> first-warp final shuffle
```

The barrier protecting shared-buffer reuse after all threads consume the mean is retained.

### Validation

```text
clean build: PASS
full pytest: 419 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

ptxas:

```text
V1:
  21 registers/thread
  1024 B shared memory/block

V2:
  23 registers/thread
  32 B shared memory/block
  0 spills
```

Selected timings:

| Shape | V1 | V2 | PyTorch |
|---:|---:|---:|---:|
| 128 x 512 | 10.528 us | 10.272 us | 10.240 us |
| 128 x 1024 | 11.264 us | 10.944 us | 10.240 us |
| 1024 x 128 | 14.336 us | 11.264 us | 11.008 us |
| 1024 x 512 | 15.360 us | 12.288 us | 11.264 us |
| 1024 x 4096 | 34.816 us | 31.856 us | 25.248 us |
| 128 x 8192 | 19.712 us | 19.360 us | 15.360 us |

Artifacts:

- `reports/data/layernorm_v0_v1_v2_rtx4090.csv`
- `reports/data/layernorm_v2_ptxas_sm89.txt`

### Interpretation

Warp shuffle materially helps high-row-count workloads and nearly closes the gap on several small/medium shapes. The wide-row cases retain noticeable headroom, but their reduction-tree cost is now small relative to reading data and computing both statistics.

### Next action

LayerNorm V3 should test Welford online statistics so mean and variance can be accumulated together in one numerically stable pass before considering float4/vectorized IO.

## E21 — LayerNorm V3: Welford online statistics

Status: **validated experimental result; not promoted as the fastest path**

### Hypothesis

V2 reads each row once for mean and again for variance. Welford can accumulate `count / mean / M2` in one statistically stable pass, potentially reducing one full input traversal and one reduction phase.

### First implementation

A float32 Welford state was implemented per thread and merged across lanes/warps using the standard parallel combine formula.

Ordinary correctness passed, but a dedicated large-offset stability test:

```text
x = 1000 + N(0, 0.1)
shape = 64 x 4096
```

showed that using PyTorch float32 output as the exact truth was itself an unsuitable stability criterion.

### Reference correction

The stability test was changed to use a float64 LayerNorm computation as the numerical truth.

Against that truth:

| Variant | max abs | mean abs | p99 abs |
|---|---:|---:|---:|
| PyTorch float32 | 2.390e-3 | 2.017e-4 | 1.076e-3 |
| V2 two-pass float32 | 2.543e-3 | 2.130e-4 | 1.147e-3 |
| V3 float Welford | 2.349e-3 | 1.980e-4 | 1.063e-3 |
| double-Welford experiment | 1.011e-3 | 1.214e-4 | 5.612e-4 |

The double version was measurably more accurate, confirming the numerical direction.

### Double-Welford negative experiment

Changing Welford state to double precision increased resources to:

```text
48 registers/thread
160 B shared memory/block
0 spills
```

and caused severe regressions:

```text
128 x 4096: 14.240 us -> 54.272 us
1024 x 4096: 31.744 us -> 318.320 us
128 x 8192: 19.456 us -> 83.792 us
```

Therefore the double path was rejected.

### Final float32 Welford validation

```text
full pytest: 443 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
ptxas: 30 registers/thread, 96 B shared memory, 0 spills
```

Selected V2 vs V3 float Welford:

| Shape | V2 | V3 |
|---:|---:|---:|
| 128 x 512 | 11.264 us | 11.264 us |
| 128 x 4096 | 14.336 us | 14.336 us |
| 1024 x 512 | 12.768 us | 14.336 us |
| 1024 x 4096 | 31.744 us | 31.552 us |
| 128 x 8192 | 19.456 us | 19.744 us |

Artifacts:

- `reports/data/layernorm_v0_v1_v2_v3_float_rtx4090.csv`
- `reports/data/layernorm_v3_float_ptxas_sm89.txt`
- `reports/data/layernorm_v3_numeric_stability.csv`

### Decision

Keep float32 Welford as a valid algorithmic/numerical experiment, but do not treat it as the default fastest LayerNorm implementation.

Next performance experiment: preserve V2 two-pass warp-shuffle statistics and vectorize aligned input/weight/bias/output traffic with `float4`.

## E22 — LayerNorm V4: aligned float4 IO

Status: **validated experimental fast path; shape-dependent**

### Hypothesis

After V2, reduction coordination is small. Wide rows still require three large memory passes:

```text
input -> mean
input -> variance
input + weight + bias -> output
```

V4 keeps V2's math and reductions but vectorizes aligned IO with `float4`.

### Implementation

Fast path requires:

```text
cols % 4 == 0
input / weight / bias / output 16-byte aligned
```

Otherwise V4 falls back to V2.

### Validation

```text
clean build: PASS
full pytest: 467 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

ptxas:

```text
V2:
  23 registers/thread
  32 B shared memory/block

V4:
  26 registers/thread
  32 B shared memory/block
  0 spills
```

### Warm benchmark

Selected results:

| Shape | V2 | V4 | V2 -> V4 |
|---:|---:|---:|---:|
| 128 x 4096 | 14.304 us | 11.264 us | 1.27x |
| 1024 x 4096 | 31.840 us | 25.456 us | 1.25x |
| 128 x 8192 | 19.456 us | 13.312 us | 1.46x |
| 1024 x 512 | 12.512 us | 11.888 us | 1.05x |

### Stable cold-cache methodology

A dedicated profiler was added:

`benchmarks/layernorm_stable_profile.py`

Method:

1. touch a 64 MiB CUDA buffer before every timed launch;
2. interleave V2 and V4 at the individual-sample level;
3. vary order across samples;
4. run 7 independent rounds x 80 timed launches;
5. repeat with independent seeds.

### Stable results

Two independent seeds on the initial matrix:

```text
128 x 4096: 1.407x / 1.379x
128 x 8192: 1.423x / 1.423x
1024 x 512: 1.118x / 1.109x
1024 x 1024: 1.017x / 1.052x
1024 x 4096: 1.023x / 1.022x
2048 x 4096: ~1.00x
```

Additional two-seed sweep:

```text
256 x 4096: 1.253x / 1.257x
512 x 4096: 1.181x / 1.184x
256 x 8192: 1.178x / 1.180x
512 x 8192: 1.157x / 1.156x
1536 x 512: 1.059x / 1.068x
2048 x 512: 1.037x / 1.007x
1536 x 1024: 1.065x / 1.053x
2048 x 1024: 0.932x / 0.925x
```

### Decision

V4 is a real fast path, but not a universal replacement for V2.

A V5 dispatcher should enable V4 only for shapes that pass a conservative repeated cold-cache acceptance gate and use V2 elsewhere.

Artifacts:

- `reports/data/layernorm_v0_v1_v2_v3_v4_rtx4090.csv`
- `reports/data/layernorm_v4_ptxas_sm89.txt`
- `reports/data/layernorm_stable_profile_seed1_rtx4090.csv`
- `reports/data/layernorm_stable_profile_seed2_rtx4090.csv`
- `reports/data/layernorm_stable_profile_extra_20261007_rtx4090.csv`
- `reports/data/layernorm_stable_profile_extra_20261008_rtx4090.csv`

## E23 — LayerNorm V5: profile-guided V2/V4 dispatcher

Status: **validated; final LayerNorm policy**

### Profile generation

Added:

`benchmarks/generate_layernorm_dispatch_profile.py`

Inputs are stable profiling CSVs from independent seeds. A shape is accepted for V4 only when:

```text
number of independent runs >= 2
AND
V2 / V4 speedup >= 1.05
in every run
```

Generated accepted shapes:

```text
128 x 4096
256 x 4096
512 x 4096

128 x 8192
256 x 8192
512 x 8192

1024 x 512
1536 x 512
1536 x 1024
```

Examples rejected:

```text
1024 x 4096: ~1.02x only
2048 x 4096: ~1.00x
1024 x 1024: one run below 1.05x
2048 x 512: one run near 1.00x
2048 x 1024: V4 regresses
```

### Implementation

V5 contains no new kernel body.

For accepted aligned shapes:

```text
V5 -> V4 float4 kernel
```

Otherwise:

```text
V5 -> V2 scalar warp-shuffle kernel
```

Unaligned pointers also fall back to V2.

### Validation

```text
clean build: PASS
full pytest: 484 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

Final interleaved cold-cache benchmark:

| Shape | V2 | V4 | V5 | V5 path |
|---:|---:|---:|---:|---|
| 128 x 4096 | 24.384 | 17.104 | 17.344 | V4 |
| 256 x 4096 | 29.696 | 23.552 | 23.552 | V4 |
| 512 x 4096 | 51.200 | 45.056 | 44.288 | V4 |
| 1024 x 4096 | 79.968 | 82.480 | 79.872 | V2 |
| 128 x 8192 | 37.888 | 26.016 | 25.904 | V4 |
| 512 x 8192 | 95.232 | 83.968 | 82.912 | V4 |
| 1024 x 8192 | 156.672 | 146.368 | 155.824 | V2 |
| 1024 x 512 | 18.432 | 16.544 | 17.056 | V4 |
| 1536 x 512 | 21.264 | 19.696 | 19.456 | V4 |
| 2048 x 512 | 24.576 | 23.520 | 24.592 | V2 |
| 1536 x 1024 | 34.032 | 31.744 | 31.728 | V4 |
| 2048 x 1024 | 42.304 | 43.824 | 41.920 | V2 |

Artifacts:

- `reports/data/layernorm_dispatch_profile_rtx4090.csv`
- `reports/data/layernorm_v5_final_cold_rtx4090.csv`

### LayerNorm conclusion

The final LayerNorm sequence is:

```text
V0 serial row
-> V1 block-parallel shared reductions
-> V2 warp-shuffle reductions
-> V3 Welford numerical-method experiment
-> V4 float4 memory fast path
-> V5 profile-guided dispatch
```

The central result is not a single "best kernel" but a measured optimization process: each change is isolated, validated, profiled under cache-aware methodology, and promoted only when repeated evidence supports it.

## E24 — Fused Residual + LayerNorm V0

Status: **implementation complete; awaiting Tang validation**

Scope:

- float32;
- contiguous x/residual [rows, cols];
- contiguous weight/bias [cols];
- one thread per row;
- residual addition recomputed inside mean, variance, and affine passes;
- no intermediate residual-add tensor;
- active PyTorch CUDA stream.

Reference comparison:

```python
torch.nn.functional.layer_norm(
    x + residual,
    ...
)
```

Next validation:
- clean CUDA build;
- full pytest;
- benchmark against unfused PyTorch add + LayerNorm;
- ptxas;
- Compute Sanitizer.

### E24 Operation 2 — Tang hardware validation

```text
clean CUDA 12.8 / sm_89 build: PASS
full pytest: 499 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

ptxas:

```text
26 registers/thread
0 B shared memory
0 spills
```

Selected benchmark results versus unfused PyTorch add + LayerNorm:

| Shape | Fused V0 | PyTorch unfused | Ratio |
|---:|---:|---:|---:|
| 128 x 128 | 90.112 us | 13.376 us | 6.74x slower |
| 128 x 512 | 331.616 us | 13.120 us | 25.28x slower |
| 128 x 1024 | 580.608 us | 13.264 us | 43.77x slower |
| 128 x 4096 | 1874.432 us | 14.336 us | 130.75x slower |
| 1024 x 4096 | 1882.112 us | 40.944 us | 45.97x slower |
| 128 x 8192 | 3895.296 us | 19.136 us | 203.56x slower |

Artifacts:

- `reports/data/fused_residual_layernorm_v0_rtx4090.csv`
- `reports/data/fused_residual_layernorm_v0_ptxas_sm89.txt`

### V0 conclusion

Fusion alone is not a performance win when the fused kernel serializes hidden-width work.

Next action: V1 should use one 256-thread block per row, shared-memory mean/variance reductions, and parallel affine output while keeping residual addition fused into all three passes.

## E25 — Fused Residual + LayerNorm V1: block-parallel reductions

Status: **validated; ready to merge**

V1 design:

```text
one 256-thread block per row
  -> fused x + residual local sum
  -> shared-memory mean reduction
  -> fused x + residual squared deviation
  -> shared-memory variance reduction
  -> parallel affine output
```

Validation:

```text
clean build: PASS
full pytest: 520 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
ptxas: 21 registers/thread, 1024 B shared memory, 0 spills
```

Representative fused V1 versus unfused PyTorch:

| Shape | Fused V1 | PyTorch unfused | Result |
|---:|---:|---:|---|
| 128 x 128 | 11.872 us | 13.312 us | fused faster |
| 128 x 512 | 11.264 us | 12.992 us | fused faster |
| 128 x 1024 | 12.064 us | 13.120 us | fused faster |
| 128 x 4096 | 15.632 us | 14.336 us | fused slower |
| 1024 x 4096 | 46.080 us | 40.848 us | fused slower |
| 128 x 8192 | 25.056 us | 19.088 us | fused slower |

This is the first stage where eliminating the residual intermediate translates into an end-to-end latency win.

Next action: V2 warp-shuffle reductions.

## E26 — Fused Residual + LayerNorm V2: warp-shuffle reductions

Status: **validated; ready to merge**

V2 changes only reduction coordination:

```text
V1:
  256 shared partials + repeated barriers

V2:
  warp shuffle
  -> 8 shared warp partials
  -> first warp final reduction
```

Residual addition remains fused into both statistics passes and the final affine pass.

Validation:

```text
clean build: PASS
full pytest: 541 passed
memcheck: 0 errors
racecheck: 0 hazards / 0 errors / 0 warnings
synccheck: 0 errors
```

ptxas:

```text
V1:
  21 registers/thread
  1024 B shared

V2:
  23 registers/thread
  32 B shared
  0 spills
```

Representative timings:

| Shape | V1 | V2 | PyTorch unfused |
|---:|---:|---:|---:|
| 128 x 512 | 12.016 | 12.080 | 13.312 |
| 128 x 1024 | 12.256 | 12.288 | 13.312 |
| 1024 x 128 | 15.360 | 12.288 | 14.336 |
| 1024 x 512 | 17.248 | 14.144 | 14.208 |
| 128 x 4096 | 15.968 | 15.360 | 14.336 |
| 1024 x 4096 | 46.080 | 45.024 | 40.720 |
| 128 x 8192 | 25.600 | 24.576 | 19.312 |

### Interpretation

Warp shuffle helps high-row-count workloads and makes the fused path competitive or faster on narrow/medium widths. Remaining wide-row headroom is now primarily a memory-path problem.

Next action: V3 aligned float4 fused IO.
