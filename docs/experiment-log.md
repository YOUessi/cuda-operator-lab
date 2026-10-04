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
