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

Status: **in progress**

### Hypothesis

The V4 medium-shape regression is caused by launch geometry being computed from scalar N instead of the actual `float4` work-item count.

### Isolated change

Keep the V4 kernel body unchanged and change only vector-path launch geometry:

```text
V4 blocks = ceil(N / 256)
V5 blocks = ceil((N / 4) / 256)
```

with the same 1,024-block cap.

Unaligned input and inputs with no complete float4 element continue to use the scalar V3 path.

### Required validation before merge

- clean CUDA 12.8 / SM 8.9 build;
- full regression suite;
- aligned / unaligned correctness;
- current-stream and output-reset behavior;
- Compute Sanitizer;
- V4 vs V5 hot and L2-evicted benchmarks;
- stable repeated comparison at the previously regressed 262K shape;
- record whether vector-work-based blocks actually remove the regression.

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
