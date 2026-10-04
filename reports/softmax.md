# Softmax Optimization Report

## Benchmark protocol

- Device: NVIDIA GeForce RTX 4090 Laptop GPU.
- Dtype: float32.
- Operation: row-wise Softmax over the last dimension.
- Reference: `torch.softmax(x, dim=-1, dtype=torch.float32)`.
- CUDA events measure GPU execution.
- Allocation remains outside the timed interval.
- Baseline recording: 5 warmups + 20 timed repeats.
- Correctness tracks max absolute element error and maximum row-sum error.

---

## V0 — one serial CUDA thread per row

### Algorithm

Each active CUDA thread owns exactly one row.

```text
row
 -> serial max over all columns
 -> serial exp(x - max) + denominator accumulation
 -> serial divide by denominator
```

Different rows can execute concurrently, but there is no cooperation among threads inside one row.

### Numerical stability

The row maximum is subtracted before exponentiation:

```text
exp(x_i - max(x))
```

Extreme-logit tests around ±1000 match the PyTorch reference.

### Correctness and safety

After V0 landed:

- complete repository test suite: **131 passed**;
- preallocated output: PASS;
- non-default CUDA stream: PASS;
- non-contiguous input rejection: PASS;
- Compute Sanitizer memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

### ptxas

CUDA 12.8 / SM 8.9:

- 24 registers/thread;
- 0 B shared memory;
- 0 spills;
- 0 barriers;
- 0 B stack frame.

### Measured results

| Shape | V0 (us) | PyTorch (us) | Slowdown |
|---:|---:|---:|---:|
| 1 × 128 | 18.432 | 9.216 | 2.00x |
| 32 × 128 | 40.800 | 9.216 | 4.43x |
| 128 × 128 | 60.416 | 9.216 | 6.56x |
| 128 × 512 | 214.736 | 9.376 | 22.90x |
| 128 × 1024 | 416.768 | 8.192 | 50.88x |
| 128 × 4096 | 1,595.392 | 9.328 | **171.03x** |
| 1024 × 128 | 57.344 | 9.328 | 6.15x |
| 1024 × 512 | 209.440 | 10.160 | 20.61x |
| 1024 × 4096 | 1,607.408 | 37.600 | 42.75x |

### Interpretation

The width dimension is the dominant problem.

At fixed 128 rows:

```text
128 cols  ->   60.416 us
512 cols  ->  214.736 us
1024 cols ->  416.768 us
4096 cols -> 1595.392 us
```

The per-row work is O(cols) and is executed by one thread, three times. Increasing row count creates more row-level parallelism, but it cannot remove the serial dependency within each row.

### V1 hypothesis

Use one CUDA block per row.

Each thread processes columns in a block-stride loop, and shared memory combines thread-local values:

```text
thread-local max
  -> shared-memory max reduction
  -> parallel exp + local denominator sum
  -> shared-memory sum reduction
  -> parallel normalize
```

This keeps Softmax mathematically identical while isolating the value of intra-row parallelism and block-level shared-memory reductions before introducing warp shuffle.

Raw artifacts:

- `reports/data/softmax_v0_rtx4090_laptop.csv`
- `reports/data/softmax_v0_ptxas_sm89.txt`
- `reports/data/softmax_v0_compute_sanitizer.txt`


---

## V1 — one block per row with shared-memory reductions

### Algorithm

V1 assigns one 256-thread CUDA block to each row.

Threads process columns in a block-stride loop, then cooperate twice:

1. shared-memory max reduction;
2. shared-memory denominator-sum reduction.

The normalization pass is also block-parallel.

### Racecheck finding and fix

The first sanitizer run found a real shared-memory reuse race that normal output tests did not reliably expose.

After the max reduction, threads copied `shared[0]` into `row_max`, but the buffer was reused for the denominator reduction without a barrier after the read. Racecheck reported one error with 72 hazards.

The fixed sequence is:

```cpp
const float row_max = shared[0];
__syncthreads();  // complete all reads before shared[] reuse
```

After this fix:

- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

This is retained as an engineering example of why sanitizer-based concurrency validation is required even when numerical regression tests pass.

### Resource usage

| Variant | Registers/thread | Shared memory/block | Spills |
|---|---:|---:|---:|
| V0 | 24 | 0 B | 0 |
| V1 | 21 | 1,024 B | 0 |

### Performance

10 warmups + 50 timed repeats:

| Shape | V0 (us) | V1 (us) | V0 -> V1 | PyTorch (us) | V1 / PyTorch |
|---:|---:|---:|---:|---:|---:|
| 1 x 128 | 17.408 | 10.240 | 1.70x | 9.536 | 1.07x |
| 32 x 128 | 38.912 | 15.360 | 2.53x | 8.320 | 1.85x |
| 128 x 128 | 56.352 | 9.216 | 6.11x | 10.496 | 0.88x |
| 128 x 512 | 224.256 | 11.264 | 19.91x | 10.240 | 1.10x |
| 128 x 1024 | 405.632 | 11.264 | 36.01x | 8.832 | 1.28x |
| 128 x 4096 | 1,614.752 | 14.336 | **112.64x** | 9.216 | 1.56x |
| 1024 x 128 | 57.344 | 15.360 | 3.73x | 9.216 | 1.67x |
| 1024 x 512 | 208.192 | 17.296 | 12.04x | 8.880 | 1.95x |
| 1024 x 4096 | 1,353.728 | 37.888 | 35.73x | 36.960 | **1.03x** |

### Interpretation

One-block-per-row parallelism eliminates the serial-width bottleneck.

The next visible cost is the reduction coordination itself: V1 performs two complete 256-thread shared-memory trees per row with a block synchronization after every stage.

### V2 hypothesis

Keep the one-block-per-row data traversal unchanged and replace only the max/sum shared-memory trees with two-level warp-shuffle reductions. This should reduce shared memory and barriers while preserving the same Softmax passes.

Raw artifacts:

- `reports/data/softmax_v0_v1_rtx4090_laptop.csv`
- `reports/data/softmax_v1_ptxas_sm89.txt`
- `reports/data/softmax_v1_compute_sanitizer.txt`


---

## V2 — warp-shuffle max and sum reductions

### Algorithm

V2 preserves V1's one-block-per-row policy, 256-thread block size, block-stride column traversal, and three Softmax passes. Only the two block reductions change.

```text
thread-local value
  -> warp __shfl_down_sync
  -> one partial per warp
  -> shared[8]
  -> first warp final reduction
```

The pattern is used once for row max and once for denominator sum.

The V1 Racecheck lesson is preserved: all threads consume the final row maximum before the shared warp-partial buffer is reused.

### Correctness and safety

RTX 4090 Laptop / CUDA 12.8 / SM 8.9:

- clean build: PASS;
- full repository suite: **180 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors;
- representative max absolute error: `3.73e-9`;
- representative max row-sum error: `1.19e-7`.

### Resource usage

| Variant | Registers/thread | Shared memory/block | Spills |
|---|---:|---:|---:|
| V1 | 21 | 1,024 B | 0 |
| V2 | 23 | **32 B** | 0 |

### Stable performance

20 warmups + 100 timed repeats:

| Shape | V1 (us) | V2 (us) | V1 -> V2 | PyTorch (us) |
|---:|---:|---:|---:|---:|
| 128 x 128 | 10.368 | 10.336 | ~1.00x | 8.192 |
| 128 x 512 | **10.240** | 10.592 | 0.97x | 8.176 |
| 128 x 1024 | 10.512 | **10.400** | 1.01x | 8.256 |
| 128 x 4096 | 13.664 | **13.216** | 1.03x | 9.216 |
| 1024 x 128 | 14.336 | **10.240** | **1.40x** | 8.096 |
| 1024 x 512 | 16.192 | **12.256** | **1.32x** | 8.192 |
| 1024 x 4096 | 41.792 | **39.600** | 1.06x | 36.608 |

### Interpretation

Warp shuffle is most valuable when many row blocks are active and the cost of repeated shared-memory reduction stages is visible. For the 1024-row shapes V2 is consistently better, with up to ~1.40x V1 -> V2 improvement.

For 128 rows the result is near-neutral, and 128 x 512 is slightly slower. V2 therefore remains an explicit optimization stage rather than a claim of universal superiority.

The next isolated issue is fixed block width. A 128-column row launches 256 threads, leaving half the threads without column work.

### V3 hypothesis

Keep the V2 kernel mathematics and warp-shuffle reductions, but dispatch the thread count by row width before attempting float4/vectorized Softmax.

Raw artifacts:

- `reports/data/softmax_v1_v2_50_rtx4090_laptop.csv`
- `reports/data/softmax_v1_v2_100_rtx4090_laptop.csv`
- `reports/data/softmax_v2_ptxas_sm89.txt`
- `reports/data/softmax_v2_compute_sanitizer.txt`


---

## V3 — width-aware block sizing

### Change

V3 keeps V2's one-block-per-row and warp-shuffle reduction algorithm but selects fewer threads for narrow rows:

```text
cols <= 32   -> 32 threads
cols <= 64   -> 64 threads
cols <= 128  -> 128 threads
otherwise    -> 256 threads
```

Dynamic reduction helpers use the actual launched warp count.

### Validation

- clean build: PASS;
- full repository suite: **209 passed**;
- narrow-path memcheck/racecheck/synccheck: clean;
- wide-path memcheck/racecheck/synccheck: clean;
- ptxas: 23 registers/thread, 32 B shared/block, 0 spills.

### Performance result

The result is mixed rather than monotonically positive.

Stable 100-repeat examples:

| Shape | V2 | V3 | Result |
|---:|---:|---:|---|
| 128 x 32 | **10.240 us** | 10.864 us | V3 regression |
| 128 x 64 | 10.240 us | 10.240 us | neutral |
| 128 x 128 | 10.240 us | 10.240 us | neutral |
| 1024 x 32 | 10.240 us | 10.240 us | neutral |
| 1024 x 64 | **10.240 us** | 10.992 us | V3 regression |
| 1024 x 128 | 10.464 us | **10.272 us** | ~1.02x |

### Interpretation

Reducing thread count removes idle column workers but also reduces the number of active warps in each one-row block. The stable data does not support using simple width-aware block shrinkage as the default strategy.

The stronger next design is to keep 256 threads/block while packing multiple small rows into the block, one row per warp.

Raw artifacts:

- `reports/data/softmax_v2_v3_boundaries_50_rtx4090_laptop.csv`
- `reports/data/softmax_v2_v3_100_rtx4090_laptop.csv`
- `reports/data/softmax_v3_ptxas_sm89.txt`
- `reports/data/softmax_v3_compute_sanitizer.txt`


---

## V4 — one warp per row, eight rows per block

### Design

For `cols <= 128`, V4 changes the execution layout rather than the Softmax math.

A 256-thread block is divided into eight independent warps:

```text
warp 0 -> row r+0
warp 1 -> row r+1
...
warp 7 -> row r+7
```

Each warp performs row max, exp-sum, and normalization with lane-stride access and warp shuffle only. No shared memory and no block-wide barrier are needed.

For wider rows, V4 falls back to V3.

### Validation

- clean CUDA 12.8 / SM 8.9 build: PASS;
- full repository suite: **229 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 warnings;
- synccheck: 0 errors.

### Resources

| Variant | Registers/thread | Shared memory/block | Barriers | Spills |
|---|---:|---:|---:|---:|
| V3 | 23 | 32 B | yes | 0 |
| V4 packed | 30 | **0 B** | **0** | 0 |

### Performance

20 warmups + 100 timed repeats:

| Shape | V3 | V4 | Speedup | PyTorch |
|---:|---:|---:|---:|---:|
| 4096 x 32 | 11.024 us | **10.240 us** | 1.08x | 8.016 us |
| 4096 x 64 | 11.264 us | **10.576 us** | 1.07x | 8.160 us |
| 4096 x 128 | 14.336 us | **11.136 us** | 1.29x | 8.688 us |
| 16384 x 32 | 19.392 us | **13.952 us** | 1.39x | 8.192 us |
| 16384 x 64 | 21.776 us | **15.104 us** | 1.44x | 9.216 us |
| 16384 x 128 | 32.752 us | **19.136 us** | **1.71x** | 11.424 us |

Small-row-count behavior is shape-sensitive:

```text
128 x 128:
V3 10.480 us
V4 11.152 us
```

so V4 is not yet the default path for every narrow matrix.

### Next hypothesis

Use an empirical dispatcher that selects V4 only above row-count thresholds measured independently for 32-, 64-, and 128-column regions.

Raw artifacts:

- `reports/data/softmax_v3_v4_warp_rows100_rtx4090.csv`
- `reports/data/softmax_v4_ptxas_sm89.txt`
- `reports/data/softmax_v4_compute_sanitizer.txt`
