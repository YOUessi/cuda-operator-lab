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
