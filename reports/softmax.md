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
