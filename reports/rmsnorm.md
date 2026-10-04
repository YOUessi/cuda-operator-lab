# RMSNorm Optimization Report

## Definition

For one row with hidden width N:

```text
mean_square = (1 / N) * sum(x_i^2)
inverse_rms = rsqrt(mean_square + eps)
y_i = x_i * inverse_rms * weight_i
```

Current scope is contiguous float32 CUDA input `[rows, cols]`, float32 weight `[cols]`, and positive epsilon.

## Benchmark protocol

- Device: NVIDIA GeForce RTX 4090 Laptop GPU.
- CUDA: 12.8 / SM 8.9.
- Reference: PyTorch expression `x * rsqrt(mean(x^2)+eps) * weight`.
- CUDA events.
- 10 warmups + 50 timed repeats.
- Allocation outside the timed interval.
- eps = 1e-5.

---

## V0 — one serial CUDA thread per row

### Algorithm

Rows execute independently, but each active thread performs all hidden-dimension work serially:

```text
row
  -> serial sum of squares
  -> inverse RMS
  -> serial normalize * weight
```

### Correctness and safety

- full repository suite: **268 passed**;
- large-magnitude input test: PASS;
- preallocated output: PASS;
- non-default CUDA stream: PASS;
- invalid weight shape / nonpositive epsilon validation: PASS;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

### ptxas

- 20 registers/thread;
- 0 B shared memory;
- 0 spills;
- 0 barriers;
- 0 B stack frame.

### Performance

| Shape | V0 (us) | PyTorch (us) | Slowdown |
|---:|---:|---:|---:|
| 1 x 128 | 15.680 | 23.552 | 0.67x |
| 32 x 128 | 28.512 | 23.936 | 1.19x |
| 128 x 128 | 40.960 | 24.576 | 1.67x |
| 128 x 512 | 145.104 | 24.576 | 5.90x |
| 128 x 1024 | 270.336 | 24.576 | 11.00x |
| 128 x 4096 | 1,035.216 | 27.648 | **37.44x** |
| 1024 x 4096 | 1,046.528 | 61.488 | 17.02x |
| 2048 x 4096 | 920.160 | 252.816 | 3.64x |
| 128 x 8192 | 1,691.936 | 26.624 | **63.55x** |

### Interpretation

The baseline is row-parallel but hidden-width serial. Increasing width makes both the sum-of-squares pass and output pass longer for one thread. The next optimization target is therefore intra-row parallelism, not vectorization yet.

### V1 hypothesis

Use one 256-thread block per row:

```text
thread-local sum of squares
  -> shared-memory sum reduction
  -> inverse RMS
  -> parallel normalize * weight
```

Warp shuffle will be deferred so the value of basic block-level parallelism can be measured independently.

Raw artifacts:

- `reports/data/rmsnorm_v0_rtx4090_laptop.csv`
- `reports/data/rmsnorm_v0_ptxas_sm89.txt`
- `reports/data/rmsnorm_v0_compute_sanitizer.txt`


---

## V1 — one block per row with shared-memory sum reduction

### Algorithm

V1 assigns one 256-thread block to each row.

```text
thread-local sum(x^2) over strided columns
  -> shared[256]
  -> shared-memory tree sum
  -> inverse RMS
  -> parallel x * inverse_rms * weight
```

No warp shuffle, vectorized loads, or shape dispatch are used.

### Correctness and safety

- full repository suite: **290 passed**;
- memcheck: 0 errors;
- racecheck: 0 hazards / 0 errors;
- synccheck: 0 errors.

### Resource usage

| Variant | Registers/thread | Shared memory/block | Spills |
|---|---:|---:|---:|
| V0 | 20 | 0 B | 0 |
| V1 | 18 | 1,024 B | 0 |

### Performance

10 warmups + 50 timed repeats:

| Shape | V0 (us) | V1 (us) | V0 -> V1 |
|---:|---:|---:|---:|
| 128 x 128 | 41.424 | **11.264** | 3.68x |
| 128 x 512 | 144.560 | **10.272** | 14.07x |
| 128 x 1024 | 279.552 | **10.464** | 26.72x |
| 128 x 4096 | 1,037.120 | **14.080** | **73.66x** |
| 1024 x 4096 | 868.352 | **27.440** | 31.65x |
| 128 x 8192 | 1,692.320 | **17.408** | **97.22x** |

The benchmark also records the PyTorch expression reference, but that reference is composed from separate PyTorch operators and is not an optimized fused RMSNorm kernel. It is therefore a correctness/context reference, not a vendor-performance baseline.

### Interpretation

The serial hidden-width bottleneck disappears. The next isolated cost is the full shared-memory reduction tree and its repeated block synchronizations.

### V2 hypothesis

Keep the one-block-per-row traversal and scalar IO unchanged, and replace only the shared-memory tree with warp-shuffle reduction plus one shared partial per warp.

Raw artifacts:

- `reports/data/rmsnorm_v0_v1_rtx4090_laptop.csv`
- `reports/data/rmsnorm_v1_ptxas_sm89.txt`
- `reports/data/rmsnorm_v1_compute_sanitizer.txt`
