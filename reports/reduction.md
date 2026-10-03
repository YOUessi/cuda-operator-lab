# Reduction Optimization Report

## V0 — single-thread serial baseline

### Kernel

V0 launches one block with one active thread. The thread loops over all `N` float32 values and accumulates the sum in a scalar register before storing one result.

This is deliberately not a reasonable GPU implementation. It exists to make the cost of missing parallelism measurable.

### Correctness

The CUDA result is compared with `torch.sum(..., dtype=torch.float32)` on the same device input.

Current tests cover:

- `N = 0`
- `N = 1`
- values around one warp: `31, 32, 33`
- odd/non-power-of-two sizes
- `1024`, `4097`, `65537`

Result: **11 tests passed** on the RTX 4090 Laptop GPU.

For the largest benchmark size, the serial float32 accumulation shows a relative difference of roughly `6.64e-5` from PyTorch's parallel reduction. This is a useful baseline observation: reduction order affects floating-point results, so later optimized kernels must track both speed and numerical error.

### Benchmark method

- CUDA events, not CPU wall-clock time.
- Allocation is outside the timed region.
- 5 warmup launches.
- 20 timed launches.
- Report median and P95.
- Same tensor reused for our kernel and PyTorch reference.
- Effective bandwidth uses input bytes divided by measured kernel time.

### Results

| N | V0 median (us) | PyTorch median (us) | Slowdown | Effective BW (GB/s) | Relative error |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 30.672 | 11.264 | 2.72x | 0.134 | 0 |
| 16,384 | 368.336 | 11.232 | 32.79x | 0.178 | 4.09e-6 |
| 262,144 | 5,670.352 | 14.768 | 383.96x | 0.185 | 3.34e-6 |
| 4,194,304 | 79,512.783 | 16.384 | 4,853.08x | 0.211 | 2.13e-5 |
| 16,777,216 | 608,030.731 | 30.720 | 19,792.67x | 0.110 | 6.64e-5 |

### Bottleneck hypothesis

The bottleneck is not subtle:

- one active CUDA thread;
- no block-level parallelism;
- no warp-level parallelism;
- no data reuse;
- one thread issues all global-memory loads serially.

V1 should first establish grid/block parallelism before we add shared-memory or warp-specific optimization. Keeping those changes separate lets the benchmark attribute performance gains to one mechanism at a time.