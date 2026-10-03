# Reduction Optimization Report

## Benchmark protocol

- Device: NVIDIA GeForce RTX 4090 Laptop GPU.
- Dtype: float32.
- Reference: `torch.sum(..., dtype=torch.float32)`.
- Allocation stays outside the timed region.
- CUDA events measure GPU work; CPU wall-clock timing is not used.
- 5 warmup launches + 20 timed launches for the recorded V0/V1 comparison.
- Reported latency is the median; P95 is retained in the raw CSV.
- Effective bandwidth = input bytes / measured kernel time.
- Numerical error is recorded together with performance because reduction order changes floating-point results.

---

## V0 — single-thread serial baseline

### Kernel

V0 launches one block with one active thread. That thread loops over all `N` float32 values, accumulates in one scalar register, and writes one output.

The implementation is intentionally poor: it gives a clean lower bound for the cost of missing GPU parallelism.

### Observed bottleneck

- one active CUDA thread;
- no grid-level parallelism;
- no warp-level parallelism;
- one thread issues all global-memory loads serially.

The largest recorded case, `N = 16,777,216`, takes about **608.6 ms** versus **29.7 us** for `torch.sum`.

---

## V1 — grid-stride local sums + global atomic accumulation

### Change

V1 introduces 256-thread blocks and caps the launch at 1,024 blocks.

Each thread:

1. walks the input with a grid-stride loop;
2. accumulates a private `local_sum` in a register;
3. contributes that partial sum with one global `atomicAdd`.

The output scalar is reset with `cudaMemsetAsync` on the caller's current CUDA stream before the kernel launch.

This version intentionally does **not** use shared memory or warp shuffle. That keeps the first optimization focused on exposing parallelism and leaves global atomic contention visible for V2.

### Correctness

The full GPU test suite reports **26 passed**:

- V0 regression tests remain green;
- V1 covers `N = 0, 1, 31, 32, 33, 127, 255, 256, 257, 1024, 4097, 65537, 1,000,003`;
- a pre-filled output tensor verifies that V1 resets output state before accumulation;
- a non-default PyTorch CUDA stream verifies stream propagation through the ctypes/C ABI boundary.

### Measured results

| N | V0 (us) | V1 (us) | V0 → V1 speedup | torch.sum (us) | V1 / torch | V1 BW (GB/s) | V1 relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 30.576 | 13.504 | 2.26x | 11.488 | 1.18x | 0.303 | 4.67e-7 |
| 16,384 | 367.616 | 35.536 | 10.34x | 10.368 | 3.43x | 1.844 | 2.29e-6 |
| 262,144 | 5,842.944 | 384.000 | 15.22x | 12.288 | 31.25x | 2.731 | 2.93e-6 |
| 4,194,304 | 82,612.225 | 384.224 | 215.01x | 15.360 | 25.01x | 43.665 | 3.10e-6 |
| 16,777,216 | 608,578.979 | 393.216 | 1,547.70x | 29.696 | 13.24x | 170.667 | 8.34e-6 |

Raw recorded data: `reports/data/reduction_v0_v1_rtx4090_laptop.csv`.

### Observation

V1 massively improves large-input latency because many CUDA threads now load and accumulate in parallel. At `N = 16,777,216`, V1 is about **1,548x faster than V0**.

However, V1 is still **13.24x slower than PyTorch** at that size.

The most informative pattern is the V1 latency plateau:

- 262,144 elements: 384.000 us
- 4,194,304 elements: 384.224 us
- 16,777,216 elements: 393.216 us

The implementation launches at most 1,024 × 256 = **262,144 threads**, and every participating thread may execute one `atomicAdd` to the same global address. Once all those threads participate, the serialized atomic update path becomes a dominant fixed cost.

### V2 hypothesis

V2 should reduce within each block before touching the global output.

Target structure:

```text
per-thread grid-stride local sum
        ↓
shared-memory block reduction
        ↓
one block result
        ↓
one global atomicAdd per block
```

With the current launch cap, the maximum number of global atomics falls from **262,144** to **1,024** — up to a **256x reduction in atomic operations**.

That is the next isolated optimization to measure before introducing warp shuffle.
