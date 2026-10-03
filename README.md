# CUDA Operator Optimization & Profiling Lab

A profile-guided CUDA operator engineering project on a real NVIDIA GPU.

## Goal

Build a reproducible optimization loop:

**PyTorch Reference → Naive CUDA → Correctness → Benchmark → Profiling → Bottleneck Analysis → Optimization → Re-benchmark**

The repository keeps meaningful intermediate kernels instead of publishing only a final implementation. Each optimization should answer three questions:

1. What profiler/benchmark observation exposed the bottleneck?
2. What concrete CUDA change targets that bottleneck?
3. Did correctness remain valid and did the measured result improve?

## Operators

- Reduction
- Softmax
- RMSNorm
- GEMM
- Fused Residual + RMSNorm

## Current milestone: Reduction V1

Reduction now has two deliberately separated stages:

- **V0 serial:** one CUDA thread sums all values.
- **V1 parallel atomic:** 256-thread blocks perform grid-stride local sums, then every participating thread contributes one global `atomicAdd`.

V1 establishes real GPU parallelism without hiding the next bottleneck behind shared-memory reduction. The next version can therefore measure the effect of reducing global atomic traffic independently.

Implemented now:

- PyTorch `torch.sum` reference.
- CUDA V0 + V1 kernels behind a small C ABI.
- Zero-copy PyTorch/ctypes binding using raw CUDA device pointers and the current PyTorch CUDA stream.
- Correctness coverage for empty, warp-adjacent, block-adjacent, odd and million-element inputs.
- Reused-output reset test and non-default CUDA stream test.
- CUDA-event benchmark with warmup, repeated measurements, median/P95 latency, effective bandwidth and PyTorch comparison.
- Reproducible CUDA 12.8 toolkit assembly from the CUDA packages already present on Tang.

### RTX 4090 Laptop: V0 → V1

Measured on the local NVIDIA GeForce RTX 4090 Laptop GPU, float32, 5 warmups + 20 timed repeats:

| N | V0 serial | V1 parallel atomic | V0 → V1 | torch.sum | V1 / torch | V1 effective BW |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 30.576 us | 13.504 us | 2.26x | 11.488 us | 1.18x | 0.303 GB/s |
| 16,384 | 367.616 us | 35.536 us | 10.34x | 10.368 us | 3.43x | 1.844 GB/s |
| 262,144 | 5,842.944 us | 384.000 us | 15.22x | 12.288 us | 31.25x | 2.731 GB/s |
| 4,194,304 | 82,612.225 us | 384.224 us | 215.01x | 15.360 us | 25.01x | 43.665 GB/s |
| 16,777,216 | 608,578.979 us | 393.216 us | 1,547.70x | 29.696 us | 13.24x | 170.667 GB/s |

V1 closes the catastrophic serial-parallel gap, but it deliberately issues up to **262,144 global atomic additions** into one output scalar. For large inputs the latency plateaus around 384–393 us even while effective bandwidth rises, which makes the next bottleneck explicit.

**V2 target:** perform an intra-block reduction first, then issue only one global atomic per block. With the current launch cap, that reduces worst-case global atomic traffic from 262,144 operations to at most 1,024.

## Quick start

```bash
./scripts/build.sh
./scripts/test.sh

PYTHONPATH=$PWD/python \
python3 benchmarks/reduction_benchmark.py
```

## Local target

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Compute capability: 8.9 (Ada)
- Driver: 580.178.04
- PyTorch: 2.10.0+cu128
- CUDA runtime reported by PyTorch: 12.8

The machine-wide `/usr/bin/nvcc` is CUDA 11.5 and does not support `sm_89`. The repository therefore assembles a local CUDA 12.8 toolkit view under `.cuda-toolkit/` from the already-installed Anaconda package cache. Generated toolkit files are ignored by Git.
