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

## Current milestone: Reduction V0

The first baseline is intentionally bad: **one CUDA thread serially sums the entire tensor**. It provides a clean lower bound before introducing block-level parallelism, shared-memory reduction, warp shuffle, and vectorized loads.

Implemented now:

- PyTorch `torch.sum` reference.
- CUDA `v0_serial` kernel behind a small C ABI.
- Zero-copy PyTorch/ctypes binding: PyTorch owns CUDA tensors and streams, the kernel receives raw device pointers.
- Correctness tests for empty, boundary, odd, warp-adjacent, and larger sizes.
- CUDA-event benchmark with warmup, repeated measurements, median/P95 latency, effective bandwidth, and PyTorch comparison.
- Reproducible CUDA 12.8 toolkit assembly from the CUDA packages already present on Tang.

### RTX 4090 Laptop baseline

Measured on the local NVIDIA GeForce RTX 4090 Laptop GPU, float32, 5 warmups + 20 timed repeats:

| N | V0 serial | torch.sum | Slowdown | Effective BW |
|---:|---:|---:|---:|---:|
| 1,024 | 30.672 us | 11.264 us | 2.72x | 0.134 GB/s |
| 16,384 | 368.336 us | 11.232 us | 32.79x | 0.178 GB/s |
| 262,144 | 5,670.352 us | 14.768 us | 383.96x | 0.185 GB/s |
| 4,194,304 | 79,512.783 us | 16.384 us | 4,853.08x | 0.211 GB/s |
| 16,777,216 | 608,030.731 us | 30.720 us | 19,792.67x | 0.110 GB/s |

This result is expected: V0 leaves essentially the whole GPU idle and gives us an intentionally poor baseline to improve.

## Quick start

```bash
cd /home/you/projects/cuda-operator-lab

./scripts/build.sh
./scripts/test.sh

PYTHONPATH=$PWD/python \
python3 benchmarks/reduction_baseline.py
```

## Local target

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Compute capability: 8.9 (Ada)
- Driver: 580.178.04
- PyTorch: 2.10.0+cu128
- CUDA runtime reported by PyTorch: 12.8

The machine-wide `/usr/bin/nvcc` is CUDA 11.5 and does not support `sm_89`. The repository therefore assembles a local CUDA 12.8 toolkit view under `.cuda-toolkit/` from the already-installed Anaconda package cache. Generated toolkit files are ignored by Git.