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

## Current milestone: Reduction V2

Reduction currently has three deliberately separated stages:

- **V0 serial:** one CUDA thread sums all values.
- **V1 parallel atomic:** grid-stride local sums plus one global `atomicAdd` per participating thread.
- **V2 shared memory:** one shared-memory tree reduction per block, then one global `atomicAdd` per block.

V2 reduces the worst-case global atomic count from 262,144 to at most 1,024 while keeping the same 256-thread / 1,024-block launch cap.

Implemented now:

- PyTorch `torch.sum` reference.
- CUDA V0 / V1 / V2 kernels behind a small C ABI.
- Zero-copy PyTorch/ctypes binding using raw CUDA device pointers and the active PyTorch CUDA stream.
- Correctness coverage for empty, warp-adjacent, block-adjacent, odd, signed and million-element inputs.
- Reused-output reset and non-default CUDA stream tests.
- CUDA-event benchmark with warmup, repeated measurements, median/P95 latency and numerical error tracking.
- Hot-cache and L2-evicted benchmark modes.
- CUDA 12.8 / SM 8.9 ptxas resource capture.
- Reproducible CUDA 12.8 toolkit assembly from the CUDA packages already present on Tang.

### RTX 4090 Laptop: V1 → V2

Measured on the local NVIDIA GeForce RTX 4090 Laptop GPU, float32, 5 warmups + 20 timed repeats.

#### L2-evicted mode

Before each timed launch the benchmark touches a 128 MiB buffer, twice the 64 MiB L2 cache size reported by the GPU.

| N | V1 parallel atomic | V2 shared memory | V1 → V2 | torch.sum | V2 / torch | Logical input throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 5.472 us | 4.128 us | 1.33x | 5.120 us | 0.81x | 0.992 GB/s |
| 16,384 | 29.904 us | 5.120 us | 5.84x | 7.168 us | 0.71x | 12.800 GB/s |
| 262,144 | 428.032 us | 9.216 us | **46.44x** | 10.768 us | 0.86x | 113.778 GB/s |
| 4,194,304 | 443.376 us | 48.128 us | 9.21x | 50.176 us | 0.96x | 348.596 GB/s |
| 16,777,216 | 448.144 us | 177.152 us | 2.53x | 172.032 us | **1.03x** | 378.821 GB/s |

At the largest 64 MiB input, V2 is within about 3% of `torch.sum` in this benchmark.

The reported throughput is **logical input throughput**, not a claim of measured DRAM bandwidth. The repository keeps hot-cache and L2-evicted results separate because the 64 MiB input can fit in the GPU's 64 MiB L2 cache.

### Kernel resource footprint

CUDA 12.8 ptxas for `sm_89` reports for V2:

- 12 registers per thread;
- 1,024 bytes shared memory per block;
- zero spills;
- no stack frame.

Raw results live under `reports/data/`.

## Quick start

```bash
./scripts/build.sh
./scripts/test.sh

PYTHONPATH=$PWD/python \
python3 benchmarks/reduction_benchmark.py \
  --variants v1_parallel_atomic v2_shared_memory \
  --cache-mode cold
```

## Local target

- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- Compute capability: 8.9 (Ada)
- L2 cache: 64 MiB
- Driver: 580.178.04
- PyTorch: 2.10.0+cu128
- CUDA runtime reported by PyTorch: 12.8

The machine-wide `/usr/bin/nvcc` is CUDA 11.5 and does not support `sm_89`. The repository therefore assembles a local CUDA 12.8 toolkit view under `.cuda-toolkit/` from the already-installed Anaconda package cache. Generated toolkit files are ignored by Git.
