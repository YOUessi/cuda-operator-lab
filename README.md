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

## Current milestone: Reduction V3

Reduction now has four deliberately separated implementations:

- **V0 serial:** one CUDA thread sums all values.
- **V1 parallel atomic:** grid-stride local sums plus one global `atomicAdd` per participating thread.
- **V2 shared memory:** full shared-memory tree reduction per block, then one global `atomicAdd` per block.
- **V3 warp shuffle:** reduce inside each warp with `__shfl_down_sync`, store only one value per warp in shared memory, then use the first warp to finish the block reduction.

V3 preserves the same 256-thread / 1,024-block launch policy as V2 so the measured difference isolates the reduction primitive rather than changing launch geometry.

Implemented now:

- PyTorch `torch.sum` reference.
- CUDA V0 / V1 / V2 / V3 kernels behind a small C ABI.
- Zero-copy PyTorch/ctypes binding using raw CUDA pointers and the active PyTorch CUDA stream.
- Correctness coverage for empty, warp-boundary, block-boundary, odd, signed and million-element inputs.
- Reused-output reset and non-default CUDA stream tests.
- CUDA-event benchmark with warmup, repeated measurements, P50/P95 and numerical-error tracking.
- Hot-cache and L2-evicted benchmark modes.
- CUDA 12.8 / SM 8.9 ptxas resource capture.
- CUDA 12.8 Compute Sanitizer memcheck + synccheck validation.

### V3 resource change

| Variant | Registers / thread | Shared memory / block | Global atomics / block |
|---|---:|---:|---:|
| V2 shared memory | 12 | 1,024 B | 1 |
| V3 warp shuffle | 13 | **32 B** | 1 |

V3 reduces block shared-memory footprint by **32x** while keeping zero spills.

### RTX 4090 Laptop: V2 → V3

For a more stable comparison, the canonical L2-evicted large-shape run uses 20 warmups + 100 timed repeats.

| N | V2 shared memory | V3 warp shuffle | V2 → V3 | torch.sum | V3 / torch |
|---:|---:|---:|---:|---:|---:|
| 262,144 | 8.192 us | **7.168 us** | **1.14x** | 10.416 us | 0.69x |
| 4,194,304 | 47.840 us | **47.104 us** | 1.02x | 48.128 us | 0.98x |
| 16,777,216 | 177.152 us | 177.200 us | ~1.00x | 171.008 us | 1.04x |

The result is intentionally not presented as “warp shuffle is always faster.” V3 helps when block-reduction overhead is still material, but the gain disappears at 64 MiB where the reduction is dominated by moving the input data rather than coordinating threads.

That is the useful conclusion: **after V2, large-shape reduction is already primarily memory-throughput limited; V3 mainly reduces synchronization/shared-memory overhead for smaller and medium shapes.**

### Compute Sanitizer

CUDA 12.8 Compute Sanitizer on a representative `N=1,000,003` signed float32 input:

- memcheck: 0 errors
- synccheck: 0 errors

## Quick start

```bash
./scripts/build.sh
./scripts/test.sh

PYTHONPATH=$PWD/python \
python3 benchmarks/reduction_benchmark.py \
  --variants v2_shared_memory v3_warp_shuffle \
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
