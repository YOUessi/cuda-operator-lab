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

## Current milestone: Softmax V1

Reduction is complete as the first optimization case study. It contains six deliberately separated implementations:

- **V0 serial:** one CUDA thread sums all values.
- **V1 parallel atomic:** grid-stride local sums plus one global `atomicAdd` per participating thread.
- **V2 shared memory:** full shared-memory tree reduction per block, then one global `atomicAdd` per block.
- **V3 warp shuffle:** reduce inside each warp with `__shfl_down_sync`, store only one value per warp in shared memory, then use the first warp to finish the block reduction.
- **V4 float4 loads:** keep the V3 reduction tree but consume aligned input four floats at a time with a 128-bit global load; unaligned contiguous tensors safely fall back to V3.
- **V5 shape-aware dispatch:** size the vector grid from N/4 work items and use the vector path only from an empirically validated 512K crossover; smaller or unaligned inputs use V3.

V5 keeps the V4 kernel body unchanged and fixes dispatch/launch policy: vector blocks are computed from N/4 float4 work items rather than scalar N. A conservative 524,288-element crossover is used because cross-regime sweeps showed small-shape results were cache-state sensitive.

Implemented now:

- PyTorch `torch.sum` reference.
- CUDA V0 / V1 / V2 / V3 / V4 / V5 kernels behind a small C ABI.
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

### V4 vectorized-load result

The aligned V4 path is verified in generated SM 8.9 SASS:

```text
LDG.E.128
```

so the compiler does emit a 128-bit global load for the `float4` path. A scalar load remains for the 0–3 element tail. V4 uses 16 registers/thread, 32 B shared memory/block, and zero spills.

Canonical L2-evicted comparison, 20 warmups + 100 timed repeats:

| N | V3 warp shuffle | V4 float4 | V3 → V4 | torch.sum | V4 / torch |
|---:|---:|---:|---:|---:|---:|
| 262,144 | **7.168 us** | 8.192 us | 0.88x | 10.240 us | 0.80x |
| 4,194,304 | 46.896 us | **46.080 us** | 1.02x | 48.128 us | 0.96x |
| 16,777,216 | 177.152 us | **168.960 us** | **1.05x** | 171.008 us | 0.99x |

The optimization is shape-dependent. At 16M elements the 128-bit load path improves V3 by about 4.8% and matches PyTorch within roughly 1%. At 262K it regresses because the unchanged launch geometry now gives only one quarter of the threads vector work, while the rest still participate in block reduction.

That gives the next bottleneck directly: V5 should make launch geometry / elements-per-thread shape-aware rather than blindly applying vector loads to every shape.

### V5 shape-aware dispatch result

V5 fixes the V4 grid mismatch without changing the float4 kernel body.

```text
N < 524,288
  -> V3 scalar warp-shuffle path

N >= 524,288 and pointer % 16 == 0
  -> V4 float4 kernel
  -> blocks computed from N/4 vector work items

unaligned
  -> V3 fallback
```

Final clean validation: **116 tests passed**.

L2-evicted, 20 warmups + 100 repeats:

| N | V3 scalar | V5 dispatch | torch.sum | V5 logical GB/s |
|---:|---:|---:|---:|---:|
| 524,288 | 10.256 us | 10.240 us | 12.288 us | 204.800 |
| 1,048,576 | 17.008 us | **15.376 us** | 17.408 us | 272.783 |
| 4,194,304 | 47.104 us | **45.152 us** | 48.128 us | 371.572 |
| 16,777,216 | 176.128 us | **167.936 us** | 170.896 us | 399.610 |

The detailed step-by-step record, including preliminary thresholds and all raw benchmark artifacts, is in `docs/experiment-log.md`.

### Compute Sanitizer

CUDA 12.8 Compute Sanitizer on a representative `N=1,000,003` signed float32 input:

- memcheck: 0 errors
- racecheck: 0 hazards / 0 errors
- synccheck: 0 errors
- unaligned contiguous V4 fallback memcheck: 0 errors


### Softmax V0 -> V1

The second operator has started with an intentionally simple row-wise baseline:

```text
one CUDA thread per row
-> serial max
-> serial exp + sum
-> serial normalize
```

Clean RTX 4090 Laptop validation:

- full repository test suite: **131 passed**;
- Compute Sanitizer memcheck / racecheck / synccheck: clean;
- ptxas: 24 registers/thread, 0 spills.

The V0 width bottleneck is intentionally obvious: `128 × 4096` takes about 1.6 ms. V1 assigns one 256-thread block per row and uses shared-memory max/sum reductions. The post-race-fix benchmark reduces `128 × 4096` from **1,614.752 us to 14.336 us** (>112x), while `1024 × 4096` reaches **37.888 us vs 36.960 us** for PyTorch.

V1 hardware validation also found and fixed a shared-memory reuse race through Compute Sanitizer Racecheck; post-fix memcheck/racecheck/synccheck are clean.

Detailed operation history and raw artifacts are in `docs/experiment-log.md` and `reports/data/softmax_v0_*`.

## Quick start

```bash
./scripts/build.sh
./scripts/test.sh

PYTHONPATH=$PWD/python \
python3 benchmarks/reduction_benchmark.py \
  --variants v3_warp_shuffle v5_shape_aware_float4 \
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
