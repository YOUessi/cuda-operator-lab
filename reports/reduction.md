# Reduction Optimization Report

## Benchmark protocol

- Device: NVIDIA GeForce RTX 4090 Laptop GPU.
- Dtype: float32.
- Reference: `torch.sum(..., dtype=torch.float32)`.
- Allocation stays outside the timed region.
- CUDA events measure GPU work; CPU wall-clock timing is not used.
- Recorded runs use 5 warmups + 20 timed launches.
- Median and P95 are retained.
- Numerical error is tracked together with performance.

Two cache regimes are reported:

- **hot:** repeatedly reuse the same input normally;
- **cold / L2-evicted:** touch a 128 MiB buffer before each timed launch, which is 2x the 64 MiB L2 cache reported by the device.

The `logical_input_gbps` column is input bytes divided by kernel time. It is intentionally **not called DRAM bandwidth** because hot-cache runs can be served from L2 and may exceed the physical DRAM peak.

---

## V0 — single-thread serial baseline

V0 launches one block with one active thread. That thread loops over all `N` float32 values and performs the complete sum serially.

Observed bottleneck:

- one active CUDA thread;
- no grid-level parallelism;
- no warp-level parallelism;
- one thread issues all global-memory loads serially.

The largest recorded V0 case, `N = 16,777,216`, takes about 608.6 ms.

---

## V1 — grid-stride local sums + global atomic accumulation

V1 introduces 256-thread blocks and caps the launch at 1,024 blocks.

Each thread:

1. walks the input with a grid-stride loop;
2. accumulates a register-local partial sum;
3. performs one global `atomicAdd` into the final output.

This closes the serial/parallel gap but can generate as many as 262,144 atomic updates to the same scalar.

The V1 latency plateau around 400–450 us exposed global atomic contention as the dominant next bottleneck.

---

## V2 — shared-memory block reduction

### Change

V2 keeps the same grid-stride local accumulation, but the per-thread partial sums are first reduced inside each block:

```text
thread local_sum
      ↓
shared[256]
      ↓
128 + 128
      ↓
64 + 64
      ↓
32 + 32
      ↓
...
      ↓
1 block sum
      ↓
1 global atomicAdd
```

With at most 1,024 blocks, V2 reduces worst-case global atomics:

```text
V1: 262,144 atomics
V2:   1,024 atomics
```

That is up to **256x fewer global atomic operations**.

V2 intentionally keeps a full shared-memory tree with `__syncthreads()` at every level. Warp shuffle is reserved for V3 so its impact can be measured independently.

### Correctness

After V2 landed, the complete GPU suite reports **45 passed**.

V2 specifically covers:

- empty input;
- warp boundaries: 31 / 32 / 33;
- block boundaries: 255 / 256 / 257;
- 511 / 512 / 513;
- odd and million-element sizes;
- signed random input;
- pre-filled output reset;
- non-default CUDA stream propagation.

### ptxas resource summary

CUDA 12.8, `sm_89`:

| Variant | Registers / thread | Shared memory / block | Spills |
|---|---:|---:|---:|
| V0 | 14 | 0 B | 0 |
| V1 | 12 | 0 B | 0 |
| V2 | 12 | 1,024 B | 0 |

Raw ptxas output: `reports/data/reduction_v2_ptxas_sm89.txt`.

### Hot-cache result

| N | V1 (us) | V2 (us) | V1 → V2 | torch.sum (us) | V2 / torch |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 15.136 | 14.336 | 1.06x | 19.456 | 0.74x |
| 16,384 | 35.824 | 14.320 | 2.50x | 11.488 | 1.25x |
| 262,144 | 439.200 | 21.056 | 20.86x | 12.848 | 1.64x |
| 4,194,304 | 441.344 | 19.568 | 22.55x | 16.384 | 1.19x |
| 16,777,216 | 450.480 | 43.008 | 10.47x | 33.792 | 1.27x |

The 64 MiB input matches the device's 64 MiB L2 cache, so hot-cache logical throughput is not a DRAM-bandwidth measurement.

### L2-evicted result

| N | V1 (us) | V2 (us) | V1 → V2 | torch.sum (us) | V2 / torch | V2 logical GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 5.472 | 4.128 | 1.33x | 5.120 | 0.81x | 0.992 |
| 16,384 | 29.904 | 5.120 | 5.84x | 7.168 | 0.71x | 12.800 |
| 262,144 | 428.032 | 9.216 | **46.44x** | 10.768 | 0.86x | 113.778 |
| 4,194,304 | 443.376 | 48.128 | 9.21x | 50.176 | 0.96x | 348.596 |
| 16,777,216 | 448.144 | 177.152 | 2.53x | 172.032 | **1.03x** | 378.821 |

At `N = 16,777,216`, V2 is only about **3% slower than PyTorch** under the L2-evicted benchmark.

### Interpretation

The V2 result confirms the V1 hypothesis:

- the ~440 us atomic-contention plateau disappears;
- the shared-memory tree makes large reductions memory-throughput dominated again;
- numerical error also drops materially because far fewer unordered global atomic updates occur.

The small-input cold numbers are faster than their hot counterparts because the 128 MiB cache-preparation kernel also keeps the GPU in a sustained active clock state. Therefore the two cache modes are separate regimes and must not be mixed into one speedup claim.

### V3 hypothesis

V2 still performs `__syncthreads()` after every tree level, including the final 32 active threads.

V3 will keep shared memory for cross-warp aggregation but replace the final warp's shared-memory/barrier stages with warp shuffle primitives:

```text
shared-memory reduction to one value per warp
        ↓
warp-level __shfl_down_sync
        ↓
one block sum
        ↓
one global atomic
```

That isolates the value of warp-level programming before later experiments with vectorized loads or multi-element-per-thread scheduling.

Raw benchmark files:

- `reports/data/reduction_v1_v2_hot_rtx4090_laptop.csv`
- `reports/data/reduction_v1_v2_cold_rtx4090_laptop.csv`
