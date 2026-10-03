# Reduction Optimization Report

## Benchmark protocol

- Device: NVIDIA GeForce RTX 4090 Laptop GPU.
- Dtype: float32.
- Reference: `torch.sum(..., dtype=torch.float32)`.
- Allocation stays outside the timed region.
- CUDA events measure GPU work; CPU wall-clock timing is not used.
- Numerical error is tracked together with performance.

Two cache regimes are kept separate:

- **hot:** repeatedly reuse the same input normally;
- **L2-evicted:** touch a 128 MiB buffer before every timed launch, twice the 64 MiB L2 capacity reported by the GPU.

The CSV field `logical_input_gbps` is input bytes divided by kernel time. It is not labeled DRAM bandwidth because hot-cache runs may be served substantially from L2.

---

## V0 — serial baseline

One CUDA thread performs the full sum.

Largest recorded case:

```text
N = 16,777,216
V0 ≈ 608.6 ms
```

This establishes the cost of missing GPU parallelism.

---

## V1 — grid-stride local sums + per-thread global atomic

V1 introduces 256-thread blocks and up to 1,024 blocks.

Each thread computes a register-local grid-stride sum and then performs one `atomicAdd` into the final output.

Worst-case global atomic count:

```text
1,024 blocks × 256 threads = 262,144 atomics
```

The large-input latency plateau around 400–450 us exposed same-address global atomic contention as the next bottleneck.

---

## V2 — shared-memory block reduction

V2 keeps the same per-thread local sums, then reduces 256 partial values inside each block using shared memory and a power-of-two tree.

Only thread 0 of each block performs the global atomic.

Worst-case global atomic count falls to:

```text
1,024 atomics
```

This removes the V1 atomic plateau.

L2-evicted, `N = 16,777,216`:

```text
V1      448.144 us
V2      177.152 us
PyTorch 172.032 us
```

V2 is therefore within roughly 3% of PyTorch for the largest recorded cold-cache case.

---

## V3 — warp-shuffle block reduction

### Change

V3 isolates warp-level programming while keeping the V2 launch policy unchanged.

Per block:

```text
256 thread-local sums
        ↓
8 independent warp reductions
using __shfl_down_sync
        ↓
lane 0 of each warp writes 8 values
to shared memory
        ↓
one __syncthreads()
        ↓
first warp loads the 8 warp sums
        ↓
second warp-shuffle reduction
        ↓
one global atomicAdd
```

Compared with V2:

- shared-memory entries fall from 256 floats to 8 floats;
- shared memory falls from 1,024 B to 32 B;
- the explicit tree no longer synchronizes at every reduction level;
- global atomics remain one per block.

### Correctness and safety

After V3 landed, the full GPU test suite reports **67 passed**.

V3-specific coverage includes:

- empty input;
- 31 / 32 / 33 and 63 / 64 / 65 warp-boundary sizes;
- 255 / 256 / 257 block-boundary sizes;
- 511 / 512 / 513;
- odd and million-element sizes;
- signed random input;
- reused/pre-filled output reset;
- non-default CUDA stream.

CUDA 12.8 Compute Sanitizer:

```text
memcheck:  0 errors
synccheck: 0 errors
```

Representative sanitizer input: 1,000,003 signed float32 values.

### ptxas resource summary

CUDA 12.8, `sm_89`:

| Variant | Registers / thread | Shared memory / block | Spills |
|---|---:|---:|---:|
| V2 | 12 | 1,024 B | 0 |
| V3 | 13 | **32 B** | 0 |

V3 trades one additional register for a 32x reduction in shared-memory footprint.

### Hot-cache results

50 timed repeats:

| N | V2 (us) | V3 (us) | V2 → V3 | torch.sum (us) |
|---:|---:|---:|---:|---:|
| 1,024 | 14.336 | 13.968 | 1.03x | 11.392 |
| 16,384 | 14.112 | 14.336 | 0.98x | 11.136 |
| 262,144 | 14.256 | 13.696 | 1.04x | 13.264 |
| 4,194,304 | 19.568 | 17.408 | **1.12x** | 16.384 |
| 16,777,216 | 43.088 | 41.184 | 1.05x | 34.448 |

Hot-cache measurements are useful for kernel-overhead comparisons but are not used to claim DRAM bandwidth.

### L2-evicted full-shape results

50 timed repeats:

| N | V2 (us) | V3 (us) | V2 → V3 | torch.sum (us) |
|---:|---:|---:|---:|---:|
| 1,024 | 4.096 | 4.096 | 1.00x | 4.912 |
| 16,384 | 4.592 | 4.288 | 1.07x | 7.168 |
| 262,144 | 8.192 | 7.280 | **1.13x** | 10.240 |
| 4,194,304 | 47.104 | 48.944 | 0.96x | 48.128 |
| 16,777,216 | 177.152 | 178.176 | 0.99x | 171.008 |

Because the large-shape V2/V3 differences are small, a second focused run used 20 warmups + 100 timed repeats.

### Stable large-shape L2-evicted comparison

| N | V2 (us) | V3 (us) | V2 → V3 | torch.sum (us) |
|---:|---:|---:|---:|---:|
| 262,144 | 8.192 | **7.168** | **1.14x** | 10.416 |
| 4,194,304 | 47.840 | **47.104** | 1.02x | 48.128 |
| 16,777,216 | 177.152 | 177.200 | ~1.00x | 171.008 |

### Interpretation

The experiment gives a more useful result than a blanket “V3 is faster” statement.

For medium shapes, warp shuffle removes enough synchronization/shared-memory overhead to matter: the stable 262K case improves by about **14%**.

For 4M elements, the gain shrinks to about **1.6%**.

For 16M elements, V2 and V3 are effectively tied.

That means the dominant bottleneck has moved again:

```text
V0: insufficient parallelism
 ↓
V1: global atomic contention
 ↓
V2: contention largely removed
 ↓
V3: coordination overhead reduced
 ↓
large inputs: input-memory movement dominates
```

This is the expected transition from synchronization-bound work toward a memory-throughput-bound reduction.

### Next hypothesis

V4 should target the input path rather than the reduction tree.

The next isolated experiment will keep the V3 warp-shuffle reduction structure and change only how each thread consumes input, beginning with multi-element/vectorized loads where alignment permits. The goal is to test whether fewer load instructions / more work per thread helps before changing block size or introducing more aggressive scheduling.

Raw artifacts:

- `reports/data/reduction_v2_v3_hot_rtx4090_laptop.csv`
- `reports/data/reduction_v2_v3_cold_rtx4090_laptop.csv`
- `reports/data/reduction_v2_v3_cold100_rtx4090_laptop.csv`
- `reports/data/reduction_v3_ptxas_sm89.txt`
- `reports/data/reduction_v3_compute_sanitizer.txt`
