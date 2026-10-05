# Final Performance Summary

本页汇总项目在 **NVIDIA GeForce RTX 4090 Laptop GPU（Ada, SM 8.9）** 上已经提交并验证的代表性性能结果。

环境：

```text
GPU: NVIDIA GeForce RTX 4090 Laptop GPU
Architecture: Ada / SM 8.9
Driver: 580.178.04
CUDA compiler: 12.8.93
PyTorch: 2.10.0+cu128
L2 cache: 64 MiB
```

> 注意：不同表格采用各自对应的公平 benchmark 口径。cold-cache 数据不会与早期 warm-cache PyTorch 数字直接混成“加速比”。所有原始 CSV 均保存在 `reports/data/`。

## 1. 基础算子：从 naive baseline 到优化实现

| Operator | Representative shape | Baseline | Optimized path | Baseline → optimized | Framework reference |
|---|---:|---:|---:|---:|---:|
| Reduction | N=16,777,216 | V0 608,030.731 us | V5 168.064 us | **3,617.9×** | torch.sum 169.984 us |
| Softmax | 128×4096 | V0 1,595.392 us | V2 warp-shuffle 13.216 us | **120.7×** | PyTorch 9.216 us |
| RMSNorm | 128×8192 | V0 1,692.512 us | V4 dispatch 12.032 us | **140.7×** | PyTorch expression 26.624 us |
| LayerNorm | 128×4096 | V0 1,076.352 us | V4 float4 11.264 us | **95.6×** | PyTorch 13.184 us |
| Residual+LayerNorm | 128×4096 | V0 fused serial 1,873.408 us | V3 float4 fused 12.448 us | **150.5×** | PyTorch unfused 16.192 us |

### 说明

- Reduction 的最终数字来自 L2-evicted cold-cache run；V5 与 PyTorch 在该大输入上基本同级。
- Softmax 的 128×4096 最终执行实际上不需要 narrow-row packing，因此用直接 warp-shuffle path 展示宽行优化效果；V5 在适合的小宽度/高 row-count 区域会自动切换到 packed warp-per-row。
- RMSNorm 的 PyTorch 对照是多算子表达式 reference，不等价于一个 vendor fused RMSNorm kernel，因此只用于工程参考。
- LayerNorm / Residual+LayerNorm 表中的 framework reference 与 optimized path 来自同一次 warm benchmark，适合直接比较。

## 2. Pointwise fusion

| Operator | Shape | PyTorch unfused | Fused optimized | End-to-end speedup |
|---|---:|---:|---:|---:|
| Bias + GELU | 1024×4096 | 33.792 us | V1 float4 20.480 us | **1.65×** |
| Bias + GELU | 2048×4096 | 183.296 us | V1 float4 36.768 us | **4.99×** |
| SwiGLU | 1024×4096 | 35.920 us | V1 float4 20.480 us | **1.75×** |
| SwiGLU | 2048×4096 | 278.528 us | V0 scalar fused 183.296 us | **1.52×** |

这些结果说明 pointwise fusion 的价值主要来自：

```text
更少 kernel launch
+ 更少中间 tensor
+ 更少 global memory round-trip
```

而不是“float4 一定更快”。例如 SwiGLU 2048×4096 的 float4 版本反而比 scalar fused 略慢，因此最终使用 profile-guided policy。

## 3. GEMM + Bias + GELU

### 从 standalone epilogue 到 cuBLASLt fused epilogue

| M×K×N | PyTorch matching prealloc | V6 robust cuBLASLt autotune | Relative result |
|---:|---:|---:|---:|
| 32×128×256 | 18.112 us | 14.336 us | **1.26× faster** |
| 128×512×512 | 22.288 us | 20.480 us | **1.09× faster** |
| 128×1024×4096 | 78.848 us | 81.920 us | 0.96× |
| 512×1024×4096 | 295.936 us | 271.360 us | **1.09× faster** |
| 512×4096×4096 | 1,188.864 us | 1,174.528 us | ~1.01× faster |

这条 case study 的主要结论不是单纯“cuBLASLt 更快”，而是：

- epilogue fusion；
- descriptor / plan lifetime；
- workspace budget；
- heuristic candidate；
- empirical online autotuning；
- algorithm cache；

这些 runtime 决策会直接决定 GEMM 性能。

## 4. GEMM + SwiGLU

### 4.1 两次 GEMM → packed single GEMM

| M×K×N | Two-GEMM V0 | Packed single-GEMM V2 | Speedup |
|---:|---:|---:|---:|
| 32×128×256 | 27.648 us | 16.624 us | **1.66×** |
| 128×512×512 | 35.840 us | 27.648 us | **1.30×** |
| 128×1024×4096 | 154.096 us | 137.216 us | **1.12×** |
| 512×1024×4096 | 592.896 us | 495.616 us | **1.20×** |
| 512×4096×4096 | 2,486.272 us | 2,241.024 us | **1.11×** |

主要收益来自把：

```text
SGEMM gate
+
SGEMM up
```

变为：

```text
one packed [2N,K] GEMM
```

而不是后处理 kernel。

### 4.2 FP32 packed → BF16 Tensor Core

| M×K×N | FP32 packed V2 | BF16 Tensor Core V4 | Speedup |
|---:|---:|---:|---:|
| 32×128×256 | 15.360 us | 15.120 us | 1.02× |
| 128×512×512 | 27.648 us | 18.352 us | **1.51×** |
| 128×1024×4096 | 122.880 us | 53.328 us | **2.30×** |
| 512×1024×4096 | 435.984 us | 217.088 us | **2.01×** |
| 512×4096×4096 | 2,246.656 us | 669.696 us | **3.35×** |

V4 使用：

```text
BF16 input
BF16 packed weight
FP32 accumulate
FP32 workspace
FP32 final output
```

这是当前大 GEMM 的高吞吐主路径。

### 4.3 自定义 WMMA vs cuBLAS

自定义 WMMA 的目标不是在所有 shape 上替代 cuBLAS。

V6–V9 证明：

- 小 GEMM / launch-dominated：workspace-free custom WMMA 可以赢；
- 大 GEMM：cuBLAS 的 tiling / scheduling / throughput 仍明显更强；
- shared A reuse 有价值；
- naive shared A+B staging 是负优化；
- 更大的 warp tile 也不是无条件更好。

### 4.4 Hybrid runtime

最终 V10/V11 策略：

```text
M <= 128
K <= 128
N <= 2048
+ alignment constraints
    -> custom V7 WMMA

otherwise
    -> cuBLAS BF16 V4
```

代表结果：

| M×K×N | V4 cuBLAS | Hybrid V10 | Speedup |
|---:|---:|---:|---:|
| 16×64×64 | 14.960 us | 12.192 us | **1.23×** |
| 32×128×256 | 14.336 us | 12.944 us | **1.11×** |
| 64×128×1024 | 15.360 us | 13.088 us | **1.17×** |
| 128×64×2048 | 17.216 us | 13.056 us | **1.32×** |
| 128×1024×4096 | 53.248 us | 53.248 us | fallback, no regression |
| 512×1024×4096 | 174.080 us | 174.080 us | fallback, no regression |

## 5. Generated hardware policy

V11 不再手写 crossover 阈值。

输入：

```text
120 measured shapes
×
2 independent profiling runs
```

接受规则：

```text
custom speedup >= 1.05x
in every run
```

自动生成：

```text
generated_hybrid_policy.h
```

并复现出当前 RTX 4090 Laptop 的安全区域：

```text
M <= 128
K <= 128
N <= 2048
```

V10 / V11 对照：

| M×K×N | V10 | V11 | Output diff |
|---:|---:|---:|---:|
| 16×64×64 | 12.288 | 12.288 | 0 |
| 32×128×256 | 12.784 | 12.544 | 0 |
| 64×128×1024 | 13.312 | 13.312 | 0 |
| 128×1024×4096 | 54.224 | 54.272 | 0 |
| 512×1024×4096 | 174.816 | 174.928 | 0 |

性能差异属于测量噪声；两者调用完全相同的底层 kernel。

## 6. Benchmark methodology

最终结论优先使用以下方法：

- CUDA Event timing；
- warmup；
- median + P95；
- preallocated framework baselines；
- L2 eviction（关键 memory-bound / dispatch 实验）；
- sample-level interleaving；
- multiple rounds；
- independent seeds；
- repeated-evidence acceptance gate。

典型 dispatch gate：

```text
speedup >= 1.05x
in every independent run
```

未测 / 不稳定 shape 默认 fallback，而不是外推一个“看起来合理”的阈值。

## 7. Validation

当前最新：

```text
816 passed
```

核心 CUDA 路径经过：

```text
Compute Sanitizer
├── memcheck
├── racecheck
└── synccheck
```

并保存 ptxas / SASS / raw CSV 证据。

## 8. Raw artifacts

完整原始数据：

- `reports/data/`
- `docs/experiment-log.md`
- `docs/engineering-log.md`

统一 benchmark：

```bash
bash scripts/benchmark_all.sh
```

该命令会重新运行一组代表性核心算子并生成新的本机汇总，不会覆盖本页已经提交的 canonical RTX 4090 Laptop 历史数据。
