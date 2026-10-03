# CUDA Operator Optimization & Profiling Lab

A profile-guided CUDA operator engineering project.

## Goal

Build a reproducible optimization loop for GPU operators:

Reference -> Naive CUDA -> Correctness -> Benchmark -> Nsight Profiling -> Bottleneck Analysis -> Optimization -> Re-benchmark.

## Operators

- Reduction
- Softmax
- RMSNorm
- GEMM
- Fused Residual + RMSNorm

## Engineering principles

1. Keep every meaningful optimization version instead of only the final kernel.
2. Never report performance without correctness checks.
3. Benchmark with warmup and CUDA events; report repeated measurements.
4. Compare identical shapes/dtypes against trusted references.
5. Explain each optimization with profiler evidence.
6. Keep raw benchmark/profiler artifacts reproducible.

## Local target

Primary development machine: NVIDIA GeForce RTX 4090 Laptop GPU (SM 8.9).

The system `/usr/bin/nvcc` is CUDA 11.5 and is not the preferred compiler for Ada.
Development uses CUDA 12.8 nvcc available locally.
