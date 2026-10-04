#pragma once

#include <cstdint>

extern "C" int cuda_operator_rmsnorm_v0(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);

extern "C" int cuda_operator_rmsnorm_v1(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);

extern "C" int cuda_operator_rmsnorm_v2(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);

extern "C" int cuda_operator_rmsnorm_v3(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);

extern "C" int cuda_operator_rmsnorm_v4(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);