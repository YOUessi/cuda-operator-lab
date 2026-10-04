#pragma once

#include <cstdint>

extern "C" int cuda_operator_layernorm_v0(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);


extern "C" int cuda_operator_layernorm_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);


extern "C" int cuda_operator_layernorm_v2(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);


extern "C" int cuda_operator_layernorm_v3(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);


extern "C" int cuda_operator_layernorm_v4(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);


extern "C" int cuda_operator_layernorm_v5(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);
