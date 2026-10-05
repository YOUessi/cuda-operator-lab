#pragma once

#include <cstdint>

extern "C" int cuda_operator_fused_bias_gelu_v0(
    const float* input,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream);


extern "C" int cuda_operator_fused_bias_gelu_v1(
    const float* input,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream);
