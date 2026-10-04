#pragma once

#include <cstdint>

extern "C" int cuda_operator_fused_residual_layernorm_v0(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream);
