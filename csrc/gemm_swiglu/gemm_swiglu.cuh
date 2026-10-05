#pragma once

#include <cstdint>

extern "C" int cuda_operator_gemm_swiglu_v0(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_swiglu_v1(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_swiglu_v2(
    const float* input,
    const float* packed_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);
