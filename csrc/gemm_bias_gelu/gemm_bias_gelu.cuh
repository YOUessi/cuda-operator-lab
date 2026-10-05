#pragma once

#include <cstdint>

extern "C" int cuda_operator_gemm_bias_gelu_v0(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_bias_gelu_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_bias_gelu_v2(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_bias_gelu_v3(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_bias_gelu_v4(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_bias_gelu_v5(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);
