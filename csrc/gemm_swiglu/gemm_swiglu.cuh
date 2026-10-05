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


extern "C" int cuda_operator_gemm_swiglu_v3(
    const float* input,
    const float* packed_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_swiglu_v4(
    const void* input_bf16,
    const void* packed_weight_bf16,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_swiglu_v5(
    const void* input_bf16,
    const void* packed_weight_bf16,
    void* workspace_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);


extern "C" int cuda_operator_gemm_swiglu_v6(
    const void* input_bf16,
    const void* gate_weight_bf16,
    const void* up_weight_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream);
