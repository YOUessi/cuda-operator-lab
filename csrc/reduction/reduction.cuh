#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

extern "C" int cuda_operator_reduction_v0(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream);

extern "C" int cuda_operator_reduction_v1(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream);

extern "C" int cuda_operator_reduction_v2(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream);

extern "C" int cuda_operator_reduction_v3(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream);

extern "C" int cuda_operator_reduction_v4(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream);

extern "C" const char* cuda_operator_error_string(int code);
