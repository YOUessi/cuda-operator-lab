#pragma once

#include <cstdint>

extern "C" int cuda_operator_swiglu_v0(
    const float* gate,
    const float* up,
    float* output,
    std::uint64_t elements,
    void* stream);


extern "C" int cuda_operator_swiglu_v1(
    const float* gate,
    const float* up,
    float* output,
    std::uint64_t elements,
    void* stream);
