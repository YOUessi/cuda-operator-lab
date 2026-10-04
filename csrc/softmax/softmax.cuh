#pragma once

#include <cstdint>

extern "C" int cuda_operator_softmax_v0(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream);
