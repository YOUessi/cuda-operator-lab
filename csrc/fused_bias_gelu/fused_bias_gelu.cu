#include "fused_bias_gelu.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr float kInvSqrt2 = 0.70710678118654752440F;

__device__ __forceinline__ float gelu_exact(float x) {
  return 0.5F * x * (1.0F + erff(x * kInvSqrt2));
}

__global__ void fused_bias_gelu_v0_kernel(
    const float* input,
    const float* bias,
    float* output,
    std::uint64_t elements,
    std::uint64_t cols) {
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t col = idx % cols;
    const float value = input[idx] + bias[col];
    output[idx] = gelu_exact(value);
  }
}

int validate_arguments(
    const float* input,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || bias == nullptr || output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_fused_bias_gelu_v0(
    const float* input,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation = validate_arguments(input, bias, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const std::uint64_t elements = rows * cols;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  fused_bias_gelu_v0_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      input, bias, output, elements, cols);
  return static_cast<int>(cudaGetLastError());
}
