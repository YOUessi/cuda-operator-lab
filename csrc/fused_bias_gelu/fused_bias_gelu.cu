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

__global__ void fused_bias_gelu_v1_float4_kernel(
    const float4* input4,
    const float4* bias4,
    float4* output4,
    std::uint64_t vec_elements,
    std::uint64_t vec_cols) {
  for (std::uint64_t idx4 =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx4 < vec_elements;
       idx4 += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t col4 = idx4 % vec_cols;
    const float4 x = input4[idx4];
    const float4 b = bias4[col4];
    float4 y;
    y.x = gelu_exact(x.x + b.x);
    y.y = gelu_exact(x.y + b.y);
    y.z = gelu_exact(x.z + b.z);
    y.w = gelu_exact(x.w + b.w);
    output4[idx4] = y;
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


extern "C" int cuda_operator_fused_bias_gelu_v1(
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

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(input) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(bias) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if ((cols % 4 == 0) && aligned) {
    const std::uint64_t vec_cols = cols / 4;
    const std::uint64_t vec_elements = rows * vec_cols;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    fused_bias_gelu_v1_float4_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<float4*>(output),
        vec_elements,
        vec_cols);
    return static_cast<int>(cudaGetLastError());
  }

  const std::uint64_t elements = rows * cols;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  fused_bias_gelu_v0_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      input, bias, output, elements, cols);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_fused_bias_gelu_v2(
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

  const bool profiled_float4 =
      (rows == 512 && cols == 4096) ||
      (rows == 1024 && cols == 4096) ||
      (rows == 2048 && cols == 1024) ||
      (rows == 2048 && cols == 4096);

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(input) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(bias) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if (profiled_float4 && aligned && (cols % 4 == 0)) {
    const std::uint64_t vec_cols = cols / 4;
    const std::uint64_t vec_elements = rows * vec_cols;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    fused_bias_gelu_v1_float4_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<const float4*>(bias),
        reinterpret_cast<float4*>(output),
        vec_elements,
        vec_cols);
    return static_cast<int>(cudaGetLastError());
  }

  const std::uint64_t elements = rows * cols;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  fused_bias_gelu_v0_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      input, bias, output, elements, cols);
  return static_cast<int>(cudaGetLastError());
}
