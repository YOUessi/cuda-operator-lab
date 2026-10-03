#include "reduction.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kReductionThreads = 256;
constexpr int kReductionV1MaxBlocks = 1024;

__global__ void reduction_v0_serial_kernel(
    const float* input,
    float* output,
    std::uint64_t n) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  float sum = 0.0F;
  for (std::uint64_t i = 0; i < n; ++i) {
    sum += input[i];
  }
  output[0] = sum;
}

__global__ void reduction_v1_parallel_atomic_kernel(
    const float* input,
    float* output,
    std::uint64_t n) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0F;
  for (std::uint64_t i = index; i < n; i += stride) {
    local_sum += input[i];
  }

  // V1 intentionally uses one global atomic per participating thread.
  // The next reduction version will reduce within each block first so that
  // global atomic traffic falls from O(threads) to O(blocks).
  if (local_sum != 0.0F) {
    atomicAdd(output, local_sum);
  }
}

int validate_reduction_arguments(
    const float* input,
    float* output,
    std::uint64_t n) {
  if (output == nullptr || (input == nullptr && n != 0)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_reduction_v0(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_reduction_arguments(input, output, n);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  reduction_v0_serial_kernel<<<1, 1, 0, cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_reduction_v1(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_reduction_arguments(input, output, n);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  cudaError_t status = cudaMemsetAsync(output, 0, sizeof(float), cuda_stream);
  if (status != cudaSuccess || n == 0) {
    return static_cast<int>(status);
  }

  const std::uint64_t required_blocks =
      (n + kReductionThreads - 1) / kReductionThreads;
  const std::uint64_t capped_blocks =
      required_blocks < static_cast<std::uint64_t>(kReductionV1MaxBlocks)
          ? required_blocks
          : static_cast<std::uint64_t>(kReductionV1MaxBlocks);
  const int blocks = static_cast<int>(capped_blocks);

  reduction_v1_parallel_atomic_kernel<<<
      blocks,
      kReductionThreads,
      0,
      cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* cuda_operator_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
