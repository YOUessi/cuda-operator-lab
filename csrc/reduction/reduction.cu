#include "reduction.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kReductionThreads = 256;
constexpr int kReductionMaxBlocks = 1024;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kReductionThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

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

  if (local_sum != 0.0F) {
    atomicAdd(output, local_sum);
  }
}

__global__ void reduction_v2_shared_memory_kernel(
    const float* input,
    float* output,
    std::uint64_t n) {
  __shared__ float block_sums[kReductionThreads];

  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0F;
  for (std::uint64_t i = index; i < n; i += stride) {
    local_sum += input[i];
  }

  block_sums[threadIdx.x] = local_sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      block_sums[threadIdx.x] += block_sums[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(output, block_sums[0]);
  }
}

__global__ void reduction_v3_warp_shuffle_kernel(
    const float* input,
    float* output,
    std::uint64_t n) {
  __shared__ float warp_sums[kWarpsPerBlock];

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0F;
  for (std::uint64_t i = index; i < n; i += stride) {
    local_sum += input[i];
  }

  local_sum = warp_reduce_sum(local_sum);

  if (lane == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      atomicAdd(output, block_sum);
    }
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

int reduction_block_count(std::uint64_t n) {
  const std::uint64_t required_blocks =
      (n + kReductionThreads - 1) / kReductionThreads;
  const std::uint64_t capped_blocks =
      required_blocks < static_cast<std::uint64_t>(kReductionMaxBlocks)
          ? required_blocks
          : static_cast<std::uint64_t>(kReductionMaxBlocks);
  return static_cast<int>(capped_blocks);
}

cudaError_t reset_output(float* output, cudaStream_t stream) {
  return cudaMemsetAsync(output, 0, sizeof(float), stream);
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
  const cudaError_t status = reset_output(output, cuda_stream);
  if (status != cudaSuccess || n == 0) {
    return static_cast<int>(status);
  }

  reduction_v1_parallel_atomic_kernel<<<
      reduction_block_count(n),
      kReductionThreads,
      0,
      cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_reduction_v2(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_reduction_arguments(input, output, n);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t status = reset_output(output, cuda_stream);
  if (status != cudaSuccess || n == 0) {
    return static_cast<int>(status);
  }

  reduction_v2_shared_memory_kernel<<<
      reduction_block_count(n),
      kReductionThreads,
      0,
      cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_reduction_v3(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_reduction_arguments(input, output, n);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const cudaError_t status = reset_output(output, cuda_stream);
  if (status != cudaSuccess || n == 0) {
    return static_cast<int>(status);
  }

  reduction_v3_warp_shuffle_kernel<<<
      reduction_block_count(n),
      kReductionThreads,
      0,
      cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* cuda_operator_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
