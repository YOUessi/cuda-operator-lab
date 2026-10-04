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
constexpr std::uintptr_t kFloat4Alignment = 16U;
constexpr std::uint64_t kVectorDispatchMinElements = 32768;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ void finish_warp_block_reduction(
    float local_sum,
    float* warp_sums,
    float* output) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

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

  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;

  float local_sum = 0.0F;
  for (std::uint64_t i = index; i < n; i += stride) {
    local_sum += input[i];
  }

  finish_warp_block_reduction(local_sum, warp_sums, output);
}

__global__ void reduction_v4_float4_kernel(
    const float* input,
    float* output,
    std::uint64_t n) {
  __shared__ float warp_sums[kWarpsPerBlock];

  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  const std::uint64_t vector_count = n / 4;
  const auto* vector_input = reinterpret_cast<const float4*>(input);

  float local_sum = 0.0F;
  for (std::uint64_t i = index; i < vector_count; i += stride) {
    const float4 values = vector_input[i];
    local_sum += (values.x + values.y) + (values.z + values.w);
  }

  const std::uint64_t tail_start = vector_count * 4;
  for (std::uint64_t i = tail_start + index; i < n; i += stride) {
    local_sum += input[i];
  }

  finish_warp_block_reduction(local_sum, warp_sums, output);
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

int reduction_block_count_for_work_items(std::uint64_t work_items) {
  const std::uint64_t required_blocks =
      (work_items + kReductionThreads - 1) / kReductionThreads;
  const std::uint64_t capped_blocks =
      required_blocks < static_cast<std::uint64_t>(kReductionMaxBlocks)
          ? required_blocks
          : static_cast<std::uint64_t>(kReductionMaxBlocks);
  return static_cast<int>(capped_blocks);
}

cudaError_t reset_output(float* output, cudaStream_t stream) {
  return cudaMemsetAsync(output, 0, sizeof(float), stream);
}

bool is_float4_aligned(const float* input) {
  return (reinterpret_cast<std::uintptr_t>(input) &
          (kFloat4Alignment - 1U)) == 0U;
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
      reduction_block_count_for_work_items(n),
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
      reduction_block_count_for_work_items(n),
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
      reduction_block_count_for_work_items(n),
      kReductionThreads,
      0,
      cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_reduction_v4(
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

  if (is_float4_aligned(input)) {
    reduction_v4_float4_kernel<<<
        reduction_block_count_for_work_items(n),
        kReductionThreads,
        0,
        cuda_stream>>>(input, output, n);
  } else {
    // A contiguous tensor may still have a non-zero storage offset. Falling
    // back keeps the public API correct rather than issuing a misaligned
    // 16-byte float4 load.
    reduction_v3_warp_shuffle_kernel<<<
        reduction_block_count_for_work_items(n),
        kReductionThreads,
        0,
        cuda_stream>>>(input, output, n);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_reduction_v5(
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

  const std::uint64_t vector_count = n / 4;
  if (is_float4_aligned(input) &&
      n >= kVectorDispatchMinElements &&
      vector_count != 0) {
    // V5 isolates dispatch/launch geometry: the kernel body is exactly the
    // V4 float4 kernel, but the grid is sized from actual vector work items.
    // The 32K crossover is empirical: hot-cache sweeps showed a small-shape
    // regression at 16K and no regression from 32K upward.
    reduction_v4_float4_kernel<<<
        reduction_block_count_for_work_items(vector_count),
        kReductionThreads,
        0,
        cuda_stream>>>(input, output, n);
  } else {
    reduction_v3_warp_shuffle_kernel<<<
        reduction_block_count_for_work_items(n),
        kReductionThreads,
        0,
        cuda_stream>>>(input, output, n);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* cuda_operator_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
