#include "layernorm.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kLayerNormV0Threads = 128;
constexpr int kLayerNormV1Threads = 256;
constexpr int kLayerNormV2Threads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kLayerNormV2Threads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__global__ void layernorm_v0_serial_row_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float sum = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    sum += row_input[col];
  }
  const float mean = sum / static_cast<float>(cols);

  float sum_squared_deviation = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    const float centered = row_input[col] - mean;
    sum_squared_deviation += centered * centered;
  }

  const float variance =
      sum_squared_deviation / static_cast<float>(cols);
  const float inverse_std = rsqrtf(variance + eps);

  for (std::uint64_t col = 0; col < cols; ++col) {
    const float normalized = (row_input[col] - mean) * inverse_std;
    row_output[col] = normalized * weight[col] + bias[col];
  }
}

__global__ void layernorm_v1_block_row_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  __shared__ float shared[kLayerNormV1Threads];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_sum += row_input[col];
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    shared[0] /= static_cast<float>(cols);
  }
  __syncthreads();

  const float mean = shared[0];

  // All threads must consume shared[0] before the shared buffer is reused
  // for the variance reduction.
  __syncthreads();

  float local_squared_deviation = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float centered = row_input[col] - mean;
    local_squared_deviation += centered * centered;
  }

  shared[tid] = local_squared_deviation;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float variance = shared[0] / static_cast<float>(cols);
    shared[0] = rsqrtf(variance + eps);
  }
  __syncthreads();

  const float inverse_std = shared[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float normalized = (row_input[col] - mean) * inverse_std;
    row_output[col] = normalized * weight[col] + bias[col];
  }
}

__global__ void layernorm_v2_warp_row_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  __shared__ float warp_sums[kWarpsPerBlock];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp_id = tid / kWarpSize;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_sum += row_input[col];
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
      warp_sums[0] = block_sum / static_cast<float>(cols);
    }
  }
  __syncthreads();

  const float mean = warp_sums[0];

  // Protect shared partials from reuse until every thread has consumed mean.
  __syncthreads();

  float local_squared_deviation = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float centered = row_input[col] - mean;
    local_squared_deviation += centered * centered;
  }

  local_squared_deviation = warp_reduce_sum(local_squared_deviation);
  if (lane == 0) {
    warp_sums[warp_id] = local_squared_deviation;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float variance = block_sum / static_cast<float>(cols);
      warp_sums[0] = rsqrtf(variance + eps);
    }
  }
  __syncthreads();

  const float inverse_std = warp_sums[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float normalized = (row_input[col] - mean) * inverse_std;
    row_output[col] = normalized * weight[col] + bias[col];
  }
}

int validate_layernorm_arguments(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!(eps > 0.0F)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_layernorm_v0(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_layernorm_arguments(input, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::uint64_t required_blocks =
      (rows + kLayerNormV0Threads - 1) / kLayerNormV0Threads;

  layernorm_v0_serial_row_kernel<<<
      static_cast<unsigned int>(required_blocks),
      kLayerNormV0Threads,
      0,
      cuda_stream>>>(input, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_layernorm_v1(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_layernorm_arguments(input, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  layernorm_v1_block_row_kernel<<<
      static_cast<unsigned int>(rows),
      kLayerNormV1Threads,
      0,
      cuda_stream>>>(input, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_layernorm_v2(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_layernorm_arguments(input, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  layernorm_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kLayerNormV2Threads,
      0,
      cuda_stream>>>(input, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}
