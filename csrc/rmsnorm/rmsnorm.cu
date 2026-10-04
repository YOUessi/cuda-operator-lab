#include "rmsnorm.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kRmsNormV0Threads = 128;
constexpr int kRmsNormV1Threads = 256;
constexpr int kRmsNormV2Threads = 256;
constexpr int kRmsNormV3Threads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kRmsNormV2Threads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__global__ void rmsnorm_v0_serial_row_kernel(
    const float* input,
    const float* weight,
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

  float sum_squares = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    const float value = row_input[col];
    sum_squares += value * value;
  }

  const float mean_square = sum_squares / static_cast<float>(cols);
  const float inverse_rms = rsqrtf(mean_square + eps);

  for (std::uint64_t col = 0; col < cols; ++col) {
    row_output[col] = row_input[col] * inverse_rms * weight[col];
  }
}

__global__ void rmsnorm_v1_block_row_kernel(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  __shared__ float shared[kRmsNormV1Threads];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_sum_squares = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col];
    local_sum_squares += value * value;
  }

  shared[tid] = local_sum_squares;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float mean_square = shared[0] / static_cast<float>(cols);
    shared[0] = rsqrtf(mean_square + eps);
  }
  __syncthreads();

  const float inverse_rms = shared[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] = row_input[col] * inverse_rms * weight[col];
  }
}

__global__ void rmsnorm_v2_warp_row_kernel(
    const float* input,
    const float* weight,
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

  float local_sum_squares = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col];
    local_sum_squares += value * value;
  }

  local_sum_squares = warp_reduce_sum(local_sum_squares);
  if (lane == 0) {
    warp_sums[warp_id] = local_sum_squares;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float mean_square = block_sum / static_cast<float>(cols);
      warp_sums[0] = rsqrtf(mean_square + eps);
    }
  }
  __syncthreads();

  const float inverse_rms = warp_sums[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] = row_input[col] * inverse_rms * weight[col];
  }
}


__global__ void rmsnorm_v3_float4_row_kernel(
    const float4* input4,
    const float4* weight4,
    float4* output4,
    std::uint64_t rows,
    std::uint64_t vec_cols,
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
  const float4* row_input4 = input4 + row * vec_cols;
  float4* row_output4 = output4 + row * vec_cols;

  float local_sum_squares = 0.0F;
  for (std::uint64_t col4 = tid; col4 < vec_cols; col4 += blockDim.x) {
    const float4 value = row_input4[col4];
    local_sum_squares += value.x * value.x;
    local_sum_squares += value.y * value.y;
    local_sum_squares += value.z * value.z;
    local_sum_squares += value.w * value.w;
  }

  local_sum_squares = warp_reduce_sum(local_sum_squares);
  if (lane == 0) {
    warp_sums[warp_id] = local_sum_squares;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_sum = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0F;
    block_sum = warp_reduce_sum(block_sum);
    if (lane == 0) {
      const float mean_square = block_sum / static_cast<float>(cols);
      warp_sums[0] = rsqrtf(mean_square + eps);
    }
  }
  __syncthreads();

  const float inverse_rms = warp_sums[0];
  for (std::uint64_t col4 = tid; col4 < vec_cols; col4 += blockDim.x) {
    const float4 x = row_input4[col4];
    const float4 w = weight4[col4];
    float4 y;
    y.x = x.x * inverse_rms * w.x;
    y.y = x.y * inverse_rms * w.y;
    y.z = x.z * inverse_rms * w.z;
    y.w = x.w * inverse_rms * w.w;
    row_output4[col4] = y;
  }
}

int validate_rmsnorm_arguments(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || weight == nullptr || output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!(eps > 0.0F)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_rmsnorm_v0(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_rmsnorm_arguments(input, weight, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::uint64_t required_blocks =
      (rows + kRmsNormV0Threads - 1) / kRmsNormV0Threads;

  rmsnorm_v0_serial_row_kernel<<<
      static_cast<unsigned int>(required_blocks),
      kRmsNormV0Threads,
      0,
      cuda_stream>>>(input, weight, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_rmsnorm_v1(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_rmsnorm_arguments(input, weight, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  rmsnorm_v1_block_row_kernel<<<
      static_cast<unsigned int>(rows),
      kRmsNormV1Threads,
      0,
      cuda_stream>>>(input, weight, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_rmsnorm_v2(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_rmsnorm_arguments(input, weight, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  rmsnorm_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kRmsNormV2Threads,
      0,
      cuda_stream>>>(input, weight, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_rmsnorm_v3(
    const float* input,
    const float* weight,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_rmsnorm_arguments(input, weight, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(input) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(weight) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  if ((cols % 4 == 0) && aligned) {
    const std::uint64_t vec_cols = cols / 4;
    rmsnorm_v3_float4_row_kernel<<<
        static_cast<unsigned int>(rows),
        kRmsNormV3Threads,
        0,
        cuda_stream>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<const float4*>(weight),
            reinterpret_cast<float4*>(output),
            rows,
            vec_cols,
            cols,
            eps);
    return static_cast<int>(cudaGetLastError());
  }

  rmsnorm_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kRmsNormV2Threads,
      0,
      cuda_stream>>>(input, weight, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}
