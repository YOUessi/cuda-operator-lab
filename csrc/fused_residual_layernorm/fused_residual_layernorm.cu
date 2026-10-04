#include "fused_residual_layernorm.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kFusedResidualLayerNormV0Threads = 128;
constexpr int kFusedResidualLayerNormV1Threads = 256;
constexpr int kFusedResidualLayerNormV2Threads = 256;
constexpr int kFusedResidualLayerNormV3Threads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kFusedResidualLayerNormV2Threads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__global__ void fused_residual_layernorm_v0_serial_row_kernel(
    const float* input,
    const float* residual,
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
  const float* row_residual = residual + row * cols;
  float* row_output = output + row * cols;

  float sum = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    sum += row_input[col] + row_residual[col];
  }
  const float mean = sum / static_cast<float>(cols);

  float sum_squared_deviation = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    const float value = row_input[col] + row_residual[col];
    const float centered = value - mean;
    sum_squared_deviation += centered * centered;
  }

  const float variance =
      sum_squared_deviation / static_cast<float>(cols);
  const float inverse_std = rsqrtf(variance + eps);

  for (std::uint64_t col = 0; col < cols; ++col) {
    const float value = row_input[col] + row_residual[col];
    const float normalized = (value - mean) * inverse_std;
    row_output[col] = normalized * weight[col] + bias[col];
  }
}

__global__ void fused_residual_layernorm_v1_block_row_kernel(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  __shared__ float shared[kFusedResidualLayerNormV1Threads];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  const float* row_residual = residual + row * cols;
  float* row_output = output + row * cols;

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_sum += row_input[col] + row_residual[col];
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
  __syncthreads();

  float local_sq = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col] + row_residual[col];
    const float centered = value - mean;
    local_sq += centered * centered;
  }

  shared[tid] = local_sq;
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

  const float inv_std = shared[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col] + row_residual[col];
    row_output[col] = (value - mean) * inv_std * weight[col] + bias[col];
  }
}

__global__ void fused_residual_layernorm_v2_warp_row_kernel(
    const float* input,
    const float* residual,
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
  const float* row_residual = residual + row * cols;
  float* row_output = output + row * cols;

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_sum += row_input[col] + row_residual[col];
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
  __syncthreads();

  float local_sq = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col] + row_residual[col];
    const float centered = value - mean;
    local_sq += centered * centered;
  }

  local_sq = warp_reduce_sum(local_sq);
  if (lane == 0) {
    warp_sums[warp_id] = local_sq;
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

  const float inv_std = warp_sums[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = row_input[col] + row_residual[col];
    row_output[col] = (value - mean) * inv_std * weight[col] + bias[col];
  }
}

__global__ void fused_residual_layernorm_v3_float4_row_kernel(
    const float4* input4,
    const float4* residual4,
    const float4* weight4,
    const float4* bias4,
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
  const float4* row_residual4 = residual4 + row * vec_cols;
  float4* row_output4 = output4 + row * vec_cols;

  float local_sum = 0.0F;
  for (std::uint64_t col4 = tid; col4 < vec_cols; col4 += blockDim.x) {
    const float4 x = row_input4[col4];
    const float4 r = row_residual4[col4];
    local_sum +=
        (x.x + r.x) + (x.y + r.y) + (x.z + r.z) + (x.w + r.w);
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
  __syncthreads();

  float local_sq = 0.0F;
  for (std::uint64_t col4 = tid; col4 < vec_cols; col4 += blockDim.x) {
    const float4 x = row_input4[col4];
    const float4 r = row_residual4[col4];
    const float vx = x.x + r.x - mean;
    const float vy = x.y + r.y - mean;
    const float vz = x.z + r.z - mean;
    const float vw = x.w + r.w - mean;
    local_sq += vx * vx + vy * vy + vz * vz + vw * vw;
  }

  local_sq = warp_reduce_sum(local_sq);
  if (lane == 0) {
    warp_sums[warp_id] = local_sq;
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

  const float inv_std = warp_sums[0];
  for (std::uint64_t col4 = tid; col4 < vec_cols; col4 += blockDim.x) {
    const float4 x = row_input4[col4];
    const float4 r = row_residual4[col4];
    const float4 w = weight4[col4];
    const float4 b = bias4[col4];
    float4 y;
    y.x = (x.x + r.x - mean) * inv_std * w.x + b.x;
    y.y = (x.y + r.y - mean) * inv_std * w.y + b.y;
    y.z = (x.z + r.z - mean) * inv_std * w.z + b.z;
    y.w = (x.w + r.w - mean) * inv_std * w.w + b.w;
    row_output4[col4] = y;
  }
}

int validate_arguments(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || residual == nullptr || weight == nullptr ||
      bias == nullptr || output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!(eps > 0.0F)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_fused_residual_layernorm_v0(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_arguments(input, residual, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::uint64_t blocks =
      (rows + kFusedResidualLayerNormV0Threads - 1) /
      kFusedResidualLayerNormV0Threads;

  fused_residual_layernorm_v0_serial_row_kernel<<<
      static_cast<unsigned int>(blocks),
      kFusedResidualLayerNormV0Threads,
      0,
      cuda_stream>>>(
          input, residual, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_fused_residual_layernorm_v1(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_arguments(input, residual, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  fused_residual_layernorm_v1_block_row_kernel<<<
      static_cast<unsigned int>(rows),
      kFusedResidualLayerNormV1Threads,
      0,
      cuda_stream>>>(
          input, residual, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_fused_residual_layernorm_v2(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_arguments(input, residual, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  fused_residual_layernorm_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kFusedResidualLayerNormV2Threads,
      0,
      cuda_stream>>>(
          input, residual, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_fused_residual_layernorm_v3(
    const float* input,
    const float* residual,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    float eps,
    void* stream) {
  const int validation =
      validate_arguments(input, residual, weight, bias, output, rows, cols, eps);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(input) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(residual) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(weight) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(bias) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if ((cols % 4 == 0) && aligned) {
    const std::uint64_t vec_cols = cols / 4;
    fused_residual_layernorm_v3_float4_row_kernel<<<
        static_cast<unsigned int>(rows),
        kFusedResidualLayerNormV3Threads,
        0,
        cuda_stream>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<const float4*>(residual),
            reinterpret_cast<const float4*>(weight),
            reinterpret_cast<const float4*>(bias),
            reinterpret_cast<float4*>(output),
            rows,
            vec_cols,
            cols,
            eps);
    return static_cast<int>(cudaGetLastError());
  }

  fused_residual_layernorm_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kFusedResidualLayerNormV2Threads,
      0,
      cuda_stream>>>(
          input, residual, weight, bias, output, rows, cols, eps);
  return static_cast<int>(cudaGetLastError());
}
