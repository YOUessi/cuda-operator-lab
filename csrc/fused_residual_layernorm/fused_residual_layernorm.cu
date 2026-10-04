#include "fused_residual_layernorm.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kFusedResidualLayerNormV0Threads = 128;
constexpr int kFusedResidualLayerNormV1Threads = 256;

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
