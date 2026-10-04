#include "rmsnorm.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kRmsNormV0Threads = 128;
constexpr int kRmsNormV1Threads = 256;

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
