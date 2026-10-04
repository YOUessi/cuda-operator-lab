#include "softmax.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kSoftmaxRowThreads = 128;

__global__ void softmax_v0_serial_row_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float row_max = -FLT_MAX;
  for (std::uint64_t col = 0; col < cols; ++col) {
    row_max = fmaxf(row_max, row_input[col]);
  }

  float denominator = 0.0F;
  for (std::uint64_t col = 0; col < cols; ++col) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    denominator += value;
  }

  const float inverse_denominator = 1.0F / denominator;
  for (std::uint64_t col = 0; col < cols; ++col) {
    row_output[col] *= inverse_denominator;
  }
}

}  // namespace

extern "C" int cuda_operator_softmax_v0(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::uint64_t required_blocks =
      (rows + kSoftmaxRowThreads - 1) / kSoftmaxRowThreads;
  const int blocks = static_cast<int>(required_blocks);

  softmax_v0_serial_row_kernel<<<
      blocks,
      kSoftmaxRowThreads,
      0,
      cuda_stream>>>(input, output, rows, cols);
  return static_cast<int>(cudaGetLastError());
}
