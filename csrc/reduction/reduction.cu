#include "reduction.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

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

}  // namespace

extern "C" int cuda_operator_reduction_v0(
    const float* input,
    float* output,
    std::uint64_t n,
    void* stream) {
  if (output == nullptr || (input == nullptr && n != 0)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  reduction_v0_serial_kernel<<<1, 1, 0, cuda_stream>>>(input, output, n);
  return static_cast<int>(cudaGetLastError());
}

extern "C" const char* cuda_operator_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}