#include "swiglu.cuh"

#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float silu(float x) {
  return x / (1.0F + expf(-x));
}

__global__ void swiglu_v0_kernel(
    const float* gate,
    const float* up,
    float* output,
    std::uint64_t elements) {
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    output[idx] = silu(gate[idx]) * up[idx];
  }
}

}  // namespace

extern "C" int cuda_operator_swiglu_v0(
    const float* gate,
    const float* up,
    float* output,
    std::uint64_t elements,
    void* stream) {
  if (elements == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (gate == nullptr || up == nullptr || output == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  swiglu_v0_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      gate, up, output, elements);
  return static_cast<int>(cudaGetLastError());
}
