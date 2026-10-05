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

__global__ void swiglu_v1_float4_kernel(
    const float4* gate4,
    const float4* up4,
    float4* output4,
    std::uint64_t vec_elements) {
  for (std::uint64_t idx4 =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx4 < vec_elements;
       idx4 += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const float4 g = gate4[idx4];
    const float4 u = up4[idx4];
    float4 y;
    y.x = silu(g.x) * u.x;
    y.y = silu(g.y) * u.y;
    y.z = silu(g.z) * u.z;
    y.w = silu(g.w) * u.w;
    output4[idx4] = y;
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


extern "C" int cuda_operator_swiglu_v1(
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

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(gate) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(up) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if ((elements % 4 == 0) && aligned) {
    const std::uint64_t vec_elements = elements / 4;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    swiglu_v1_float4_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<const float4*>(gate),
        reinterpret_cast<const float4*>(up),
        reinterpret_cast<float4*>(output),
        vec_elements);
    return static_cast<int>(cudaGetLastError());
  }

  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_v0_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      gate, up, output, elements);
  return static_cast<int>(cudaGetLastError());
}
