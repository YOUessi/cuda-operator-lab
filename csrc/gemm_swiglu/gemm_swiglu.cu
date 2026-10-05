#include "gemm_swiglu.cuh"

#include <cstdint>
#include <dlfcn.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

using cublasHandle_t = void*;
using cublasStatus_t = int;
using cublasOperation_t = int;
using cudaDataType_t = int;
using cublasComputeType_t = int;
using cublasGemmAlgo_t = int;

constexpr cublasStatus_t kCublasSuccess = 0;
constexpr cublasOperation_t kCublasOpN = 0;
constexpr cublasOperation_t kCublasOpT = 1;
constexpr cudaDataType_t kCudaR32F = 0;
constexpr cudaDataType_t kCudaR16BF = 14;
constexpr cublasComputeType_t kCublasCompute32F = 68;
constexpr cublasGemmAlgo_t kCublasGemmDefaultTensorOp = 99;

using CublasCreateFn = cublasStatus_t (*)(cublasHandle_t*);
using CublasDestroyFn = cublasStatus_t (*)(cublasHandle_t);
using CublasSetStreamFn = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
using CublasSgemmFn = cublasStatus_t (*)(
    cublasHandle_t,
    cublasOperation_t,
    cublasOperation_t,
    int,
    int,
    int,
    const float*,
    const float*,
    int,
    const float*,
    int,
    const float*,
    float*,
    int);

using CublasGemmExFn = cublasStatus_t (*)(
    cublasHandle_t,
    cublasOperation_t,
    cublasOperation_t,
    int,
    int,
    int,
    const void*,
    const void*,
    cudaDataType_t,
    int,
    const void*,
    cudaDataType_t,
    int,
    const void*,
    void*,
    cudaDataType_t,
    int,
    cublasComputeType_t,
    cublasGemmAlgo_t);

struct CublasApi {
  void* library = nullptr;
  cublasHandle_t handle = nullptr;
  CublasCreateFn create = nullptr;
  CublasDestroyFn destroy = nullptr;
  CublasSetStreamFn set_stream = nullptr;
  CublasSgemmFn sgemm = nullptr;
  CublasGemmExFn gemm_ex = nullptr;
  bool ready = false;

  CublasApi() {
    library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) return;

    create = reinterpret_cast<CublasCreateFn>(dlsym(library, "cublasCreate_v2"));
    destroy = reinterpret_cast<CublasDestroyFn>(dlsym(library, "cublasDestroy_v2"));
    set_stream = reinterpret_cast<CublasSetStreamFn>(dlsym(library, "cublasSetStream_v2"));
    sgemm = reinterpret_cast<CublasSgemmFn>(dlsym(library, "cublasSgemm_v2"));
    gemm_ex = reinterpret_cast<CublasGemmExFn>(dlsym(library, "cublasGemmEx"));

    if (create == nullptr || destroy == nullptr ||
        set_stream == nullptr || sgemm == nullptr) {
      return;
    }
    ready = create(&handle) == kCublasSuccess;
  }

  ~CublasApi() {
    if (handle != nullptr && destroy != nullptr) destroy(handle);
    if (library != nullptr) dlclose(library);
  }
};

CublasApi& cublas_api() {
  static CublasApi api;
  return api;
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0F + expf(-x));
}

__global__ void swiglu_inplace_kernel(
    float* gate,
    const float* up,
    std::uint64_t elements) {
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    gate[idx] = silu(gate[idx]) * up[idx];
  }
}

__global__ void swiglu_inplace_float4_kernel(
    float4* gate4,
    const float4* up4,
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
    gate4[idx4] = y;
  }
}

int validate_arguments(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n) {
  if (m == 0 || n == 0) return static_cast<int>(cudaSuccess);
  if (input == nullptr || gate_weight == nullptr || up_weight == nullptr ||
      workspace == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

cublasStatus_t row_major_gemm(
    CublasApi& api,
    const float* input,
    const float* weight,
    float* output,
    int m,
    int k,
    int n) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return api.sgemm(
      api.handle,
      kCublasOpT,
      kCublasOpN,
      n,
      m,
      k,
      &alpha,
      weight,
      k,
      input,
      k,
      &beta,
      output,
      n);
}

cublasStatus_t row_major_gemm_bf16_fp32(
    CublasApi& api,
    const void* input_bf16,
    const void* weight_bf16,
    float* output,
    int m,
    int k,
    int n) {
  if (api.gemm_ex == nullptr) {
    return -1;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  return api.gemm_ex(
      api.handle,
      kCublasOpT,
      kCublasOpN,
      n,
      m,
      k,
      &alpha,
      weight_bf16,
      kCudaR16BF,
      k,
      input_bf16,
      kCudaR16BF,
      k,
      &beta,
      output,
      kCudaR32F,
      n,
      kCublasCompute32F,
      kCublasGemmDefaultTensorOp);
}

cublasStatus_t row_major_gemm_bf16_bf16(
    CublasApi& api,
    const void* input_bf16,
    const void* weight_bf16,
    void* output_bf16,
    int m,
    int k,
    int n) {
  if (api.gemm_ex == nullptr) {
    return -1;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  return api.gemm_ex(
      api.handle,
      kCublasOpT,
      kCublasOpN,
      n,
      m,
      k,
      &alpha,
      weight_bf16,
      kCudaR16BF,
      k,
      input_bf16,
      kCudaR16BF,
      k,
      &beta,
      output_bf16,
      kCudaR16BF,
      n,
      kCublasCompute32F,
      kCublasGemmDefaultTensorOp);
}

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16);
}

__global__ void swiglu_packed_split_bf16_kernel(
    const std::uint16_t* packed_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t n) {
  const std::uint64_t elements = m * n;
  const std::uint64_t packed_cols = 2ULL * n;
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t row = idx / n;
    const std::uint64_t col = idx - row * n;
    const std::uint64_t base = row * packed_cols;
    const float gate = bf16_bits_to_float(packed_bf16[base + col]);
    const float up = bf16_bits_to_float(packed_bf16[base + n + col]);
    output[idx] = silu(gate) * up;
  }
}

__global__ void swiglu_packed_split_kernel(
    const float* packed,
    float* output,
    std::uint64_t m,
    std::uint64_t n) {
  const std::uint64_t elements = m * n;
  const std::uint64_t packed_cols = 2ULL * n;
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t row = idx / n;
    const std::uint64_t col = idx - row * n;
    const std::uint64_t base = row * packed_cols;
    const float gate = packed[base + col];
    const float up = packed[base + n + col];
    output[idx] = silu(gate) * up;
  }
}

__global__ void swiglu_packed_split_float4_kernel(
    const float4* packed4,
    float4* output4,
    std::uint64_t m,
    std::uint64_t vec_n) {
  const std::uint64_t vec_elements = m * vec_n;
  const std::uint64_t packed_vec_cols = 2ULL * vec_n;
  for (std::uint64_t idx4 =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx4 < vec_elements;
       idx4 += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t row = idx4 / vec_n;
    const std::uint64_t col4 = idx4 - row * vec_n;
    const std::uint64_t base4 = row * packed_vec_cols;
    const float4 g = packed4[base4 + col4];
    const float4 u = packed4[base4 + vec_n + col4];
    float4 y;
    y.x = silu(g.x) * u.x;
    y.y = silu(g.y) * u.y;
    y.z = silu(g.z) * u.z;
    y.w = silu(g.w) * u.w;
    output4[idx4] = y;
  }
}

int validate_packed_arguments(
    const float* input,
    const float* packed_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n) {
  if (m == 0 || n == 0) return static_cast<int>(cudaSuccess);
  if (input == nullptr || packed_weight == nullptr ||
      workspace == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX / 2) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_gemm_swiglu_v0(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_arguments(
      input, gate_weight, up_weight, workspace, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = cublas_api();
  if (!api.ready) return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int ni = static_cast<int>(n);

  // output stores gate projection; workspace stores up projection.
  if (row_major_gemm(api, input, gate_weight, output, mi, ki, ni) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  if (row_major_gemm(api, input, up_weight, workspace, mi, ki, ni) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_inplace_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      output, workspace, elements);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v1(
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_arguments(
      input, gate_weight, up_weight, workspace, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = cublas_api();
  if (!api.ready) return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int ni = static_cast<int>(n);

  if (row_major_gemm(api, input, gate_weight, output, mi, ki, ni) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  if (row_major_gemm(api, input, up_weight, workspace, mi, ki, ni) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(workspace) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  if ((elements % 4 == 0) && aligned) {
    const std::uint64_t vec_elements = elements / 4;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    swiglu_inplace_float4_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<float4*>(output),
        reinterpret_cast<const float4*>(workspace),
        vec_elements);
    return static_cast<int>(cudaGetLastError());
  }

  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_inplace_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      output, workspace, elements);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v2(
    const float* input,
    const float* packed_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_packed_arguments(
      input, packed_weight, workspace, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = cublas_api();
  if (!api.ready) return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int packed_n = static_cast<int>(2ULL * n);

  if (row_major_gemm(
          api, input, packed_weight, workspace, mi, ki, packed_n) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_packed_split_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      workspace, output, m, n);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v3(
    const float* input,
    const float* packed_weight,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation = validate_packed_arguments(
      input, packed_weight, workspace, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = cublas_api();
  if (!api.ready) return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int packed_n = static_cast<int>(2ULL * n);

  if (row_major_gemm(
          api, input, packed_weight, workspace, mi, ki, packed_n) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(workspace) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0);

  if ((n % 4 == 0) && aligned) {
    const std::uint64_t vec_n = n / 4;
    const std::uint64_t vec_elements = m * vec_n;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    swiglu_packed_split_float4_kernel<<<
        capped_blocks, kThreads, 0, cuda_stream>>>(
            reinterpret_cast<const float4*>(workspace),
            reinterpret_cast<float4*>(output),
            m,
            vec_n);
    return static_cast<int>(cudaGetLastError());
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_packed_split_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      workspace, output, m, n);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v4(
    const void* input_bf16,
    const void* packed_weight_bf16,
    float* workspace,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || packed_weight_bf16 == nullptr ||
      workspace == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX / 2) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto& api = cublas_api();
  if (!api.ready || api.gemm_ex == nullptr) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int packed_n = static_cast<int>(2ULL * n);

  if (row_major_gemm_bf16_fp32(
          api,
          input_bf16,
          packed_weight_bf16,
          workspace,
          mi,
          ki,
          packed_n) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_packed_split_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      workspace, output, m, n);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v5(
    const void* input_bf16,
    const void* packed_weight_bf16,
    void* workspace_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || packed_weight_bf16 == nullptr ||
      workspace_bf16 == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX / 2) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  auto& api = cublas_api();
  if (!api.ready || api.gemm_ex == nullptr) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int packed_n = static_cast<int>(2ULL * n);

  if (row_major_gemm_bf16_bf16(
          api,
          input_bf16,
          packed_weight_bf16,
          workspace_bf16,
          mi,
          ki,
          packed_n) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  swiglu_packed_split_bf16_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      reinterpret_cast<const std::uint16_t*>(workspace_bf16),
      output,
      m,
      n);
  return static_cast<int>(cudaGetLastError());
}
