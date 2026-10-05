#include "gemm_bias_gelu.cuh"

#include <cstdint>
#include <dlfcn.h>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;
constexpr float kInvSqrt2 = 0.70710678118654752440F;

using cublasHandle_t = void*;
using cublasStatus_t = int;
using cublasOperation_t = int;

constexpr cublasStatus_t kCublasSuccess = 0;
constexpr cublasOperation_t kCublasOpN = 0;
constexpr cublasOperation_t kCublasOpT = 1;

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

struct CublasApi {
  void* library = nullptr;
  cublasHandle_t handle = nullptr;
  CublasCreateFn create = nullptr;
  CublasDestroyFn destroy = nullptr;
  CublasSetStreamFn set_stream = nullptr;
  CublasSgemmFn sgemm = nullptr;
  bool ready = false;

  CublasApi() {
    library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return;
    }

    create = reinterpret_cast<CublasCreateFn>(dlsym(library, "cublasCreate_v2"));
    destroy = reinterpret_cast<CublasDestroyFn>(dlsym(library, "cublasDestroy_v2"));
    set_stream =
        reinterpret_cast<CublasSetStreamFn>(dlsym(library, "cublasSetStream_v2"));
    sgemm = reinterpret_cast<CublasSgemmFn>(dlsym(library, "cublasSgemm_v2"));

    if (create == nullptr || destroy == nullptr ||
        set_stream == nullptr || sgemm == nullptr) {
      return;
    }

    ready = create(&handle) == kCublasSuccess;
  }

  ~CublasApi() {
    if (handle != nullptr && destroy != nullptr) {
      destroy(handle);
    }
    if (library != nullptr) {
      dlclose(library);
    }
  }
};

CublasApi& cublas_api() {
  static CublasApi api;
  return api;
}

__device__ __forceinline__ float gelu_exact(float x) {
  return 0.5F * x * (1.0F + erff(x * kInvSqrt2));
}

__global__ void bias_gelu_epilogue_kernel(
    float* output,
    const float* bias,
    std::uint64_t elements,
    std::uint64_t n) {
  for (std::uint64_t idx =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < elements;
       idx += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t col = idx % n;
    output[idx] = gelu_exact(output[idx] + bias[col]);
  }
}

int validate_arguments(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_gemm_bias_gelu_v0(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation =
      validate_arguments(input, weight, bias, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = cublas_api();
  if (!api.ready) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (api.set_stream(api.handle, cuda_stream) != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int mi = static_cast<int>(m);
  const int ki = static_cast<int>(k);
  const int ni = static_cast<int>(n);
  const float alpha = 1.0F;
  const float beta = 0.0F;

  // Row-major Y[M,N] is the same memory layout as column-major Y^T[N,M].
  // input[M,K] is interpreted as column-major X^T[K,M].
  // weight[N,K] is interpreted as column-major W^T[K,N], then transposed.
  const cublasStatus_t gemm_status = api.sgemm(
      api.handle,
      kCublasOpT,
      kCublasOpN,
      ni,
      mi,
      ki,
      &alpha,
      weight,
      ki,
      input,
      ki,
      &beta,
      output,
      ni);
  if (gemm_status != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;

  bias_gelu_epilogue_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      output, bias, elements, n);
  return static_cast<int>(cudaGetLastError());
}
