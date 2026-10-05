#include "gemm_bias_gelu.cuh"

#include <cstddef>
#include <cstdint>
#include <dlfcn.h>

#include <cuda_runtime.h>

namespace {

using CublasLtHandle = void*;
using CublasLtMatmulDesc = void*;
using CublasLtMatrixLayout = void*;
using CublasStatus = int;

constexpr CublasStatus kSuccess = 0;
constexpr int kCudaR32F = 0;
constexpr int kCublasCompute32F = 68;
constexpr int kOpN = 0;
constexpr int kOpT = 1;
constexpr int kAttrTransA = 3;
constexpr int kAttrTransB = 4;
constexpr int kAttrEpilogue = 7;
constexpr int kAttrBiasPointer = 8;
constexpr std::uint32_t kEpilogueGeluBias = 36;  // GELU(32) | BIAS(4)

using LtCreateFn = CublasStatus (*)(CublasLtHandle*);
using LtDestroyFn = CublasStatus (*)(CublasLtHandle);
using LtMatmulDescCreateFn =
    CublasStatus (*)(CublasLtMatmulDesc*, int, int);
using LtMatmulDescDestroyFn =
    CublasStatus (*)(CublasLtMatmulDesc);
using LtMatmulDescSetAttributeFn =
    CublasStatus (*)(CublasLtMatmulDesc, int, const void*, std::size_t);
using LtMatrixLayoutCreateFn =
    CublasStatus (*)(
        CublasLtMatrixLayout*,
        int,
        std::uint64_t,
        std::uint64_t,
        std::int64_t);
using LtMatrixLayoutDestroyFn =
    CublasStatus (*)(CublasLtMatrixLayout);
using LtMatmulFn = CublasStatus (*)(
    CublasLtHandle,
    CublasLtMatmulDesc,
    const void*,
    const void*,
    CublasLtMatrixLayout,
    const void*,
    CublasLtMatrixLayout,
    const void*,
    const void*,
    CublasLtMatrixLayout,
    void*,
    CublasLtMatrixLayout,
    const void*,
    void*,
    std::size_t,
    cudaStream_t);

struct CublasLtApi {
  void* library = nullptr;
  CublasLtHandle handle = nullptr;
  LtCreateFn create = nullptr;
  LtDestroyFn destroy = nullptr;
  LtMatmulDescCreateFn desc_create = nullptr;
  LtMatmulDescDestroyFn desc_destroy = nullptr;
  LtMatmulDescSetAttributeFn desc_set = nullptr;
  LtMatrixLayoutCreateFn layout_create = nullptr;
  LtMatrixLayoutDestroyFn layout_destroy = nullptr;
  LtMatmulFn matmul = nullptr;
  bool ready = false;

  CublasLtApi() {
    library = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return;
    }

    create = reinterpret_cast<LtCreateFn>(dlsym(library, "cublasLtCreate"));
    destroy = reinterpret_cast<LtDestroyFn>(dlsym(library, "cublasLtDestroy"));
    desc_create = reinterpret_cast<LtMatmulDescCreateFn>(
        dlsym(library, "cublasLtMatmulDescCreate"));
    desc_destroy = reinterpret_cast<LtMatmulDescDestroyFn>(
        dlsym(library, "cublasLtMatmulDescDestroy"));
    desc_set = reinterpret_cast<LtMatmulDescSetAttributeFn>(
        dlsym(library, "cublasLtMatmulDescSetAttribute"));
    layout_create = reinterpret_cast<LtMatrixLayoutCreateFn>(
        dlsym(library, "cublasLtMatrixLayoutCreate"));
    layout_destroy = reinterpret_cast<LtMatrixLayoutDestroyFn>(
        dlsym(library, "cublasLtMatrixLayoutDestroy"));
    matmul =
        reinterpret_cast<LtMatmulFn>(dlsym(library, "cublasLtMatmul"));

    if (create == nullptr || destroy == nullptr ||
        desc_create == nullptr || desc_destroy == nullptr ||
        desc_set == nullptr || layout_create == nullptr ||
        layout_destroy == nullptr || matmul == nullptr) {
      return;
    }

    ready = create(&handle) == kSuccess;
  }

  ~CublasLtApi() {
    if (handle != nullptr && destroy != nullptr) {
      destroy(handle);
    }
    if (library != nullptr) {
      dlclose(library);
    }
  }
};

CublasLtApi& lt_api() {
  static CublasLtApi api;
  return api;
}

struct LtPlan {
  CublasLtMatmulDesc operation = nullptr;
  CublasLtMatrixLayout weight_layout = nullptr;
  CublasLtMatrixLayout input_layout = nullptr;
  CublasLtMatrixLayout output_layout = nullptr;
  std::uint64_t m = 0;
  std::uint64_t k = 0;
  std::uint64_t n = 0;
  bool ready = false;

  void reset(CublasLtApi& api) {
    if (weight_layout != nullptr) {
      api.layout_destroy(weight_layout);
    }
    if (input_layout != nullptr) {
      api.layout_destroy(input_layout);
    }
    if (output_layout != nullptr) {
      api.layout_destroy(output_layout);
    }
    if (operation != nullptr) {
      api.desc_destroy(operation);
    }
    operation = nullptr;
    weight_layout = nullptr;
    input_layout = nullptr;
    output_layout = nullptr;
    ready = false;
  }

  bool ensure(
      CublasLtApi& api,
      std::uint64_t next_m,
      std::uint64_t next_k,
      std::uint64_t next_n) {
    if (ready && m == next_m && k == next_k && n == next_n) {
      return true;
    }

    reset(api);

    if (api.desc_create(
            &operation,
            kCublasCompute32F,
            kCudaR32F) != kSuccess) {
      reset(api);
      return false;
    }

    const int trans_a = kOpT;
    const int trans_b = kOpN;
    const std::uint32_t epilogue = kEpilogueGeluBias;

    if (api.desc_set(
            operation,
            kAttrTransA,
            &trans_a,
            sizeof(trans_a)) != kSuccess ||
        api.desc_set(
            operation,
            kAttrTransB,
            &trans_b,
            sizeof(trans_b)) != kSuccess ||
        api.desc_set(
            operation,
            kAttrEpilogue,
            &epilogue,
            sizeof(epilogue)) != kSuccess) {
      reset(api);
      return false;
    }

    // Same row-major/column-major equivalence used by V0/V1:
    // weight[N,K] -> column-major [K,N]
    // input[M,K]  -> column-major [K,M]
    // output[M,N] -> column-major [N,M]
    if (api.layout_create(
            &weight_layout,
            kCudaR32F,
            next_k,
            next_n,
            static_cast<std::int64_t>(next_k)) != kSuccess ||
        api.layout_create(
            &input_layout,
            kCudaR32F,
            next_k,
            next_m,
            static_cast<std::int64_t>(next_k)) != kSuccess ||
        api.layout_create(
            &output_layout,
            kCudaR32F,
            next_n,
            next_m,
            static_cast<std::int64_t>(next_n)) != kSuccess) {
      reset(api);
      return false;
    }

    m = next_m;
    k = next_k;
    n = next_n;
    ready = true;
    return true;
  }
};

thread_local LtPlan g_plan;

int validate(
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
      k > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int cuda_operator_gemm_bias_gelu_v2(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  const int validation = validate(input, weight, bias, output, m, k, n);
  if (validation != static_cast<int>(cudaSuccess) || m == 0 || n == 0) {
    return validation;
  }

  auto& api = lt_api();
  if (!api.ready || !g_plan.ensure(api, m, k, n)) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  const void* bias_pointer = bias;
  if (api.desc_set(
          g_plan.operation,
          kAttrBiasPointer,
          &bias_pointer,
          sizeof(bias_pointer)) != kSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  const CublasStatus status = api.matmul(
      api.handle,
      g_plan.operation,
      &alpha,
      weight,
      g_plan.weight_layout,
      input,
      g_plan.input_layout,
      &beta,
      output,
      g_plan.output_layout,
      output,
      g_plan.output_layout,
      nullptr,
      nullptr,
      0,
      cuda_stream);

  if (status != kSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return static_cast<int>(cudaSuccess);
}
