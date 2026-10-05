#include "gemm_bias_gelu.cuh"

#include <cstdint>
#include <dlfcn.h>
#include <memory>
#include <tuple>
#include <unordered_map>

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

using cublasLtHandle_t = void*;
using cublasLtMatmulDesc_t = void*;
using cublasLtMatrixLayout_t = void*;
struct cublasLtMatmulAlgo_t {
  std::uint64_t data[8];
};
using cublasLtMatmulPreference_t = void*;

struct cublasLtMatmulHeuristicResult_t {
  cublasLtMatmulAlgo_t algo;
  std::size_t workspace_size;
  cublasStatus_t state;
  float waves_count;
  int reserved[4];
};

using CublasLtCreateFn = cublasStatus_t (*)(cublasLtHandle_t*);
using CublasLtDestroyFn = cublasStatus_t (*)(cublasLtHandle_t);
using CublasLtMatmulDescCreateFn =
    cublasStatus_t (*)(cublasLtMatmulDesc_t*, int, int);
using CublasLtMatmulDescDestroyFn = cublasStatus_t (*)(cublasLtMatmulDesc_t);
using CublasLtMatmulDescSetAttributeFn =
    cublasStatus_t (*)(cublasLtMatmulDesc_t, int, const void*, std::size_t);
using CublasLtMatrixLayoutCreateFn =
    cublasStatus_t (*)(cublasLtMatrixLayout_t*, int, std::uint64_t, std::uint64_t, std::int64_t);
using CublasLtMatrixLayoutDestroyFn =
    cublasStatus_t (*)(cublasLtMatrixLayout_t);
using CublasLtMatmulFn = cublasStatus_t (*)(
    cublasLtHandle_t,
    cublasLtMatmulDesc_t,
    const void*,
    const void*,
    cublasLtMatrixLayout_t,
    const void*,
    cublasLtMatrixLayout_t,
    const void*,
    const void*,
    cublasLtMatrixLayout_t,
    void*,
    cublasLtMatrixLayout_t,
    const cublasLtMatmulAlgo_t*,
    void*,
    std::size_t,
    cudaStream_t);

using CublasLtPreferenceCreateFn =
    cublasStatus_t (*)(cublasLtMatmulPreference_t*);
using CublasLtPreferenceDestroyFn =
    cublasStatus_t (*)(cublasLtMatmulPreference_t);
using CublasLtPreferenceSetAttributeFn =
    cublasStatus_t (*)(cublasLtMatmulPreference_t, int, const void*, std::size_t);
using CublasLtAlgoGetHeuristicFn = cublasStatus_t (*)(
    cublasLtHandle_t,
    cublasLtMatmulDesc_t,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t,
    cublasLtMatmulPreference_t,
    int,
    cublasLtMatmulHeuristicResult_t*,
    int*);

constexpr int kCudaR32F = 0;
constexpr int kCublasCompute32F = 68;
constexpr int kLtDescTransA = 3;
constexpr int kLtDescTransB = 4;
constexpr int kLtDescEpilogue = 7;
constexpr int kLtDescBiasPointer = 8;
constexpr std::uint32_t kLtEpilogueGeluBias = 36U;
constexpr int kLtPrefMaxWorkspaceBytes = 1;
constexpr std::size_t kLtWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

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

struct CublasLtApi {
  void* library = nullptr;
  cublasLtHandle_t handle = nullptr;
  CublasLtCreateFn create = nullptr;
  CublasLtDestroyFn destroy = nullptr;
  CublasLtMatmulDescCreateFn matmul_desc_create = nullptr;
  CublasLtMatmulDescDestroyFn matmul_desc_destroy = nullptr;
  CublasLtMatmulDescSetAttributeFn matmul_desc_set_attribute = nullptr;
  CublasLtMatrixLayoutCreateFn matrix_layout_create = nullptr;
  CublasLtMatrixLayoutDestroyFn matrix_layout_destroy = nullptr;
  CublasLtMatmulFn matmul = nullptr;
  CublasLtPreferenceCreateFn preference_create = nullptr;
  CublasLtPreferenceDestroyFn preference_destroy = nullptr;
  CublasLtPreferenceSetAttributeFn preference_set_attribute = nullptr;
  CublasLtAlgoGetHeuristicFn algo_get_heuristic = nullptr;
  bool ready = false;

  CublasLtApi() {
    library = dlopen("libcublasLt.so.12", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return;
    }

    create = reinterpret_cast<CublasLtCreateFn>(dlsym(library, "cublasLtCreate"));
    destroy = reinterpret_cast<CublasLtDestroyFn>(dlsym(library, "cublasLtDestroy"));
    matmul_desc_create = reinterpret_cast<CublasLtMatmulDescCreateFn>(
        dlsym(library, "cublasLtMatmulDescCreate"));
    matmul_desc_destroy = reinterpret_cast<CublasLtMatmulDescDestroyFn>(
        dlsym(library, "cublasLtMatmulDescDestroy"));
    matmul_desc_set_attribute = reinterpret_cast<CublasLtMatmulDescSetAttributeFn>(
        dlsym(library, "cublasLtMatmulDescSetAttribute"));
    matrix_layout_create = reinterpret_cast<CublasLtMatrixLayoutCreateFn>(
        dlsym(library, "cublasLtMatrixLayoutCreate"));
    matrix_layout_destroy = reinterpret_cast<CublasLtMatrixLayoutDestroyFn>(
        dlsym(library, "cublasLtMatrixLayoutDestroy"));
    matmul = reinterpret_cast<CublasLtMatmulFn>(dlsym(library, "cublasLtMatmul"));
    preference_create = reinterpret_cast<CublasLtPreferenceCreateFn>(
        dlsym(library, "cublasLtMatmulPreferenceCreate"));
    preference_destroy = reinterpret_cast<CublasLtPreferenceDestroyFn>(
        dlsym(library, "cublasLtMatmulPreferenceDestroy"));
    preference_set_attribute = reinterpret_cast<CublasLtPreferenceSetAttributeFn>(
        dlsym(library, "cublasLtMatmulPreferenceSetAttribute"));
    algo_get_heuristic = reinterpret_cast<CublasLtAlgoGetHeuristicFn>(
        dlsym(library, "cublasLtMatmulAlgoGetHeuristic"));

    if (create == nullptr || destroy == nullptr ||
        matmul_desc_create == nullptr || matmul_desc_destroy == nullptr ||
        matmul_desc_set_attribute == nullptr ||
        matrix_layout_create == nullptr || matrix_layout_destroy == nullptr ||
        matmul == nullptr || preference_create == nullptr ||
        preference_destroy == nullptr || preference_set_attribute == nullptr ||
        algo_get_heuristic == nullptr) {
      return;
    }

    ready = create(&handle) == kCublasSuccess;
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

CublasLtApi& cublaslt_api() {
  static CublasLtApi api;
  return api;
}

struct LtDescriptors {
  CublasLtApi* api = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  const float* bias = nullptr;
  cublasLtMatmulAlgo_t algo{};
  bool has_algo = false;
  void* workspace = nullptr;
  std::size_t workspace_size = 0;

  ~LtDescriptors() {
    if (api == nullptr) {
      return;
    }
    if (workspace != nullptr) cudaFree(workspace);
    if (a != nullptr) api->matrix_layout_destroy(a);
    if (b != nullptr) api->matrix_layout_destroy(b);
    if (c != nullptr) api->matrix_layout_destroy(c);
    if (d != nullptr) api->matrix_layout_destroy(d);
    if (op != nullptr) api->matmul_desc_destroy(op);
  }
};


struct LtPlanKey {
  std::uint64_t m;
  std::uint64_t k;
  std::uint64_t n;

  bool operator==(const LtPlanKey& other) const {
    return m == other.m && k == other.k && n == other.n;
  }
};

struct LtPlanKeyHash {
  std::size_t operator()(const LtPlanKey& key) const {
    std::size_t h = static_cast<std::size_t>(key.m);
    h ^= static_cast<std::size_t>(key.k) + 0x9e3779b9U + (h << 6) + (h >> 2);
    h ^= static_cast<std::size_t>(key.n) + 0x9e3779b9U + (h << 6) + (h >> 2);
    return h;
  }
};

std::unique_ptr<LtDescriptors> create_lt_plan(
    CublasLtApi& api,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    const float* bias,
    bool choose_algo) {
  auto desc = std::make_unique<LtDescriptors>();
  desc->api = &api;

  if (api.matmul_desc_create(&desc->op, kCublasCompute32F, kCudaR32F) !=
      kCublasSuccess) {
    return nullptr;
  }

  const int trans_a = kCublasOpT;
  const int trans_b = kCublasOpN;
  const std::uint32_t epilogue = kLtEpilogueGeluBias;

  if (api.matmul_desc_set_attribute(
          desc->op, kLtDescTransA, &trans_a, sizeof(trans_a)) != kCublasSuccess ||
      api.matmul_desc_set_attribute(
          desc->op, kLtDescTransB, &trans_b, sizeof(trans_b)) != kCublasSuccess ||
      api.matmul_desc_set_attribute(
          desc->op, kLtDescEpilogue, &epilogue, sizeof(epilogue)) !=
          kCublasSuccess) {
    return nullptr;
  }

  if (api.matrix_layout_create(&desc->a, kCudaR32F, k, n, k) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc->b, kCudaR32F, k, m, k) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc->c, kCudaR32F, n, m, n) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc->d, kCudaR32F, n, m, n) !=
          kCublasSuccess) {
    return nullptr;
  }

  if (bias != nullptr) {
    const void* bias_ptr = bias;
    if (api.matmul_desc_set_attribute(
            desc->op,
            kLtDescBiasPointer,
            &bias_ptr,
            sizeof(bias_ptr)) != kCublasSuccess) {
      return nullptr;
    }
    desc->bias = bias;
  }

  if (choose_algo) {
    cublasLtMatmulPreference_t preference = nullptr;
    if (api.preference_create(&preference) != kCublasSuccess) {
      return nullptr;
    }

    const std::size_t max_workspace = kLtWorkspaceBytes;
    const cublasStatus_t pref_status = api.preference_set_attribute(
        preference,
        kLtPrefMaxWorkspaceBytes,
        &max_workspace,
        sizeof(max_workspace));
    if (pref_status != kCublasSuccess) {
      api.preference_destroy(preference);
      return nullptr;
    }

    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int returned = 0;
    const cublasStatus_t heuristic_status = api.algo_get_heuristic(
        api.handle,
        desc->op,
        desc->a,
        desc->b,
        desc->c,
        desc->d,
        preference,
        8,
        candidates,
        &returned);
    api.preference_destroy(preference);

    if (heuristic_status != kCublasSuccess || returned <= 0) {
      return nullptr;
    }

    for (int i = 0; i < returned; ++i) {
      if (candidates[i].state == kCublasSuccess &&
          candidates[i].workspace_size <= kLtWorkspaceBytes) {
        desc->algo = candidates[i].algo;
        desc->workspace_size = candidates[i].workspace_size;
        desc->has_algo = true;
        break;
      }
    }

    if (!desc->has_algo) {
      return nullptr;
    }

    if (desc->workspace_size > 0) {
      if (cudaMalloc(&desc->workspace, desc->workspace_size) != cudaSuccess) {
        return nullptr;
      }
    }
  }

  return desc;
}

LtDescriptors* cached_lt_plan(
    CublasLtApi& api,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    const float* bias,
    bool choose_algo) {
  thread_local std::unordered_map<
      LtPlanKey,
      std::unique_ptr<LtDescriptors>,
      LtPlanKeyHash> cache;

  const LtPlanKey key{m, k, n};
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second.get();
  }

  auto plan = create_lt_plan(api, m, k, n, bias, choose_algo);
  if (!plan) {
    return nullptr;
  }
  LtDescriptors* raw = plan.get();
  cache.emplace(key, std::move(plan));
  return raw;
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

__global__ void bias_gelu_epilogue_float4_kernel(
    float4* output4,
    const float4* bias4,
    std::uint64_t vec_elements,
    std::uint64_t vec_n) {
  for (std::uint64_t idx4 =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx4 < vec_elements;
       idx4 += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    const std::uint64_t col4 = idx4 % vec_n;
    const float4 y = output4[idx4];
    const float4 b = bias4[col4];
    float4 out;
    out.x = gelu_exact(y.x + b.x);
    out.y = gelu_exact(y.y + b.y);
    out.z = gelu_exact(y.z + b.z);
    out.w = gelu_exact(y.w + b.w);
    output4[idx4] = out;
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


extern "C" int cuda_operator_gemm_bias_gelu_v1(
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

  const bool aligned =
      (reinterpret_cast<std::uintptr_t>(output) % alignof(float4) == 0) &&
      (reinterpret_cast<std::uintptr_t>(bias) % alignof(float4) == 0);

  if ((n % 4 == 0) && aligned) {
    const std::uint64_t vec_n = n / 4;
    const std::uint64_t vec_elements = m * vec_n;
    const unsigned int blocks = static_cast<unsigned int>(
        (vec_elements + kThreads - 1) / kThreads);
    const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
    bias_gelu_epilogue_float4_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
        reinterpret_cast<float4*>(output),
        reinterpret_cast<const float4*>(bias),
        vec_elements,
        vec_n);
    return static_cast<int>(cudaGetLastError());
  }

  const std::uint64_t elements = m * n;
  const unsigned int blocks = static_cast<unsigned int>(
      (elements + kThreads - 1) / kThreads);
  const unsigned int capped_blocks = blocks > 4096U ? 4096U : blocks;
  bias_gelu_epilogue_kernel<<<capped_blocks, kThreads, 0, cuda_stream>>>(
      output, bias, elements, n);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_bias_gelu_v2(
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

  auto& api = cublaslt_api();
  if (!api.ready) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  LtDescriptors desc;
  desc.api = &api;

  if (api.matmul_desc_create(&desc.op, kCublasCompute32F, kCudaR32F) !=
      kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const int trans_a = kCublasOpT;
  const int trans_b = kCublasOpN;
  const std::uint32_t epilogue = kLtEpilogueGeluBias;
  const void* bias_ptr = bias;

  if (api.matmul_desc_set_attribute(
          desc.op, kLtDescTransA, &trans_a, sizeof(trans_a)) != kCublasSuccess ||
      api.matmul_desc_set_attribute(
          desc.op, kLtDescTransB, &trans_b, sizeof(trans_b)) != kCublasSuccess ||
      api.matmul_desc_set_attribute(
          desc.op, kLtDescEpilogue, &epilogue, sizeof(epilogue)) != kCublasSuccess ||
      api.matmul_desc_set_attribute(
          desc.op, kLtDescBiasPointer, &bias_ptr, sizeof(bias_ptr)) !=
          kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  // Row-major memory is viewed as column-major transposes:
  // W[N,K] -> A storage KxN, op(A)=A^T -> NxK
  // X[M,K] -> B storage KxM, op(B)=B -> KxM
  // Y[M,N] -> D storage NxM.
  if (api.matrix_layout_create(&desc.a, kCudaR32F, k, n, k) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc.b, kCudaR32F, k, m, k) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc.c, kCudaR32F, n, m, n) !=
          kCublasSuccess ||
      api.matrix_layout_create(&desc.d, kCudaR32F, n, m, n) !=
          kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  const cublasStatus_t status = api.matmul(
      api.handle,
      desc.op,
      &alpha,
      weight,
      desc.a,
      input,
      desc.b,
      &beta,
      output,
      desc.c,
      output,
      desc.d,
      nullptr,
      nullptr,
      0,
      cuda_stream);

  if (status != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_bias_gelu_v3(
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

  auto& api = cublaslt_api();
  if (!api.ready) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  LtDescriptors* desc = cached_lt_plan(api, m, k, n, bias, false);
  if (desc == nullptr) {
    return static_cast<int>(cudaErrorUnknown);
  }

  if (desc->bias != bias) {
    const void* bias_ptr = bias;
    if (api.matmul_desc_set_attribute(
            desc->op,
            kLtDescBiasPointer,
            &bias_ptr,
            sizeof(bias_ptr)) != kCublasSuccess) {
      return static_cast<int>(cudaErrorUnknown);
    }
    desc->bias = bias;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  const cublasStatus_t status = api.matmul(
      api.handle,
      desc->op,
      &alpha,
      weight,
      desc->a,
      input,
      desc->b,
      &beta,
      output,
      desc->c,
      output,
      desc->d,
      nullptr,
      nullptr,
      0,
      cuda_stream);

  if (status != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_bias_gelu_v4(
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

  auto& api = cublaslt_api();
  if (!api.ready) {
    return static_cast<int>(cudaErrorSharedObjectSymbolNotFound);
  }

  // Use a distinct thread-local key space from V3 by toggling the high bit of n.
  // The real descriptors still use the original n value.
  thread_local std::unordered_map<
      LtPlanKey,
      std::unique_ptr<LtDescriptors>,
      LtPlanKeyHash> tuned_cache;

  const LtPlanKey key{m, k, n};
  LtDescriptors* desc = nullptr;
  auto it = tuned_cache.find(key);
  if (it == tuned_cache.end()) {
    auto plan = create_lt_plan(api, m, k, n, bias, true);
    if (!plan) {
      return static_cast<int>(cudaErrorUnknown);
    }
    desc = plan.get();
    tuned_cache.emplace(key, std::move(plan));
  } else {
    desc = it->second.get();
  }

  if (desc->bias != bias) {
    const void* bias_ptr = bias;
    if (api.matmul_desc_set_attribute(
            desc->op,
            kLtDescBiasPointer,
            &bias_ptr,
            sizeof(bias_ptr)) != kCublasSuccess) {
      return static_cast<int>(cudaErrorUnknown);
    }
    desc->bias = bias;
  }

  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  const cublasStatus_t status = api.matmul(
      api.handle,
      desc->op,
      &alpha,
      weight,
      desc->a,
      input,
      desc->b,
      &beta,
      output,
      desc->c,
      output,
      desc->d,
      &desc->algo,
      desc->workspace,
      desc->workspace_size,
      cuda_stream);

  if (status != kCublasSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return static_cast<int>(cudaGetLastError());
}
