#include "gemm_swiglu.cuh"

#include <cstdint>
#include <dlfcn.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

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

__global__ void gemm_swiglu_v6_wmma_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    float* output,
    int m,
    int k,
    int n) {
  using namespace nvcuda;

  const int tile_n = static_cast<int>(blockIdx.x) * 16;
  const int tile_m = static_cast<int>(blockIdx.y) * 16;

  if (tile_m >= m || tile_n >= n) {
    return;
  }

  wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::row_major> a_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> gate_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> up_frag;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> gate_acc;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> up_acc;

  wmma::fill_fragment(gate_acc, 0.0F);
  wmma::fill_fragment(up_acc, 0.0F);

  for (int kk = 0; kk < k; kk += 16) {
    const __nv_bfloat16* a_ptr =
        input + static_cast<std::uint64_t>(tile_m) * k + kk;

    // Row-major W[N,K] is the same memory as column-major W^T[K,N].
    const __nv_bfloat16* gate_ptr =
        gate_weight + static_cast<std::uint64_t>(tile_n) * k + kk;
    const __nv_bfloat16* up_ptr =
        up_weight + static_cast<std::uint64_t>(tile_n) * k + kk;

    wmma::load_matrix_sync(a_frag, a_ptr, k);
    wmma::load_matrix_sync(gate_frag, gate_ptr, k);
    wmma::load_matrix_sync(up_frag, up_ptr, k);

    wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
    wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);
  }

  #pragma unroll
  for (int i = 0; i < gate_acc.num_elements; ++i) {
    gate_acc.x[i] = silu(gate_acc.x[i]) * up_acc.x[i];
  }

  float* out_ptr =
      output + static_cast<std::uint64_t>(tile_m) * n + tile_n;
  wmma::store_matrix_sync(
      out_ptr,
      gate_acc,
      n,
      wmma::mem_row_major);
}

__global__ void gemm_swiglu_v7_shared_a_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    float* output,
    int m,
    int k,
    int n) {
  using namespace nvcuda;

  __shared__ __nv_bfloat16 shared_a[16 * 16];

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int tile_m = static_cast<int>(blockIdx.y) * 16;
  const int tile_n = static_cast<int>(blockIdx.x) * 64 + warp_id * 16;

  if (tile_m >= m || tile_n >= n) {
    return;
  }

  wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::row_major> a_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> gate_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> up_frag;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> gate_acc;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> up_acc;

  wmma::fill_fragment(gate_acc, 0.0F);
  wmma::fill_fragment(up_acc, 0.0F);

  for (int kk = 0; kk < k; kk += 16) {
    // 128 threads cooperatively load 256 BF16 values: two values/thread.
    const int first = static_cast<int>(threadIdx.x);
    const int second = first + 128;

    const int first_row = first / 16;
    const int first_col = first - first_row * 16;
    shared_a[first] =
        input[static_cast<std::uint64_t>(tile_m + first_row) * k +
              kk + first_col];

    const int second_row = second / 16;
    const int second_col = second - second_row * 16;
    shared_a[second] =
        input[static_cast<std::uint64_t>(tile_m + second_row) * k +
              kk + second_col];

    __syncthreads();

    wmma::load_matrix_sync(a_frag, shared_a, 16);

    const __nv_bfloat16* gate_ptr =
        gate_weight + static_cast<std::uint64_t>(tile_n) * k + kk;
    const __nv_bfloat16* up_ptr =
        up_weight + static_cast<std::uint64_t>(tile_n) * k + kk;

    wmma::load_matrix_sync(gate_frag, gate_ptr, k);
    wmma::load_matrix_sync(up_frag, up_ptr, k);

    wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
    wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);

    // All warps must finish consuming shared_a before the next K tile overwrites it.
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < gate_acc.num_elements; ++i) {
    gate_acc.x[i] = silu(gate_acc.x[i]) * up_acc.x[i];
  }

  float* out_ptr =
      output + static_cast<std::uint64_t>(tile_m) * n + tile_n;
  wmma::store_matrix_sync(
      out_ptr,
      gate_acc,
      n,
      wmma::mem_row_major);

  (void)lane;
}

__global__ void gemm_swiglu_v8_shared_ab_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    float* output,
    int m,
    int k,
    int n) {
  using namespace nvcuda;

  __shared__ __nv_bfloat16 shared_a[2 * 16 * 16];
  __shared__ __nv_bfloat16 shared_gate[4 * 16 * 16];
  __shared__ __nv_bfloat16 shared_up[4 * 16 * 16];

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int row_group = warp_id / 4;
  const int col_group = warp_id - row_group * 4;

  const int block_m = static_cast<int>(blockIdx.y) * 32;
  const int block_n = static_cast<int>(blockIdx.x) * 64;
  const int tile_m = block_m + row_group * 16;
  const int tile_n = block_n + col_group * 16;

  wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::row_major> a_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> gate_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> up_frag;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> gate_acc;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> up_acc;

  wmma::fill_fragment(gate_acc, 0.0F);
  wmma::fill_fragment(up_acc, 0.0F);

  for (int kk = 0; kk < k; kk += 16) {
    // Stage 512 A values + 1024 gate-B values + 1024 up-B values.
    for (int idx = static_cast<int>(threadIdx.x);
         idx < 2560;
         idx += static_cast<int>(blockDim.x)) {
      if (idx < 512) {
        const int rg = idx / 256;
        const int local = idx - rg * 256;
        const int i = local / 16;
        const int j = local - i * 16;
        shared_a[idx] =
            input[static_cast<std::uint64_t>(block_m + rg * 16 + i) * k +
                  kk + j];
      } else if (idx < 1536) {
        const int local_all = idx - 512;
        const int cg = local_all / 256;
        const int local = local_all - cg * 256;
        const int j = local / 16;
        const int i = local - j * 16;
        shared_gate[local_all] =
            gate_weight[
                static_cast<std::uint64_t>(block_n + cg * 16 + j) * k +
                kk + i];
      } else {
        const int local_all = idx - 1536;
        const int cg = local_all / 256;
        const int local = local_all - cg * 256;
        const int j = local / 16;
        const int i = local - j * 16;
        shared_up[local_all] =
            up_weight[
                static_cast<std::uint64_t>(block_n + cg * 16 + j) * k +
                kk + i];
      }
    }

    __syncthreads();

    wmma::load_matrix_sync(
        a_frag,
        shared_a + row_group * 256,
        16);
    wmma::load_matrix_sync(
        gate_frag,
        shared_gate + col_group * 256,
        16);
    wmma::load_matrix_sync(
        up_frag,
        shared_up + col_group * 256,
        16);

    wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
    wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < gate_acc.num_elements; ++i) {
    gate_acc.x[i] = silu(gate_acc.x[i]) * up_acc.x[i];
  }

  float* out_ptr =
      output + static_cast<std::uint64_t>(tile_m) * n + tile_n;
  wmma::store_matrix_sync(
      out_ptr,
      gate_acc,
      n,
      wmma::mem_row_major);
}

__global__ void gemm_swiglu_v9_shared_a_8warp_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weight,
    const __nv_bfloat16* up_weight,
    float* output,
    int m,
    int k,
    int n) {
  using namespace nvcuda;

  __shared__ __nv_bfloat16 shared_a[16 * 16];

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int tile_m = static_cast<int>(blockIdx.y) * 16;
  const int tile_n = static_cast<int>(blockIdx.x) * 128 + warp_id * 16;

  wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::row_major> a_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> gate_frag;
  wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major> up_frag;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> gate_acc;
  wmma::fragment<
      wmma::accumulator,
      16,
      16,
      16,
      float> up_acc;

  wmma::fill_fragment(gate_acc, 0.0F);
  wmma::fill_fragment(up_acc, 0.0F);

  for (int kk = 0; kk < k; kk += 16) {
    const int idx = static_cast<int>(threadIdx.x);
    const int row = idx / 16;
    const int col = idx - row * 16;
    shared_a[idx] =
        input[static_cast<std::uint64_t>(tile_m + row) * k + kk + col];

    __syncthreads();

    wmma::load_matrix_sync(a_frag, shared_a, 16);

    const __nv_bfloat16* gate_ptr =
        gate_weight + static_cast<std::uint64_t>(tile_n) * k + kk;
    const __nv_bfloat16* up_ptr =
        up_weight + static_cast<std::uint64_t>(tile_n) * k + kk;

    wmma::load_matrix_sync(gate_frag, gate_ptr, k);
    wmma::load_matrix_sync(up_frag, up_ptr, k);

    wmma::mma_sync(gate_acc, a_frag, gate_frag, gate_acc);
    wmma::mma_sync(up_acc, a_frag, up_frag, up_acc);

    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < gate_acc.num_elements; ++i) {
    gate_acc.x[i] = silu(gate_acc.x[i]) * up_acc.x[i];
  }

  float* out_ptr =
      output + static_cast<std::uint64_t>(tile_m) * n + tile_n;
  wmma::store_matrix_sync(
      out_ptr,
      gate_acc,
      n,
      wmma::mem_row_major);
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


extern "C" int cuda_operator_gemm_swiglu_v6(
    const void* input_bf16,
    const void* gate_weight_bf16,
    const void* up_weight_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || gate_weight_bf16 == nullptr ||
      up_weight_bf16 == nullptr || output == nullptr ||
      k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((m % 16) != 0 || (k % 16) != 0 || (n % 16) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 grid(
      static_cast<unsigned int>(n / 16),
      static_cast<unsigned int>(m / 16),
      1U);
  const dim3 block(32U, 1U, 1U);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  gemm_swiglu_v6_wmma_kernel<<<grid, block, 0, cuda_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gate_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(up_weight_bf16),
      output,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(n));
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v7(
    const void* input_bf16,
    const void* gate_weight_bf16,
    const void* up_weight_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || gate_weight_bf16 == nullptr ||
      up_weight_bf16 == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((m % 16) != 0 || (k % 16) != 0 || (n % 64) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 grid(
      static_cast<unsigned int>(n / 64),
      static_cast<unsigned int>(m / 16),
      1U);
  const dim3 block(128U, 1U, 1U);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  gemm_swiglu_v7_shared_a_kernel<<<grid, block, 0, cuda_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gate_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(up_weight_bf16),
      output,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(n));
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v8(
    const void* input_bf16,
    const void* gate_weight_bf16,
    const void* up_weight_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || gate_weight_bf16 == nullptr ||
      up_weight_bf16 == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((m % 32) != 0 || (k % 16) != 0 || (n % 64) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 grid(
      static_cast<unsigned int>(n / 64),
      static_cast<unsigned int>(m / 32),
      1U);
  const dim3 block(256U, 1U, 1U);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  gemm_swiglu_v8_shared_ab_kernel<<<grid, block, 0, cuda_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gate_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(up_weight_bf16),
      output,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(n));
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_gemm_swiglu_v9(
    const void* input_bf16,
    const void* gate_weight_bf16,
    const void* up_weight_bf16,
    float* output,
    std::uint64_t m,
    std::uint64_t k,
    std::uint64_t n,
    void* stream) {
  if (m == 0 || n == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input_bf16 == nullptr || gate_weight_bf16 == nullptr ||
      up_weight_bf16 == nullptr || output == nullptr || k == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if ((m % 16) != 0 || (k % 16) != 0 || (n % 128) != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (m > static_cast<std::uint64_t>(INT32_MAX) ||
      n > static_cast<std::uint64_t>(INT32_MAX) ||
      k > static_cast<std::uint64_t>(INT32_MAX)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 grid(
      static_cast<unsigned int>(n / 128),
      static_cast<unsigned int>(m / 16),
      1U);
  const dim3 block(256U, 1U, 1U);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  gemm_swiglu_v9_shared_a_8warp_kernel<<<grid, block, 0, cuda_stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input_bf16),
      reinterpret_cast<const __nv_bfloat16*>(gate_weight_bf16),
      reinterpret_cast<const __nv_bfloat16*>(up_weight_bf16),
      output,
      static_cast<int>(m),
      static_cast<int>(k),
      static_cast<int>(n));
  return static_cast<int>(cudaGetLastError());
}
