#include "softmax.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kSoftmaxV0Threads = 128;
constexpr int kSoftmaxBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarpsPerBlock = kSoftmaxBlockThreads / kWarpSize;
constexpr unsigned int kFullWarpMask = 0xffffffffU;

__device__ __forceinline__ float warp_reduce_max(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_reduce_max_fixed(
    float value,
    float* warp_partials) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value =
        lane < kMaxWarpsPerBlock ? warp_partials[lane] : -FLT_MAX;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  const float result = warp_partials[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_reduce_sum_fixed(
    float value,
    float* warp_partials) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value =
        lane < kMaxWarpsPerBlock ? warp_partials[lane] : 0.0F;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  return warp_partials[0];
}

__device__ __forceinline__ float block_reduce_max_dynamic(
    float value,
    float* warp_partials) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warp_count = blockDim.x / kWarpSize;

  value = warp_reduce_max(value);
  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value = lane < warp_count ? warp_partials[lane] : -FLT_MAX;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  const float result = warp_partials[0];
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_reduce_sum_dynamic(
    float value,
    float* warp_partials) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const int warp_count = blockDim.x / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value = lane < warp_count ? warp_partials[lane] : 0.0F;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  return warp_partials[0];
}

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

__global__ void softmax_v1_block_row_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  __shared__ float shared[kSoftmaxBlockThreads];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -FLT_MAX;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_input[col]);
  }

  shared[tid] = local_max;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
    }
    __syncthreads();
  }

  const float row_max = shared[0];
  __syncthreads();

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }

  shared[tid] = local_sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      shared[tid] += shared[tid + offset];
    }
    __syncthreads();
  }

  const float inverse_denominator = 1.0F / shared[0];
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] *= inverse_denominator;
  }
}

__global__ void softmax_v2_warp_row_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  __shared__ float warp_partials[kMaxWarpsPerBlock];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -FLT_MAX;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_input[col]);
  }
  const float row_max = block_reduce_max_fixed(local_max, warp_partials);

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }
  const float denominator = block_reduce_sum_fixed(local_sum, warp_partials);

  const float inverse_denominator = 1.0F / denominator;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] *= inverse_denominator;
  }
}

__global__ void softmax_v3_width_aware_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  __shared__ float warp_partials[kMaxWarpsPerBlock];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -FLT_MAX;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    local_max = fmaxf(local_max, row_input[col]);
  }
  const float row_max = block_reduce_max_dynamic(local_max, warp_partials);

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }
  const float denominator = block_reduce_sum_dynamic(local_sum, warp_partials);

  const float inverse_denominator = 1.0F / denominator;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] *= inverse_denominator;
  }
}


__global__ void softmax_v4_packed_warp_rows_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;
  const std::uint64_t row =
      static_cast<std::uint64_t>(blockIdx.x) * kMaxWarpsPerBlock + warp_id;

  if (row >= rows) {
    return;
  }

  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -FLT_MAX;
  for (std::uint64_t col = lane; col < cols; col += kWarpSize) {
    local_max = fmaxf(local_max, row_input[col]);
  }

  const float reduced_max = warp_reduce_max(local_max);
  const float row_max = __shfl_sync(kFullWarpMask, reduced_max, 0);

  float local_sum = 0.0F;
  for (std::uint64_t col = lane; col < cols; col += kWarpSize) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }

  const float reduced_sum = warp_reduce_sum(local_sum);
  const float denominator = __shfl_sync(kFullWarpMask, reduced_sum, 0);
  const float inverse_denominator = 1.0F / denominator;

  for (std::uint64_t col = lane; col < cols; col += kWarpSize) {
    row_output[col] *= inverse_denominator;
  }
}

int validate_softmax_arguments(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  if (rows == 0) {
    return static_cast<int>(cudaSuccess);
  }
  if (input == nullptr || output == nullptr || cols == 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaSuccess);
}

int softmax_v3_thread_count(std::uint64_t cols) {
  if (cols <= 32) {
    return 32;
  }
  if (cols <= 64) {
    return 64;
  }
  if (cols <= 128) {
    return 128;
  }
  return kSoftmaxBlockThreads;
}

}  // namespace

extern "C" int cuda_operator_softmax_v0(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation =
      validate_softmax_arguments(input, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  const std::uint64_t required_blocks =
      (rows + kSoftmaxV0Threads - 1) / kSoftmaxV0Threads;
  const int blocks = static_cast<int>(required_blocks);

  softmax_v0_serial_row_kernel<<<
      blocks,
      kSoftmaxV0Threads,
      0,
      cuda_stream>>>(input, output, rows, cols);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_softmax_v1(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation =
      validate_softmax_arguments(input, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  softmax_v1_block_row_kernel<<<
      static_cast<unsigned int>(rows),
      kSoftmaxBlockThreads,
      0,
      cuda_stream>>>(input, output, rows, cols);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_softmax_v2(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation =
      validate_softmax_arguments(input, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  softmax_v2_warp_row_kernel<<<
      static_cast<unsigned int>(rows),
      kSoftmaxBlockThreads,
      0,
      cuda_stream>>>(input, output, rows, cols);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int cuda_operator_softmax_v3(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation =
      validate_softmax_arguments(input, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  softmax_v3_width_aware_kernel<<<
      static_cast<unsigned int>(rows),
      softmax_v3_thread_count(cols),
      0,
      cuda_stream>>>(input, output, rows, cols);
  return static_cast<int>(cudaGetLastError());
}


extern "C" int cuda_operator_softmax_v4(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    void* stream) {
  const int validation =
      validate_softmax_arguments(input, output, rows, cols);
  if (validation != static_cast<int>(cudaSuccess) || rows == 0) {
    return validation;
  }

  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  if (cols <= 128) {
    const std::uint64_t required_blocks =
        (rows + kMaxWarpsPerBlock - 1) / kMaxWarpsPerBlock;
    softmax_v4_packed_warp_rows_kernel<<<
        static_cast<unsigned int>(required_blocks),
        kSoftmaxBlockThreads,
        0,
        cuda_stream>>>(input, output, rows, cols);
  } else {
    softmax_v2_warp_row_kernel<<<
        static_cast<unsigned int>(rows),
        kSoftmaxBlockThreads,
        0,
        cuda_stream>>>(input, output, rows, cols);
  }

  return static_cast<int>(cudaGetLastError());
}
