#include "softmax.cuh"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

namespace {

constexpr int kSoftmaxV0Threads = 128;
constexpr int kSoftmaxBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kSoftmaxBlockThreads / kWarpSize;
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

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_max_width(
    float value,
    float* warp_partials) {
  constexpr int kBlockWarps = BlockThreads / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_max(value);

  if constexpr (kBlockWarps == 1) {
    // A one-warp block needs no shared-memory handoff. Broadcasting lane 0's
    // final register value also avoids a read/write race on warp_partials[0]
    // under independent thread scheduling.
    return __shfl_sync(kFullWarpMask, value, 0);
  }

  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value =
        lane < kBlockWarps ? warp_partials[lane] : -FLT_MAX;
    block_value = warp_reduce_max(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  const float result = warp_partials[0];

  // The same scratch space is reused by the denominator reduction. Ensure all
  // threads finish reading row_max before any warp overwrites the partials.
  __syncthreads();
  return result;
}

template <int BlockThreads>
__device__ __forceinline__ float block_reduce_sum_width(
    float value,
    float* warp_partials) {
  constexpr int kBlockWarps = BlockThreads / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_id = threadIdx.x / kWarpSize;

  value = warp_reduce_sum(value);

  if constexpr (kBlockWarps == 1) {
    return __shfl_sync(kFullWarpMask, value, 0);
  }

  if (lane == 0) {
    warp_partials[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    float block_value =
        lane < kBlockWarps ? warp_partials[lane] : 0.0F;
    block_value = warp_reduce_sum(block_value);
    if (lane == 0) {
      warp_partials[0] = block_value;
    }
  }
  __syncthreads();

  return warp_partials[0];
}

__device__ __forceinline__ float block_reduce_max(
    float value,
    float* warp_partials) {
  return block_reduce_max_width<kSoftmaxBlockThreads>(value, warp_partials);
}

__device__ __forceinline__ float block_reduce_sum(
    float value,
    float* warp_partials) {
  return block_reduce_sum_width<kSoftmaxBlockThreads>(value, warp_partials);
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
  __shared__ float warp_partials[kWarpsPerBlock];

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
  const float row_max = block_reduce_max(local_max, warp_partials);

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }
  const float denominator = block_reduce_sum(local_sum, warp_partials);

  const float inverse_denominator = 1.0F / denominator;
  for (std::uint64_t col = tid; col < cols; col += blockDim.x) {
    row_output[col] *= inverse_denominator;
  }
}

template <int BlockThreads>
__global__ void softmax_v3_width_row_kernel(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols) {
  static_assert(BlockThreads % kWarpSize == 0);
  constexpr int kBlockWarps = BlockThreads / kWarpSize;
  __shared__ float warp_partials[kBlockWarps];

  const std::uint64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  const unsigned int tid = threadIdx.x;
  const float* row_input = input + row * cols;
  float* row_output = output + row * cols;

  float local_max = -FLT_MAX;
  for (std::uint64_t col = tid; col < cols; col += BlockThreads) {
    local_max = fmaxf(local_max, row_input[col]);
  }
  const float row_max =
      block_reduce_max_width<BlockThreads>(local_max, warp_partials);

  float local_sum = 0.0F;
  for (std::uint64_t col = tid; col < cols; col += BlockThreads) {
    const float value = expf(row_input[col] - row_max);
    row_output[col] = value;
    local_sum += value;
  }
  const float denominator =
      block_reduce_sum_width<BlockThreads>(local_sum, warp_partials);

  const float inverse_denominator = 1.0F / denominator;
  for (std::uint64_t col = tid; col < cols; col += BlockThreads) {
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

template <int BlockThreads>
void launch_softmax_v3(
    const float* input,
    float* output,
    std::uint64_t rows,
    std::uint64_t cols,
    cudaStream_t stream) {
  softmax_v3_width_row_kernel<BlockThreads><<<
      static_cast<unsigned int>(rows),
      BlockThreads,
      0,
      stream>>>(input, output, rows, cols);
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
  if (cols <= 32) {
    launch_softmax_v3<32>(input, output, rows, cols, cuda_stream);
  } else if (cols <= 64) {
    launch_softmax_v3<64>(input, output, rows, cols, cuda_stream);
  } else if (cols <= 128) {
    launch_softmax_v3<128>(input, output, rows, cols, cuda_stream);
  } else {
    // Preserve V2 exactly for wider rows so this experiment isolates the
    // effect of reducing block width on narrow rows.
    softmax_v2_warp_row_kernel<<<
        static_cast<unsigned int>(rows),
        kSoftmaxBlockThreads,
        0,
        cuda_stream>>>(input, output, rows, cols);
  }
  return static_cast<int>(cudaGetLastError());
}
