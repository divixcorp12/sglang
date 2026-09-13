#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

namespace sglang {

constexpr int kExpertTransferBlockSize = 256;
constexpr int kExpertTransferGridSize = 8;
constexpr int kExpertTransferWarpSize = 32;

__device__ __forceinline__ uint32_t load_expert_host_word_noncoherent(const uint32_t* address) {
#ifdef USE_ROCM
  return *address;
#else
  uint32_t value;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
#endif
}

__device__ __forceinline__ void store_expert_device_word_cached(uint32_t* address, uint32_t value) {
#ifdef USE_ROCM
  *address = value;
#else
  asm volatile("st.global.cg.u32 [%0], %1;" : : "l"(address), "r"(value));
#endif
}

__device__ __forceinline__ void copy_expert_row_lane(
    const uint8_t* src_row, uint8_t* dst_row, int64_t row_bytes, int64_t lane, int64_t lanes) {
  const bool word_aligned =
      ((reinterpret_cast<uintptr_t>(src_row) | reinterpret_cast<uintptr_t>(dst_row)) & 3) == 0;
  const int64_t word_bytes = word_aligned ? (row_bytes / sizeof(uint32_t)) * sizeof(uint32_t) : 0;

  if (word_aligned) {
    const auto src_words = reinterpret_cast<const uint32_t*>(src_row);
    auto dst_words = reinterpret_cast<uint32_t*>(dst_row);
    const int64_t word_count = word_bytes / sizeof(uint32_t);
    for (int64_t word = lane; word < word_count; word += lanes) {
      store_expert_device_word_cached(dst_words + word, load_expert_host_word_noncoherent(src_words + word));
    }
  }

  for (int64_t byte = word_bytes + lane; byte < row_bytes; byte += lanes) {
    dst_row[byte] = src_row[byte];
  }
}

// Every launched thread works on the rows the device-side count selects. With
// fewer rows than threads, each row owns a contiguous range of threads, one lane
// per thread, so a handful of misses reads host memory with hundreds of
// concurrent lanes; contiguity keeps each launch block on one row's host pages.
// With more rows than threads, each thread walks one row in every `threads`.
__global__ __launch_bounds__(kExpertTransferBlockSize, 1) void copy_expert_rows_gpu_kernel(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count,
    int64_t row_bytes) {
  const int64_t total_threads = static_cast<int64_t>(gridDim.x) * kExpertTransferBlockSize;
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * kExpertTransferBlockSize + threadIdx.x;
  const int64_t active_count = count[0];
  if (active_count <= 0) {
    return;
  }

  if (active_count <= total_threads) {
    const int64_t row = thread * active_count / total_threads;
    const int64_t first_thread = (row * total_threads + active_count - 1) / active_count;
    const int64_t next_thread = ((row + 1) * total_threads + active_count - 1) / active_count;
    const int64_t lane = thread - first_thread;
    const int64_t lanes = next_thread - first_thread;
    copy_expert_row_lane(
        source + source_rows[row] * row_bytes,
        destination + static_cast<int64_t>(destination_slots[row]) * row_bytes,
        row_bytes,
        lane,
        lanes);
    return;
  }

  for (int64_t row = thread; row < active_count; row += total_threads) {
    copy_expert_row_lane(
        source + source_rows[row] * row_bytes,
        destination + static_cast<int64_t>(destination_slots[row]) * row_bytes,
        row_bytes,
        0,
        1);
  }
}

void copy_expert_rows_gpu(
    tvm::ffi::TensorView source,
    tvm::ffi::TensorView destination,
    tvm::ffi::TensorView source_rows,
    tvm::ffi::TensorView destination_slots,
    tvm::ffi::TensorView count) {
  const int64_t element_bytes = source.dtype().bits * source.dtype().lanes / 8;
  const int64_t row_bytes = source.stride(0) * element_bytes;
  const auto device = host::LaunchKernel::resolve_device(destination.device());
  host::LaunchKernel(kExpertTransferGridSize, kExpertTransferBlockSize, device)(
      copy_expert_rows_gpu_kernel,
      static_cast<const uint8_t*>(source.data_ptr()),
      static_cast<uint8_t*>(destination.data_ptr()),
      static_cast<const int64_t*>(source_rows.data_ptr()),
      static_cast<const int32_t*>(destination_slots.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      row_bytes);
}

}  // namespace sglang
