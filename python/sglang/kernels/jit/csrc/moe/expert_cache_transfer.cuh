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

// A segment is {source address, destination address, row bytes}.
__device__ __forceinline__ void copy_expert_row_segments_lane(
    const int64_t* segments,
    int64_t segment_count,
    int64_t source_row,
    int64_t destination_slot,
    int64_t lane,
    int64_t lanes) {
  for (int64_t segment = 0; segment < segment_count; ++segment) {
    const int64_t* entry = segments + 3 * segment;
    const int64_t row_bytes = entry[2];
    copy_expert_row_lane(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(entry[0])) + source_row * row_bytes,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(entry[1])) + destination_slot * row_bytes,
        row_bytes,
        lane,
        lanes);
  }
}

// Every launched thread works on the rows the device-side count selects. With
// fewer rows than threads, each row owns a contiguous range of threads, one lane
// per thread, so a handful of misses reads host memory with hundreds of
// concurrent lanes; contiguity keeps each launch block on one row's host pages.
// With more rows than threads, each thread walks one row in every `threads`.
__device__ __forceinline__ void copy_expert_rows_for_thread(
    const int64_t* segments,
    int64_t segment_count,
    const int64_t* source_rows,
    const int32_t* destination_slots,
    int64_t active_count,
    int64_t thread,
    int64_t total_threads) {
  if (active_count <= 0) {
    return;
  }

  if (active_count <= total_threads) {
    const int64_t row = thread * active_count / total_threads;
    const int64_t first_thread = (row * total_threads + active_count - 1) / active_count;
    const int64_t next_thread = ((row + 1) * total_threads + active_count - 1) / active_count;
    copy_expert_row_segments_lane(
        segments,
        segment_count,
        source_rows[row],
        static_cast<int64_t>(destination_slots[row]),
        thread - first_thread,
        next_thread - first_thread);
    return;
  }

  for (int64_t row = thread; row < active_count; row += total_threads) {
    copy_expert_row_segments_lane(
        segments, segment_count, source_rows[row], static_cast<int64_t>(destination_slots[row]), 0, 1);
  }
}

__global__ __launch_bounds__(kExpertTransferBlockSize, 1) void copy_expert_rows_gpu_kernel(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count,
    int64_t row_bytes) {
  const int64_t segment[3] = {
      static_cast<int64_t>(reinterpret_cast<intptr_t>(source)),
      static_cast<int64_t>(reinterpret_cast<intptr_t>(destination)),
      row_bytes};
  copy_expert_rows_for_thread(
      segment,
      1,
      source_rows,
      destination_slots,
      count[0],
      static_cast<int64_t>(blockIdx.x) * kExpertTransferBlockSize + threadIdx.x,
      static_cast<int64_t>(gridDim.x) * kExpertTransferBlockSize);
}

__global__ __launch_bounds__(kExpertTransferBlockSize, 1) void copy_expert_row_segments_gpu_kernel(
    const int64_t* __restrict__ segments,
    int64_t segment_count,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count) {
  copy_expert_rows_for_thread(
      segments,
      segment_count,
      source_rows,
      destination_slots,
      count[0],
      static_cast<int64_t>(blockIdx.x) * kExpertTransferBlockSize + threadIdx.x,
      static_cast<int64_t>(gridDim.x) * kExpertTransferBlockSize);
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

void copy_expert_row_segments_gpu(
    tvm::ffi::TensorView segments,
    tvm::ffi::TensorView source_rows,
    tvm::ffi::TensorView destination_slots,
    tvm::ffi::TensorView count) {
  const auto device = host::LaunchKernel::resolve_device(segments.device());
  host::LaunchKernel(kExpertTransferGridSize, kExpertTransferBlockSize, device)(
      copy_expert_row_segments_gpu_kernel,
      static_cast<const int64_t*>(segments.data_ptr()),
      static_cast<int64_t>(segments.size(0)),
      static_cast<const int64_t*>(source_rows.data_ptr()),
      static_cast<const int32_t*>(destination_slots.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()));
}

}  // namespace sglang
