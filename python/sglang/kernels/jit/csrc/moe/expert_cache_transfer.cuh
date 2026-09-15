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

__device__ __forceinline__ void copy_expert_host_unit16(const uint8_t* src, uint8_t* dst) {
#ifdef USE_ROCM
  reinterpret_cast<uint64_t*>(dst)[0] = reinterpret_cast<const uint64_t*>(src)[0];
  reinterpret_cast<uint64_t*>(dst)[1] = reinterpret_cast<const uint64_t*>(src)[1];
#else
  uint64_t lo, hi;
  asm volatile("ld.global.nc.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" : : "l"(dst), "l"(lo), "l"(hi) : "memory");
#endif
}

// Copies lane `lane` of warp `warp` out of `warps` warps sharing one row.
// Unit i of the row belongs to lane i % 32 of warp (i / 32) % warps, so the 32
// lanes of a warp touch 32 adjacent 16-byte units on every step and the device
// batches them into one tile read instead of issuing scattered 4-byte loads.
// Bytes past the last whole 16-byte unit, or rows whose addresses are not
// 16-byte aligned, fall back to 4-byte words and then single bytes in the same
// lane pattern.
__device__ __forceinline__ void copy_expert_row_lane(
    const uint8_t* src_row, uint8_t* dst_row, int64_t row_bytes, int64_t warp, int64_t warps, int64_t lane) {
  const int64_t stride = warps * kExpertTransferWarpSize;
  const int64_t first = warp * kExpertTransferWarpSize + lane;
  const bool unit_aligned =
      ((reinterpret_cast<uintptr_t>(src_row) | reinterpret_cast<uintptr_t>(dst_row)) & 15) == 0;
  const int64_t unit_count = unit_aligned ? row_bytes / 16 : 0;
  for (int64_t unit = first; unit < unit_count; unit += stride) {
    copy_expert_host_unit16(src_row + 16 * unit, dst_row + 16 * unit);
  }

  const int64_t tail = 16 * unit_count;
  const bool word_aligned =
      ((reinterpret_cast<uintptr_t>(src_row + tail) | reinterpret_cast<uintptr_t>(dst_row + tail)) & 3) == 0;
  const int64_t word_count = word_aligned ? (row_bytes - tail) / 4 : 0;
  const auto src_words = reinterpret_cast<const uint32_t*>(src_row + tail);
  auto dst_words = reinterpret_cast<uint32_t*>(dst_row + tail);
  for (int64_t word = first; word < word_count; word += stride) {
    store_expert_device_word_cached(dst_words + word, load_expert_host_word_noncoherent(src_words + word));
  }

  const int64_t byte_tail = tail + 4 * word_count;
  for (int64_t byte = byte_tail + first; byte < row_bytes; byte += stride) {
    dst_row[byte] = src_row[byte];
  }
}

// A segment is {source address, destination address, row bytes}.
__device__ __forceinline__ void copy_expert_row_segments_lane(
    const int64_t* segments,
    int64_t segment_count,
    int64_t source_row,
    int64_t destination_slot,
    int64_t warp,
    int64_t warps,
    int64_t lane) {
  for (int64_t segment = 0; segment < segment_count; ++segment) {
    const int64_t* entry = segments + 3 * segment;
    const int64_t row_bytes = entry[2];
    copy_expert_row_lane(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(entry[0])) + source_row * row_bytes,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(entry[1])) + destination_slot * row_bytes,
        row_bytes,
        warp,
        warps,
        lane);
  }
}

// Every launched thread works on the rows the device-side count selects. With
// no more rows than warps, each row owns a contiguous range of whole warps, so
// every lane of a warp reads the same row and adjacent host units stay in one
// warp. With more rows than warps, each warp walks one row in every `warps`,
// all 32 of its lanes on that row.
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

  const int64_t warp = thread / kExpertTransferWarpSize;
  const int64_t lane = thread % kExpertTransferWarpSize;
  const int64_t total_warps = total_threads / kExpertTransferWarpSize;

  if (active_count <= total_warps) {
    const int64_t row = warp * active_count / total_warps;
    const int64_t first_warp = (row * total_warps + active_count - 1) / active_count;
    const int64_t next_warp = ((row + 1) * total_warps + active_count - 1) / active_count;
    copy_expert_row_segments_lane(
        segments,
        segment_count,
        source_rows[row],
        static_cast<int64_t>(destination_slots[row]),
        warp - first_warp,
        next_warp - first_warp,
        lane);
    return;
  }

  for (int64_t row = warp; row < active_count; row += total_warps) {
    copy_expert_row_segments_lane(
        segments, segment_count, source_rows[row], static_cast<int64_t>(destination_slots[row]), 0, 1, lane);
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
