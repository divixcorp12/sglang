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

__global__ __launch_bounds__(kExpertTransferBlockSize, 1) void copy_expert_rows_gpu_kernel(
    const uint8_t* __restrict__ source,
    uint8_t* __restrict__ destination,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count,
    int64_t row_bytes) {
  constexpr int kWarpsPerBlock = kExpertTransferBlockSize / kExpertTransferWarpSize;
  const int lane = threadIdx.x % kExpertTransferWarpSize;
  const int warp = blockIdx.x * kWarpsPerBlock + threadIdx.x / kExpertTransferWarpSize;
  const int total_warps = gridDim.x * kWarpsPerBlock;
  const int active_count = count[0];

  for (int plan_index = warp; plan_index < active_count; plan_index += total_warps) {
    const auto src_row = source + source_rows[plan_index] * row_bytes;
    auto dst_row = destination + static_cast<int64_t>(destination_slots[plan_index]) * row_bytes;
    const bool word_aligned =
        ((reinterpret_cast<uintptr_t>(src_row) | reinterpret_cast<uintptr_t>(dst_row)) & 3) == 0;
    const int64_t word_bytes = word_aligned ? (row_bytes / sizeof(uint32_t)) * sizeof(uint32_t) : 0;

    if (word_aligned) {
      const auto src_words = reinterpret_cast<const uint32_t*>(src_row);
      auto dst_words = reinterpret_cast<uint32_t*>(dst_row);
      const int64_t word_count = word_bytes / sizeof(uint32_t);
      for (int64_t word = lane; word < word_count; word += kExpertTransferWarpSize) {
        store_expert_device_word_cached(
            dst_words + word, load_expert_host_word_noncoherent(src_words + word));
      }
    }

    for (int64_t byte = word_bytes + lane; byte < row_bytes; byte += kExpertTransferWarpSize) {
      dst_row[byte] = src_row[byte];
    }
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
