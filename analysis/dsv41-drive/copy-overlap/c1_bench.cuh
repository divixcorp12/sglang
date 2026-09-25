// Bench-only launchers for the C1 copy: production kernel at any grid, and an
// unrolled variant that keeps U 16-byte loads in flight per thread.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "moe/expert_cache_transfer.cuh"

namespace sglang {

// Every unit is 16 B and both ends are 16-byte aligned (the six segment row sizes are multiples of 512 B).
template <int U>
__device__ __forceinline__ void bench_copy_row_unrolled(
    const uint8_t* src_row, uint8_t* dst_row, int64_t row_bytes, int64_t warp, int64_t warps, int64_t lane) {
  const int64_t stride = warps * kExpertTransferWarpSize;
  const int64_t units = row_bytes / 16;
  const auto src = reinterpret_cast<const uint4*>(src_row);
  auto dst = reinterpret_cast<uint4*>(dst_row);
  for (int64_t base = warp * kExpertTransferWarpSize + lane; base < units; base += stride * U) {
    uint4 v[U];
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int64_t u = base + k * stride;
      if (u < units) {
        asm("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4];"
            : "=r"(v[k].x), "=r"(v[k].y), "=r"(v[k].z), "=r"(v[k].w)
            : "l"(src + u));
      }
    }
#pragma unroll
    for (int k = 0; k < U; ++k) {
      const int64_t u = base + k * stride;
      if (u < units) {
        asm volatile("st.global.cg.v4.u32 [%0],{%1,%2,%3,%4};" : : "l"(dst + u), "r"(v[k].x), "r"(v[k].y), "r"(v[k].z), "r"(v[k].w) : "memory");
      }
    }
  }
}

template <int U>
__global__ __launch_bounds__(kExpertTransferBlockSize, 1) void bench_copy_unrolled_kernel(
    const int64_t* __restrict__ segments,
    int64_t segment_count,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count) {
  const int64_t active = count[0];
  const int64_t thread = static_cast<int64_t>(blockIdx.x) * kExpertTransferBlockSize + threadIdx.x;
  const int64_t total_warps = static_cast<int64_t>(gridDim.x) * kExpertTransferBlockSize / kExpertTransferWarpSize;
  const int64_t warp = thread / kExpertTransferWarpSize;
  const int64_t lane = thread % kExpertTransferWarpSize;
  if (active <= 0) return;
  int64_t row, w, ws;
  if (active <= total_warps) {
    row = warp * active / total_warps;
    const int64_t first = (row * total_warps + active - 1) / active;
    const int64_t next = ((row + 1) * total_warps + active - 1) / active;
    w = warp - first;
    ws = next - first;
  } else {
    row = warp;
    w = 0;
    ws = 1;
  }
  for (; row < active; row += (active <= total_warps ? active : total_warps)) {
    for (int64_t s = 0; s < segment_count; ++s) {
      const int64_t* e = segments + 3 * s;
      const int64_t rb = e[2];
      bench_copy_row_unrolled<U>(
          reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + source_rows[row] * rb,
          reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + static_cast<int64_t>(destination_slots[row]) * rb,
          rb, w, ws, lane);
    }
  }
}

void bench_copy_segments(
    tvm::ffi::TensorView segments,
    tvm::ffi::TensorView source_rows,
    tvm::ffi::TensorView destination_slots,
    tvm::ffi::TensorView count,
    int64_t grid,
    int64_t unroll) {
  const auto device = host::LaunchKernel::resolve_device(segments.device());
  const auto seg = static_cast<const int64_t*>(segments.data_ptr());
  const auto n = static_cast<int64_t>(segments.size(0));
  const auto rows = static_cast<const int64_t*>(source_rows.data_ptr());
  const auto slots = static_cast<const int32_t*>(destination_slots.data_ptr());
  const auto cnt = static_cast<const int32_t*>(count.data_ptr());
  const int g = static_cast<int>(grid);
  switch (unroll) {
    case 0:
      host::LaunchKernel(g, kExpertTransferBlockSize, device)(copy_expert_row_segments_gpu_kernel, seg, n, rows, slots, cnt);
      break;
    case 1:
      host::LaunchKernel(g, kExpertTransferBlockSize, device)(bench_copy_unrolled_kernel<1>, seg, n, rows, slots, cnt);
      break;
    case 2:
      host::LaunchKernel(g, kExpertTransferBlockSize, device)(bench_copy_unrolled_kernel<2>, seg, n, rows, slots, cnt);
      break;
    case 4:
      host::LaunchKernel(g, kExpertTransferBlockSize, device)(bench_copy_unrolled_kernel<4>, seg, n, rows, slots, cnt);
      break;
    case 8:
      host::LaunchKernel(g, kExpertTransferBlockSize, device)(bench_copy_unrolled_kernel<8>, seg, n, rows, slots, cnt);
      break;
    default:
      host::RuntimeCheck(false, "unroll must be 0 (production kernel), 1, 2, 4 or 8");
  }
}

}  // namespace sglang
