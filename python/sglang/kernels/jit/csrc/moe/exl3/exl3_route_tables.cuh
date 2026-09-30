// REQUIRED BUILD FLAG: none; never -use_fast_math. It implies -ftz, which could flush subnormal products and inputs
// that torch keeps (the __fmul_rn and __float2half_rn below), and bit parity with the torch chain breaks.
// The EXL3 fused MoE's route tables and input staging (exl3_fused_moe.route_tables and the copies around it) in one
// launch per layer, bit for bit: integer bookkeeping plus two exact float conversions, so nothing reorders a sum.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "../expert_stream/tensor_checks.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>

namespace sglang::exl3 {

__device__ __forceinline__ float route_tables_to_float(float v) {
  return v;
}
__device__ __forceinline__ float route_tables_to_float(__half v) {
  return __half2float(v);
}
__device__ __forceinline__ float route_tables_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// exl3_fused_moe.route_tables plus the copies around it in Exl3FusedMoE.run: the int64 remap, x -> fp16, the zeroed
// fp32 output, per-slot route counts (zero when keep is 0), inv_order, the keep-scaled fp16 weights in slot order,
// and the deterministic table stack [start, start, count > 0] over slots + 1 columns.
//
// Routes are ranked stably (ties by route index). torch.argsort(remap) is not stable, so the two orders can differ
// only when two routes share a slot, which a BS1 remap does not do: hits are distinct slots and DIRECT's miss lanes
// take distinct victims.
//
// CPU experts (plan 2026-09-29-dsv41-cpu-experts; cpu_lanes non-null): the lanes of cpu_lanes[0] were computed by the
// CPU expert thread, whose partial sum cpu_out (host memory, [hidden] fp32) seeds the output instead of zero. A route
// whose slot is such a lane's dst_slots entry is ranked as if its slot were past every column: its slot's count is 0
// (the fused kernel and the gather skip it) and every other slot's start, which the fused kernel recomputes as a
// running sum of the counts to index weight_sorted, is unchanged by it. Its remap64_out entry keeps the real slot.
template <typename RemapT, typename WeightT, typename XT>
__global__ void exl3_moe_route_tables_kernel(
    const RemapT* __restrict__ remap,
    int top_k,
    const WeightT* __restrict__ weights,
    const float* __restrict__ keep,
    const XT* __restrict__ x,
    int64_t hidden,
    int64_t columns,
    int64_t* __restrict__ remap64_out,
    __half* __restrict__ x16_out,
    float* __restrict__ out_zero,
    int64_t* __restrict__ expert_count,
    int64_t* __restrict__ inv_order,
    __half* __restrict__ weight_sorted,
    int64_t* __restrict__ det,
    const int32_t* __restrict__ cpu_lanes,
    const int32_t* __restrict__ dst_slots,
    int64_t dst_count,
    const float* __restrict__ cpu_out) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const bool kept = keep[0] > 0.0f;
  const uint32_t cpu = cpu_lanes != nullptr ? static_cast<uint32_t>(cpu_lanes[0]) : 0u;
  // A route's slot as the tables rank it: past every column when the CPU computed it.
  auto ranked = [&](int i) -> int64_t {
    const int64_t r = static_cast<int64_t>(remap[i]);
    for (uint32_t lanes = cpu; lanes != 0; lanes &= lanes - 1) {
      const int lane = __ffs(lanes) - 1;
      if (lane < dst_count && static_cast<int64_t>(dst_slots[lane]) == r) return columns;
    }
    return r;
  };
  const bool seeded = kept && cpu != 0;
  for (int64_t i = tid; i < hidden; i += stride) {
    x16_out[i] = __float2half_rn(route_tables_to_float(x[i]));
    out_zero[i] = seeded ? __ldcv(cpu_out + i) : 0.0f;
  }
  for (int64_t s = tid; s < columns; s += stride) {
    int64_t count = 0;
    int64_t before = 0;
    for (int i = 0; i < top_k; ++i) {
      const int64_t r = ranked(i);
      count += r == s;
      before += r < s;
    }
    count = kept ? count : 0;
    const int64_t start = kept ? before : 0;
    expert_count[s] = count;
    det[s] = start;
    det[columns + s] = start;
    det[2 * columns + s] = count > 0;
  }
  if (tid < top_k) {
    const int64_t r = ranked(static_cast<int>(tid));
    int64_t rank = 0;
    for (int i = 0; i < top_k; ++i) {
      const int64_t other = ranked(i);
      rank += other < r || (other == r && i < tid);
    }
    remap64_out[tid] = static_cast<int64_t>(remap[tid]);
    inv_order[tid] = rank;
    weight_sorted[rank] = __float2half_rn(__fmul_rn(route_tables_to_float(weights[tid]), keep[0]));
  }
}

/// \brief Checked launcher for `exl3_moe_route_tables_kernel`: the fused MoE's route tables and input staging.
///
/// `x`, `x16_out` and `out_zero` are the one decode token's `[1, hidden]` rows; `det` is the `[3, slots + 1]` stack.
template <typename RemapT, typename WeightT, typename XT>
void exl3_moe_route_tables_gpu(
    tvm::ffi::TensorView remap,
    tvm::ffi::TensorView weights,
    tvm::ffi::TensorView keep,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView remap64_out,
    tvm::ffi::TensorView x16_out,
    tvm::ffi::TensorView out_zero,
    tvm::ffi::TensorView expert_count,
    tvm::ffi::TensorView inv_order,
    tvm::ffi::TensorView weight_sorted,
    tvm::ffi::TensorView det,
    tvm::ffi::TensorView cpu_lanes,
    tvm::ffi::TensorView dst_slots,
    int64_t cpu_out) {
  using namespace host;
  static_assert(std::is_same_v<RemapT, int32_t> || std::is_same_v<RemapT, int64_t>, "remap is int32 or int64");
  static_assert(
      (std::is_same_v<WeightT, fp32_t> || std::is_same_v<WeightT, fp16_t> || std::is_same_v<WeightT, bf16_t>) &&
          (std::is_same_v<XT, fp32_t> || std::is_same_v<XT, fp16_t> || std::is_same_v<XT, bf16_t>),
      "weights and x are fp32, fp16 or bf16");
  constexpr int64_t kMaxRoutes = 32;
  auto K_ = SymbolicSize{"routes"};
  auto H_ = SymbolicSize{"hidden"};
  auto C_ = SymbolicSize{"columns"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named(
      "remap", TensorMatcher({K_}).with_dtype<RemapT>().template with_device<kDLCUDA>(device), remap);
  expert_stream::verify_named(
      "weights", TensorMatcher({K_}).with_dtype<WeightT>().template with_device<kDLCUDA>(device), weights);
  expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);
  expert_stream::verify_named("x", TensorMatcher({1, H_}).with_dtype<XT>().template with_device<kDLCUDA>(device), x);
  expert_stream::verify_named(
      "remap64_out", TensorMatcher({K_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), remap64_out);
  expert_stream::verify_named(
      "x16_out", TensorMatcher({1, H_}).with_dtype<fp16_t>().with_device<kDLCUDA>(device), x16_out);
  expert_stream::verify_named(
      "out_zero", TensorMatcher({1, H_}).with_dtype<float>().with_device<kDLCUDA>(device), out_zero);
  expert_stream::verify_named(
      "expert_count", TensorMatcher({C_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), expert_count);
  expert_stream::verify_named(
      "inv_order", TensorMatcher({K_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), inv_order);
  expert_stream::verify_named(
      "weight_sorted", TensorMatcher({K_}).with_dtype<fp16_t>().with_device<kDLCUDA>(device), weight_sorted);
  expert_stream::verify_named("det", TensorMatcher({3, C_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), det);
  RuntimeCheck(0 < K_.unwrap() && K_.unwrap() <= kMaxRoutes, "remap must hold 1-32 routes");
  // CPU experts: cpu_lanes is one word (empty when off), dst_slots the plan's lane slots, cpu_out the host row.
  expert_stream::verify_named(
      "cpu_lanes", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), cpu_lanes);
  expert_stream::verify_named(
      "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);
  const bool cpu_on = cpu_lanes.size(0) == 1;
  RuntimeCheck(cpu_lanes.size(0) <= 1, "cpu_lanes: one word, or empty when CPU experts are off");
  RuntimeCheck(!cpu_on || (cpu_out != 0 && cpu_out % 16 == 0), "cpu_out: the CPU partial's host row, 16-byte aligned");
  const auto stream = host::LaunchKernel::resolve_device(remap.device());
  const int64_t hidden = x.numel();
  const int64_t columns = expert_count.numel();
  constexpr int kThreads = 256;
  const int64_t work = hidden > columns ? hidden : columns;
  const int blocks = static_cast<int>((work + kThreads - 1) / kThreads);
  host::LaunchKernel(blocks, kThreads, stream)(
      exl3_moe_route_tables_kernel<RemapT, WeightT, XT>,
      static_cast<const RemapT*>(remap.data_ptr()),
      static_cast<int>(remap.numel()),
      static_cast<const WeightT*>(weights.data_ptr()),
      static_cast<const float*>(keep.data_ptr()),
      static_cast<const XT*>(x.data_ptr()),
      hidden,
      columns,
      static_cast<int64_t*>(remap64_out.data_ptr()),
      static_cast<__half*>(x16_out.data_ptr()),
      static_cast<float*>(out_zero.data_ptr()),
      static_cast<int64_t*>(expert_count.data_ptr()),
      static_cast<int64_t*>(inv_order.data_ptr()),
      static_cast<__half*>(weight_sorted.data_ptr()),
      static_cast<int64_t*>(det.data_ptr()),
      cpu_on ? static_cast<const int32_t*>(cpu_lanes.data_ptr()) : nullptr,
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      dst_slots.size(0),
      reinterpret_cast<const float*>(cpu_out));
}

}  // namespace sglang::exl3
