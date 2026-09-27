#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include "expert_stream/tensor_checks.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <type_traits>
#include <utility>

// Three single-launch replacements for the per-layer torch op chains of the DSV4.1 EXL3 graph decode
// (SGLANG_DSV41_ENABLE_LAYER_FUSION). Each one reproduces its chain's results bit for bit: the chains are integer
// bookkeeping plus two exact float conversions, so nothing here reorders a floating-point sum.

namespace sglang {

constexpr int kLayerFusionWarp = 32;

__device__ __forceinline__ float layer_fusion_to_float(float v) {
  return v;
}
__device__ __forceinline__ float layer_fusion_to_float(__half v) {
  return __half2float(v);
}
__device__ __forceinline__ float layer_fusion_to_float(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

// GpuResidencyUpdater.gather_destinations, for one layer. One warp; lane j owns shortlist entry j (width <= 32).
//
// A shortlist entry is usable when it is valid and no route of this forward reads its slot. Usable entries move to
// the front in shortlist order, the others follow in shortlist order (the stable argsort of the torch chain). Lane k
// of the reordered list is live when it is usable and k < miss_count; a live lane's destination is its slot, any other
// lane's is 0. A remap entry at or past scratch_base is a miss lane's rank and becomes that lane's destination.
template <typename IdT, typename RemapInT, typename RemapOutT>
__global__ __launch_bounds__(kLayerFusionWarp, 1) void direct_gather_destinations_kernel(
    const IdT* __restrict__ topk_ids,
    int top_k,
    const int64_t* __restrict__ expert_to_slot,
    const int64_t* __restrict__ victims,
    const bool* __restrict__ victim_valid,
    int width,
    const int32_t* __restrict__ miss_count,
    const RemapInT* __restrict__ remap_in,
    int64_t scratch_base,
    int32_t* __restrict__ destination_slots_out,
    int64_t* __restrict__ destinations_out,
    bool* __restrict__ live_out,
    RemapOutT* __restrict__ remap_out) {
  __shared__ int64_t usable[kLayerFusionWarp];
  __shared__ bool usable_valid[kLayerFusionWarp];
  __shared__ int64_t destinations[kLayerFusionWarp];
  const unsigned lane = threadIdx.x;
  const bool entry = static_cast<int>(lane) < width;
  const int64_t victim = entry ? victims[lane] : 0;
  const bool valid = entry && victim_valid[lane];
  bool hazard = false;
  if (entry) {
    for (int i = 0; i < top_k; ++i) {
      hazard |= expert_to_slot[static_cast<int64_t>(topk_ids[i])] == victim;
    }
  }
  const bool good = valid && !hazard;
  const unsigned good_mask = __ballot_sync(0xffffffffu, entry && good);
  const unsigned bad_mask = __ballot_sync(0xffffffffu, entry && !good);
  const unsigned earlier = (1u << lane) - 1u;
  if (entry) {
    const unsigned position = good ? __popc(good_mask & earlier) : __popc(good_mask) + __popc(bad_mask & earlier);
    usable[position] = victim;
    usable_valid[position] = good;
  }
  __syncwarp();
  if (entry) {
    const bool live = static_cast<int>(lane) < miss_count[0] && usable_valid[lane];
    const int64_t destination = live ? usable[lane] : 0;
    destinations[lane] = destination;
    destinations_out[lane] = destination;
    destination_slots_out[lane] = static_cast<int32_t>(destination);
    live_out[lane] = live;
  }
  __syncwarp();
  if (static_cast<int>(lane) < top_k) {
    const int64_t remap = static_cast<int64_t>(remap_in[lane]);
    int64_t rank = remap - scratch_base;
    rank = rank < 0 ? 0 : (rank > width - 1 ? width - 1 : rank);
    remap_out[lane] = static_cast<RemapOutT>(remap >= scratch_base ? destinations[rank] : remap);
  }
}

// GpuResidencyUpdater.commit_gather + _commit_gather, for one layer. One thread, in the torch chain's order: the
// chain is a sequence of scatters whose later writes overwrite earlier ones on shared indices (the dump columns), so
// running it serially is what keeps every final value, dump columns included, identical. width is top_k (6).
//
// delivered is null for a backend without leased delivery; the truncation tripwire then reads miss_count instead.
__global__ __launch_bounds__(1, 1) void direct_commit_gather_kernel(
    const int64_t* __restrict__ destinations,
    const bool* __restrict__ live_in,
    const int64_t* __restrict__ new_experts,
    int width,
    int64_t num_experts,
    int64_t slot_dump,
    int64_t* __restrict__ mapping,
    int64_t* __restrict__ slot_to_expert,
    uint8_t* __restrict__ slot_state,
    int64_t* __restrict__ slot_generations,
    int64_t* __restrict__ gather_insertions,
    int64_t* __restrict__ gather_evictions,
    int64_t* __restrict__ insertion_truncated,
    const int32_t* __restrict__ delivered,
    const float* __restrict__ keep,
    const int32_t* __restrict__ miss_count,
    uint8_t ready,
    uint8_t free_state) {
  bool live[kLayerFusionWarp];
  bool evicted[kLayerFusionWarp];
  int64_t old_expert[kLayerFusionWarp];
  const bool good = delivered == nullptr || keep[0] > 0.0f;
  int64_t live_sum = 0;
  int64_t evicted_sum = 0;
  for (int j = 0; j < width; ++j) {
    live[j] = live_in[j] && (delivered == nullptr || (j < delivered[0] && good));
    old_expert[j] = slot_to_expert[destinations[j]];
    evicted[j] = live[j] && old_expert[j] >= 0;
    live_sum += live[j];
    evicted_sum += evicted[j];
  }
  for (int j = 0; j < width; ++j) {
    mapping[evicted[j] ? old_expert[j] : num_experts] = -1;
  }
  for (int j = 0; j < width; ++j) {
    mapping[live[j] ? new_experts[j] : num_experts] = live[j] ? destinations[j] : slot_dump;
  }
  for (int j = 0; j < width; ++j) {
    slot_to_expert[live[j] ? destinations[j] : slot_dump] = live[j] ? new_experts[j] : -1;
  }
  for (int j = 0; j < width; ++j) {
    slot_state[live[j] ? destinations[j] : slot_dump] = ready;
  }
  for (int j = 0; j < width; ++j) {
    slot_generations[live[j] ? destinations[j] : slot_dump] += live[j];
  }
  slot_state[slot_dump] = free_state;
  slot_to_expert[slot_dump] = -1;
  slot_generations[slot_dump] = 0;
  gather_insertions[0] += live_sum;
  gather_evictions[0] += evicted_sum;
  if (delivered != nullptr) {
    insertion_truncated[0] += (static_cast<int64_t>(delivered[0]) > live_sum) && good;
  } else {
    insertion_truncated[0] += static_cast<int64_t>(miss_count[0]) > live_sum;
  }
}

// exl3_fused_moe.route_tables plus the copies around it in Exl3FusedMoE.run: the int64 remap, x -> fp16, the zeroed
// fp32 output, per-slot route counts (zero when keep is 0), inv_order, the keep-scaled fp16 weights in slot order,
// and the deterministic table stack [start, start, count > 0] over slots + 1 columns.
//
// Routes are ranked stably (ties by route index). torch.argsort(remap) is not stable, so the two orders can differ
// only when two routes share a slot, which a BS1 remap does not do: hits are distinct slots and DIRECT's miss lanes
// take distinct victims.
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
    int64_t* __restrict__ det) {
  const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = tid; i < hidden; i += stride) {
    x16_out[i] = __float2half_rn(layer_fusion_to_float(x[i]));
    out_zero[i] = 0.0f;
  }
  const bool kept = keep[0] > 0.0f;
  for (int64_t s = tid; s < columns; s += stride) {
    int64_t count = 0;
    int64_t before = 0;
    for (int i = 0; i < top_k; ++i) {
      const int64_t r = static_cast<int64_t>(remap[i]);
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
    const int64_t r = static_cast<int64_t>(remap[tid]);
    int64_t rank = 0;
    for (int i = 0; i < top_k; ++i) {
      const int64_t other = static_cast<int64_t>(remap[i]);
      rank += other < r || (other == r && i < tid);
    }
    remap64_out[tid] = r;
    inv_order[tid] = rank;
    weight_sorted[rank] = __float2half_rn(__fmul_rn(layer_fusion_to_float(weights[tid]), keep[0]));
  }
}

/// \brief `verify_named` for a tensor the kernel reads as `bool` (`kDLBool` has no dtype trait).
inline void verify_bool_named(const char* name, host::TensorMatcher&& matcher, tvm::ffi::TensorView view) {
  expert_stream::verify_named(name, std::move(matcher), view);
  host::RuntimeCheck(view.dtype().code == kDLBool && view.dtype().bits == 8, name, ": must be a bool tensor");
}

/// \brief Checked launcher for `direct_gather_destinations_kernel`: one layer's DIRECT gather destinations.
///
/// Precondition, not checked: every `topk_ids` entry indexes `expert_to_slot`. Reading the ids would need a device
/// sync, which a graph capture refuses, and the route planner indexes `expert_to_slot` with the same ids earlier in
/// the forward (`expert_route_plan.cuh`); a bad id has already read out of bounds by the time this launch runs.
template <typename IdT, typename RemapInT, typename RemapOutT>
void direct_gather_destinations_gpu(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView expert_to_slot,
    tvm::ffi::TensorView victims,
    tvm::ffi::TensorView victim_valid,
    tvm::ffi::TensorView miss_count,
    tvm::ffi::TensorView remap_in,
    int64_t scratch_base,
    tvm::ffi::TensorView destination_slots_out,
    tvm::ffi::TensorView destinations_out,
    tvm::ffi::TensorView live_out,
    tvm::ffi::TensorView remap_out) {
  using namespace host;
  static_assert(
      (std::is_same_v<IdT, int32_t> || std::is_same_v<IdT, int64_t>) &&
          (std::is_same_v<RemapInT, int32_t> || std::is_same_v<RemapInT, int64_t>) &&
          (std::is_same_v<RemapOutT, int32_t> || std::is_same_v<RemapOutT, int64_t>),
      "direct_gather_destinations: ids and remaps are int32 or int64");
  auto K_ = SymbolicSize{"routes"};
  auto W_ = SymbolicSize{"width"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named(
      "topk_ids", TensorMatcher({K_}).with_dtype<IdT>().template with_device<kDLCUDA>(device), topk_ids);
  expert_stream::verify_named(
      "expert_to_slot", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), expert_to_slot);
  expert_stream::verify_named(
      "victims", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), victims);
  verify_bool_named("victim_valid", TensorMatcher({W_}).with_device<kDLCUDA>(device), victim_valid);
  expert_stream::verify_named(
      "miss_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), miss_count);
  expert_stream::verify_named(
      "remap_in", TensorMatcher({K_}).with_dtype<RemapInT>().template with_device<kDLCUDA>(device), remap_in);
  expert_stream::verify_named(
      "destination_slots_out",
      TensorMatcher({W_}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
      destination_slots_out);
  expert_stream::verify_named(
      "destinations_out", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), destinations_out);
  verify_bool_named("live_out", TensorMatcher({W_}).with_device<kDLCUDA>(device), live_out);
  expert_stream::verify_named(
      "remap_out", TensorMatcher({K_}).with_dtype<RemapOutT>().template with_device<kDLCUDA>(device), remap_out);
  RuntimeCheck(
      0 < W_.unwrap() && W_.unwrap() <= kLayerFusionWarp && 0 < K_.unwrap() && K_.unwrap() <= kLayerFusionWarp,
      "the shortlist and the routes must hold 1-32 entries");
  const auto stream = host::LaunchKernel::resolve_device(topk_ids.device());
  host::LaunchKernel(1, kLayerFusionWarp, stream)(
      direct_gather_destinations_kernel<IdT, RemapInT, RemapOutT>,
      static_cast<const IdT*>(topk_ids.data_ptr()),
      static_cast<int>(topk_ids.numel()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<const int64_t*>(victims.data_ptr()),
      static_cast<const bool*>(victim_valid.data_ptr()),
      static_cast<int>(victims.numel()),
      static_cast<const int32_t*>(miss_count.data_ptr()),
      static_cast<const RemapInT*>(remap_in.data_ptr()),
      scratch_base,
      static_cast<int32_t*>(destination_slots_out.data_ptr()),
      static_cast<int64_t*>(destinations_out.data_ptr()),
      static_cast<bool*>(live_out.data_ptr()),
      static_cast<RemapOutT*>(remap_out.data_ptr()));
}

/// \brief Checked launcher for `direct_commit_gather_kernel`: one layer's DIRECT residency commit.
///
/// The width bound keeps the kernel's 32-entry lane arrays in range. Precondition, not checked (it would need a
/// device sync): every `destinations` entry indexes `slot_to_expert`, and every live lane's `new_experts` entry
/// indexes `mapping`; the gather kernel and the planner produce both.
void direct_commit_gather_gpu(
    tvm::ffi::TensorView destinations,
    tvm::ffi::TensorView live,
    tvm::ffi::TensorView new_experts,
    int64_t num_experts,
    int64_t slot_dump,
    tvm::ffi::TensorView mapping,
    tvm::ffi::TensorView slot_to_expert,
    tvm::ffi::TensorView slot_state,
    tvm::ffi::TensorView slot_generations,
    tvm::ffi::TensorView gather_insertions,
    tvm::ffi::TensorView gather_evictions,
    tvm::ffi::TensorView insertion_truncated,
    tvm::ffi::Optional<tvm::ffi::TensorView> delivered,
    tvm::ffi::Optional<tvm::ffi::TensorView> keep,
    tvm::ffi::TensorView miss_count,
    int64_t ready,
    int64_t free_state) {
  using namespace host;
  auto W_ = SymbolicSize{"width"};
  auto E_ = SymbolicSize{"mapping_columns"};
  auto S_ = SymbolicSize{"slot_columns"};
  auto device = SymbolicDevice{};
  expert_stream::verify_named(
      "destinations", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), destinations);
  verify_bool_named("live", TensorMatcher({W_}).with_device<kDLCUDA>(device), live);
  expert_stream::verify_named(
      "new_experts", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), new_experts);
  expert_stream::verify_named(
      "mapping", TensorMatcher({E_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), mapping);
  expert_stream::verify_named(
      "slot_to_expert", TensorMatcher({S_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), slot_to_expert);
  expert_stream::verify_named(
      "slot_state", TensorMatcher({S_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), slot_state);
  expert_stream::verify_named(
      "slot_generations", TensorMatcher({S_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), slot_generations);
  expert_stream::verify_named(
      "gather_insertions", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_insertions);
  expert_stream::verify_named(
      "gather_evictions", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), gather_evictions);
  expert_stream::verify_named(
      "insertion_truncated",
      TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device),
      insertion_truncated);
  RuntimeCheck(delivered.has_value() == keep.has_value(), "delivered and keep go together");
  if (delivered.has_value()) {
    expert_stream::verify_named(
        "delivered", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), delivered.value());
    expert_stream::verify_named(
        "keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep.value());
  }
  expert_stream::verify_named(
      "miss_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), miss_count);
  RuntimeCheck(0 < W_.unwrap() && W_.unwrap() <= kLayerFusionWarp, "the commit must cover 1-32 lanes");
  RuntimeCheck(num_experts == E_.unwrap() - 1, "num_experts must be mapping's size minus the dump column");
  RuntimeCheck(slot_dump == S_.unwrap() - 1, "slot_dump must be slot_to_expert's last column");
  const auto stream = host::LaunchKernel::resolve_device(destinations.device());
  host::LaunchKernel(1, 1, stream)(
      direct_commit_gather_kernel,
      static_cast<const int64_t*>(destinations.data_ptr()),
      static_cast<const bool*>(live.data_ptr()),
      static_cast<const int64_t*>(new_experts.data_ptr()),
      static_cast<int>(destinations.numel()),
      num_experts,
      slot_dump,
      static_cast<int64_t*>(mapping.data_ptr()),
      static_cast<int64_t*>(slot_to_expert.data_ptr()),
      static_cast<uint8_t*>(slot_state.data_ptr()),
      static_cast<int64_t*>(slot_generations.data_ptr()),
      static_cast<int64_t*>(gather_insertions.data_ptr()),
      static_cast<int64_t*>(gather_evictions.data_ptr()),
      static_cast<int64_t*>(insertion_truncated.data_ptr()),
      delivered.has_value() ? static_cast<const int32_t*>(delivered.value().data_ptr()) : nullptr,
      keep.has_value() ? static_cast<const float*>(keep.value().data_ptr()) : nullptr,
      static_cast<const int32_t*>(miss_count.data_ptr()),
      static_cast<uint8_t>(ready),
      static_cast<uint8_t>(free_state));
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
    tvm::ffi::TensorView det) {
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
      static_cast<int64_t*>(det.data_ptr()));
}

}  // namespace sglang
