#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <stdint.h>

namespace sglang {

constexpr int kExpertRoutePlanWarpSize = 32;

// One warp plans a BS1, top_k<=32 gather in a single pass. Lane `lane` owns
// route `lane`; inactive lanes (lane >= top_k) still take part in every warp
// collective with `hot_hit`, `prefetched` and `residual` all false, so the
// ballot masks below only ever set bits for active routes.
//
// `misses` marks residual (uncovered nonresident) lanes; `rank` is the count
// of residual lanes before this one and `total` the residual count for the
// whole warp. A residual route's compacted position is `rank`; a hot-hit or
// prefetch-covered route's is `total + lane - rank`, i.e. its own rank among
// non-residual lanes. This puts residual experts first, in first-appearance
// order, and leaves every other route in its original relative order, so the
// compacted vector covers positions `[0, top_k)` with no gaps.
template <typename IdT, typename RemapT>
__global__ __launch_bounds__(kExpertRoutePlanWarpSize, 1) void plan_unique_routes_kernel(
    const IdT* __restrict__ topk_ids,
    const int64_t* __restrict__ expert_to_slot,
    int top_k,
    int32_t scratch_base,
    int64_t* __restrict__ source_rows_out,
    int32_t* __restrict__ slots_out,
    int32_t* __restrict__ count_out,
    RemapT* __restrict__ remap_out,
    int64_t* __restrict__ graph_counters,
    int64_t* __restrict__ graph_unique_counters,
    float* __restrict__ route_counts,
    const int64_t* __restrict__ prefetch_expert,
    const int32_t* __restrict__ prefetch_count,
    int32_t prefetch_slot) {
  const unsigned lane = threadIdx.x;
  const bool active = static_cast<int>(lane) < top_k;
  const int64_t expert = active ? static_cast<int64_t>(topk_ids[lane]) : 0;
  const int64_t slot = active ? expert_to_slot[expert] : -1;
  const bool hot_hit = active && slot >= 0;
  const bool prefetched =
      active && !hot_hit && prefetch_count[0] == 1 && expert == prefetch_expert[0];
  const bool residual = active && !hot_hit && !prefetched;

  const unsigned active_mask = __ballot_sync(0xffffffffu, active);
  const unsigned hit_mask = __ballot_sync(0xffffffffu, hot_hit);
  const unsigned miss_mask = __ballot_sync(0xffffffffu, residual);
  const unsigned earlier = (1u << lane) - 1u;
  const unsigned rank = __popc(miss_mask & earlier);
  const unsigned total = __popc(miss_mask);

  if (active) {
    const int32_t destination = hot_hit
                                     ? static_cast<int32_t>(slot)
                                     : (prefetched ? prefetch_slot : scratch_base + static_cast<int32_t>(rank));
    const unsigned dest_pos = residual ? rank : (total + lane - rank);
    source_rows_out[dest_pos] = expert;
    slots_out[dest_pos] = destination;
    remap_out[lane] = static_cast<RemapT>(destination);
    if (route_counts != nullptr) {
      atomicAdd(route_counts + expert, 1.0f);
    }
  }

  if (lane == 0) {
    count_out[0] = static_cast<int32_t>(total);
    if (graph_counters != nullptr) {
      auto* counters = reinterpret_cast<unsigned long long*>(graph_counters);
      atomicAdd(counters, static_cast<unsigned long long>(__popc(active_mask)));
      atomicAdd(counters + 1, static_cast<unsigned long long>(total));
    }
    if (graph_unique_counters != nullptr) {
      auto* unique_counters = reinterpret_cast<unsigned long long*>(graph_unique_counters);
      atomicAdd(unique_counters, static_cast<unsigned long long>(__popc(hit_mask)));
      atomicAdd(unique_counters + 1, static_cast<unsigned long long>(total));
    }
  }
}

// Launches `plan_unique_routes_kernel` as exactly one block of 32 threads.
// `graph_counters`, `graph_unique_counters` and `route_counts` are optional
// accumulators: absent, they are left untouched. `prefetch_expert` is
// int64[1] and `prefetch_count` is int32[1]; a zero `prefetch_count` means no
// route can ever be a covered miss, whatever `prefetch_slot` is.
template <typename IdT, typename RemapT>
void plan_unique_routes_gpu(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView expert_to_slot,
    int64_t scratch_base,
    tvm::ffi::TensorView source_rows_out,
    tvm::ffi::TensorView slots_out,
    tvm::ffi::TensorView count_out,
    tvm::ffi::TensorView remap_out,
    tvm::ffi::Optional<tvm::ffi::TensorView> graph_counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> graph_unique_counters,
    tvm::ffi::Optional<tvm::ffi::TensorView> route_counts,
    tvm::ffi::TensorView prefetch_expert,
    tvm::ffi::TensorView prefetch_count,
    int64_t prefetch_slot) {
  const auto stream = host::LaunchKernel::resolve_device(topk_ids.device());
  int64_t* graph_counters_ptr =
      graph_counters.has_value() ? static_cast<int64_t*>(graph_counters.value().data_ptr()) : nullptr;
  int64_t* graph_unique_counters_ptr = graph_unique_counters.has_value()
                                           ? static_cast<int64_t*>(graph_unique_counters.value().data_ptr())
                                           : nullptr;
  float* route_counts_ptr =
      route_counts.has_value() ? static_cast<float*>(route_counts.value().data_ptr()) : nullptr;
  host::LaunchKernel(1, kExpertRoutePlanWarpSize, stream)(
      plan_unique_routes_kernel<IdT, RemapT>,
      static_cast<const IdT*>(topk_ids.data_ptr()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<int>(topk_ids.numel()),
      static_cast<int32_t>(scratch_base),
      static_cast<int64_t*>(source_rows_out.data_ptr()),
      static_cast<int32_t*>(slots_out.data_ptr()),
      static_cast<int32_t*>(count_out.data_ptr()),
      static_cast<RemapT*>(remap_out.data_ptr()),
      graph_counters_ptr,
      graph_unique_counters_ptr,
      route_counts_ptr,
      static_cast<const int64_t*>(prefetch_expert.data_ptr()),
      static_cast<const int32_t*>(prefetch_count.data_ptr()),
      static_cast<int32_t>(prefetch_slot));
}

}  // namespace sglang
