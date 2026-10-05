// Slot-map protocol kernels, independent of the streamed row's format.
//
//   PostParams / exl3_ram_miss_post_kernel   the post: applies the row's pending map delta, types the lanes and
//                                            publishes the demand record
//   BulkApplyParams / ..._map_bulk_apply_kernel  applies every row's pending delta and the eager paths' map entries
//                                            while the service is paused
//   LeaseProtocolKernel                      the checked host launchers for both
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The chain".
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "tensor_checks.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sglang {

// Arguments of the post kernel, passed as a __grid_constant__ so a captured graph freezes them. Pointers address
// device memory unless noted; the launcher verifies each one.
struct PostParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  int64_t lanes;  // the plan's buffers: planned and dst_slots hold this many
  const int64_t* routes;
  int64_t route_count;
  int64_t row;
  int64_t experts;
  uint8_t* lease;  // the completion block, then the delta block at Wire::kDeltaBase
  int64_t timeout_ns;
  uint8_t* hot_page;
  int64_t hot_stride;
  const int64_t* hot_slots;
  int64_t hot_capacity;
  const int32_t* dst_slots;
  int64_t captured;
  // The device's map bank (ExpertStreamDevice.map_bank): row-major [rows, experts] and
  // [rows, Wire::kNodes * Wire::kLanes] int32, int64 [rows] chain words, per-row eligibility. A graph replay reads
  // what the previous deltas left here.
  int32_t* ram_slot;
  int32_t* staging;
  int64_t* map_chain;
  int64_t* map_applied;
  const uint8_t* ce_ok;
  const uint8_t* cpu_ok;
  const int32_t* dst_rows;
  uint32_t row_capacity;
  int64_t hit_copy_ce;  // SGLANG_DSV41_RAM_HIT_COPY=ce: an armed captured hit goes to the copy engine
  int64_t cpu_on;
  int64_t cpu_misses;
  // The post's outputs for the later kernels of the chain (C1 copies SM hits, S streams misses, CW waits for the copy
  // engine): each lane's kind, source slot and home node, and C1's compacted list of SM hits.
  int32_t* lane_kind;
  int32_t* lane_slot;
  int32_t* lane_node;
  int32_t* go_1;
  int64_t* host_rows_1;
  int32_t* dst_slots_1;
  // CPU experts. With cpu_x_dst set, the block stages the layer's input row there as fp16 (cpu_hidden elements of
  // cpu_x_src, of dtype cpu_x_dtype) when a lane is a CPU lane, and the record carries each lane's routing weight from
  // cpu_weights (cpu_weights_count routes aligned with `routes`, dtype cpu_weights_dtype). Null cpu_x_dst: none of it.
  const void* cpu_x_src;
  int64_t cpu_x_dtype;
  uint8_t* cpu_x_dst;
  int64_t cpu_hidden;
  const void* cpu_weights;
  int64_t cpu_weights_dtype;
  int64_t cpu_weights_count;
};

// Element dtypes of the CPU experts' staged input and routing weights (PostParams).
constexpr int64_t kCpuDtypeF16 = 0;
constexpr int64_t kCpuDtypeBf16 = 1;
constexpr int64_t kCpuDtypeF32 = 2;

// Element `i` of a CPU-experts input or weight array of dtype `dtype` (kCpuDtype*), as fp32.
SGL_DEVICE float cpu_input_value(const void* src, int64_t dtype, int64_t i) {
  if (dtype == kCpuDtypeF16) return __half2float(static_cast<const __half*>(src)[i]);
  if (dtype == kCpuDtypeBf16) return __bfloat162float(static_cast<const __nv_bfloat16*>(src)[i]);
  return static_cast<const float*>(src)[i];
}

// Stages the layer's input row as fp16 into the host row, 8 elements per 16-byte store, using the whole block. The
// launcher checks hidden % 8 == 0 and the alignment. Each thread fences its own stores at system scope before the
// barrier, so thread 0's later release of the record and demand_head orders all of them.
SGL_DEVICE void stage_cpu_input(const PostParams& p) {
  const int64_t vectors = p.cpu_hidden / 8;
  for (int64_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    __align__(16) __half h[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
      h[k] = __float2half_rn(cpu_input_value(p.cpu_x_src, p.cpu_x_dtype, 8 * v + k));
    __stwt(reinterpret_cast<uint4*>(p.cpu_x_dst + 16 * v), *reinterpret_cast<const uint4*>(h));
  }
  __threadfence_system();
  __syncthreads();
}

// The post kernel, one block of kBlock threads. Thread 0 applies the row's pending map delta, types the lanes from
// the device's map and publishes the demand record; the whole block stages the CPU input when a lane is the CPU's.
//
// Programmatic dependent launch (kUsePDL = SGLANG_DSV41_ENABLE_LEASE_PDL, used by every chain kernel but C1 and CC):
// the wait is the kernel's first statement and the trigger its second, so every word is read and written after the
// primary grid completed and flushed, as without PDL. The ordering is transitive down the chain
// (analysis/dsv41-drive/LEASE_PROTOCOL.md, "PDL").
template <bool kUsePDL>
__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_post_kernel(
    const __grid_constant__ PostParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  using namespace device::expert_stream;
  __shared__ int any_cpu;
  __shared__ TypedLanes typed;
  int32_t* __restrict__ const state = p.state;
  const uint64_t deadline = global_ns() + static_cast<uint64_t>(p.timeout_ns);
  const int64_t count = max(static_cast<int64_t>(p.count[0]), static_cast<int64_t>(0));
  if (threadIdx.x == 0) {
    if (count > Wire::kLanes || count > p.lanes) __trap();  // the record and the plan's buffers hold no more
    any_cpu = 0;
    if (count > 0) {
      const RowMap map{
          .ram_slot = p.ram_slot + p.row * p.experts,
          .staging = p.staging + p.row * (Wire::kNodes * Wire::kLanes),
          .map_chain = p.map_chain + p.row,
          .map_applied = p.map_applied + p.row,
          .experts = p.experts,
          .row_capacity = p.row_capacity,
      };
      LanePolicy policy{
          .host_lanes = false,
          .hit_copy_ce = p.hit_copy_ce != 0,
          .cpu_on = p.cpu_on != 0,
          .cpu_misses = p.cpu_misses != 0,
          .ce_ok = p.ce_ok[p.row] != 0,
          .cpu_ok = p.cpu_ok[p.row] != 0,
          .dst_rows = p.dst_rows[p.row],
      };
      // Loaded before the tag spin: the host stores split relaxed at any time, ordered by nothing (set_cpu_split in
      // host/ram_tier.h). Only a post that can have eligible lanes reads it; otherwise split stays zero and type_lanes
      // never indexes it.
      if (p.captured != 0 && policy.cpu_on && policy.cpu_ok) load_split(p.lease + Wire::kSplit, policy.split);
      const uint8_t* delta = p.lease + Wire::kDeltaBase + p.row * Wire::kDeltaStride;
      const bool pending = await_map_delta(delta, map, deadline);
      MapDelta d;
      if (pending) d = load_map_delta(delta);
      // Ordered after the tag's acquire; issued while the delta's loads are in flight.
      policy.host_lanes = p.captured != 0 && ld_acquire_sys(p.lease + Wire::kCopyArmed) == 1u;
      if (pending) apply_map_delta(d, map);
      type_lanes(LanePlan{.planned = p.planned, .dst = p.dst_slots, .count = count}, map, policy, typed);
      for (int64_t j = 0; j < count; ++j)
        any_cpu |= is_cpu_kind(typed.kind[j]) ? 1 : 0;
    }
  }
  __syncthreads();
  if (any_cpu != 0) {
    if (p.cpu_x_dst == nullptr) __trap();  // a CPU lane needs the staged input
    stage_cpu_input(p);
  }
  if (threadIdx.x != 0) return;
  int32_t protect[Wire::kLanes];
  int protect_count = 0;
  for (int64_t i = 0; i < p.route_count && protect_count < Wire::kLanes; ++i) {
    const int32_t expert = static_cast<int32_t>(p.routes[i]);
    if (expert >= 0 && expert < p.experts && !listed(protect, protect_count, expert)) protect[protect_count++] = expert;
  }
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) {
    seq = 1;
    state[kEpoch] += 1;
  }
  state[kPosted] = static_cast<int32_t>(seq);
  const uint32_t epoch = static_cast<uint32_t>(state[kEpoch]);
  if (p.hot_page != nullptr) {
    uint8_t* hot = p.hot_page + static_cast<int64_t>((seq - 1u) % Wire::kHotRecords) * p.hot_stride;
    st_relaxed_sys<uint32_t>(hot, 0u);
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    uint8_t* bits = hot + Wire::kHotHeaderBytes;
    for (int64_t byte = 0; byte < (p.experts + 7) / 8; ++byte) {
      uint8_t mask = 0;
      for (int64_t slot = 0; slot < p.hot_capacity; ++slot) {
        const int64_t expert = p.hot_slots[slot];
        if (expert >= byte * 8 && expert < byte * 8 + 8) mask |= static_cast<uint8_t>(1u << (expert - byte * 8));
      }
      st_relaxed_sys(bits + byte, mask);
    }
    st_release_sys(hot, seq);  // orders the bitmap before the seq
  }
  bool any_miss = false;
  int64_t go = 0;
  float weight[Wire::kLanes];
  for (int64_t j = 0; j < Wire::kLanes; ++j) {
    const bool used = j < count;
    p.lane_kind[j] = used ? typed.kind[j] : 0;
    p.lane_slot[j] = used ? typed.slot[j] : -1;
    p.lane_node[j] = used ? typed.node[j] : 0;
    if (!used) continue;
    any_miss = any_miss || typed.kind[j] == Wire::kKindMissGpu || typed.kind[j] == Wire::kKindMissCpu;
    if (typed.kind[j] == Wire::kKindHitSm) {
      p.host_rows_1[go] = static_cast<int64_t>(typed.slot[j]);
      p.dst_slots_1[go] = p.dst_slots[j];
      ++go;
    }
    // The lane expert's routing weight: its route's (batch-size-1 routes are distinct experts). 0 when CPU experts
    // are off.
    weight[j] = 0.0f;
    if (p.cpu_x_dst != nullptr) {
      for (int64_t r = 0; r < p.route_count && r < p.cpu_weights_count; ++r) {
        if (p.routes[r] == p.planned[j]) weight[j] += cpu_input_value(p.cpu_weights, p.cpu_weights_dtype, r);
      }
    }
  }
  p.go_1[0] = static_cast<int32_t>(go);
  uint64_t chain = 0;
  if (any_miss) {
    p.map_chain[p.row] += 1;  // the host publishes this chain's delta under this number
    chain = static_cast<uint64_t>(p.map_chain[p.row]);
  }
  uint8_t* record = p.page + Wire::kDemandRing + ring_index(seq) * Wire::kRecordBytes;
  write_record(
      record,
      seq,
      RecordFields{
          .row = p.row,
          .flags = p.captured != 0 ? Wire::kRecFlagCaptured : 0u,
          .chain = chain,
          .epoch = epoch,
          .protect = protect,
          .protect_count = protect_count,
          .count = count,
          .planned = p.planned,
          .dst = p.dst_slots,
          .weight = weight,
          .lanes = &typed,
      });
  // A release orders every earlier store of this thread: the hot page and the record come first.
  st_release_sys(p.page + Wire::kDemandHead, seq);
  state[kPending] = count > 0 ? static_cast<int32_t>(seq) : 0;
  state[kPendingEpoch] = state[kEpoch];
  store_deadline(state, global_ns() + static_cast<uint64_t>(p.timeout_ns));
}

// Arguments of the bulk map apply kernel.
//
// Run by Exl3RamMissService.after_host_use with the service paused and the stream synchronized: first every row's
// pending decode delta, then the eager paths' entries {row, expert, slot} (`entries`, int32 [entry_count, 3]). One
// thread does it all, since the work is a few dozen words.
struct BulkApplyParams {
  const uint8_t* lease;
  int32_t* ram_slot;
  int32_t* staging;
  int64_t* map_chain;
  int64_t* map_applied;
  const int32_t* entries;
  int64_t entry_count;
  int64_t rows;
  int64_t experts;
  const int32_t* row_capacity;
};

// The bulk map apply: see BulkApplyParams. Traps on an entry outside the map bank.
__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_map_bulk_apply_kernel(
    const __grid_constant__ BulkApplyParams p) {
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  for (int64_t row = 0; row < p.rows; ++row) {
    const uint8_t* delta = p.lease + Wire::kDeltaBase + row * Wire::kDeltaStride;
    // The paused host has already published every delta it owes, so nothing is waited for here.
    if (ld_acquire_sys64(delta + Wire::kDeltaTag) != static_cast<uint64_t>(p.map_chain[row])) continue;
    const RowMap map{
        .ram_slot = p.ram_slot + row * p.experts,
        .staging = p.staging + row * (Wire::kNodes * Wire::kLanes),
        .map_chain = p.map_chain + row,
        .map_applied = p.map_applied + row,
        .experts = p.experts,
        .row_capacity = static_cast<uint32_t>(p.row_capacity[row]),
    };
    if (await_map_delta(delta, map, global_ns())) apply_map_delta(load_map_delta(delta), map);
  }
  for (int64_t i = 0; i < p.entry_count; ++i) {
    const int32_t row = p.entries[3 * i];
    const int32_t expert = p.entries[3 * i + 1];
    const int32_t slot = p.entries[3 * i + 2];
    if (row < 0 || row >= p.rows || expert < 0 || expert >= p.experts || slot < -1 || slot >= p.row_capacity[row])
      __trap();
    p.ram_slot[static_cast<int64_t>(row) * p.experts + expert] = slot;
  }
}

/// \brief Checked host launchers for the slot-map kernels above: post and map_bulk_apply.
///
/// Every tensor argument is verified with `TensorMatcher` (named via `verify_named`) and every address with
/// `RuntimeCheck` before the params struct is built and the kernel launched. Both launch on the stream of the
/// tensors' device.
struct LeaseProtocolKernel {
  /// The node count this module was compiled for (-DSGLANG_EXPERT_STREAM_NODES).
  static int64_t wire_nodes() {
    return expert_stream::wire::Wire::kNodes;
  }

  /// Launches the post kernel with `use_pdl` selecting the PDL instantiation. Argument meanings follow PostParams;
  /// `lease_address` and `cpu_x_dst` are raw addresses of the pinned completion block and the staged input row, and
  /// `hot_address` of the optional hot page (0: none).
  static void post(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      tvm::ffi::TensorView planned,
      tvm::ffi::TensorView count,
      tvm::ffi::TensorView routes,
      int64_t row,
      int64_t experts,
      int64_t lease_address,
      int64_t timeout_ns,
      int64_t hot_address,
      int64_t hot_stride,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView dst_slots,
      int64_t captured,
      tvm::ffi::TensorView ram_slot,
      tvm::ffi::TensorView staging,
      tvm::ffi::TensorView map_chain,
      tvm::ffi::TensorView map_applied,
      tvm::ffi::TensorView ce_ok,
      tvm::ffi::TensorView cpu_ok,
      tvm::ffi::TensorView dst_rows,
      int64_t row_capacity,
      int64_t hit_copy_ce,
      int64_t cpu_on,
      int64_t cpu_misses,
      tvm::ffi::TensorView lane_kind,
      tvm::ffi::TensorView lane_slot,
      tvm::ffi::TensorView lane_node,
      tvm::ffi::TensorView go_1,
      tvm::ffi::TensorView host_rows_1,
      tvm::ffi::TensorView dst_slots_1,
      tvm::ffi::TensorView cpu_x,
      int64_t cpu_x_dst,
      tvm::ffi::TensorView cpu_weights,
      int64_t use_pdl) {
    using namespace host;
    using namespace expert_stream::wire;
    // The record carries i16 ids and is written with 16-byte stores (lease_layout.h).
    RuntimeCheck(experts <= Wire::kRecIdMax, "experts: a demand record carries expert ids up to ", Wire::kRecIdMax);
    RuntimeCheck(
        row_capacity <= Wire::kRecIdMax, "row_capacity: a demand record carries slots up to ", Wire::kRecIdMax);
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(page.data_ptr()) % 128 == 0,
        "page: must be 128-byte aligned, so each record's two cache lines are one prefetch pair (128-byte block)");
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto R_ = SymbolicSize{"routes"};
    auto Rows_ = SymbolicSize{"rows"};

    expert_stream::verify_named(
        "page",
        TensorMatcher({Wire::kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host),
        page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "planned", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "routes", TensorMatcher({R_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), routes);
    expert_stream::verify_named(
        "hot_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), hot_slots);
    if (hot_address != 0) {
      RuntimeCheck(hot_slots.size(0) >= hot_capacity, "hot_slots: size must be at least hot_capacity");
    }
    expert_stream::verify_named(
        "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, experts}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "staging",
        TensorMatcher({Rows_, Wire::kNodes * Wire::kLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        staging);
    expert_stream::verify_named(
        "map_chain", TensorMatcher({Rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), map_chain);
    expert_stream::verify_named(
        "map_applied", TensorMatcher({Rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), map_applied);
    expert_stream::verify_named(
        "ce_ok", TensorMatcher({Rows_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), ce_ok);
    expert_stream::verify_named(
        "cpu_ok", TensorMatcher({Rows_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), cpu_ok);
    expert_stream::verify_named(
        "dst_rows", TensorMatcher({Rows_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_rows);
    RuntimeCheck(row >= 0 && row < Rows_.unwrap(), "row: outside the map bank");
    for (auto [name, t] :
         {std::pair{"lane_kind", lane_kind},
          std::pair{"lane_slot", lane_slot},
          std::pair{"lane_node", lane_node},
          std::pair{"dst_slots_1", dst_slots_1}}) {
      expert_stream::verify_named(
          name, TensorMatcher({Wire::kLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), t);
    }
    expert_stream::verify_named("go_1", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_1);
    expert_stream::verify_named(
        "host_rows_1", TensorMatcher({Wire::kLanes}).with_dtype<int64_t>().with_device<kDLCUDA>(device), host_rows_1);
    RuntimeCheck(experts > 0, "experts: must be positive");
    // CPU experts: cpu_x is the layer's input row [1, hidden] (empty when off), cpu_weights the routes' weights.
    auto cpu_dtype = [](tvm::ffi::TensorView t) -> int64_t {
      const DLDataType d = t.dtype();
      if (d.code == kDLFloat && d.bits == 16) return kCpuDtypeF16;
      if (d.code == kDLBfloat && d.bits == 16) return kCpuDtypeBf16;
      if (d.code == kDLFloat && d.bits == 32) return kCpuDtypeF32;
      return -1;
    };
    const bool cpu_input = cpu_x_dst != 0;
    int64_t cpu_hidden = 0;
    if (cpu_input) {
      RuntimeCheck(captured != 0, "CPU experts: only a captured post stages the input");
      RuntimeCheck(
          cpu_x.device().device_type == kDLCUDA && cpu_weights.device().device_type == kDLCUDA,
          "CPU experts: cpu_x and cpu_weights live on the device");
      RuntimeCheck(
          cpu_x.is_contiguous() && cpu_weights.is_contiguous(),
          "CPU experts: cpu_x and cpu_weights must be contiguous");
      RuntimeCheck(cpu_dtype(cpu_x) >= 0 && cpu_dtype(cpu_weights) >= 0, "CPU experts: fp16, bf16 or fp32 inputs");
      RuntimeCheck(cpu_x.dim() == 2 && cpu_x.size(0) == 1, "CPU experts: cpu_x is one row [1, hidden]");
      cpu_hidden = cpu_x.size(1);
      RuntimeCheck(cpu_hidden > 0 && cpu_hidden % 8 == 0, "CPU experts: the hidden size must be a multiple of 8");
      RuntimeCheck(cpu_x_dst % 16 == 0, "CPU experts: the staged row must be 16-byte aligned");
    }
    RuntimeCheck(cpu_on == 0 || cpu_input || captured == 0, "CPU experts: a captured post needs the CPU input");
    RuntimeCheck(
        lease_address != 0 && lease_address % Wire::kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of Wire::kLeaseBlockAlign");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = PostParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .lanes = std::min<int64_t>(planned.size(0), dst_slots.size(0)),
        .routes = static_cast<const int64_t*>(routes.data_ptr()),
        .route_count = routes.size(0),
        .row = row,
        .experts = experts,
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .timeout_ns = timeout_ns,
        .hot_page = reinterpret_cast<uint8_t*>(hot_address),
        .hot_stride = hot_stride,
        .hot_slots = static_cast<const int64_t*>(hot_slots.data_ptr()),
        .hot_capacity = hot_capacity,
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .captured = captured,
        .ram_slot = static_cast<int32_t*>(ram_slot.data_ptr()),
        .staging = static_cast<int32_t*>(staging.data_ptr()),
        .map_chain = static_cast<int64_t*>(map_chain.data_ptr()),
        .map_applied = static_cast<int64_t*>(map_applied.data_ptr()),
        .ce_ok = static_cast<const uint8_t*>(ce_ok.data_ptr()),
        .cpu_ok = static_cast<const uint8_t*>(cpu_ok.data_ptr()),
        .dst_rows = static_cast<const int32_t*>(dst_rows.data_ptr()),
        .row_capacity = expert_stream::checked_row_capacity(row_capacity),
        .hit_copy_ce = hit_copy_ce,
        .cpu_on = cpu_on,
        .cpu_misses = cpu_misses,
        .lane_kind = static_cast<int32_t*>(lane_kind.data_ptr()),
        .lane_slot = static_cast<int32_t*>(lane_slot.data_ptr()),
        .lane_node = static_cast<int32_t*>(lane_node.data_ptr()),
        .go_1 = static_cast<int32_t*>(go_1.data_ptr()),
        .host_rows_1 = static_cast<int64_t*>(host_rows_1.data_ptr()),
        .dst_slots_1 = static_cast<int32_t*>(dst_slots_1.data_ptr()),
        .cpu_x_src = cpu_input ? cpu_x.data_ptr() : nullptr,
        .cpu_x_dtype = cpu_input ? cpu_dtype(cpu_x) : 0,
        .cpu_x_dst = reinterpret_cast<uint8_t*>(cpu_x_dst),
        .cpu_hidden = cpu_hidden,
        .cpu_weights = cpu_input ? cpu_weights.data_ptr() : nullptr,
        .cpu_weights_dtype = cpu_input ? cpu_dtype(cpu_weights) : 0,
        .cpu_weights_count = cpu_input ? cpu_weights.numel() : 0,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)
        .enable_pdl(use_pdl != 0)(
            use_pdl != 0 ? exl3_ram_miss_post_kernel<true> : exl3_ram_miss_post_kernel<false>, params);
  }

  /// Launches the bulk map apply kernel; see BulkApplyParams. `lease_address` is the completion block's address.
  static void map_bulk_apply(
      int64_t lease_address,
      tvm::ffi::TensorView ram_slot,
      tvm::ffi::TensorView staging,
      tvm::ffi::TensorView map_chain,
      tvm::ffi::TensorView map_applied,
      tvm::ffi::TensorView entries,
      tvm::ffi::TensorView row_capacity) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto Rows_ = SymbolicSize{"rows"};
    auto E_ = SymbolicSize{"experts"};
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, E_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "staging",
        TensorMatcher({Rows_, Wire::kNodes * Wire::kLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        staging);
    expert_stream::verify_named(
        "map_chain", TensorMatcher({Rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), map_chain);
    expert_stream::verify_named(
        "map_applied", TensorMatcher({Rows_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), map_applied);
    expert_stream::verify_named(
        "entries", TensorMatcher({-1, 3}).with_dtype<int32_t>().with_device<kDLCUDA>(device), entries);
    expert_stream::verify_named(
        "row_capacity", TensorMatcher({Rows_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), row_capacity);
    RuntimeCheck(
        lease_address != 0 && lease_address % Wire::kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of Wire::kLeaseBlockAlign");
    const auto stream = LaunchKernel::resolve_device(ram_slot.device());
    const auto params = BulkApplyParams{
        .lease = reinterpret_cast<const uint8_t*>(lease_address),
        .ram_slot = static_cast<int32_t*>(ram_slot.data_ptr()),
        .staging = static_cast<int32_t*>(staging.data_ptr()),
        .map_chain = static_cast<int64_t*>(map_chain.data_ptr()),
        .map_applied = static_cast<int64_t*>(map_applied.data_ptr()),
        .entries = static_cast<const int32_t*>(entries.data_ptr()),
        .entry_count = entries.size(0),
        .rows = Rows_.unwrap(),
        .experts = E_.unwrap(),
        .row_capacity = static_cast<const int32_t*>(row_capacity.data_ptr()),
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_map_bulk_apply_kernel, params);
  }
};

}  // namespace sglang
