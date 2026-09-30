// Lease-protocol kernels: the post and W1, the hit wait (analysis/dsv41-drive/LEASE_PROTOCOL.md). Format-free.
#pragma once

#include <sgl_kernel/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "lease_device.cuh"
#include "tensor_checks.h"

namespace sglang {

struct PostParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int64_t* routes;
  int64_t route_count;
  int64_t row;
  int64_t experts;
  uint8_t* lease;
  int64_t timeout_ns;
  uint8_t* hot_page;
  int64_t hot_stride;
  const int64_t* hot_slots;
  int64_t hot_capacity;
  const int32_t* dst_slots;
  int64_t dst_count;
  int64_t captured;
  // CPU experts: with cpu_x_dst set, the block stages the layer's input row there (cpu_hidden elements of cpu_x_src,
  // of dtype cpu_x_dtype, as fp16) and the LaneRequest carries each lane's routing weight from cpu_weights
  // (cpu_weights_count routes aligned with `routes`, dtype cpu_weights_dtype). Null cpu_x_dst: none of it.
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

SGL_DEVICE float cpu_input_value(const void* src, int64_t dtype, int64_t i) {
  if (dtype == kCpuDtypeF16) return __half2float(static_cast<const __half*>(src)[i]);
  if (dtype == kCpuDtypeBf16) return __bfloat162float(static_cast<const __nv_bfloat16*>(src)[i]);
  return static_cast<const float*>(src)[i];
}

// The whole block: the layer's input row as fp16, 8 elements per 16-byte store to the host row (the launcher checks
// hidden % 8 == 0 and the alignment). Each thread fences its own stores at system scope before the barrier, so thread
// 0's later release of the LaneRequest and demand_head orders all of them.
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

// Lease-chain PDL (kUsePDL = SGLANG_DSV41_ENABLE_LEASE_PDL, every chain kernel but C1 and CC): the wait is the
// kernel's first statement and the trigger its second, so every word is read and written after the primary grid
// completed and flushed, as without PDL; the ordering is transitive down the chain (LEASE_PROTOCOL.md, "PDL").
template <bool kUsePDL>
__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_post_kernel(
    const __grid_constant__ PostParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  using namespace device::expert_stream;
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const bool cpu_experts = p.cpu_x_dst != nullptr;
  if (cpu_experts) stage_cpu_input(p);
  if (threadIdx.x != 0) return;
  int32_t protect[kMaxIds];
  int protect_count = 0;
  for (int64_t i = 0; i < p.route_count && protect_count < kMaxIds; ++i) {
    const int32_t expert = static_cast<int32_t>(p.routes[i]);
    if (expert >= 0 && expert < p.experts && !listed(protect, protect_count, expert)) protect[protect_count++] = expert;
  }
  // W1 traps on a plan wider than kMaxIds; the clamp only keeps this kernel inside the LaneRequest's arrays.
  const int64_t planned_count =
      min(max(static_cast<int64_t>(p.count[0]), static_cast<int64_t>(0)), static_cast<int64_t>(kMaxIds));
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) {
    seq = 1;
    state[kEpoch] += 1;
  }
  state[kPosted] = static_cast<int32_t>(seq);
  const uint64_t generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kEpoch])) << 32) | seq;
  const int64_t idx = ring_index(seq);
  if (p.hot_page != nullptr) {
    uint8_t* hot = p.hot_page + static_cast<int64_t>((seq - 1u) % kHotRecords) * p.hot_stride;
    st_relaxed_sys<uint32_t>(hot, 0u);
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    uint8_t* bits = hot + kHotHeaderBytes;
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
  // Every request with lanes is armed: the service leases hits as well as misses, and reads a request's lanes only
  // from an armed record, so an unarmed request with lanes would be copied from slots nobody leased.
  const bool armed = planned_count > 0;
  // LaneRequest: the seqlock shape of write_record. Only an armed request's is read, and the device posts request
  // G + 16 only after G's chain ended, so the service never meets a half-written one it would act on.
  uint8_t* request = p.lease + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
  st_relaxed_sys<uint64_t>(request + kLeaseLrGen, 0ull);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(request + kLeaseLrCount, static_cast<uint32_t>(planned_count));
  st_relaxed_sys<uint32_t>(request + kLeaseLrFlags, p.captured != 0 ? kLeaseLrFlagCaptured : 0u);
  for (int i = 0; i < kMaxIds; ++i) {
    st_relaxed_sys<int32_t>(
        request + kLeaseLrExpert + 4 * i, i < planned_count ? static_cast<int32_t>(p.planned[i]) : -1);
    st_relaxed_sys<int32_t>(
        request + kLeaseLrDst + 4 * i, p.dst_slots != nullptr && i < planned_count && i < p.dst_count ? p.dst_slots[i] : -1);
    // The lane expert's routing weight: its route's (BS1 routes are distinct experts). 0 past the plan or when off.
    float weight = 0.0f;
    if (cpu_experts && i < planned_count) {
      for (int64_t r = 0; r < p.route_count && r < p.cpu_weights_count; ++r) {
        if (p.routes[r] == p.planned[i]) weight += cpu_input_value(p.cpu_weights, p.cpu_weights_dtype, r);
      }
    }
    st_relaxed_sys<uint32_t>(request + kLeaseLrWeight + 4 * i, __float_as_uint(weight));
  }
  st_release_sys64(request + kLeaseLrGen, generation);
  uint8_t* record = page + kDemandRing + idx * kRecordBytes;
  write_record(record, seq, p.row, protect, protect_count, armed ? 1u : 0u);
  // A release orders every earlier store of this thread: the hot page, the LaneRequest and the record come first.
  st_release_sys(page + kDemandHead, seq);
  state[kPending] = armed ? static_cast<int32_t>(seq) : 0;
  state[kPendingEpoch] = state[kEpoch];
  store_deadline(state, global_ns() + static_cast<uint64_t>(p.timeout_ns));
}

// W1, the hit wait. Polls each planned lane's RowResult for at most `budget_ns` and hands the ones the service
// granted inside its reservation hold (before read()) to C1, compacted: READY lanes into host_rows_1/dst_slots_1 in
// one order (source and destination compacted apart would send a lane's bytes to another lane's slot), COPYING and
// CPU lanes marked for CW. It never waits on demand_done: a LOADING lane is S's, and so is any lane still unpublished
// when the budget runs out. `claimed` is S's complement: 0 S's, 1 C1's, 2 CW's.
struct HitWaitParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t lanes;
  int64_t* host_rows_1;
  int32_t* dst_slots_1;
  const uint8_t* lease;
  int32_t* go_1;
  int32_t* claimed;
  int64_t budget_ns;
  uint32_t row_capacity;  // the row's pinned slots: fixed for the process, so a captured argument is sound
};

template <bool kUsePDL>
__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_hit_wait_kernel(
    const __grid_constant__ HitWaitParams p) {
  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  const uint64_t start = global_ns();
  p.go_1[0] = 0;  // C1 reads it as its count: written on every path
  for (int64_t i = 0; i < kLeaseLanes; ++i)
    p.claimed[i] = 0;
  const int64_t planned_count = max(static_cast<int64_t>(p.count[0]), static_cast<int64_t>(0));
  // The post clamps at kMaxIds lanes and the plan's buffers hold `lanes`: a wider plan would lose lanes silently.
  if (planned_count > kLeaseLanes || planned_count > p.lanes) __trap();
  if (planned_count == 0) return;  // an unarmed post: kPending is 0 and nothing was leased
  const uint32_t seq = static_cast<uint32_t>(p.state[kPending]);
  const uint64_t generation = pending_generation(p.state);
  const uint8_t* results = p.lease + kLeaseRowResult + ring_index(seq) * kLeaseLanes * kLeaseRowResultBytes;
  int32_t slots[kMaxIds];
  int64_t settled = 0;  // lanes claimed, or LOADING (S's for good: a LOADING lane never turns READY)
  for (;;) {
    for (int64_t i = 0; i < planned_count; ++i) {
      if (p.claimed[i] != 0) continue;
      const LaneRead r = lane_read(results + i * kLeaseRowResultBytes, generation, p.row_capacity);
      if (r.tag == kLeaseTagReady || copy_owned(r.tag)) {
        p.claimed[i] = copy_owned(r.tag) ? 2 : 1;
        slots[i] = r.host_slot;
        ++settled;
      } else if (r.tag == kLeaseTagLoading) {
        p.claimed[i] = 3;  // not a claim: marks the lane settled for this loop, and is reset to 0 below
        ++settled;
      }
    }
    if (settled == planned_count) break;
    // Once served, every lane is published, so a later poll can discover nothing new.
    if (reached(ld_acquire_sys(p.page + kDemandDone), seq)) break;
    // Time, not passes: a pass reads every unclaimed lane across PCIe, measured up to 212 us for 8 passes.
    if (static_cast<int64_t>(global_ns() - start) >= p.budget_ns) break;
    __nanosleep(256);
  }
  int64_t n = 0;
  for (int64_t i = 0; i < planned_count; ++i) {
    if (p.claimed[i] == 3) p.claimed[i] = 0;
    if (p.claimed[i] != 1) continue;
    p.host_rows_1[n] = static_cast<int64_t>(slots[i]);
    p.dst_slots_1[n] = p.dst_slots[i];
    ++n;
  }
  p.go_1[0] = static_cast<int32_t>(n);
}

/// \brief Checked host launchers for the lease-protocol kernels above: post and lease_hit_wait.
///
/// Every tensor argument is verified with `TensorMatcher` (named via `verify_named`) and every address with
/// `RuntimeCheck` before the params struct is built and the kernel launched.
struct LeaseProtocolKernel {
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
      tvm::ffi::TensorView cpu_x,
      int64_t cpu_x_dst,
      tvm::ffi::TensorView cpu_weights,
      int64_t use_pdl) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto R_ = SymbolicSize{"routes"};

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
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
    RuntimeCheck(experts > 0, "experts: must be positive");
    // CPU experts: cpu_x is the layer's input row [1, hidden] (or empty when off), cpu_weights the routes' weights.
    auto cpu_dtype = [](tvm::ffi::TensorView t) -> int64_t {
      const DLDataType d = t.dtype();
      if (d.code == kDLFloat && d.bits == 16) return kCpuDtypeF16;
      if (d.code == kDLBfloat && d.bits == 16) return kCpuDtypeBf16;
      if (d.code == kDLFloat && d.bits == 32) return kCpuDtypeF32;
      return -1;
    };
    const bool cpu_on = cpu_x_dst != 0;
    int64_t cpu_hidden = 0;
    if (cpu_on) {
      RuntimeCheck(captured != 0, "CPU experts: only a captured post stages the input");
      RuntimeCheck(cpu_x.device().device_type == kDLCUDA && cpu_weights.device().device_type == kDLCUDA,
                   "CPU experts: cpu_x and cpu_weights live on the device");
      RuntimeCheck(cpu_x.is_contiguous() && cpu_weights.is_contiguous(), "CPU experts: cpu_x and cpu_weights must be contiguous");
      RuntimeCheck(cpu_dtype(cpu_x) >= 0 && cpu_dtype(cpu_weights) >= 0, "CPU experts: fp16, bf16 or fp32 inputs");
      RuntimeCheck(cpu_x.dim() == 2 && cpu_x.size(0) == 1, "CPU experts: cpu_x is one row [1, hidden]");
      cpu_hidden = cpu_x.size(1);
      RuntimeCheck(cpu_hidden > 0 && cpu_hidden % 8 == 0, "CPU experts: the hidden size must be a multiple of 8");
      RuntimeCheck(cpu_x_dst % 16 == 0, "CPU experts: the staged row must be 16-byte aligned");
    }
    RuntimeCheck(
        lease_address != 0 && lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of kLeaseBlockAlign");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = PostParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
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
        .dst_slots = dst_slots.size(0) > 0 ? static_cast<const int32_t*>(dst_slots.data_ptr()) : nullptr,
        .dst_count = dst_slots.size(0),
        .captured = captured,
        .cpu_x_src = cpu_on ? cpu_x.data_ptr() : nullptr,
        .cpu_x_dtype = cpu_on ? cpu_dtype(cpu_x) : 0,
        .cpu_x_dst = reinterpret_cast<uint8_t*>(cpu_x_dst),
        .cpu_hidden = cpu_hidden,
        .cpu_weights = cpu_on ? cpu_weights.data_ptr() : nullptr,
        .cpu_weights_dtype = cpu_on ? cpu_dtype(cpu_weights) : 0,
        .cpu_weights_count = cpu_on ? cpu_weights.numel() : 0,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(use_pdl != 0)(
        use_pdl != 0 ? exl3_ram_miss_post_kernel<true> : exl3_ram_miss_post_kernel<false>, params);
  }

  static void lease_hit_wait(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      tvm::ffi::TensorView planned,
      tvm::ffi::TensorView count,
      tvm::ffi::TensorView dst_slots,
      tvm::ffi::TensorView host_rows_1,
      tvm::ffi::TensorView dst_slots_1,
      int64_t lease_address,
      tvm::ffi::TensorView go_1,
      tvm::ffi::TensorView claimed,
      int64_t budget_ns,
      int64_t row_capacity,
      int64_t use_pdl) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    // The lane bound covers every lane-indexed array the kernel reads: dst_slots is the plan's, host_rows_1 kLeaseLanes.
    const int64_t lanes = std::min<int64_t>(host_rows_1.size(0), dst_slots.size(0));

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    RuntimeCheck(P_.unwrap() >= lanes, "planned: must have at least as many lanes as host_rows_1/dst_slots");
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);
    expert_stream::verify_named(
        "host_rows_1", TensorMatcher({kLeaseLanes}).with_dtype<int64_t>().with_device<kDLCUDA>(device), host_rows_1);
    expert_stream::verify_named(
        "dst_slots_1", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots_1);
    expert_stream::verify_named("go_1", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_1);
    expert_stream::verify_named(
        "claimed", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), claimed);
    RuntimeCheck(
        lease_address != 0 && lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be a nonzero multiple of kLeaseBlockAlign");

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = HitWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .lanes = lanes,
        .host_rows_1 = static_cast<int64_t*>(host_rows_1.data_ptr()),
        .dst_slots_1 = static_cast<int32_t*>(dst_slots_1.data_ptr()),
        .lease = reinterpret_cast<const uint8_t*>(lease_address),
        .go_1 = static_cast<int32_t*>(go_1.data_ptr()),
        .claimed = static_cast<int32_t*>(claimed.data_ptr()),
        .budget_ns = budget_ns,
        .row_capacity = expert_stream::checked_row_capacity(row_capacity),
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(use_pdl != 0)(
        use_pdl != 0 ? exl3_ram_miss_lease_hit_wait_kernel<true> : exl3_ram_miss_lease_hit_wait_kernel<false>, params);
  }
};

}  // namespace sglang
