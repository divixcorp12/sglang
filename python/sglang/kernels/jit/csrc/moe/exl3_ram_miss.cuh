// Device side of the option C RAM-miss service (DSV41 Phase 3b plan, D10-D15, D22).
//
// post: one block; thread 0 builds the layer's request (need = planned VRAM misses
// whose host-mapped slot map entry is -1; protect = every routed expert), writes it
// into the page's demand ring as a seqlock (seq cleared and fenced, volatile payload,
// seq release-stored) and release-stores demand_head. The record is posted for every MoE layer (the thread
// uses touch-only records for LRU recency); the wait is armed only when something is
// needed or advisories are on, and the record says so (kRecArmed): the thread only
// touches for an unarmed record, since nothing orders it before the next gathers. With `advise`, it also remembers this token's routes
// for `row` and posts the previous token's routes of `next_row` that are not in RAM
// as an advisory record.
// wait: one block; thread 0 polls demand_done with ld.acquire.sys and __nanosleep
// until it reaches the armed sequence or `timeout_ns` of %globaltimer passes, then
// translates the planned experts to pinned slots from the slot map. A timeout, a
// failed request, a planned row still not in RAM, or a fatal word already raised on
// the page raises the page's fatal word
// (sticky: later posts post nothing and later waits return at once) and sets keep
// to 0, which drops the layer's routed output for this forward.
//
// Lease mode (LEASE_PROTOCOL.md; the wrappers take a lease block, and a null one leaves everything above as it was):
// post also writes the request's LaneRequest into the lease block and arms every request that has planned lanes;
// lease_wait replaces wait's translate half: it validates each lane's RowResult and commits `go_count[0] = count`
// or fails closed with go_count 0 and a Terminal record; lease_ack is launched after the copy kernel, in the
// same stream, and release-stores one LaneAck word per committed lane (section 6.4 says why it is a separate kernel).
// Copy engine (LEASE_PROTOCOL.md 7.6): the service may publish a hit lane as COPYING and copy it with the DMA engine
// itself; copy_wait then waits for the service's CopyDone word instead of any kernel copying or acknowledging it.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>

#include "expert_stream/lease_kernels.cuh"
#include "expert_stream/row_copy_kernels.cuh"

namespace sglang {

void exl3_ram_miss_post(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView routes,
    int64_t row,
    int64_t advise,
    tvm::ffi::TensorView last_routes,
    int64_t next_row,
    int64_t lease_address,
    int64_t lease_d,
    int64_t timeout_ns,
    int64_t hot_address,
    int64_t hot_stride,
    tvm::ffi::TensorView hot_slots,
    int64_t hot_capacity,
    tvm::ffi::TensorView dst_slots,
    int64_t copy_engine) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_post_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int64_t*>(routes.data_ptr()),
      routes.size(0),
      row,
      slot_map.size(1),
      advise,
      static_cast<int32_t*>(last_routes.data_ptr()),
      next_row,
      reinterpret_cast<uint8_t*>(lease_address),  // zero: no lease block, today's protocol
      lease_d,
      timeout_ns,
      reinterpret_cast<uint8_t*>(hot_address),
      hot_stride,
      static_cast<const int64_t*>(hot_slots.data_ptr()),
      hot_capacity,
      dst_slots.size(0) > 0 ? static_cast<const int32_t*>(dst_slots.data_ptr()) : nullptr,  // empty: no plan slots
      dst_slots.size(0),
      copy_engine);
}

void exl3_ram_miss_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView slot_map,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    int64_t row,
    tvm::ffi::TensorView host_rows,
    tvm::ffi::TensorView keep,
    tvm::ffi::TensorView ram_miss,
    int64_t timeout_ns) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(slot_map.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      row,
      slot_map.size(1),
      host_rows.size(0),
      static_cast<int64_t*>(host_rows.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      timeout_ns);
}

void exl3_ram_miss_lease_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    int64_t row,
    tvm::ffi::TensorView host_rows,
    tvm::ffi::TensorView keep,
    tvm::ffi::TensorView ram_miss,
    int64_t timeout_ns,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView go_count,
    tvm::ffi::TensorView lane_ctx) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_lease_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      row,
      host_rows.size(0),
      static_cast<int64_t*>(host_rows.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      timeout_ns,
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<int32_t*>(go_count.data_ptr()),
      static_cast<int64_t*>(lane_ctx.data_ptr()));
}

void exl3_ram_miss_lease_ack(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView go_count,
    tvm::ffi::TensorView lane_ctx,
    tvm::ffi::TensorView keep) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, static_cast<int>(exl3_ram_miss_device::kLeaseLanes), stream)(
      exl3_ram_miss_lease_ack_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<const int32_t*>(go_count.data_ptr()),
      static_cast<const int64_t*>(lane_ctx.data_ptr()),
      static_cast<float*>(keep.data_ptr()));
}

void exl3_ram_miss_lease_hit_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView dst_slots,
    int64_t row,
    tvm::ffi::TensorView host_rows_1,
    tvm::ffi::TensorView dst_slots_1,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView go_1,
    tvm::ffi::TensorView lane_ctx_1,
    tvm::ffi::TensorView origin_1,
    tvm::ffi::TensorView claimed,
    tvm::ffi::TensorView violated,
    int64_t budget_ns) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_lease_hit_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      row,
      // The lane bound must cover EVERY lane-indexed array the kernel reads, not just its own
      // staging buffer: dst_slots is the plan's, of length capacity, while host_rows_1 is LANES.
      // Bounding by the staging buffer alone lets a device-side count above capacity, up to LANES, read
      // dst_slots out of bounds. The planner clamps count today, but count lives on the device
      // precisely because no host check can bound it.
      std::min<int64_t>(host_rows_1.size(0), dst_slots.size(0)),
      static_cast<int64_t*>(host_rows_1.data_ptr()),
      static_cast<int32_t*>(dst_slots_1.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<int32_t*>(go_1.data_ptr()),
      static_cast<int64_t*>(lane_ctx_1.data_ptr()),
      static_cast<int32_t*>(origin_1.data_ptr()),
      static_cast<int32_t*>(claimed.data_ptr()),
      static_cast<int32_t*>(violated.data_ptr()),
      budget_ns);
}

void exl3_ram_miss_lease_rest_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView dst_slots,
    int64_t row,
    tvm::ffi::TensorView host_rows_2,
    tvm::ffi::TensorView dst_slots_2,
    tvm::ffi::TensorView ram_miss,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView claimed,
    tvm::ffi::TensorView go_2,
    tvm::ffi::TensorView lane_ctx_2,
    tvm::ffi::TensorView origin_2) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_lease_rest_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      row,
      // The lane bound must cover EVERY lane-indexed array the kernel reads, not just its own
      // staging buffer: dst_slots is the plan's, of length capacity, while host_rows_2 is LANES.
      // Bounding by the staging buffer alone lets a device-side count above capacity, up to LANES, read
      // dst_slots out of bounds. The planner clamps count today, but count lives on the device
      // precisely because no host check can bound it.
      std::min<int64_t>(host_rows_2.size(0), dst_slots.size(0)),
      static_cast<int64_t*>(host_rows_2.data_ptr()),
      static_cast<int32_t*>(dst_slots_2.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<const int32_t*>(claimed.data_ptr()),
      static_cast<int32_t*>(go_2.data_ptr()),
      static_cast<int64_t*>(lane_ctx_2.data_ptr()),
      static_cast<int32_t*>(origin_2.data_ptr()));
}

void exl3_ram_miss_lease_stage_ack(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView go_count,
    tvm::ffi::TensorView lane_ctx,
    tvm::ffi::TensorView origin,
    tvm::ffi::TensorView violated) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, static_cast<int>(exl3_ram_miss_device::kLeaseLanes), stream)(
      exl3_ram_miss_lease_stage_ack_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<const int32_t*>(go_count.data_ptr()),
      static_cast<const int64_t*>(lane_ctx.data_ptr()),
      static_cast<const int32_t*>(origin.data_ptr()),
      static_cast<int32_t*>(violated.data_ptr()));
}

void exl3_ram_miss_lease_finalize(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView go_1,
    tvm::ffi::TensorView go_2,
    tvm::ffi::TensorView go_ce,
    tvm::ffi::TensorView violated,
    tvm::ffi::TensorView keep,
    int64_t lease_address,
    int64_t lease_d) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_lease_finalize_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(go_1.data_ptr()),
      static_cast<const int32_t*>(go_2.data_ptr()),
      static_cast<const int32_t*>(go_ce.data_ptr()),
      static_cast<const int32_t*>(violated.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d);
}

void exl3_ram_miss_lease_copy_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView count,
    int64_t lease_address,
    int64_t lease_c,
    int64_t lease_d,
    int64_t sm_table_address,
    int64_t sm_count,
    tvm::ffi::TensorView go_ce) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  const int threads = sm_count > 0 ? exl3_ram_miss_device::kCopyWaitThreads : exl3_ram_miss_device::kBlock;
  host::LaunchKernel(1, threads, stream)(
      exl3_ram_miss_lease_copy_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_c,
      lease_d,
      reinterpret_cast<const int64_t*>(sm_table_address),
      sm_count,
      static_cast<int32_t*>(go_ce.data_ptr()));
}

void exl3_ram_miss_lease_stream_hit_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView dst_slots,
    int64_t row,
    tvm::ffi::TensorView host_rows_1,
    tvm::ffi::TensorView dst_slots_1,
    int64_t lease_address,
    int64_t lease_d,
    tvm::ffi::TensorView go_1,
    tvm::ffi::TensorView lane_ctx_1,
    tvm::ffi::TensorView origin_1,
    tvm::ffi::TensorView claimed,
    tvm::ffi::TensorView violated,
    int64_t budget_ns,
    tvm::ffi::TensorView go_2,
    tvm::ffi::TensorView stream_count,
    tvm::ffi::TensorView stream_abort) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, exl3_ram_miss_device::kBlock, stream)(
      exl3_ram_miss_lease_stream_hit_wait_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      row,
      std::min<int64_t>(host_rows_1.size(0), dst_slots.size(0)),  // every lane-indexed array, as the two-phase W1
      static_cast<int64_t*>(host_rows_1.data_ptr()),
      static_cast<int32_t*>(dst_slots_1.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      static_cast<int32_t*>(go_1.data_ptr()),
      static_cast<int64_t*>(lane_ctx_1.data_ptr()),
      static_cast<int32_t*>(origin_1.data_ptr()),
      static_cast<int32_t*>(claimed.data_ptr()),
      static_cast<int32_t*>(violated.data_ptr()),
      budget_ns,
      static_cast<int32_t*>(go_2.data_ptr()),
      static_cast<uint32_t*>(stream_count.data_ptr()),
      static_cast<int32_t*>(stream_abort.data_ptr()));
}

void exl3_ram_miss_lease_stream(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView planned,
    tvm::ffi::TensorView count,
    tvm::ffi::TensorView dst_slots,
    int64_t row,
    int64_t experts,
    tvm::ffi::TensorView host_rows_2,
    tvm::ffi::TensorView dst_slots_2,
    tvm::ffi::TensorView ram_miss,
    int64_t lease_address,
    int64_t lease_d,
    int64_t lease_p,
    tvm::ffi::TensorView claimed,
    tvm::ffi::TensorView go_2,
    tvm::ffi::TensorView lane_ctx_2,
    tvm::ffi::TensorView origin_2,
    tvm::ffi::TensorView stream_count,
    tvm::ffi::TensorView stream_abort,
    tvm::ffi::TensorView segments,
    tvm::ffi::TensorView segment_map,
    int64_t row_segments,
    tvm::ffi::TensorView piece_runs,
    tvm::ffi::TensorView fault) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(exl3_ram_miss_device::kStreamBlocks, exl3_ram_miss_device::kStreamThreads, stream)(
      exl3_ram_miss_lease_stream_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(planned.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      static_cast<const int32_t*>(dst_slots.data_ptr()),
      row,
      std::min<int64_t>(host_rows_2.size(0), dst_slots.size(0)),  // every lane-indexed array, as W2
      experts,
      static_cast<int64_t*>(host_rows_2.data_ptr()),
      static_cast<int32_t*>(dst_slots_2.data_ptr()),
      static_cast<int64_t*>(ram_miss.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d,
      lease_p,
      static_cast<const int32_t*>(claimed.data_ptr()),
      static_cast<int32_t*>(go_2.data_ptr()),
      static_cast<int64_t*>(lane_ctx_2.data_ptr()),
      static_cast<int32_t*>(origin_2.data_ptr()),
      static_cast<uint32_t*>(stream_count.data_ptr()),
      static_cast<int32_t*>(stream_abort.data_ptr()),
      static_cast<const int64_t*>(segments.data_ptr()),
      static_cast<int64_t>(segments.size(0)),
      static_cast<const int32_t*>(segment_map.data_ptr()),
      row_segments,
      static_cast<const int32_t*>(piece_runs.data_ptr()),
      static_cast<const int32_t*>(fault.data_ptr()));
}

}  // namespace sglang
