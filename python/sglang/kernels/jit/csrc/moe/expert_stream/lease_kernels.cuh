// Lease-protocol kernels: post, the batched and lease waits, the two-phase hit/rest waits, and the
// acknowledgement and finalize kernels (a split of the EXL3 instantiation's former exl3_ram_miss.cuh). Format-free.
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "tensor_checks.h"

namespace sglang {

struct PostParams {
  uint8_t* page;
  int32_t* state;
  const int32_t* slot_map;
  const int64_t* planned;
  const int32_t* count;
  const int64_t* routes;
  int64_t route_count;
  int64_t row;
  int64_t experts;
  int64_t advise;
  int32_t* last_routes;
  int64_t next_row;
  uint8_t* lease;
  int64_t lease_d;
  int64_t timeout_ns;
  uint8_t* hot_page;
  int64_t hot_stride;
  const int64_t* hot_slots;
  int64_t hot_capacity;
  const int32_t* dst_slots;
  int64_t dst_count;
  int64_t copy_engine;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_post_kernel(
    const __grid_constant__ PostParams p) {
  const device::expert_stream::TestPdlEntry test_pdl_entry(1);  // test builds only (kTestPdl)
  device::expert_stream::test_pdl_trigger();  // test builds only (kTestPdlEarly): see results.md, chain PDL
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int32_t* __restrict__ const slot_map = p.slot_map;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int64_t* __restrict__ const routes = p.routes;
  const int64_t route_count = p.route_count;
  const int64_t row = p.row;
  const int64_t experts = p.experts;
  const int64_t advise = p.advise;
  int32_t* __restrict__ const last_routes = p.last_routes;
  const int64_t next_row = p.next_row;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int64_t timeout_ns = p.timeout_ns;
  uint8_t* __restrict__ const hot_page = p.hot_page;
  const int64_t hot_stride = p.hot_stride;
  const int64_t* __restrict__ const hot_slots = p.hot_slots;
  const int64_t hot_capacity = p.hot_capacity;
  const int32_t* __restrict__ const dst_slots = p.dst_slots;
  const int64_t dst_count = p.dst_count;
  const int64_t copy_engine = p.copy_engine;
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  if (state[kSticky] != 0 || ld_acquire_sys(page + kFatal) != 0) {
    state[kSticky] = 1;
    state[kPending] = 0;
    return;
  }
  const int32_t* map_row = slot_map + row * experts;
  int32_t need[kMaxIds];
  int32_t protect[kMaxIds];
  int need_count = 0;
  int protect_count = 0;
  const int64_t planned_count =
      min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), static_cast<int64_t>(kMaxIds));
  for (int64_t i = 0; i < planned_count; ++i) {
    const int32_t expert = static_cast<int32_t>(planned[i]);
    if (expert >= 0 && expert < experts && ld_relaxed_sys(map_row + expert) < 0 && !listed(need, need_count, expert)) {
      need[need_count++] = expert;
    }
  }
  for (int64_t i = 0; i < route_count && protect_count < kMaxIds; ++i) {
    const int32_t expert = static_cast<int32_t>(routes[i]);
    if (expert >= 0 && expert < experts && !listed(protect, protect_count, expert)) protect[protect_count++] = expert;
  }
  for (int i = 0; i < need_count && protect_count < kMaxIds; ++i) {
    if (!listed(protect, protect_count, need[i])) protect[protect_count++] = need[i];
  }
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) {
    seq = 1;
    state[kEpoch] +=
        1;  // seq32 wrapped (LEASE_PROTOCOL.md 11.3); the wait and ack kernels read it back as kPendingEpoch
  }
  state[kPosted] = static_cast<int32_t>(seq);
  uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
  if (hot_page != nullptr) {
    uint8_t* hot = hot_page + static_cast<int64_t>((seq - 1u) % kHotRecords) * hot_stride;
    st_relaxed_sys<uint32_t>(hot, 0u);
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    st_relaxed_sys<uint32_t>(hot + 4, static_cast<uint32_t>(experts));
    uint8_t* bits = hot + kHotHeaderBytes;
    for (int64_t byte = 0; byte < (experts + 7) / 8; ++byte) {
      uint8_t mask = 0;
      for (int64_t slot = 0; slot < hot_capacity; ++slot) {
        const int64_t expert = hot_slots[slot];
        if (expert >= byte * 8 && expert < byte * 8 + 8) mask |= static_cast<uint8_t>(1u << (expert - byte * 8));
      }
      st_relaxed_sys(bits + byte, mask);
    }
    st_release_sys(hot, seq);  // orders the bitmap before the seq; no separate fence
  }
  // Lease mode arms every request that has planned lanes, not only those with a miss (LEASE_PROTOCOL.md section 15:
  // the all-hit handshake is where the lease is granted, so an unarmed record is reachable only for count == 0). The
  // service leases RAM hits too (7.2) and reads a request's lanes only when the record is armed; an unarmed request
  // with lanes would be copied from slots nobody leased. The extra service round trip per layer is unmeasured (OPEN
  // 11).
  const bool armed = need_count > 0 || advise != 0 || (lease != nullptr && planned_count > 0);
  const uint32_t lanes = static_cast<uint32_t>(max(count[0], 0));  // the plan's lanes, unclamped
  if (lease != nullptr) {
    // LaneRequest (6.3): the seqlock shape of write_record, before the demand record and demand_head are published.
    const uint64_t generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kEpoch])) << 32) | seq;
    uint8_t* request = lease + lease_d + kLeaseLaneRequest +
                       static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLaneRequestBytes;
    st_relaxed_sys<uint64_t>(request + kLeaseLrGen, 0ull);
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
    st_relaxed_sys<uint32_t>(request + kLeaseLrCount, static_cast<uint32_t>(planned_count));
    st_relaxed_sys<uint32_t>(request + kLeaseLrRow, static_cast<uint32_t>(row));
    for (int i = 0; i < kMaxIds; ++i)
      st_relaxed_sys<int32_t>(
          request + kLeaseLrExpert + 4 * i, i < planned_count ? static_cast<int32_t>(planned[i]) : -1);
    for (int i = 0; i < kMaxIds; ++i) {
      st_relaxed_sys<int32_t>(
          request + kLeaseLrDst + 4 * i,
          dst_slots != nullptr && i < planned_count && i < dst_count ? dst_slots[i] : -1);
    }
    st_relaxed_sys<uint32_t>(request + kLeaseLrFlags, copy_engine != 0 ? kLeaseLrFlagCopyEngine : 0u);
    st_release_sys64(request + kLeaseLrGen, tagged_word(kLeaseTagDemand, generation));
  }
  write_record(record, seq, row, need, need_count, protect, protect_count, 0, armed ? 1u : 0u, lanes);
  // A release store orders every earlier store of this thread, so demand_head is published after the hot page, the
  // LaneRequest and the record.
  st_release_sys(page + kDemandHead, seq);
  state[kPending] = armed ? static_cast<int32_t>(seq) : 0;
  state[kPendingEpoch] = state[kEpoch];
  // D5. The one deadline both stages compare against, absolute rather than per-stage: two stages each timing from
  // their own start would let a request run to twice the configured timeout.
  store_deadline(state, global_ns() + static_cast<uint64_t>(timeout_ns));
  state[kReqFailed] = 0;  // D6, per request: kSticky is a process-lifetime latch and cannot be reused for this
  state[kFailReason] = 0;
  if (advise == 0) return;
  for (int i = 0; i < kMaxIds; ++i)
    last_routes[row * kMaxIds + i] = i < protect_count ? protect[i] : -1;
  if (next_row < 0) return;
  const int32_t* next_map = slot_map + next_row * experts;
  int32_t ahead[kMaxIds];
  int ahead_count = 0;
  for (int i = 0; i < kMaxIds; ++i) {
    const int32_t expert = last_routes[next_row * kMaxIds + i];
    if (expert >= 0 && expert < experts && ld_relaxed_sys(next_map + expert) < 0) ahead[ahead_count++] = expert;
  }
  if (ahead_count == 0) return;
  uint32_t advice = static_cast<uint32_t>(state[kAdvised]) + 1u;
  if (advice == 0) advice = 1;
  state[kAdvised] = static_cast<int32_t>(advice);
  uint8_t* advice_record = page + kAdviseRing + static_cast<int64_t>((advice - 1u) % kAdviseRecords) * kRecordBytes;
  write_record(advice_record, advice, next_row, ahead, ahead_count, ahead, ahead_count, seq, 1u, ahead_count);
  st_release_sys(page + kAdviseHead, advice);
}

struct WaitParams {
  uint8_t* page;
  int32_t* state;
  const int32_t* slot_map;
  const int64_t* planned;
  const int32_t* count;
  int64_t row;
  int64_t experts;
  int64_t lanes;
  int64_t* host_rows;
  float* keep;
  int64_t* ram_miss;
  int64_t timeout_ns;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_wait_kernel(
    const __grid_constant__ WaitParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int32_t* __restrict__ const slot_map = p.slot_map;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int64_t row = p.row;
  const int64_t experts = p.experts;
  const int64_t lanes = p.lanes;
  int64_t* __restrict__ const host_rows = p.host_rows;
  float* __restrict__ const keep = p.keep;
  int64_t* __restrict__ const ram_miss = p.ram_miss;
  const int64_t timeout_ns = p.timeout_ns;
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  // D15: the page's fatal word too, not only this device's sticky flag.
  bool ok = state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0;
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  if (ok && seq != 0) {
    state[kWaits] += 1;
    const uint64_t start = global_ns();
    int64_t polls = 0;
    uint32_t done = ld_acquire_sys(page + kDemandDone);
    while (!reached(done, seq) && static_cast<int64_t>(global_ns() - start) < timeout_ns) {
      __nanosleep(256);
      ++polls;
      done = ld_acquire_sys(page + kDemandDone);
    }
    const int64_t total = static_cast<int64_t>(state[kPolls]) + polls;
    state[kPolls] = static_cast<int32_t>(total < 0x7fffffffLL ? total : 0x7fffffffLL);
    if (!reached(done, seq)) {
      state[kTimeouts] += 1;
      raise_fatal(page, seq);
      ok = false;
    } else {
      // `done` came from an acquire load, which orders every load below after it: no fence needed.
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = ld_relaxed_sys<uint16_t>(record + kRecStatus);
      if (status != kServed) {
        state[kFailures] += 1;
        raise_fatal(page, seq);
        ok = false;
      }
    }
  }
  state[kPending] = 0;
  const int32_t* map_row = slot_map + row * experts;
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), lanes);
  int64_t misses = 0;
  for (int64_t i = 0; i < lanes; ++i) {
    int64_t slot = 0;
    if (i < planned_count) {
      const int64_t expert = planned[i];
      slot = expert >= 0 && expert < experts ? ld_relaxed_sys(map_row + expert) : -1;
      if (slot < 0) {
        ++misses;
        slot = 0;
      }
    }
    host_rows[i] = slot;
  }
  if (misses > 0 && ok) {
    // A served (or unarmed) request left a planned row out of RAM: never expected.
    state[kUnservedMisses] += static_cast<int32_t>(misses);
    raise_fatal(page, 0xFFFFFFFFu);
    ok = false;
  }
  if (!ok) state[kSticky] = 1;
  ram_miss[0] += misses;
  keep[0] = ok ? 1.0f : 0.0f;
}

// LEASE_PROTOCOL.md 7.3. Replaces the translate half of exl3_ram_miss_wait_kernel for lease mode: host_rows comes
// from the lanes' RowResults, never from the slot map. `go_count[0]` is zero on entry and is written exactly once,
// as the last store of a commit; every other exit leaves it zero, so the copy kernel that reads it as its active
// count copies nothing and the ack kernel that reads it acknowledges nothing.
struct LeaseWaitParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  int64_t row;
  int64_t lanes;
  int64_t* host_rows;
  float* keep;
  int64_t* ram_miss;
  int64_t timeout_ns;
  uint8_t* lease;
  int64_t lease_d;
  int32_t* go_count;
  int64_t* lane_ctx;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_wait_kernel(
    const __grid_constant__ LeaseWaitParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int64_t row = p.row;
  const int64_t lanes = p.lanes;
  int64_t* __restrict__ const host_rows = p.host_rows;
  float* __restrict__ const keep = p.keep;
  int64_t* __restrict__ const ram_miss = p.ram_miss;
  const int64_t timeout_ns = p.timeout_ns;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  int32_t* __restrict__ const go_count = p.go_count;
  int64_t* __restrict__ const lane_ctx = p.lane_ctx;
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  go_count[0] = 0;  // fail closed
  bool ok =
      state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0 && ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
  uint32_t reason = ok ? 0u : kLeaseReasonAborted;
  uint32_t fatal_word = 0;  // what to raise, when this wait is the one that fails the page
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  // The post kernel clamps a request at kMaxIds lanes, so a plan of more would silently lose the rest. The host_rows
  // buffer is the other bound: never write past it.
  if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
    ok = false;
    reason = kLeaseReasonCount;
    fatal_word = 0xFFFFFFFFu;
  }
  if (ok && seq != 0) {
    state[kWaits] += 1;
    const uint64_t start = global_ns();
    int64_t polls = 0;
    bool aborted = false;
    uint32_t done = ld_acquire_sys(page + kDemandDone);
    while (!reached(done, seq) && static_cast<int64_t>(global_ns() - start) < timeout_ns) {
      // A fatal word or the header's shutdown ends the wait early (D4): nobody will serve this request.
      if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) {
        aborted = true;
        break;
      }
      __nanosleep(256);
      ++polls;
      done = ld_acquire_sys(page + kDemandDone);
    }
    const int64_t total = static_cast<int64_t>(state[kPolls]) + polls;
    state[kPolls] = static_cast<int32_t>(total < 0x7fffffffLL ? total : 0x7fffffffLL);
    if (!reached(done, seq)) {
      ok = false;
      if (aborted) {
        reason = kLeaseReasonAborted;
      } else {
        state[kTimeouts] += 1;
        reason = kLeaseReasonTimeout;
        fatal_word = seq;
      }
    } else {
      // `done` came from an acquire load, which orders every load below after it: no fence needed.
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = ld_relaxed_sys<uint16_t>(record + kRecStatus);
      if (status != kServed) {
        state[kFailures] += 1;
        ok = false;
        reason = kLeaseReasonFailed;
        fatal_word = seq;
      }
    }
  } else if (ok && planned_count > 0) {
    // Lanes to copy and no armed request to have leased them: the post kernel arms every request with lanes in lease
    // mode, so this is a protocol error. There is no generation to name in a Terminal, and nothing was leased.
    state[kFailures] += 1;
    ok = false;
    fatal_word = 0xFFFFFFFFu;
  }
  state[kPending] = 0;

  int32_t slots[kMaxIds];
  uint32_t slot_generations[kMaxIds];
  int64_t misses = 0;
  if (ok && planned_count > 0) {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
    const uint32_t capacity = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
    for (int64_t i = 0; i < planned_count; ++i) {
      bool ready_seen = false;
      if (!lane_result_valid(
              results + i * kLeaseRowResultBytes,
              generation,
              planned[i],
              capacity,
              &slots[i],
              &slot_generations[i],
              &ready_seen)) {
        ++misses;
      }
    }
    if (misses > 0) {
      ok = false;
      reason = kLeaseReasonIdentity;
      fatal_word = 0xFFFFFFFFu;
      state[kUnservedMisses] += static_cast<int32_t>(misses);
    }
  } else if (!ok) {
    misses = planned_count;  // nothing was served for these lanes
  }

  if (ok) {
    for (int64_t i = 0; i < lanes; ++i)
      host_rows[i] = i < planned_count ? static_cast<int64_t>(slots[i]) : 0;
    for (int64_t i = 0; i < planned_count; ++i) {
      lane_ctx[4 * i + 0] = static_cast<int64_t>(generation);
      lane_ctx[4 * i + 1] = static_cast<int64_t>(slot_generations[i]);
      lane_ctx[4 * i + 2] = row;
      lane_ctx[4 * i + 3] = static_cast<int64_t>(slots[i]);
    }
    keep[0] = 1.0f;
    ram_miss[0] += misses;
    go_count[0] = static_cast<int32_t>(planned_count);  // the single commit point
    return;
  }
  for (int64_t i = 0; i < lanes; ++i)
    host_rows[i] = 0;
  // Terminal first, then the fatal word (F2): the service must be able to tell a give-up from a slow serve.
  if (seq != 0 && planned_count > 0) {
    const int64_t named = planned_count < kLeaseLanes ? planned_count : kLeaseLanes;
    publish_terminal(lease, lease_d, seq, generation, static_cast<uint32_t>((1u << named) - 1u), reason);
  }
  if (fatal_word != 0) raise_fatal(page, fatal_word);
  state[kSticky] = 1;
  ram_miss[0] += misses;
  keep[0] = 0.0f;
}

// V1 two-phase, stage 1 (D1). Polls each planned lane's RowResult and copies out the ones the service has already
// published -- the resident ("hit") lanes, which serve() grants inside its reservation hold, before read() returns.
// It MUST NOT wait on kDemandDone: waiting for the request to be served is the serialisation this task removes.
//
// The two cases a copy of the batched wait gets wrong, and they are opposite here:
//   - a lane whose ready word is NOT set is NOT a failure. The service has not published it yet, it is a miss lane,
//     and it belongs to stage 2.
//   - a lane that IS published but invalid (wrong expert, host_slot out of range, torn seqlock) is a hard failure,
//     exactly as in the batched wait.
//
// Compaction: the copy kernel takes base pointers plus a count, so this stage's lanes must be contiguous. Source
// rows and destination slots are compacted in ONE order; compacting one without the other sends a lane's bytes to
// another lane's destination, which is the failure mode compaction introduces and `ord` would not have (T9).
// `origin` carries each compacted entry back to the lane it came from, because the acknowledgement word is keyed by
// LANE in the lease block and retire_leases reads it by lane: keying it by compacted position instead would
// acknowledge the wrong lease.
//
// It writes neither `keep` nor `ram_miss`: `keep` has exactly one writer (the finalize kernel) and `ram_miss` is
// owned by stage 2, the stage that knows the true miss count once the request has been served.

struct HitWaitParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t row;
  int64_t lanes;
  int64_t* host_rows_1;
  int32_t* dst_slots_1;
  uint8_t* lease;
  int64_t lease_d;
  int32_t* go_1;
  int64_t* lane_ctx_1;
  int32_t* origin_1;
  int32_t* claimed;
  int32_t* violated;
  int64_t budget_ns;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_hit_wait_kernel(
    const __grid_constant__ HitWaitParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int32_t* __restrict__ const dst_slots = p.dst_slots;
  const int64_t row = p.row;
  const int64_t lanes = p.lanes;
  int64_t* __restrict__ const host_rows_1 = p.host_rows_1;
  int32_t* __restrict__ const dst_slots_1 = p.dst_slots_1;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  int32_t* __restrict__ const go_1 = p.go_1;
  int64_t* __restrict__ const lane_ctx_1 = p.lane_ctx_1;
  int32_t* __restrict__ const origin_1 = p.origin_1;
  int32_t* __restrict__ const claimed = p.claimed;
  int32_t* __restrict__ const violated = p.violated;
  const int64_t budget_ns = p.budget_ns;
  if (threadIdx.x != 0) return;
  device::expert_stream::lease_hit_wait_body(
      page,
      state,
      planned,
      count,
      dst_slots,
      row,
      lanes,
      host_rows_1,
      dst_slots_1,
      lease,
      lease_d,
      go_1,
      lane_ctx_1,
      origin_1,
      claimed,
      violated,
      budget_ns);
}

// Piece streaming's W1: stage 1 exactly, after resetting every word the stream kernel and the finalize kernel read
// from it (plan 5, C1). go_2 and the counter are written by S only on a commit and by its blocks' counts, so without
// this a replay could read an earlier replay's values; this is the first kernel of the chain that owns them.
struct StreamHitWaitParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t row;
  int64_t lanes;
  int64_t* host_rows_1;
  int32_t* dst_slots_1;
  uint8_t* lease;
  int64_t lease_d;
  int32_t* go_1;
  int64_t* lane_ctx_1;
  int32_t* origin_1;
  int32_t* claimed;
  int32_t* violated;
  int64_t budget_ns;
  int32_t* go_2;
  uint32_t* stream_count;
  int32_t* stream_abort;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_stream_hit_wait_kernel(
    const __grid_constant__ StreamHitWaitParams p) {
  const device::expert_stream::TestPdlEntry test_pdl_entry(2);  // test builds only (kTestPdl)
  device::expert_stream::test_pdl_trigger();  // test builds only (kTestPdlEarly): see results.md, chain PDL
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int32_t* __restrict__ const dst_slots = p.dst_slots;
  const int64_t row = p.row;
  const int64_t lanes = p.lanes;
  int64_t* __restrict__ const host_rows_1 = p.host_rows_1;
  int32_t* __restrict__ const dst_slots_1 = p.dst_slots_1;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  int32_t* __restrict__ const go_1 = p.go_1;
  int64_t* __restrict__ const lane_ctx_1 = p.lane_ctx_1;
  int32_t* __restrict__ const origin_1 = p.origin_1;
  int32_t* __restrict__ const claimed = p.claimed;
  int32_t* __restrict__ const violated = p.violated;
  const int64_t budget_ns = p.budget_ns;
  int32_t* __restrict__ const go_2 = p.go_2;
  uint32_t* __restrict__ const stream_count = p.stream_count;
  int32_t* __restrict__ const stream_abort = p.stream_abort;
  if (threadIdx.x != 0) return;
  go_2[0] = 0;
  stream_count[0] = 0u;
  stream_abort[0] = 0;
  device::expert_stream::lease_hit_wait_body(
      page,
      state,
      planned,
      count,
      dst_slots,
      row,
      lanes,
      host_rows_1,
      dst_slots_1,
      lease,
      lease_d,
      go_1,
      lane_ctx_1,
      origin_1,
      claimed,
      violated,
      budget_ns);
}

// V1 two-phase, stage 2 (D2). The rest of the request: it waits on kDemandDone as the batched wait does, because
// the missing rows are not published until read() returns, and builds its plan over the COMPLEMENT of `claimed`.
//
// Three things it deliberately does not do, each of which the batched wait does:
//   - it does not write `keep`. With two stages a stage-2 success would overwrite a keep = 0 that stage 1's
//     acknowledgement already wrote on a violation, so `keep` has exactly one writer: the finalize kernel.
//   - it does not publish a terminal and does not raise the fatal word. The mask must name only the lanes no stage
//     acknowledged, which is unknown until both acknowledgements have run; the finalize kernel publishes it and
//     raises the fatal word after it, preserving the terminal-before-fatal order.
//   - it owns `ram_miss` across the two stages, so the count is not doubled.
//
// `state[kPending]` is not cleared here even though this is the later of the two waits: the finalize kernel still
// needs the request's generation, so the clear moves there and both stages read it.
struct RestWaitParams {
  uint8_t* page;
  int32_t* state;
  const int64_t* planned;
  const int32_t* count;
  const int32_t* dst_slots;
  int64_t row;
  int64_t lanes;
  int64_t* host_rows_2;
  int32_t* dst_slots_2;
  int64_t* ram_miss;
  uint8_t* lease;
  int64_t lease_d;
  const int32_t* claimed;
  int32_t* go_2;
  int64_t* lane_ctx_2;
  int32_t* origin_2;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_rest_wait_kernel(
    const __grid_constant__ RestWaitParams p) {
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int64_t* __restrict__ const planned = p.planned;
  const int32_t* __restrict__ const count = p.count;
  const int32_t* __restrict__ const dst_slots = p.dst_slots;
  const int64_t row = p.row;
  const int64_t lanes = p.lanes;
  int64_t* __restrict__ const host_rows_2 = p.host_rows_2;
  int32_t* __restrict__ const dst_slots_2 = p.dst_slots_2;
  int64_t* __restrict__ const ram_miss = p.ram_miss;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int32_t* __restrict__ const claimed = p.claimed;
  int32_t* __restrict__ const go_2 = p.go_2;
  int64_t* __restrict__ const lane_ctx_2 = p.lane_ctx_2;
  int32_t* __restrict__ const origin_2 = p.origin_2;
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  go_2[0] = 0;  // fail closed
  for (int64_t i = 0; i < lanes; ++i)
    host_rows_2[i] = 0;
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  int64_t unclaimed = 0;
  for (int64_t i = 0; i < planned_count; ++i)
    if (claimed[i] == 0) ++unclaimed;

  bool ok = state[kSticky] == 0 && state[kReqFailed] == 0 && ld_acquire_sys(page + kFatal) == 0 &&
            ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
  uint32_t reason = ok ? 0u : kLeaseReasonAborted;
  if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
    ok = false;
    reason = kLeaseReasonCount;
  }
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;

  if (ok && seq != 0) {
    state[kWaits] += 1;
    const uint64_t deadline = load_deadline(state);
    int64_t polls = 0;
    bool aborted = false;
    uint32_t done = ld_acquire_sys(page + kDemandDone);
    while (!reached(done, seq) && static_cast<int64_t>(global_ns() - deadline) < 0) {
      if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) {
        aborted = true;
        break;
      }
      __nanosleep(256);
      ++polls;
      done = ld_acquire_sys(page + kDemandDone);
    }
    const int64_t total = static_cast<int64_t>(state[kPolls]) + polls;
    state[kPolls] = static_cast<int32_t>(total < 0x7fffffffLL ? total : 0x7fffffffLL);
    if (!reached(done, seq)) {
      ok = false;
      if (aborted) {
        reason = kLeaseReasonAborted;
      } else {
        state[kTimeouts] += 1;
        reason = kLeaseReasonTimeout;
      }
    } else {
      // `done` came from an acquire load, which orders every load below after it: no fence needed.
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = ld_relaxed_sys<uint16_t>(record + kRecStatus);
      if (status != kServed) {
        state[kFailures] += 1;
        ok = false;
        reason = kLeaseReasonFailed;
      }
    }
  } else if (ok && planned_count > 0) {
    // Lanes to copy and no armed request to have leased them: the post kernel arms every request with lanes in
    // lease mode, so this is a protocol error.
    state[kFailures] += 1;
    ok = false;
  }

  if (!ok) {
    state[kReqFailed] = 1;
    if (state[kFailReason] == 0) state[kFailReason] = static_cast<int32_t>(reason);
    ram_miss[0] += unclaimed;  // nothing was served for the lanes stage 1 did not claim
    return;
  }
  if (planned_count == 0) return;

  const uint8_t* results =
      lease + kLeaseRowResult + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLanes * kLeaseRowResultBytes;
  const uint32_t capacity = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
  int64_t n = 0;
  int64_t misses = 0;
  for (int64_t i = 0; i < planned_count; ++i) {
    if (claimed[i] != 0) continue;  // stage 1 copied and acknowledged this lane already
    int32_t host_slot = 0;
    uint32_t slot_generation = 0;
    bool ready_seen = false;
    if (!lane_result_valid(
            results + i * kLeaseRowResultBytes,
            generation,
            planned[i],
            capacity,
            &host_slot,
            &slot_generation,
            &ready_seen)) {
      ++misses;
      continue;
    }
    host_rows_2[n] = static_cast<int64_t>(host_slot);
    dst_slots_2[n] = dst_slots[i];
    origin_2[n] = static_cast<int32_t>(i);
    lane_ctx_2[4 * n + 0] = static_cast<int64_t>(generation);
    lane_ctx_2[4 * n + 1] = static_cast<int64_t>(slot_generation);
    lane_ctx_2[4 * n + 2] = row;
    lane_ctx_2[4 * n + 3] = static_cast<int64_t>(host_slot);
    ++n;
  }
  ram_miss[0] += misses;
  if (misses > 0) {
    // After demand_done every lane is published, so an unpublished or invalid one here is a protocol violation.
    state[kUnservedMisses] += static_cast<int32_t>(misses);
    state[kReqFailed] = 1;
    state[kFailReason] = static_cast<int32_t>(kLeaseReasonIdentity);
    return;  // go_2 stays 0
  }
  go_2[0] = static_cast<int32_t>(n);  // the single commit point
}

// V1 two-phase acknowledgement (D3), launched after each stage's copy kernel in the same stream. One thread per
// COMPACTED entry; `origin` maps it back to the lane whose acknowledgement word it must write, because
// retire_leases reads those words by lane. Every effect is guarded by `lane < n` with n = go_s[0], so an empty
// stage emits nothing: not an acknowledgement, not a violation, not a fatal word.
//
// It records a violation in `violated` rather than writing `keep`, which only the finalize kernel writes.
struct StageAckParams {
  uint8_t* page;
  uint8_t* lease;
  int64_t lease_d;
  const int32_t* go_count;
  const int64_t* lane_ctx;
  const int32_t* origin;
  int32_t* violated;
};

__global__ __launch_bounds__(device::expert_stream::kLeaseLanes, 1) void exl3_ram_miss_lease_stage_ack_kernel(
    const __grid_constant__ StageAckParams p) {
  const device::expert_stream::TestPdlEntry test_pdl_entry(3);  // test builds only (kTestPdl)
  device::expert_stream::test_pdl_trigger();  // test builds only (kTestPdlEarly): see results.md, chain PDL
  uint8_t* __restrict__ const page = p.page;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int32_t* __restrict__ const go_count = p.go_count;
  const int64_t* __restrict__ const lane_ctx = p.lane_ctx;
  const int32_t* __restrict__ const origin = p.origin;
  int32_t* __restrict__ const violated = p.violated;
  using namespace device::expert_stream;
  __shared__ int any_violated;
  if (threadIdx.x == 0) any_violated = 0;
  __syncthreads();
  const int64_t entry = threadIdx.x;
  const int64_t n = go_count[0];
  if (entry < n && entry < kLeaseLanes) {
    const uint64_t generation = static_cast<uint64_t>(lane_ctx[4 * entry + 0]);
    const uint32_t slot_generation = static_cast<uint32_t>(lane_ctx[4 * entry + 1]);
    const int64_t row = lane_ctx[4 * entry + 2];
    const int64_t slot = lane_ctx[4 * entry + 3];
    const int64_t lane = static_cast<int64_t>(origin[entry]);
    const uint32_t base = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes);
    const uint32_t current = ld_acquire_sys(lease + kLeaseSlotGen + 4 * (static_cast<int64_t>(base) + slot));
    const bool consumed = current == slot_generation;
    const int64_t idx = static_cast<int64_t>((static_cast<uint32_t>(generation) - 1u) % kDemandRecords);
    st_release_sys64(
        lease + lease_d + kLeaseLaneAck + (idx * kLeaseLanes + lane) * kLeaseLaneAckBytes,
        tagged_word(consumed ? kLeaseTagConsumed : kLeaseTagViolated, generation));
    if (!consumed) any_violated = 1;
  }
  // Also orders every lane's LaneAck store before thread 0's fatal release; a warp vote would not document that.
  __syncthreads();
  if (threadIdx.x == 0 && any_violated != 0) {
    violated[0] = 1;
    raise_fatal(page, static_cast<uint32_t>(lane_ctx[0]));
  }
}

// V1 two-phase, finalize (D4). Stream-ordered after every copy and every acknowledgement, and before the fused
// MoE. It is the ONLY writer of `keep` (PER_ROW_TRANSFER.md DECIDE 4).
//
// On failure it publishes a PARTIAL terminal mask: the lanes no stage acknowledged. The batched wait's
// whole-request mask would void the hit lanes stage 1 has already acknowledged; retire_leases's per-lane state
// machine keeps that from corrupting anything, but it records it only as kLeaseDoubleSignal, so the wrong mask
// would otherwise be invisible.
//
// No `lanes` field on purpose. The only bound this kernel needs is kLeaseLanes, because the array it walks is
// the per-record LaneAck table, which is kLeaseLanes wide by construction -- not a staging buffer. Passing the
// staging width here would read as the bound and be wrong.
struct FinalizeParams {
  uint8_t* page;
  int32_t* state;
  const int32_t* count;
  const int32_t* go_1;
  const int32_t* go_2;
  const int32_t* go_ce;
  const int32_t* violated;
  float* keep;
  uint8_t* lease;
  int64_t lease_d;
};

__global__ __launch_bounds__(device::expert_stream::kBlock, 1) void exl3_ram_miss_lease_finalize_kernel(
    const __grid_constant__ FinalizeParams p) {
  const device::expert_stream::TestPdlEntry test_pdl_entry(6);  // test builds only (kTestPdl)
  device::expert_stream::test_pdl_trigger();  // test builds only (kTestPdlEarly): see results.md, chain PDL
  uint8_t* __restrict__ const page = p.page;
  int32_t* __restrict__ const state = p.state;
  const int32_t* __restrict__ const count = p.count;
  const int32_t* __restrict__ const go_1 = p.go_1;
  const int32_t* __restrict__ const go_2 = p.go_2;
  const int32_t* __restrict__ const go_ce = p.go_ce;
  const int32_t* __restrict__ const violated = p.violated;
  float* __restrict__ const keep = p.keep;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  using namespace device::expert_stream;
  if (threadIdx.x != 0) return;
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
  const int64_t copied = static_cast<int64_t>(go_1[0]) + static_cast<int64_t>(go_2[0]) + static_cast<int64_t>(go_ce[0]);
  const bool served =
      state[kReqFailed] == 0 && violated[0] == 0 && copied == planned_count && ld_acquire_sys(page + kFatal) == 0;
  state[kPending] = 0;  // the last kernel of the chain, so the clear lands here rather than in either wait
  if (served) {
    keep[0] = 1.0f;
    return;
  }
  // The mask names every planned lane carrying no acknowledgement for this generation, copy-engine lanes included:
  // the service releases those on the copy's completion alone and ignores their bits. A VIOLATED acknowledgement
  // is still an acknowledgement: retire_leases released that lease already, and naming it again would be exactly
  // the double signal this partial mask exists to avoid.
  if (seq != 0 && planned_count > 0) {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* acks = lease + lease_d + kLeaseLaneAck + idx * kLeaseLanes * kLeaseLaneAckBytes;
    uint32_t mask = 0;
    const int64_t named = planned_count < kLeaseLanes ? planned_count : kLeaseLanes;
    for (int64_t lane = 0; lane < named; ++lane) {
      const uint64_t word = ld_acquire_sys64(acks + lane * kLeaseLaneAckBytes);
      const bool acknowledged = (word >> 56) != 0 && (word & ((1ull << 56) - 1)) == generation;
      if (!acknowledged) mask |= 1u << lane;
    }
    const uint32_t reason = state[kFailReason] != 0 ? static_cast<uint32_t>(state[kFailReason]) : kLeaseReasonFailed;
    publish_terminal(lease, lease_d, seq, generation, mask, reason);
  }
  // Terminal first, then the fatal word (F2): the service must be able to tell a give-up from a slow serve.
  // With no armed request there is no seq to name; a protocol error must still fail stop, as the batched wait does,
  // while an abort (shutdown, or a fatal already raised) raises nothing new.
  if (seq != 0) {
    raise_fatal(page, seq);
  } else if (state[kFailReason] != static_cast<int32_t>(kLeaseReasonAborted)) {
    raise_fatal(page, 0xFFFFFFFFu);
  }
  state[kSticky] = 1;
  keep[0] = 0.0f;
}

// LEASE_PROTOCOL.md 7.4, launched after the copy kernel in the same stream (6.4). One block, one thread per lane.
// Every effect below is guarded by `lane < n` with n = go_count[0], so n == 0 emits nothing: not an
// acknowledgement, not a keep write, not a fatal word. The kernel does not test "was the copy skipped"; the
// one word that gated the copy gates this too.
struct LeaseAckParams {
  uint8_t* page;
  uint8_t* lease;
  int64_t lease_d;
  const int32_t* go_count;
  const int64_t* lane_ctx;
  float* keep;
};

__global__ __launch_bounds__(device::expert_stream::kLeaseLanes, 1) void exl3_ram_miss_lease_ack_kernel(
    const __grid_constant__ LeaseAckParams p) {
  uint8_t* __restrict__ const page = p.page;
  uint8_t* __restrict__ const lease = p.lease;
  const int64_t lease_d = p.lease_d;
  const int32_t* __restrict__ const go_count = p.go_count;
  const int64_t* __restrict__ const lane_ctx = p.lane_ctx;
  float* __restrict__ const keep = p.keep;
  using namespace device::expert_stream;
  __shared__ int violated;
  if (threadIdx.x == 0) violated = 0;
  __syncthreads();
  const int64_t lane = threadIdx.x;
  const int64_t n = go_count[0];
  if (lane < n && lane < kLeaseLanes) {
    const uint64_t generation = static_cast<uint64_t>(lane_ctx[4 * lane + 0]);
    const uint32_t slot_generation = static_cast<uint32_t>(lane_ctx[4 * lane + 1]);
    const int64_t row = lane_ctx[4 * lane + 2];
    const int64_t slot = lane_ctx[4 * lane + 3];
    const uint32_t base = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes);
    const uint32_t current = ld_acquire_sys(lease + kLeaseSlotGen + 4 * (static_cast<int64_t>(base) + slot));
    const bool consumed = current == slot_generation;
    const int64_t idx = static_cast<int64_t>((static_cast<uint32_t>(generation) - 1u) % kDemandRecords);
    st_release_sys64(
        lease + lease_d + kLeaseLaneAck + (idx * kLeaseLanes + lane) * kLeaseLaneAckBytes,
        tagged_word(consumed ? kLeaseTagConsumed : kLeaseTagViolated, generation));
    if (!consumed) violated = 1;
  }
  __syncthreads();  // as in the stage ack: also orders the lanes' LaneAck stores before the fatal release
  if (threadIdx.x == 0 && violated != 0) {
    keep[0] = 0.0f;
    raise_fatal(page, static_cast<uint32_t>(lane_ctx[0]));
  }
}

/// \brief Checked host launchers for the lease-protocol kernels above: format-free post, wait, lease_wait,
/// lease_ack, lease_hit_wait, lease_rest_wait, lease_stage_ack, lease_finalize and lease_stream_hit_wait.
///
/// Every tensor argument is verified with `TensorMatcher` (named via `verify_named`) and every address/offset with
/// `RuntimeCheck` before the params struct is built and the kernel launched. FFI signatures are unchanged from the
/// free launchers they replace, so Python call sites do not change.
struct LeaseProtocolKernel {
#ifdef EXL3_RAM_MISS_TEST_PDL_STAMP
  // Test builds only: the chain-PDL stamp ring (lease_device.cuh) into int64 `out` [kPdlStampSlots * 4 + 1], on the
  // host, its last entry the number of stamps written; then the ring's count is reset. Synchronizes the device.
  static void pdl_stamps(tvm::ffi::TensorView out) {
    using namespace device::expert_stream;
    host::RuntimeCheck(out.size(0) == kPdlStampSlots * 4 + 1, "pdl_stamps: out holds ", kPdlStampSlots * 4 + 1, " int64");
    CHECK_CUDA(cudaDeviceSynchronize()) << "pdl_stamps";
    auto* dst = static_cast<int64_t*>(out.data_ptr());
    CHECK_CUDA(cudaMemcpyFromSymbol(dst, g_pdl_stamp, sizeof(g_pdl_stamp))) << "pdl_stamps";
    uint32_t seq = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&seq, g_pdl_seq, sizeof(seq))) << "pdl_stamps";
    dst[kPdlStampSlots * 4] = seq;
    seq = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(g_pdl_seq, &seq, sizeof(seq))) << "pdl_stamps";
  }
#endif
  static void post(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto L_ = SymbolicSize{"layers"};
    auto E_ = SymbolicSize{"experts"};
    auto P_ = SymbolicSize{"planned"};
    auto R_ = SymbolicSize{"routes"};

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "slot_map", TensorMatcher({L_, E_}).with_dtype<int32_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), slot_map);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "routes", TensorMatcher({R_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), routes);
    expert_stream::verify_named(
        "last_routes", TensorMatcher({L_, kMaxIds}).with_dtype<int32_t>().with_device<kDLCUDA>(device), last_routes);
    expert_stream::verify_named(
        "hot_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), hot_slots);
    if (hot_address != 0) {
      RuntimeCheck(hot_slots.size(0) >= hot_capacity, "hot_slots: size must be at least hot_capacity");
    }
    expert_stream::verify_named(
        "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = PostParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .slot_map = static_cast<const int32_t*>(slot_map.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .routes = static_cast<const int64_t*>(routes.data_ptr()),
        .route_count = routes.size(0),
        .row = row,
        .experts = slot_map.size(1),
        .advise = advise,
        .last_routes = static_cast<int32_t*>(last_routes.data_ptr()),
        .next_row = next_row,
        .lease = reinterpret_cast<uint8_t*>(lease_address),  // zero: no lease block, today's protocol
        .lease_d = lease_d,
        .timeout_ns = timeout_ns,
        .hot_page = reinterpret_cast<uint8_t*>(hot_address),
        .hot_stride = hot_stride,
        .hot_slots = static_cast<const int64_t*>(hot_slots.data_ptr()),
        .hot_capacity = hot_capacity,
        .dst_slots = dst_slots.size(0) > 0 ? static_cast<const int32_t*>(dst_slots.data_ptr())
                                           : nullptr,  // empty: no plan slots
        .dst_count = dst_slots.size(0),
        .copy_engine = copy_engine,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(device::expert_stream::kTestPdl)(exl3_ram_miss_post_kernel, params);
  }

  static void wait(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto L_ = SymbolicSize{"layers"};
    auto E_ = SymbolicSize{"experts"};
    auto P_ = SymbolicSize{"planned"};
    const int64_t lanes = host_rows.size(0);

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "slot_map", TensorMatcher({L_, E_}).with_dtype<int32_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), slot_map);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    RuntimeCheck(P_.unwrap() >= lanes, "planned: must have at least as many lanes as host_rows");
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "host_rows", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), host_rows);
    expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);
    expert_stream::verify_named(
        "ram_miss", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), ram_miss);

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = WaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .slot_map = static_cast<const int32_t*>(slot_map.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .row = row,
        .experts = slot_map.size(1),
        .lanes = lanes,
        .host_rows = static_cast<int64_t*>(host_rows.data_ptr()),
        .keep = static_cast<float*>(keep.data_ptr()),
        .ram_miss = static_cast<int64_t*>(ram_miss.data_ptr()),
        .timeout_ns = timeout_ns,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_wait_kernel, params);
  }

  static void lease_wait(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    const int64_t lanes = host_rows.size(0);

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    RuntimeCheck(P_.unwrap() >= lanes, "planned: must have at least as many lanes as host_rows");
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "host_rows", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), host_rows);
    expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);
    expert_stream::verify_named(
        "ram_miss", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), ram_miss);
    expert_stream::verify_named(
        "go_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_count);
    expert_stream::verify_named(
        "lane_ctx", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = LeaseWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .row = row,
        .lanes = lanes,
        .host_rows = static_cast<int64_t*>(host_rows.data_ptr()),
        .keep = static_cast<float*>(keep.data_ptr()),
        .ram_miss = static_cast<int64_t*>(ram_miss.data_ptr()),
        .timeout_ns = timeout_ns,
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .go_count = static_cast<int32_t*>(go_count.data_ptr()),
        .lane_ctx = static_cast<int64_t*>(lane_ctx.data_ptr()),
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_lease_wait_kernel, params);
  }

  static void lease_ack(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      int64_t lease_address,
      int64_t lease_d,
      tvm::ffi::TensorView go_count,
      tvm::ffi::TensorView lane_ctx,
      tvm::ffi::TensorView keep) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "go_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_count);
    expert_stream::verify_named(
        "lane_ctx", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx);
    expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = LeaseAckParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .go_count = static_cast<const int32_t*>(go_count.data_ptr()),
        .lane_ctx = static_cast<const int64_t*>(lane_ctx.data_ptr()),
        .keep = static_cast<float*>(keep.data_ptr()),
    };
    LaunchKernel(1, static_cast<int>(device::expert_stream::kLeaseLanes), stream)(
        exl3_ram_miss_lease_ack_kernel, params);
  }

  static void lease_hit_wait(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    // The lane bound must cover EVERY lane-indexed array the kernel reads, not just its own staging buffer:
    // dst_slots is the plan's, of length capacity, while host_rows_1 is LANES. Bounding by the staging buffer alone
    // lets a device-side count above capacity, up to LANES, read dst_slots out of bounds.
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
        "lane_ctx_1", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx_1);
    expert_stream::verify_named(
        "origin_1", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), origin_1);
    expert_stream::verify_named(
        "claimed", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), claimed);
    expert_stream::verify_named(
        "violated", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), violated);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = HitWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .row = row,
        .lanes = lanes,
        .host_rows_1 = static_cast<int64_t*>(host_rows_1.data_ptr()),
        .dst_slots_1 = static_cast<int32_t*>(dst_slots_1.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .go_1 = static_cast<int32_t*>(go_1.data_ptr()),
        .lane_ctx_1 = static_cast<int64_t*>(lane_ctx_1.data_ptr()),
        .origin_1 = static_cast<int32_t*>(origin_1.data_ptr()),
        .claimed = static_cast<int32_t*>(claimed.data_ptr()),
        .violated = static_cast<int32_t*>(violated.data_ptr()),
        .budget_ns = budget_ns,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_lease_hit_wait_kernel, params);
  }

  static void lease_stream_hit_wait(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
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
        "lane_ctx_1", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx_1);
    expert_stream::verify_named(
        "origin_1", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), origin_1);
    expert_stream::verify_named(
        "claimed", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), claimed);
    expert_stream::verify_named(
        "violated", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), violated);
    expert_stream::verify_named("go_2", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_2);
    expert_stream::verify_named(
        "stream_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), stream_count);
    expert_stream::verify_named(
        "stream_abort", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), stream_abort);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = StreamHitWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .row = row,
        .lanes = lanes,
        .host_rows_1 = static_cast<int64_t*>(host_rows_1.data_ptr()),
        .dst_slots_1 = static_cast<int32_t*>(dst_slots_1.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .go_1 = static_cast<int32_t*>(go_1.data_ptr()),
        .lane_ctx_1 = static_cast<int64_t*>(lane_ctx_1.data_ptr()),
        .origin_1 = static_cast<int32_t*>(origin_1.data_ptr()),
        .claimed = static_cast<int32_t*>(claimed.data_ptr()),
        .violated = static_cast<int32_t*>(violated.data_ptr()),
        .budget_ns = budget_ns,
        .go_2 = static_cast<int32_t*>(go_2.data_ptr()),
        .stream_count = static_cast<uint32_t*>(stream_count.data_ptr()),
        .stream_abort = static_cast<int32_t*>(stream_abort.data_ptr()),
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(device::expert_stream::kTestPdl)(exl3_ram_miss_lease_stream_hit_wait_kernel, params);
  }

  static void lease_rest_wait(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();
    auto P_ = SymbolicSize{"planned"};
    const int64_t lanes = std::min<int64_t>(host_rows_2.size(0), dst_slots.size(0));

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "planned", TensorMatcher({P_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    RuntimeCheck(P_.unwrap() >= lanes, "planned: must have at least as many lanes as host_rows_2/dst_slots");
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "dst_slots", TensorMatcher({-1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots);
    expert_stream::verify_named(
        "host_rows_2", TensorMatcher({kLeaseLanes}).with_dtype<int64_t>().with_device<kDLCUDA>(device), host_rows_2);
    expert_stream::verify_named(
        "dst_slots_2", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), dst_slots_2);
    expert_stream::verify_named(
        "ram_miss", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), ram_miss);
    expert_stream::verify_named(
        "claimed", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), claimed);
    expert_stream::verify_named("go_2", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_2);
    expert_stream::verify_named(
        "lane_ctx_2", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx_2);
    expert_stream::verify_named(
        "origin_2", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), origin_2);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = RestWaitParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .dst_slots = static_cast<const int32_t*>(dst_slots.data_ptr()),
        .row = row,
        .lanes = lanes,
        .host_rows_2 = static_cast<int64_t*>(host_rows_2.data_ptr()),
        .dst_slots_2 = static_cast<int32_t*>(dst_slots_2.data_ptr()),
        .ram_miss = static_cast<int64_t*>(ram_miss.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .claimed = static_cast<const int32_t*>(claimed.data_ptr()),
        .go_2 = static_cast<int32_t*>(go_2.data_ptr()),
        .lane_ctx_2 = static_cast<int64_t*>(lane_ctx_2.data_ptr()),
        .origin_2 = static_cast<int32_t*>(origin_2.data_ptr()),
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream)(exl3_ram_miss_lease_rest_wait_kernel, params);
  }

  static void lease_stage_ack(
      tvm::ffi::TensorView page,
      tvm::ffi::TensorView state,
      int64_t lease_address,
      int64_t lease_d,
      tvm::ffi::TensorView go_count,
      tvm::ffi::TensorView lane_ctx,
      tvm::ffi::TensorView origin,
      tvm::ffi::TensorView violated) {
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named(
        "go_count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_count);
    expert_stream::verify_named(
        "lane_ctx", TensorMatcher({kLeaseLanes, 4}).with_dtype<int64_t>().with_device<kDLCUDA>(device), lane_ctx);
    expert_stream::verify_named(
        "origin", TensorMatcher({kLeaseLanes}).with_dtype<int32_t>().with_device<kDLCUDA>(device), origin);
    expert_stream::verify_named(
        "violated", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), violated);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = StageAckParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
        .go_count = static_cast<const int32_t*>(go_count.data_ptr()),
        .lane_ctx = static_cast<const int64_t*>(lane_ctx.data_ptr()),
        .origin = static_cast<const int32_t*>(origin.data_ptr()),
        .violated = static_cast<int32_t*>(violated.data_ptr()),
    };
    LaunchKernel(1, static_cast<int>(device::expert_stream::kLeaseLanes), stream).enable_pdl(device::expert_stream::kTestPdl)(
        exl3_ram_miss_lease_stage_ack_kernel, params);
  }

  static void lease_finalize(
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
    using namespace host;
    using namespace expert_stream::wire;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto on_host = SymbolicDevice{};
    on_host.set_options<kDLCPU, kDLCUDAHost>();

    expert_stream::verify_named(
        "page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(on_host), page);
    expert_stream::verify_named(
        "state",
        TensorMatcher({device::expert_stream::kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        state);
    expert_stream::verify_named("count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named("go_1", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_1);
    expert_stream::verify_named("go_2", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_2);
    expert_stream::verify_named("go_ce", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), go_ce);
    expert_stream::verify_named(
        "violated", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), violated);
    expert_stream::verify_named("keep", TensorMatcher({1}).with_dtype<float>().with_device<kDLCUDA>(device), keep);

    RuntimeCheck(
        lease_address == 0 || lease_address % kLeaseBlockAlign == 0,
        "lease_address: must be 0 or a multiple of kLeaseBlockAlign");
    if (lease_address != 0) {
      RuntimeCheck(lease_d % kLeaseBlockAlign == 0, "lease_d: must be a multiple of kLeaseBlockAlign");
    }

    const auto stream = LaunchKernel::resolve_device(state.device());
    const auto params = FinalizeParams{
        .page = static_cast<uint8_t*>(page.data_ptr()),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .go_1 = static_cast<const int32_t*>(go_1.data_ptr()),
        .go_2 = static_cast<const int32_t*>(go_2.data_ptr()),
        .go_ce = static_cast<const int32_t*>(go_ce.data_ptr()),
        .violated = static_cast<const int32_t*>(violated.data_ptr()),
        .keep = static_cast<float*>(keep.data_ptr()),
        .lease = reinterpret_cast<uint8_t*>(lease_address),
        .lease_d = lease_d,
    };
    LaunchKernel(1, device::expert_stream::kBlock, stream).enable_pdl(device::expert_stream::kTestPdl)(exl3_ram_miss_lease_finalize_kernel, params);
  }
};

}  // namespace sglang
