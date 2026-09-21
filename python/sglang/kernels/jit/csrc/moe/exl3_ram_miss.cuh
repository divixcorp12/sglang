// Device side of the option C RAM-miss service (DSV41 Phase 3b plan, D10-D15, D22).
//
// post: one block; thread 0 builds the layer's request (need = planned VRAM misses
// whose host-mapped slot map entry is -1; protect = every routed expert), writes it
// into the page's demand ring with volatile stores, fences system-wide and
// release-stores demand_head. The record is posted for every MoE layer (the thread
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

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace exl3_ram_miss_device {

constexpr int kBlock = 32;
// The page layout mirrors exl3_ram_miss_host.cpp and the Python constants;
// test_exl3_ram_miss_device_args checks all three agree.
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kRecordBytes = 128;
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kAdviseRing = kDemandRing + kDemandRecords * kRecordBytes;
constexpr uint32_t kAdviseRecords = 64;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;
constexpr int64_t kRecRow = 4;
constexpr int64_t kRecNeedCount = 6;
constexpr int64_t kRecProtectCount = 8;
constexpr int64_t kRecStatus = 10;
constexpr int64_t kRecAfter = 12;
constexpr int64_t kRecNeed = 16;
constexpr int64_t kRecProtect = 48;
constexpr int64_t kRecArmed = 80;
constexpr int64_t kRecLanes = 84;
constexpr uint16_t kServed = 1;

// The lease block beside the request page (LEASE_PROTOCOL.md section 4). Its layout is written here, in
// exl3_ram_miss_host.cpp and in ops/moe/exl3_lease_block.py; test_exl3_lease_block checks they agree. The
// publication word (tag << 56 | generation) is built in code: the layout test parses these lines with + - * only.
constexpr int64_t kLeaseRing = 16;   // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;   // == kMaxIds
constexpr int64_t kLeaseHeaderRing = 8;
constexpr int64_t kLeaseHeaderLanes = 12;
constexpr int64_t kLeaseHeaderShutdown = 20;
constexpr int64_t kLeaseHeaderSlotGenOffset = 32;
constexpr int64_t kLeaseHeaderDOffset = 36;
constexpr int64_t kLeaseRowTable = 128;
constexpr int64_t kLeaseRowResult = 4096;
constexpr int64_t kLeaseRowResultBytes = 32;
constexpr int64_t kLeaseRrReady = 0;
constexpr int64_t kLeaseRrSlotGeneration = 8;
constexpr int64_t kLeaseRrHostSlot = 12;
constexpr int64_t kLeaseRrExpert = 16;
constexpr int64_t kLeaseSlotGen = kLeaseRowResult + kLeaseRing * kLeaseLanes * kLeaseRowResultBytes;
constexpr int64_t kLeaseLaneRequest = 0;
constexpr int64_t kLeaseLaneRequestBytes = 64;
constexpr int64_t kLeaseLrGen = 0;
constexpr int64_t kLeaseLrCount = 8;
constexpr int64_t kLeaseLrRow = 12;
constexpr int64_t kLeaseLrExpert = 16;
constexpr int64_t kLeaseLaneAck = kLeaseLaneRequest + kLeaseRing * kLeaseLaneRequestBytes;
constexpr int64_t kLeaseLaneAckBytes = 8;
constexpr int64_t kLeaseTerminal = kLeaseLaneAck + kLeaseRing * kLeaseLanes * kLeaseLaneAckBytes;
constexpr int64_t kLeaseTerminalBytes = 16;
constexpr int64_t kLeaseTermSkippedMask = 0;
constexpr int64_t kLeaseTermReason = 4;
constexpr int64_t kLeaseTermGen = 8;
constexpr int64_t kLeaseRowTableBytes = 8;
// Tags of the byte above the 56-bit request generation, and the reasons a Terminal record carries (section 4.3, 13).
constexpr uint64_t kLeaseTagDemand = 1;
constexpr uint64_t kLeaseTagReady = 1;
constexpr uint64_t kLeaseTagConsumed = 1;
constexpr uint64_t kLeaseTagViolated = 2;
constexpr uint64_t kLeaseTagTerminal = 1;
constexpr uint32_t kLeaseReasonTimeout = 1;
constexpr uint32_t kLeaseReasonAborted = 2;
constexpr uint32_t kLeaseReasonFailed = 3;
constexpr uint32_t kLeaseReasonIdentity = 4;
constexpr uint32_t kLeaseReasonCount = 5;

constexpr int kPosted = 0;
constexpr int kPending = 1;
constexpr int kTimeouts = 2;
constexpr int kFailures = 3;
constexpr int kWaits = 4;
constexpr int kPolls = 5;
constexpr int kSticky = 6;
constexpr int kAdvised = 7;
constexpr int kUnservedMisses = 8;
constexpr int kEpoch = 9;         // times seq32 wrapped; the device owns it (LEASE_PROTOCOL.md 11.3)
constexpr int kPendingEpoch = 10;  // the epoch of the request kPending names

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint8_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys64(const uint8_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys64(uint8_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t tagged_word(uint64_t tag, uint64_t generation) {
  return (tag << 56) | generation;
}

__device__ __forceinline__ int32_t ld_volatile(const int32_t* address) {
  return *reinterpret_cast<const volatile int32_t*>(address);
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ __forceinline__ bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

__device__ __forceinline__ bool listed(const int32_t* ids, int count, int32_t id) {
  for (int i = 0; i < count; ++i) {
    if (ids[i] == id) return true;
  }
  return false;
}

__device__ __forceinline__ void write_record(
    uint8_t* record, uint32_t seq, int64_t row, const int32_t* need, int need_count, const int32_t* protect,
    int protect_count, uint32_t after, uint32_t armed, uint32_t lanes) {
  volatile uint32_t* words = reinterpret_cast<volatile uint32_t*>(record);
  volatile uint16_t* halves = reinterpret_cast<volatile uint16_t*>(record);
  // Seqlock writer: invalidate seq before touching the payload, so a lapped record that
  // is half rewritten never passes the thread's read_record seq re-check.
  words[kRecSeq / 4] = 0u;
  __threadfence_system();
  halves[kRecRow / 2] = static_cast<uint16_t>(row);
  halves[kRecNeedCount / 2] = static_cast<uint16_t>(need_count);
  halves[kRecProtectCount / 2] = static_cast<uint16_t>(protect_count);
  halves[kRecStatus / 2] = 0;
  words[kRecAfter / 4] = after;
  words[kRecArmed / 4] = armed;
  words[kRecLanes / 4] = lanes;
  volatile int32_t* need_out = reinterpret_cast<volatile int32_t*>(record + kRecNeed);
  volatile int32_t* protect_out = reinterpret_cast<volatile int32_t*>(record + kRecProtect);
  for (int i = 0; i < kMaxIds; ++i) {
    need_out[i] = i < need_count ? need[i] : -1;
    protect_out[i] = i < protect_count ? protect[i] : -1;
  }
  // The seqlock order the thread's read_record relies on: seq=0, fence, payload, fence, seq last.
  __threadfence_system();
  words[kRecSeq / 4] = seq;
}

__device__ __forceinline__ void raise_fatal(uint8_t* page, uint32_t seq) {
  if (ld_acquire_sys(page + kFatal) == 0) st_release_sys(page + kFatal, seq);
}

}  // namespace exl3_ram_miss_device

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_post_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int64_t* __restrict__ routes,
    int64_t route_count,
    int64_t row,
    int64_t experts,
    int64_t advise,
    int32_t* __restrict__ last_routes,
    int64_t next_row,
    uint8_t* __restrict__ lease,
    int64_t lease_d) {
  using namespace exl3_ram_miss_device;
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
  const int64_t planned_count = min(max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0)), static_cast<int64_t>(kMaxIds));
  for (int64_t i = 0; i < planned_count; ++i) {
    const int32_t expert = static_cast<int32_t>(planned[i]);
    if (expert >= 0 && expert < experts && ld_volatile(map_row + expert) < 0 && !listed(need, need_count, expert)) {
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
    state[kEpoch] += 1;  // seq32 wrapped (LEASE_PROTOCOL.md 11.3); the wait and ack kernels read it back as kPendingEpoch
  }
  state[kPosted] = static_cast<int32_t>(seq);
  uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
  // Lease mode arms every request that has planned lanes, not only those with a miss: the service leases RAM hits too
  // (7.2), and it reads a request's lanes only when the record is armed. An unarmed request with lanes would be copied
  // from slots nobody leased.
  const bool armed = need_count > 0 || advise != 0 || (lease != nullptr && planned_count > 0);
  const uint32_t lanes = static_cast<uint32_t>(max(count[0], 0));  // the plan's lanes, unclamped
  if (lease != nullptr) {
    // LaneRequest (6.3): the seqlock shape of write_record, before the demand record and demand_head are published.
    const uint64_t generation = (static_cast<uint64_t>(static_cast<uint32_t>(state[kEpoch])) << 32) | seq;
    uint8_t* request = lease + lease_d + kLeaseLaneRequest + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLaneRequestBytes;
    *reinterpret_cast<volatile uint64_t*>(request + kLeaseLrGen) = 0ull;
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(request + kLeaseLrCount) = static_cast<uint32_t>(planned_count);
    *reinterpret_cast<volatile uint32_t*>(request + kLeaseLrRow) = static_cast<uint32_t>(row);
    volatile int32_t* lane_experts = reinterpret_cast<volatile int32_t*>(request + kLeaseLrExpert);
    for (int i = 0; i < kMaxIds; ++i) lane_experts[i] = i < planned_count ? static_cast<int32_t>(planned[i]) : -1;
    __threadfence_system();
    st_release_sys64(request + kLeaseLrGen, tagged_word(kLeaseTagDemand, generation));
  }
  write_record(record, seq, row, need, need_count, protect, protect_count, 0, armed ? 1u : 0u, lanes);
  __threadfence_system();
  st_release_sys(page + kDemandHead, seq);
  state[kPending] = armed ? static_cast<int32_t>(seq) : 0;
  state[kPendingEpoch] = state[kEpoch];
  if (advise == 0) return;
  for (int i = 0; i < kMaxIds; ++i) last_routes[row * kMaxIds + i] = i < protect_count ? protect[i] : -1;
  if (next_row < 0) return;
  const int32_t* next_map = slot_map + next_row * experts;
  int32_t ahead[kMaxIds];
  int ahead_count = 0;
  for (int i = 0; i < kMaxIds; ++i) {
    const int32_t expert = last_routes[next_row * kMaxIds + i];
    if (expert >= 0 && expert < experts && ld_volatile(next_map + expert) < 0) ahead[ahead_count++] = expert;
  }
  if (ahead_count == 0) return;
  uint32_t advice = static_cast<uint32_t>(state[kAdvised]) + 1u;
  if (advice == 0) advice = 1;
  state[kAdvised] = static_cast<int32_t>(advice);
  uint8_t* advice_record = page + kAdviseRing + static_cast<int64_t>((advice - 1u) % kAdviseRecords) * kRecordBytes;
  write_record(advice_record, advice, next_row, ahead, ahead_count, ahead, ahead_count, seq, 1u, ahead_count);
  __threadfence_system();
  st_release_sys(page + kAdviseHead, advice);
}

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ slot_map,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    int64_t row,
    int64_t experts,
    int64_t lanes,
    int64_t* __restrict__ host_rows,
    float* __restrict__ keep,
    int64_t* __restrict__ ram_miss,
    int64_t timeout_ns) {
  using namespace exl3_ram_miss_device;
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
      __threadfence_system();
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
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
      slot = expert >= 0 && expert < experts ? ld_volatile(map_row + expert) : -1;
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

namespace exl3_ram_miss_device {

// A Terminal record (LEASE_PROTOCOL.md 13): the mask and the reason first, the tagged generation word last with a
// release store, so a reader that acquires the word sees the mask. Named lanes never start a copy afterwards: the
// caller has already left go_count at zero.
__device__ __forceinline__ void publish_terminal(
    uint8_t* lease, int64_t lease_d, uint32_t seq, uint64_t generation, uint32_t skipped_mask, uint32_t reason) {
  uint8_t* terminal = lease + lease_d + kLeaseTerminal + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseTerminalBytes;
  *reinterpret_cast<volatile uint32_t*>(terminal + kLeaseTermSkippedMask) = skipped_mask;
  *reinterpret_cast<volatile uint32_t*>(terminal + kLeaseTermReason) = reason;
  st_release_sys64(terminal + kLeaseTermGen, tagged_word(kLeaseTagTerminal, generation));
}

}  // namespace exl3_ram_miss_device

// LEASE_PROTOCOL.md 7.3. Replaces the translate half of exl3_ram_miss_wait_kernel for lease mode: host_rows comes
// from the lanes' RowResults, never from the slot map. `go_count[0]` is zero on entry and is written exactly once,
// as the last store of a commit; every other exit leaves it zero, so the copy kernel that reads it as its active
// count copies nothing and the ack kernel that reads it acknowledges nothing.
__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    int64_t row,
    int64_t lanes,
    int64_t* __restrict__ host_rows,
    float* __restrict__ keep,
    int64_t* __restrict__ ram_miss,
    int64_t timeout_ns,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    int32_t* __restrict__ go_count,
    int64_t* __restrict__ lane_ctx) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  go_count[0] = 0;  // fail closed
  bool ok = state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0 && ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
  uint32_t reason = ok ? 0u : kLeaseReasonAborted;
  uint32_t fatal_word = 0;  // what to raise, when this wait is the one that fails the page
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation = seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
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
      __threadfence_system();
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
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
    const uint32_t capacity = *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
    const uint64_t generation_mask = (1ull << 56) - 1;
    for (int64_t i = 0; i < planned_count; ++i) {
      const uint8_t* result = results + i * kLeaseRowResultBytes;
      const uint64_t ready = ld_acquire_sys64(result + kLeaseRrReady);
      const uint32_t slot_generation = *reinterpret_cast<const volatile uint32_t*>(result + kLeaseRrSlotGeneration);
      const int32_t host_slot = *reinterpret_cast<const volatile int32_t*>(result + kLeaseRrHostSlot);
      const int32_t expert = *reinterpret_cast<const volatile int32_t*>(result + kLeaseRrExpert);
      // The belt of 11.4: a ready word that changed while the payload was read is a protocol violation.
      const uint64_t again = ld_acquire_sys64(result + kLeaseRrReady);
      const bool valid = (ready & generation_mask) == generation && (ready >> 56) == kLeaseTagReady && again == ready &&
                         static_cast<int64_t>(expert) == planned[i] && host_slot >= 0 &&
                         static_cast<uint32_t>(host_slot) < capacity;  // also bounds the ack kernel's SlotGen read
      if (valid) {
        slots[i] = host_slot;
        slot_generations[i] = slot_generation;
      } else {
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
    for (int64_t i = 0; i < lanes; ++i) host_rows[i] = i < planned_count ? static_cast<int64_t>(slots[i]) : 0;
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
  for (int64_t i = 0; i < lanes; ++i) host_rows[i] = 0;
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

// LEASE_PROTOCOL.md 7.4, launched after the copy kernel in the same stream (6.4). One block, one thread per lane.
// Every effect below is guarded by `lane < n` with n = go_count[0], so n == 0 emits nothing: not an
// acknowledgement, not a keep write, not a fatal word. The kernel does not test "was the copy skipped"; the
// one word that gated the copy gates this too.
__global__ __launch_bounds__(exl3_ram_miss_device::kLeaseLanes, 1) void exl3_ram_miss_lease_ack_kernel(
    uint8_t* __restrict__ page,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    const int32_t* __restrict__ go_count,
    const int64_t* __restrict__ lane_ctx,
    float* __restrict__ keep) {
  using namespace exl3_ram_miss_device;
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
    const uint32_t base = *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes);
    const uint32_t current = ld_acquire_sys(lease + kLeaseSlotGen + 4 * (static_cast<int64_t>(base) + slot));
    const bool consumed = current == slot_generation;
    const int64_t idx = static_cast<int64_t>((static_cast<uint32_t>(generation) - 1u) % kDemandRecords);
    st_release_sys64(
        lease + lease_d + kLeaseLaneAck + (idx * kLeaseLanes + lane) * kLeaseLaneAckBytes,
        tagged_word(consumed ? kLeaseTagConsumed : kLeaseTagViolated, generation));
    if (!consumed) violated = 1;
  }
  __syncthreads();
  if (threadIdx.x == 0 && violated != 0) {
    keep[0] = 0.0f;
    raise_fatal(page, static_cast<uint32_t>(lane_ctx[0]));
  }
}

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
    int64_t lease_d) {
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
      lease_d);
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

}  // namespace sglang
