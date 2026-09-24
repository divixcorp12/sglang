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

#include <algorithm>
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
constexpr int64_t kHotHeaderBytes = 8;
constexpr int64_t kHotAlignment = 64;
constexpr uint32_t kHotRecords = kDemandRecords;
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
constexpr int64_t kLeaseHeaderPieceOffset = 40;
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
// StreamProbe[kLeaseRing], device-written: tagged(1, generation) once the stream kernel has copied a piece of that
// request. The only piece-streaming progress the host can see; `state` lives in device memory.
constexpr int64_t kLeaseStreamProbe = kLeaseTerminal + kLeaseRing * kLeaseTerminalBytes;
constexpr int64_t kLeaseStreamProbeBytes = 8;
constexpr int64_t kLeaseRowTableBytes = 8;
// Area P, service-written, at kLeaseHeaderPieceOffset: PieceMask[kLeaseRing][kLeaseLanes], a per-lane
// generation-tagged 8-bit readiness bitmask (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment). Each word
// gets its own 128 B line, so the device's per-lane poll never shares a line with a lane it did not ask for.
// The stream kernel reads it (ld.acquire.sys) and the service's reader owner sets its bits.
constexpr int64_t kLeasePieceMaskLineBytes = 128;
constexpr int64_t kLeasePieceMaskBytes = 8;  // one uint64 per word
constexpr int64_t kLeaseAreaPieceMaskBytes = kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
// Tags of the byte above the 56-bit request generation, and the reasons a Terminal record carries (section 4.3, 13).
constexpr uint64_t kLeaseTagDemand = 1;
constexpr uint64_t kLeaseTagReady = 1;
constexpr uint64_t kLeaseTagLoading = 2;  // RowResult.ready: leased, still loading (piece-streaming plan; task 1)
constexpr uint64_t kLeaseTagConsumed = 1;
constexpr uint64_t kLeaseTagViolated = 2;
constexpr uint64_t kLeaseTagTerminal = 1;
constexpr uint64_t kLeaseTagStreamed = 1;  // StreamProbe
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
// D5. One absolute deadline per request, written once by the post kernel and compared against by both stages, so a
// two-stage request cannot run to 2 x SGLANG_DSV41_RAM_MISS_TIMEOUT_MS. `state` is int32[], so it takes two words.
constexpr int kDeadlineLo = 11;
constexpr int kDeadlineHi = 12;
// D6. An earlier stage of THIS request failed. kSticky cannot serve here: it is a process-lifetime fail-stop latch
// with no clear site, so using it would refuse every later request in the process.
constexpr int kReqFailed = 13;
constexpr int kFailReason = 14;  // the kLeaseReason* the failing stage recorded, for the terminal F publishes
constexpr int kStreamPieces = 15;  // piece slices the stream kernel's block 0 copied, cumulative
constexpr int kStreamPolls = 16;   // stream-kernel leader passes, summed over its blocks, cumulative
constexpr int kStateWords = 17;

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

__device__ __forceinline__ void store_deadline(int32_t* state, uint64_t deadline) {
  state[kDeadlineLo] = static_cast<int32_t>(static_cast<uint32_t>(deadline & 0xFFFFFFFFull));
  state[kDeadlineHi] = static_cast<int32_t>(static_cast<uint32_t>(deadline >> 32));
}

__device__ __forceinline__ uint64_t load_deadline(const int32_t* state) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineHi])) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineLo]));
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

// The per-lane RowResult validity test (LEASE_PROTOCOL.md 11.4), shared by the batched wait and both V1 stages so
// that exactly one copy of it exists. `ready_seen` reports whether the service has published this lane at all,
// which two-phase must tell apart from an invalid publish: an unpublished lane is simply the other stage's work,
// while a published-but-invalid one is a protocol violation. A caller that conflates them fails every mixed
// request, or turns a genuine identity violation into a silent retry.
// `accept_loading` also accepts tag LOADING (piece streaming, LEASE_PROTOCOL.md E1 amendment) under the same
// contract, and `loading` then reports which tag it was. Only the stream kernel passes it: a tag-2 lane's bytes are
// final piece by piece, so every other caller must go on seeing it as unpublished.
__device__ __forceinline__ bool lane_result_valid(
    const uint8_t* result,
    uint64_t generation,
    int64_t expected_expert,
    uint32_t capacity,
    int32_t* out_slot,
    uint32_t* out_slot_generation,
    bool* ready_seen,
    bool accept_loading = false,
    bool* loading = nullptr) {
  const uint64_t ready = ld_acquire_sys64(result + kLeaseRrReady);
  const uint32_t slot_generation = *reinterpret_cast<const volatile uint32_t*>(result + kLeaseRrSlotGeneration);
  const int32_t host_slot = *reinterpret_cast<const volatile int32_t*>(result + kLeaseRrHostSlot);
  const int32_t expert = *reinterpret_cast<const volatile int32_t*>(result + kLeaseRrExpert);
  // The belt of 11.4: a ready word that changed while the payload was read is a protocol violation.
  const uint64_t again = ld_acquire_sys64(result + kLeaseRrReady);
  const uint64_t generation_mask = (1ull << 56) - 1;
  const uint64_t tag = ready >> 56;
  if (loading != nullptr) *loading = tag == kLeaseTagLoading;
  *ready_seen = (tag == kLeaseTagReady || (accept_loading && tag == kLeaseTagLoading)) &&
                (ready & generation_mask) == generation;
  const bool valid = *ready_seen && again == ready && static_cast<int64_t>(expert) == expected_expert &&
                     host_slot >= 0 && static_cast<uint32_t>(host_slot) < capacity;  // also bounds the ack SlotGen read
  if (valid) {
    *out_slot = host_slot;
    *out_slot_generation = slot_generation;
  }
  return valid;
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
    int64_t lease_d,
    int64_t timeout_ns,
    uint8_t* __restrict__ hot_page,
    int64_t hot_stride,
    const int64_t* __restrict__ hot_slots,
    int64_t hot_capacity) {
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
  if (hot_page != nullptr) {
    uint8_t* hot = hot_page + static_cast<int64_t>((seq - 1u) % kHotRecords) * hot_stride;
    *reinterpret_cast<volatile uint32_t*>(hot) = 0;
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(hot + 4) = static_cast<uint32_t>(experts);
    volatile uint8_t* bits = reinterpret_cast<volatile uint8_t*>(hot + kHotHeaderBytes);
    for (int64_t byte = 0; byte < (experts + 7) / 8; ++byte) {
      uint8_t mask = 0;
      for (int64_t slot = 0; slot < hot_capacity; ++slot) {
        const int64_t expert = hot_slots[slot];
        if (expert >= byte * 8 && expert < byte * 8 + 8) mask |= static_cast<uint8_t>(1u << (expert - byte * 8));
      }
      bits[byte] = mask;
    }
    __threadfence_system();
    st_release_sys(hot, seq);
  }
  // Lease mode arms every request that has planned lanes, not only those with a miss (LEASE_PROTOCOL.md section 15:
  // the all-hit handshake is where the lease is granted, so an unarmed record is reachable only for count == 0). The
  // service leases RAM hits too (7.2) and reads a request's lanes only when the record is armed; an unarmed request
  // with lanes would be copied from slots nobody leased. The extra service round trip per layer is unmeasured (OPEN 11).
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
  // D5. The one deadline both stages compare against, absolute rather than per-stage: two stages each timing from
  // their own start would let a request run to twice the configured timeout.
  store_deadline(state, global_ns() + static_cast<uint64_t>(timeout_ns));
  state[kReqFailed] = 0;  // D6, per request: kSticky is a process-lifetime latch and cannot be reused for this
  state[kFailReason] = 0;
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
    for (int64_t i = 0; i < planned_count; ++i) {
      bool ready_seen = false;
      if (!lane_result_valid(
              results + i * kLeaseRowResultBytes, generation, planned[i], capacity, &slots[i], &slot_generations[i],
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
namespace exl3_ram_miss_device {

// Stage 1's body, shared by the two-phase W1 and piece streaming's W1 (which also resets the stream kernel's words).
__device__ __forceinline__ void lease_hit_wait_body(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int32_t* __restrict__ dst_slots,
    int64_t row,
    int64_t lanes,
    int64_t* __restrict__ host_rows_1,
    int32_t* __restrict__ dst_slots_1,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    int32_t* __restrict__ go_1,
    int64_t* __restrict__ lane_ctx_1,
    int32_t* __restrict__ origin_1,
    int32_t* __restrict__ claimed,
    int32_t* __restrict__ violated,
    int64_t budget_ns) {
  const uint64_t start = global_ns();
  go_1[0] = 0;      // fail closed: the single commit point is the last store of this kernel
  violated[0] = 0;  // stage 1 opens the chain, so it is where the shared violation flag is cleared
  for (int64_t i = 0; i < lanes; ++i) {
    claimed[i] = 0;
    host_rows_1[i] = 0;
  }
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));

  bool ok = state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0 &&
            ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
  if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
    ok = false;
    state[kFailReason] = static_cast<int32_t>(kLeaseReasonCount);
  }
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
  if (!ok) {
    state[kReqFailed] = 1;
    return;
  }
  // No armed request, or no lanes: nothing to claim early. seq == 0 with lanes planned is a protocol error, and
  // stage 2 is where it is diagnosed, so this stage simply claims nothing.
  if (seq == 0 || planned_count == 0) return;

  const uint8_t* results = lease + kLeaseRowResult +
                           static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLanes * kLeaseRowResultBytes;
  const uint32_t capacity =
      *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
  const uint64_t deadline = load_deadline(state);
  int32_t slots[kMaxIds];
  uint32_t slot_generations[kMaxIds];
  int64_t taken = 0;
  bool hard_fail = false;

  // Bounded poll. Without a bound an all-miss request would spin here until the request was served and so pay the
  // read wait twice (T10). The bound is time, not iterations: each pass reads every unclaimed lane's result across
  // PCIe, so a pass costs microseconds that grow with the lane count, and 8 passes measured up to 212 us.
  for (;;) {
    for (int64_t i = 0; i < planned_count && !hard_fail; ++i) {
      if (claimed[i] != 0) continue;
      bool ready_seen = false;
      if (lane_result_valid(
              results + i * kLeaseRowResultBytes, generation, planned[i], capacity, &slots[i], &slot_generations[i],
              &ready_seen)) {
        claimed[i] = 1;
        ++taken;
      } else if (ready_seen) {
        hard_fail = true;  // published and invalid: a protocol violation, not a lane for stage 2
      }
    }
    if (hard_fail || taken == planned_count) break;
    // Once the request is served every lane is published, so a later poll can discover nothing new.
    if (reached(ld_acquire_sys(page + kDemandDone), seq)) break;
    const uint64_t now = global_ns();
    if (static_cast<int64_t>(now - start) >= budget_ns) break;
    if (static_cast<int64_t>(now - deadline) >= 0) break;
    if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) break;
    __nanosleep(256);
  }

  if (hard_fail) {
    state[kReqFailed] = 1;
    state[kFailReason] = static_cast<int32_t>(kLeaseReasonIdentity);
    // Nothing was committed here: go_1 stays 0, so not one lane is copied or acknowledged. `claimed` has to
    // say so as well. Stage 2 reads it as "stage 1 already served this lane" and skips those lanes, so a
    // claimed-but-uncopied lane is silently absent from `ram_miss` and from kUnservedMisses -- 4 planned
    // lanes with one invalid would report 1 miss where the truth is 4. Clearing it restores the invariant
    // rather than patching the arithmetic downstream: stage 2's `unclaimed` then counts the whole request,
    // which is what actually went unserved. The request still fails on its own account (copied != planned),
    // so this is the counters telling the truth, not a change of outcome.
    for (int64_t i = 0; i < planned_count; ++i) claimed[i] = 0;
    state[kUnservedMisses] += static_cast<int32_t>(planned_count);
    return;  // go_1 stays 0: no copy, no acknowledgement; the finalize kernel publishes the terminal
  }

  int64_t n = 0;
  for (int64_t i = 0; i < planned_count; ++i) {
    if (claimed[i] == 0) continue;
    host_rows_1[n] = static_cast<int64_t>(slots[i]);
    dst_slots_1[n] = dst_slots[i];
    origin_1[n] = static_cast<int32_t>(i);
    lane_ctx_1[4 * n + 0] = static_cast<int64_t>(generation);
    lane_ctx_1[4 * n + 1] = static_cast<int64_t>(slot_generations[i]);
    lane_ctx_1[4 * n + 2] = row;
    lane_ctx_1[4 * n + 3] = static_cast<int64_t>(slots[i]);
    ++n;
  }
  go_1[0] = static_cast<int32_t>(n);  // the single commit point
}

}  // namespace exl3_ram_miss_device

__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_hit_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int32_t* __restrict__ dst_slots,
    int64_t row,
    int64_t lanes,
    int64_t* __restrict__ host_rows_1,
    int32_t* __restrict__ dst_slots_1,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    int32_t* __restrict__ go_1,
    int64_t* __restrict__ lane_ctx_1,
    int32_t* __restrict__ origin_1,
    int32_t* __restrict__ claimed,
    int32_t* __restrict__ violated,
    int64_t budget_ns) {
  if (threadIdx.x != 0) return;
  exl3_ram_miss_device::lease_hit_wait_body(
      page, state, planned, count, dst_slots, row, lanes, host_rows_1, dst_slots_1, lease, lease_d, go_1, lane_ctx_1,
      origin_1, claimed, violated, budget_ns);
}

// Piece streaming's W1: stage 1 exactly, after resetting every word the stream kernel and the finalize kernel read
// from it (plan 5, C1). go_2 and the counter are written by S only on a commit and by its blocks' counts, so without
// this a replay could read an earlier replay's values; this is the first kernel of the chain that owns them.
__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_stream_hit_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int32_t* __restrict__ dst_slots,
    int64_t row,
    int64_t lanes,
    int64_t* __restrict__ host_rows_1,
    int32_t* __restrict__ dst_slots_1,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    int32_t* __restrict__ go_1,
    int64_t* __restrict__ lane_ctx_1,
    int32_t* __restrict__ origin_1,
    int32_t* __restrict__ claimed,
    int32_t* __restrict__ violated,
    int64_t budget_ns,
    int32_t* __restrict__ go_2,
    uint32_t* __restrict__ stream_count,
    int32_t* __restrict__ stream_abort) {
  if (threadIdx.x != 0) return;
  go_2[0] = 0;
  stream_count[0] = 0u;
  stream_abort[0] = 0;
  exl3_ram_miss_device::lease_hit_wait_body(
      page, state, planned, count, dst_slots, row, lanes, host_rows_1, dst_slots_1, lease, lease_d, go_1, lane_ctx_1,
      origin_1, claimed, violated, budget_ns);
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
__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_rest_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int32_t* __restrict__ dst_slots,
    int64_t row,
    int64_t lanes,
    int64_t* __restrict__ host_rows_2,
    int32_t* __restrict__ dst_slots_2,
    int64_t* __restrict__ ram_miss,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    const int32_t* __restrict__ claimed,
    int32_t* __restrict__ go_2,
    int64_t* __restrict__ lane_ctx_2,
    int32_t* __restrict__ origin_2) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  go_2[0] = 0;  // fail closed
  for (int64_t i = 0; i < lanes; ++i) host_rows_2[i] = 0;
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
      __threadfence_system();
      const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
      const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
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

  const uint8_t* results = lease + kLeaseRowResult +
                           static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLanes * kLeaseRowResultBytes;
  const uint32_t capacity =
      *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
  int64_t n = 0;
  int64_t misses = 0;
  for (int64_t i = 0; i < planned_count; ++i) {
    if (claimed[i] != 0) continue;  // stage 1 copied and acknowledged this lane already
    int32_t host_slot = 0;
    uint32_t slot_generation = 0;
    bool ready_seen = false;
    if (!lane_result_valid(
            results + i * kLeaseRowResultBytes, generation, planned[i], capacity, &host_slot, &slot_generation,
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

namespace exl3_ram_miss_device {

// Piece streaming's stream kernel S (piece-streaming plan section 5): kStreamBlocks blocks of kStreamThreads.
constexpr int kStreamBlocks = 8;
constexpr int kStreamThreads = 256;
constexpr int kRowPieces = 8;  // the host's kPieces: one readiness bit per piece of a row
constexpr uint32_t kAllPieces = 255u;
// S's one counter word: finished blocks in the low 16 bits, completed blocks in units of kStreamCompleted above them.
// One word, so the last block reads both counts in the atomic that makes it last.
constexpr uint32_t kStreamCompleted = 65536u;
constexpr uint32_t kStreamFinishedMask = 65535u;
// Test-only fault words (a device int32 tensor, all zero in production).
constexpr int kStreamFaultAbortBlock = 0;  // 1 + the block that takes the abort path (0: none)
constexpr int kStreamFaultAbortDelay = 1;  // ns that block spins before its abort stores
constexpr int kStreamFaultStall = 2;       // ns every leader pass stalls between reading the masks and kDemandDone
constexpr int kStreamFaultCountDelay = 3;  // ns a completing block spins before its count
constexpr int kStreamFaultWords = 4;

// state[*word] += value, clamped at INT32_MAX like W2's kPolls, from any number of blocks at once.
__device__ __forceinline__ void saturating_add(int32_t* word, int64_t value) {
  int32_t old = *reinterpret_cast<volatile int32_t*>(word);
  while (true) {
    const int64_t sum = static_cast<int64_t>(old) + value;
    const int32_t next = static_cast<int32_t>(sum < 0x7fffffffLL ? sum : 0x7fffffffLL);
    const int32_t seen = atomicCAS(word, old, next);
    if (seen == old) return;
    old = seen;
  }
}

__device__ __forceinline__ void spin_ns(int64_t ns) {
  if (ns <= 0) return;
  const uint64_t until = global_ns() + static_cast<uint64_t>(ns);
  while (static_cast<int64_t>(global_ns() - until) < 0) __nanosleep(256);
}

// ld.global.cv, never .nc: a tag-2 lane's host bytes are written while the kernel runs, and .nc may serve a line
// cached before its piece was published (LEASE_PROTOCOL.md E1 amendment).
__device__ __forceinline__ void stream_copy16(const uint8_t* src, uint8_t* dst) {
  uint64_t lo, hi;
  asm volatile("ld.global.cv.v2.b64 {%0,%1},[%2];" : "=l"(lo), "=l"(hi) : "l"(src) : "memory");
  asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(dst), "l"(lo), "l"(hi) : "memory");
}

__device__ __forceinline__ void stream_copy1(const uint8_t* src, uint8_t* dst) {
  uint16_t value;
  asm volatile("ld.global.cv.u8 %0, [%1];" : "=h"(value) : "l"(src) : "memory");
  *dst = static_cast<uint8_t>(value);
}

// This block's share of `bytes` bytes: units (16 B when both ends allow it) in chunks of kStreamThreads, the chunks
// dealt round-robin over the grid, so every block copies about 1/gridDim of a range and no two blocks write one byte.
__device__ __forceinline__ void stream_copy_slice(const uint8_t* src, uint8_t* dst, int64_t bytes) {
  const int64_t tid = threadIdx.x;
  const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15) == 0;
  const int64_t units = aligned ? bytes / 16 : bytes;
  for (int64_t chunk = blockIdx.x; chunk * kStreamThreads < units; chunk += gridDim.x) {
    const int64_t u = chunk * kStreamThreads + tid;
    if (u >= units) break;
    if (aligned) {
      stream_copy16(src + 16 * u, dst + 16 * u);
    } else {
      stream_copy1(src + u, dst + u);
    }
  }
  if (aligned && blockIdx.x == 0) {
    for (int64_t b = units * 16 + tid; b < bytes; b += kStreamThreads) stream_copy1(src + b, dst + b);
  }
}

// This block's slice of piece `piece` of one lane. `runs` is the lane's [kRowPieces][row_segments][2] table of
// name-row byte ranges; `segment_map` gives each row segment's copy-table entry (-1: none), then a flag per entry
// that no row segment names, which is copied whole with piece 0 (C2 copied every entry for a lane).
__device__ __forceinline__ void stream_copy_piece(
    const int64_t* segments,
    int64_t segment_count,
    const int32_t* segment_map,
    int64_t row_segments,
    const int32_t* runs,
    int piece,
    int64_t host_slot,
    int64_t dst_slot) {
  for (int64_t r = 0; r < row_segments; ++r) {
    const int32_t entry = segment_map[r];
    if (entry < 0) continue;
    const int64_t lo = runs[(piece * row_segments + r) * 2];
    const int64_t hi = runs[(piece * row_segments + r) * 2 + 1];
    if (hi <= lo) continue;
    const int64_t* e = segments + 3 * entry;
    const int64_t row_bytes = e[2];
    stream_copy_slice(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + host_slot * row_bytes + lo,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + dst_slot * row_bytes + lo,
        hi - lo);
  }
  if (piece != 0) return;
  for (int64_t k = 0; k < segment_count; ++k) {
    if (segment_map[row_segments + k] == 0) continue;
    const int64_t* e = segments + 3 * k;
    const int64_t row_bytes = e[2];
    stream_copy_slice(
        reinterpret_cast<const uint8_t*>(static_cast<intptr_t>(e[0])) + host_slot * row_bytes,
        reinterpret_cast<uint8_t*>(static_cast<intptr_t>(e[1])) + dst_slot * row_bytes,
        row_bytes);
  }
}

struct StreamLanes {
  int32_t mine[kLeaseLanes];  // planned and not claimed by W1: this kernel's lane
  int32_t admitted[kLeaseLanes];
  int32_t loading[kLeaseLanes];  // admitted under tag LOADING (else READY: every piece is there)
  int32_t slot[kLeaseLanes];
  uint32_t slot_generation[kLeaseLanes];
  uint32_t done[kLeaseLanes];  // pieces this block has copied its slice of
  uint32_t todo[kLeaseLanes];  // pieces to copy this pass
  int identity;                // a published lane failed validation
  int aborting;
  uint32_t reason;             // the abort's kLeaseReason*, 0 for one that names none
  int served;                  // kDemandDone >= seq, status kServed, and the re-read found every mask full
  int finished;                // served and every piece of every lane copied
  int probed;
};

// One lane's admission (plan 5): the whole lane_result_valid contract, tag LOADING accepted as well as READY.
// Returns false for a published lane that fails it; an unpublished lane is simply not admitted yet.
__device__ __forceinline__ bool stream_admit(
    StreamLanes& sh,
    int lane,
    const uint8_t* results,
    uint64_t generation,
    int64_t expected_expert,
    int64_t experts,
    uint32_t capacity) {
  bool ready_seen = false;
  bool loading = false;
  int32_t slot = 0;
  uint32_t slot_generation = 0;
  if (lane_result_valid(
          results + lane * kLeaseRowResultBytes, generation, expected_expert, capacity, &slot, &slot_generation,
          &ready_seen, /*accept_loading=*/true, &loading) &&
      expected_expert >= 0 && expected_expert < experts) {
    sh.slot[lane] = slot;
    sh.slot_generation[lane] = slot_generation;
    sh.loading[lane] = loading ? 1 : 0;
    sh.admitted[lane] = 1;
    return true;
  }
  return !ready_seen;
}

__device__ __forceinline__ uint32_t piece_bits(uint64_t word, uint64_t generation) {
  return (word >> 8) == (generation & ((1ull << 56) - 1)) ? static_cast<uint32_t>(word & 0xFFu) : 0u;
}

}  // namespace exl3_ram_miss_device

// Piece streaming's stage 2 (piece-streaming plan section 5): replaces W2 and C2. It covers every planned lane W1 did
// not claim, admits each once its RowResult validates (tag READY or LOADING), and copies each piece as its bit appears
// in the lane's PieceMask word, so the copy overlaps the read. Every block runs the same leader loop on its own and
// copies its own slice of every piece; there is no inter-block barrier, so no co-residency is assumed.
//
// Commit (plan 5, C1): each block ends on exactly one path and counts into one word. The block that makes the count
// finished commits `go_2` only if every block completed, none aborted, and (by completing) it saw kDemandDone >= seq
// with status kServed and every mask full on a re-read made after acquiring kDemandDone. Otherwise go_2 stays at W1's
// reset of 0. Like W2 it never writes `keep` (the finalize kernel's alone), a terminal or the fatal word.
__global__ __launch_bounds__(exl3_ram_miss_device::kStreamThreads, 1) void exl3_ram_miss_lease_stream_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ planned,
    const int32_t* __restrict__ count,
    const int32_t* __restrict__ dst_slots,
    int64_t row,
    int64_t lanes,
    int64_t experts,
    int64_t* __restrict__ host_rows_2,
    int32_t* __restrict__ dst_slots_2,
    int64_t* __restrict__ ram_miss,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    int64_t lease_p,
    const int32_t* __restrict__ claimed,
    int32_t* __restrict__ go_2,
    int64_t* __restrict__ lane_ctx_2,
    int32_t* __restrict__ origin_2,
    uint32_t* __restrict__ stream_count,
    int32_t* __restrict__ stream_abort,
    const int64_t* __restrict__ segments,
    int64_t segment_count,
    const int32_t* __restrict__ segment_map,
    int64_t row_segments,
    const int32_t* __restrict__ piece_runs,
    const int32_t* __restrict__ fault) {
  using namespace exl3_ram_miss_device;
  __shared__ StreamLanes sh;
  __shared__ int64_t planned_count;
  __shared__ uint32_t seq;
  __shared__ uint64_t generation;
  __shared__ uint32_t capacity;
  __shared__ int started;         // W2's `ok && seq != 0` at entry: the requests W2 counts in kWaits
  __shared__ int64_t unclaimed;   // planned lanes W1 did not claim: W2's `unclaimed`, whatever path S takes
  const int tid = threadIdx.x;

  if (tid == 0) {
    planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
    seq = static_cast<uint32_t>(state[kPending]);
    generation = seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
    sh.identity = 0;
    sh.aborting = 0;
    sh.reason = 0;
    sh.served = 0;
    sh.finished = 0;
    sh.probed = 0;
    bool ok = state[kSticky] == 0 && state[kReqFailed] == 0 && ld_acquire_sys(page + kFatal) == 0 &&
              ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
    uint32_t reason = ok ? 0u : kLeaseReasonAborted;
    if (ok && (planned_count > kLeaseLanes || planned_count > lanes)) {
      ok = false;
      reason = kLeaseReasonCount;
    }
    if (ok && seq == 0 && planned_count > 0) {
      // Lanes and no armed request to have leased them: a protocol error, as in W2 (it names no reason).
      if (blockIdx.x == 0) state[kFailures] += 1;
      ok = false;
      reason = 0;
    }
    started = ok && seq != 0 ? 1 : 0;
    unclaimed = 0;
    for (int64_t i = 0; i < min(planned_count, static_cast<int64_t>(kLeaseLanes)); ++i) {
      if (claimed[i] == 0) ++unclaimed;
    }
    if (ok && fault[kStreamFaultAbortBlock] == static_cast<int32_t>(blockIdx.x) + 1) {
      ok = false;
      reason = kLeaseReasonAborted;
    }
    if (!ok) {
      sh.aborting = 1;
      sh.reason = reason;
    } else if (seq == 0) {
      sh.served = 1;  // no request and no lanes: nothing to wait for or copy
    }
    capacity = seq != 0 ? *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4)
                        : 0u;
  }
  __syncthreads();
  if (tid < kLeaseLanes) {
    sh.mine[tid] = sh.aborting == 0 && tid < planned_count && claimed[tid] == 0 ? 1 : 0;
    sh.admitted[tid] = 0;
    sh.loading[tid] = 0;
    sh.slot[tid] = 0;
    sh.slot_generation[tid] = 0;
    sh.done[tid] = 0;
    sh.todo[tid] = 0;
  }
  __syncthreads();

  const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
  const uint8_t* results = lease + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
  const uint8_t* masks = lease + lease_p + idx * kLeaseLanes * kLeasePieceMaskLineBytes;
  const uint64_t deadline = load_deadline(state);
  const int64_t piece_stride = static_cast<int64_t>(kRowPieces) * row_segments * 2;
  int64_t polls = 0;
  int64_t pieces = 0;

  while (sh.aborting == 0 && sh.finished == 0) {
    // Admission and masks: one leader-warp lane per request lane. A READY lane (a hit W1 missed) has every piece.
    if (tid < kLeaseLanes && sh.mine[tid] != 0) {
      if (sh.admitted[tid] == 0 && !stream_admit(sh, tid, results, generation, planned[tid], experts, capacity)) {
        sh.identity = 1;
      }
      if (sh.admitted[tid] != 0) {
        const uint32_t bits =
            sh.loading[tid] != 0 ? piece_bits(ld_acquire_sys64(masks + tid * kLeasePieceMaskLineBytes), generation)
                                 : kAllPieces;
        sh.todo[tid] = bits & ~sh.done[tid];
      }
    }
    __syncthreads();
    bool copied = false;
    if (sh.identity == 0) {
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        const uint32_t todo = sh.todo[lane];
        if (todo == 0) continue;
        copied = true;
        const int32_t* runs = piece_runs + (row * experts + planned[lane]) * piece_stride;
        for (int piece = 0; piece < kRowPieces; ++piece) {
          if ((todo >> piece & 1u) == 0) continue;
          stream_copy_piece(
              segments, segment_count, segment_map, row_segments, runs, piece, sh.slot[lane], dst_slots[lane]);
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      for (int lane = 0; lane < kLeaseLanes; ++lane) {
        pieces += __popc(sh.todo[lane]);
        sh.done[lane] |= sh.todo[lane];
        sh.todo[lane] = 0;
      }
      if (copied && sh.probed == 0) {
        // The host's only view of streaming progress (G2): stored once this block's first slice is copied.
        st_release_sys64(
            lease + lease_d + kLeaseStreamProbe + idx * kLeaseStreamProbeBytes, tagged_word(kLeaseTagStreamed, generation));
        sh.probed = 1;
      }
      ++polls;
      if (sh.identity != 0) {
        sh.aborting = 1;
        sh.reason = kLeaseReasonIdentity;
      } else if (sh.served == 0) {
        spin_ns(fault[kStreamFaultStall]);
        if (static_cast<int64_t>(global_ns() - deadline) >= 0) {
          sh.aborting = 1;
          sh.reason = kLeaseReasonTimeout;
        } else if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) {
          sh.aborting = 1;
          sh.reason = kLeaseReasonAborted;
        } else if (reached(ld_acquire_sys(page + kDemandDone), seq)) {
          __threadfence_system();
          const uint8_t* record = page + kDemandRing + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kRecordBytes;
          const uint16_t status = *reinterpret_cast<const volatile uint16_t*>(record + kRecStatus);
          if (status != kServed) {
            sh.aborting = 1;
            sh.reason = kLeaseReasonFailed;
          } else {
            // The judgement, on masks re-read by this thread AFTER its acquire of kDemandDone: the host stores
            // kDemandDone after read() returned, so after every publish CAS, and a mask read earlier in the pass may
            // predate the last one. After kDemandDone every lane is granted, so an unadmitted one is a violation too.
            bool whole = true;
            for (int lane = 0; lane < kLeaseLanes && whole; ++lane) {
              if (sh.mine[lane] == 0) continue;
              if (sh.admitted[lane] == 0) stream_admit(sh, lane, results, generation, planned[lane], experts, capacity);
              if (sh.admitted[lane] == 0) {
                whole = false;
                break;
              }
              const uint32_t bits = sh.loading[lane] != 0
                                        ? piece_bits(ld_acquire_sys64(masks + lane * kLeasePieceMaskLineBytes), generation)
                                        : kAllPieces;
              if (bits != kAllPieces) whole = false;
            }
            if (whole) {
              sh.served = 1;
            } else {
              sh.aborting = 1;
              sh.reason = kLeaseReasonIdentity;
            }
          }
        } else if (!copied) {
          __nanosleep(256);
        }
      }
      // Once served there is no D5 check: termination rests on the masks staying final until the ring slot's reuse
      // re-initialises them, 16 requests later, which cannot happen while this request is still in the chain.
      if (sh.aborting == 0 && sh.served != 0) {
        bool all = true;
        for (int lane = 0; lane < kLeaseLanes; ++lane) {
          if (sh.mine[lane] != 0 && sh.done[lane] != kAllPieces) all = false;
        }
        if (all) sh.finished = 1;
      }
    }
    __syncthreads();
  }

  if (tid != 0) return;
  saturating_add(&state[kStreamPolls], polls);
  if (blockIdx.x == 0) atomicAdd(&state[kStreamPieces], static_cast<int32_t>(pieces));
  uint32_t increment;
  if (sh.aborting != 0) {
    if (fault[kStreamFaultAbortBlock] == static_cast<int32_t>(blockIdx.x) + 1) spin_ns(fault[kStreamFaultAbortDelay]);
    // The abort path: this block's own failure record, first writer of the reason wins, then abort, fence, count.
    *reinterpret_cast<volatile int32_t*>(&state[kReqFailed]) = 1;
    if (sh.reason != 0 && atomicCAS(&state[kFailReason], 0, static_cast<int32_t>(sh.reason)) == 0) {
      if (sh.reason == kLeaseReasonTimeout) state[kTimeouts] += 1;
      if (sh.reason == kLeaseReasonFailed) state[kFailures] += 1;
      if (sh.reason == kLeaseReasonIdentity) state[kUnservedMisses] += static_cast<int32_t>(unclaimed);
    }
    *reinterpret_cast<volatile int32_t*>(stream_abort) = 1;
    __threadfence();
    increment = 1u;
  } else {
    spin_ns(fault[kStreamFaultCountDelay]);
    __threadfence();  // this block's copies, before its count says it completed
    increment = 1u + kStreamCompleted;
  }
  const uint32_t old = atomicAdd(stream_count, increment);
  if ((old & kStreamFinishedMask) != gridDim.x - 1) return;
  // The last block. It decides from the value its own atomic returned, never from a second read of the counter.
  const uint32_t now = old + increment;
  __threadfence();
  int32_t aborted;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(aborted) : "l"(stream_abort) : "memory");
  if (started != 0) state[kWaits] += 1;
  const bool commit = now / kStreamCompleted == gridDim.x && aborted == 0 && sh.aborting == 0 && sh.served != 0;
  if (!commit) {
    ram_miss[0] += unclaimed;  // nothing was served for the lanes W1 did not claim
    return;
  }
  int64_t n = 0;
  for (int lane = 0; lane < kLeaseLanes; ++lane) {
    if (sh.mine[lane] == 0) continue;
    host_rows_2[n] = static_cast<int64_t>(sh.slot[lane]);
    dst_slots_2[n] = dst_slots[lane];
    origin_2[n] = static_cast<int32_t>(lane);
    lane_ctx_2[4 * n + 0] = static_cast<int64_t>(generation);
    lane_ctx_2[4 * n + 1] = static_cast<int64_t>(sh.slot_generation[lane]);
    lane_ctx_2[4 * n + 2] = row;
    lane_ctx_2[4 * n + 3] = static_cast<int64_t>(sh.slot[lane]);
    ++n;
  }
  for (int64_t i = n; i < lanes; ++i) host_rows_2[i] = 0;
  go_2[0] = static_cast<int32_t>(n);  // the single commit point
}

// V1 two-phase acknowledgement (D3), launched after each stage's copy kernel in the same stream. One thread per
// COMPACTED entry; `origin` maps it back to the lane whose acknowledgement word it must write, because
// retire_leases reads those words by lane. Every effect is guarded by `lane < n` with n = go_s[0], so an empty
// stage emits nothing: not an acknowledgement, not a violation, not a fatal word.
//
// It records a violation in `violated` rather than writing `keep`, which only the finalize kernel writes.
__global__ __launch_bounds__(exl3_ram_miss_device::kLeaseLanes, 1) void exl3_ram_miss_lease_stage_ack_kernel(
    uint8_t* __restrict__ page,
    uint8_t* __restrict__ lease,
    int64_t lease_d,
    const int32_t* __restrict__ go_count,
    const int64_t* __restrict__ lane_ctx,
    const int32_t* __restrict__ origin,
    int32_t* __restrict__ violated) {
  using namespace exl3_ram_miss_device;
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
    const uint32_t base =
        *reinterpret_cast<const volatile uint32_t*>(lease + kLeaseRowTable + row * kLeaseRowTableBytes);
    const uint32_t current = ld_acquire_sys(lease + kLeaseSlotGen + 4 * (static_cast<int64_t>(base) + slot));
    const bool consumed = current == slot_generation;
    const int64_t idx = static_cast<int64_t>((static_cast<uint32_t>(generation) - 1u) % kDemandRecords);
    st_release_sys64(
        lease + lease_d + kLeaseLaneAck + (idx * kLeaseLanes + lane) * kLeaseLaneAckBytes,
        tagged_word(consumed ? kLeaseTagConsumed : kLeaseTagViolated, generation));
    if (!consumed) any_violated = 1;
  }
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
__global__ __launch_bounds__(exl3_ram_miss_device::kBlock, 1) void exl3_ram_miss_lease_finalize_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int32_t* __restrict__ count,
    // No `lanes` parameter on purpose. The only bound this kernel needs is kLeaseLanes, because the array it
    // walks is the per-record LaneAck table, which is kLeaseLanes wide by construction -- not a staging buffer.
    // Passing the staging width here would read as the bound and be wrong.
    const int32_t* __restrict__ go_1,
    const int32_t* __restrict__ go_2,
    const int32_t* __restrict__ violated,
    float* __restrict__ keep,
    uint8_t* __restrict__ lease,
    int64_t lease_d) {
  using namespace exl3_ram_miss_device;
  if (threadIdx.x != 0) return;
  const int64_t planned_count = max(static_cast<int64_t>(count[0]), static_cast<int64_t>(0));
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  const uint64_t generation =
      seq != 0 ? (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq : 0ull;
  const int64_t copied = static_cast<int64_t>(go_1[0]) + static_cast<int64_t>(go_2[0]);
  const bool served = state[kReqFailed] == 0 && violated[0] == 0 && copied == planned_count &&
                      ld_acquire_sys(page + kFatal) == 0;
  state[kPending] = 0;  // the last kernel of the chain, so the clear lands here rather than in either wait
  if (served) {
    keep[0] = 1.0f;
    return;
  }
  // The mask names every planned lane carrying no acknowledgement for this generation. A VIOLATED acknowledgement
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
    const uint32_t reason =
        state[kFailReason] != 0 ? static_cast<uint32_t>(state[kFailReason]) : kLeaseReasonFailed;
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
    int64_t lease_d,
    int64_t timeout_ns,
    int64_t hot_address,
    int64_t hot_stride,
    tvm::ffi::TensorView hot_slots,
    int64_t hot_capacity) {
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
      hot_capacity);
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
      static_cast<const int32_t*>(violated.data_ptr()),
      static_cast<float*>(keep.data_ptr()),
      reinterpret_cast<uint8_t*>(lease_address),
      lease_d);
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
