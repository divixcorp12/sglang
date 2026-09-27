// Device-side constants and helpers shared by the lease-protocol and row-copy kernels (a split of the EXL3
// instantiation's former exl3_ram_miss.cuh).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <cuda/ptx>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "lease_layout.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {
namespace device::expert_stream {

using namespace ::sglang::expert_stream::wire;

constexpr int kBlock = 32;
constexpr int kCopyWaitThreads = 256;  // the copy wait's block when it also reads the small tensors

constexpr int kPosted = 0;
constexpr int kPending = 1;
constexpr int kTimeouts = 2;
constexpr int kFailures = 3;
constexpr int kWaits = 4;
constexpr int kPolls = 5;
constexpr int kSticky = 6;
constexpr int kAdvised = 7;
constexpr int kUnservedMisses = 8;
constexpr int kEpoch = 9;          // times seq32 wrapped; the device owns it (LEASE_PROTOCOL.md 11.3)
constexpr int kPendingEpoch = 10;  // the epoch of the request kPending names
// D5. One absolute deadline per request, written once by the post kernel and compared against by both stages, so a
// two-stage request cannot run to 2 x SGLANG_DSV41_RAM_MISS_TIMEOUT_MS. `state` is int32[], so it takes two words.
constexpr int kDeadlineLo = 11;
constexpr int kDeadlineHi = 12;
// D6. An earlier stage of THIS request failed. kSticky cannot serve here: it is a process-lifetime fail-stop latch
// with no clear site, so using it would refuse every later request in the process.
constexpr int kReqFailed = 13;
constexpr int kFailReason = 14;    // the kLeaseReason* the failing stage recorded, for the terminal F publishes
constexpr int kStreamPieces = 15;  // piece slices the stream kernel's block 0 copied, cumulative
constexpr int kStreamPolls = 16;   // stream-kernel leader passes, summed over its blocks, cumulative
constexpr int kW1Passes = 17;      // stage 1's polling passes over its unclaimed lanes, cumulative
constexpr int kCopyWaits = 18;     // requests whose copy-engine lanes the copy wait waited for, cumulative
constexpr int kCopySpun = 19;      // ... of which CopyDone was not yet published on the first read
constexpr int kStateWords = 20;

SGL_DEVICE uint32_t ld_acquire_sys(const uint8_t* address) {
  return ::sglang::device::ptx::load_acquire_sys(reinterpret_cast<const uint32_t*>(address));
}

SGL_DEVICE void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

SGL_DEVICE uint64_t ld_acquire_sys64(const uint8_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

SGL_DEVICE void st_release_sys64(uint8_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

SGL_DEVICE uint64_t tagged_word(uint64_t tag, uint64_t generation) {
  return (tag << 56) | generation;
}

// Every word the host or another kernel accesses concurrently goes through one of these, so each call site states
// its ordering. Relaxed is a volatile access: the PTX memory model treats ld/st.volatile as relaxed at system scope,
// and nvcc emits LDG/STG.E.STRONG.SYS for it. Not cuda::atomic_ref: with CUDA 13.4's libcu++, an access through a
// __grid_constant__ parameter gains a run-time local-pointer check with a byte-copy fallback, and adjacent relaxed
// accesses are merged and reordered (plan 2026-09-27-expert-stream-native-sync, Task 2).
template <typename T>
SGL_DEVICE T ld_relaxed_sys(const T* word) {
  return *reinterpret_cast<const volatile T*>(word);
}

template <typename T>
SGL_DEVICE void st_relaxed_sys(T* word, std::type_identity_t<T> value) {
  *reinterpret_cast<volatile T*>(word) = value;
}

// A wire field at a byte address (lease_layout.h offsets); T names the field's type.
template <typename T>
SGL_DEVICE T ld_relaxed_sys(const uint8_t* address) {
  return ld_relaxed_sys(reinterpret_cast<const T*>(address));
}

template <typename T>
SGL_DEVICE void st_relaxed_sys(uint8_t* address, std::type_identity_t<T> value) {
  st_relaxed_sys<T>(reinterpret_cast<T*>(address), value);
}

// Device scope, for the stream kernel's abort word, which only its own blocks write.
SGL_DEVICE int32_t ld_relaxed_gpu(const int32_t* word) {
  int32_t value;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(value) : "l"(word) : "memory");
  return value;
}

SGL_DEVICE uint64_t global_ns() {
  return cuda::ptx::get_sreg_globaltimer();
}

SGL_DEVICE void store_deadline(int32_t* state, uint64_t deadline) {
  state[kDeadlineLo] = static_cast<int32_t>(static_cast<uint32_t>(deadline & 0xFFFFFFFFull));
  state[kDeadlineHi] = static_cast<int32_t>(static_cast<uint32_t>(deadline >> 32));
}

SGL_DEVICE uint64_t load_deadline(const int32_t* state) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineHi])) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineLo]));
}

SGL_DEVICE bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

SGL_DEVICE bool listed(const int32_t* ids, int count, int32_t id) {
  for (int i = 0; i < count; ++i) {
    if (ids[i] == id) return true;
  }
  return false;
}

SGL_DEVICE void write_record(
    uint8_t* record,
    uint32_t seq,
    int64_t row,
    const int32_t* need,
    int need_count,
    const int32_t* protect,
    int protect_count,
    uint32_t after,
    uint32_t armed,
    uint32_t lanes) {
  // Seqlock writer: invalidate seq before touching the payload, so a lapped record that
  // is half rewritten never passes the thread's read_record seq re-check.
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  __threadfence_system();
  st_relaxed_sys<uint16_t>(record + kRecRow, static_cast<uint16_t>(row));
  st_relaxed_sys<uint16_t>(record + kRecNeedCount, static_cast<uint16_t>(need_count));
  st_relaxed_sys<uint16_t>(record + kRecProtectCount, static_cast<uint16_t>(protect_count));
  st_relaxed_sys<uint16_t>(record + kRecStatus, 0);
  st_relaxed_sys<uint32_t>(record + kRecAfter, after);
  st_relaxed_sys<uint32_t>(record + kRecArmed, armed);
  st_relaxed_sys<uint32_t>(record + kRecLanes, lanes);
  for (int i = 0; i < kMaxIds; ++i) {
    st_relaxed_sys<int32_t>(record + kRecNeed + 4 * i, i < need_count ? need[i] : -1);
    st_relaxed_sys<int32_t>(record + kRecProtect + 4 * i, i < protect_count ? protect[i] : -1);
  }
  // The seqlock order the thread's read_record relies on: seq=0, fence, payload, seq last. The release store is the
  // second fence: it orders every payload store above before the seq.
  st_release_sys(record + kRecSeq, seq);
}

SGL_DEVICE void raise_fatal(uint8_t* page, uint32_t seq) {
  if (ld_acquire_sys(page + kFatal) == 0) st_release_sys(page + kFatal, seq);
}

// The per-lane RowResult validity test (LEASE_PROTOCOL.md 11.4), shared by the batched wait and both V1 stages so
// that exactly one copy of it exists. `ready_seen` reports whether the service has published this lane at all,
// which two-phase must tell apart from an invalid publish: an unpublished lane is simply the other stage's work,
// while a published-but-invalid one is a protocol violation. A caller that conflates them fails every mixed
// request, or turns a genuine identity violation into a silent retry.
// `accept_loading` also accepts tag LOADING (piece streaming, LEASE_PROTOCOL.md E1 amendment) under the same
// contract. Only the stream kernel passes it: a tag-2 lane's bytes are final piece by piece, so every other caller
// must go on seeing it as unpublished. `loading` reports a LOADING tag of THIS generation whether or not it was
// accepted, so stage 1 can tell a lane it will never be able to claim from one not yet published.
//
// The belt of 11.4 is a seqlock re-read of the ready word, and it has two halves. The host clears a lane's ready
// word before rewriting its payload (grant_lane_group_locked). The reader must order its payload loads before the
// re-read with a system fence: an acquire orders only what follows it, so without the fence the re-read may be
// served before the payload and a rewrite caught half-way passes. Hence read, fence, re-read, judge; a caller that
// polls several lanes reads them all and pays one fence (stage 1).
struct LaneRead {
  uint64_t ready;
  uint32_t slot_generation;
  int32_t host_slot;
  int32_t expert;
};

SGL_DEVICE LaneRead lane_result_read(const uint8_t* result) {
  LaneRead r;
  r.ready = ld_acquire_sys64(result + kLeaseRrReady);
  r.slot_generation = ld_relaxed_sys<uint32_t>(result + kLeaseRrSlotGeneration);
  r.host_slot = ld_relaxed_sys<int32_t>(result + kLeaseRrHostSlot);
  r.expert = ld_relaxed_sys<int32_t>(result + kLeaseRrExpert);
  return r;
}

// Relaxed: the caller's fence already orders it after the payload loads, and nothing after it depends on it.
SGL_DEVICE uint64_t lane_result_reread(const uint8_t* result) {
  return ld_relaxed_sys<uint64_t>(result + kLeaseRrReady);
}

SGL_DEVICE bool lane_result_judge(
    const LaneRead& r,
    uint64_t again,
    uint64_t generation,
    int64_t expected_expert,
    uint32_t capacity,
    int32_t* out_slot,
    uint32_t* out_slot_generation,
    bool* ready_seen,
    bool accept_loading,
    bool* loading,
    bool accept_copying = false,
    bool* copying = nullptr) {
  const uint64_t generation_mask = (1ull << 56) - 1;
  const uint64_t tag = r.ready >> 56;
  const bool current = (r.ready & generation_mask) == generation;
  if (loading != nullptr) *loading = tag == kLeaseTagLoading && current;
  if (copying != nullptr) *copying = tag == kLeaseTagCopying && current;
  *ready_seen = (tag == kLeaseTagReady || (accept_loading && tag == kLeaseTagLoading) ||
                 (accept_copying && tag == kLeaseTagCopying)) &&
                current;
  const bool valid = *ready_seen && again == r.ready && static_cast<int64_t>(r.expert) == expected_expert &&
                     r.host_slot >= 0 &&
                     static_cast<uint32_t>(r.host_slot) < capacity;  // also bounds the ack SlotGen read
  if (valid) {
    *out_slot = r.host_slot;
    *out_slot_generation = r.slot_generation;
  }
  return valid;
}

SGL_DEVICE bool lane_result_valid(
    const uint8_t* result,
    uint64_t generation,
    int64_t expected_expert,
    uint32_t capacity,
    int32_t* out_slot,
    uint32_t* out_slot_generation,
    bool* ready_seen,
    bool accept_loading = false,
    bool* loading = nullptr,
    bool accept_copying = false,
    bool* copying = nullptr) {
  const LaneRead r = lane_result_read(result);
  __threadfence_system();
  return lane_result_judge(
      r,
      lane_result_reread(result),
      generation,
      expected_expert,
      capacity,
      out_slot,
      out_slot_generation,
      ready_seen,
      accept_loading,
      loading,
      accept_copying,
      copying);
}

// A Terminal record (LEASE_PROTOCOL.md 13): the mask and the reason first, the tagged generation word last with a
// release store, so a reader that acquires the word sees the mask. Named lanes never start a copy afterwards: the
// caller has already left go_count at zero.
SGL_DEVICE void publish_terminal(
    uint8_t* lease, int64_t lease_d, uint32_t seq, uint64_t generation, uint32_t skipped_mask, uint32_t reason) {
  uint8_t* terminal =
      lease + lease_d + kLeaseTerminal + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseTerminalBytes;
  st_relaxed_sys<uint32_t>(terminal + kLeaseTermSkippedMask, skipped_mask);
  st_relaxed_sys<uint32_t>(terminal + kLeaseTermReason, reason);
  st_release_sys64(terminal + kLeaseTermGen, tagged_word(kLeaseTagTerminal, generation));
}

// Stage 1's body, shared by the two-phase W1 and piece streaming's W1 (which also resets the stream kernel's words).
SGL_DEVICE void lease_hit_wait_body(
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

  bool ok =
      state[kSticky] == 0 && ld_acquire_sys(page + kFatal) == 0 && ld_acquire_sys(lease + kLeaseHeaderShutdown) == 0;
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

  const uint8_t* results =
      lease + kLeaseRowResult + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseLanes * kLeaseRowResultBytes;
  const uint32_t capacity = ld_relaxed_sys<uint32_t>(lease + kLeaseRowTable + row * kLeaseRowTableBytes + 4);
  const uint64_t deadline = load_deadline(state);
  int32_t slots[kMaxIds];
  uint32_t slot_generations[kMaxIds];
  int64_t taken = 0;
  bool hard_fail = false;

  // Bounded poll. Without a bound an all-miss request would spin here until the request was served and so pay the
  // read wait twice (T10). The bound is time, not iterations: each pass reads every unclaimed lane's result across
  // PCIe, so a pass costs microseconds that grow with the lane count, and 8 passes measured up to 212 us.
  int64_t passes = 0;
  for (;;) {
    ++passes;
    LaneRead reads[kMaxIds];
    for (int64_t i = 0; i < planned_count; ++i) {
      if (claimed[i] == 0) reads[i] = lane_result_read(results + i * kLeaseRowResultBytes);
    }
    __threadfence_system();  // one per pass: every lane's payload loads before any lane's re-read
    int64_t streaming = 0;
    for (int64_t i = 0; i < planned_count && !hard_fail; ++i) {
      if (claimed[i] != 0) continue;
      bool ready_seen = false;
      bool loading = false;
      bool copying = false;
      if (lane_result_judge(
              reads[i],
              lane_result_reread(results + i * kLeaseRowResultBytes),
              generation,
              planned[i],
              capacity,
              &slots[i],
              &slot_generations[i],
              &ready_seen,
              /*accept_loading=*/false,
              &loading,
              /*accept_copying=*/true,
              &copying)) {
        // 2: the service's copy engine owns this lane; S skips it, C1 and A1 never see it, the copy wait awaits it.
        claimed[i] = copying ? 2 : 1;
        ++taken;
      } else if (ready_seen) {
        hard_fail = true;  // published and invalid: a protocol violation, not a lane for stage 2
      } else if (loading) {
        ++streaming;
      }
    }
    // A LOADING lane is stage 2's for good: it never turns READY, so once every lane is claimed or LOADING a further
    // pass can claim nothing.
    if (hard_fail || taken + streaming == planned_count) break;
    // Once the request is served every lane is published, so a later poll can discover nothing new.
    if (reached(ld_acquire_sys(page + kDemandDone), seq)) break;
    const uint64_t now = global_ns();
    if (static_cast<int64_t>(now - start) >= budget_ns) break;
    if (static_cast<int64_t>(now - deadline) >= 0) break;
    if (ld_acquire_sys(page + kFatal) != 0 || ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0) break;
    __nanosleep(256);
  }
  const int64_t total_passes = static_cast<int64_t>(state[kW1Passes]) + passes;
  state[kW1Passes] = static_cast<int32_t>(total_passes < 0x7fffffffLL ? total_passes : 0x7fffffffLL);

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
    for (int64_t i = 0; i < planned_count; ++i)
      claimed[i] = 0;
    state[kUnservedMisses] += static_cast<int32_t>(planned_count);
    return;  // go_1 stays 0: no copy, no acknowledgement; the finalize kernel publishes the terminal
  }

  int64_t n = 0;
  for (int64_t i = 0; i < planned_count; ++i) {
    if (claimed[i] != 1) continue;
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

}  // namespace device::expert_stream
}  // namespace sglang
