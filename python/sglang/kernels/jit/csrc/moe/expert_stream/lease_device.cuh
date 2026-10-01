// Device-side constants and helpers shared by the lease-protocol and row-copy kernels.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <cuda/atomic>
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

// The device's own words (`state`, int32, device memory): never on the wire.
constexpr int kPosted = 0;
constexpr int kPending = 1;       // the seq of the request this layer's chain serves, 0 for an unarmed post
constexpr int kEpoch = 2;         // times seq32 wrapped; G = epoch << 32 | seq
constexpr int kPendingEpoch = 3;  // the epoch of the request kPending names
// S's absolute deadline, written by the post (two int32 halves): the one device spin with no other bound.
constexpr int kDeadlineLo = 4;
constexpr int kDeadlineHi = 5;
constexpr int kStateWords = 6;

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

constexpr uint64_t kGenerationMask = (1ull << 56) - 1;

// The copy wait's gate word for request `seq`: `low` is kLeaseGateClosed or kLeaseGateOpen.
SGL_DEVICE uint32_t copy_gate_word(uint32_t seq, uint32_t low) {
  return ((seq & kLeaseGateSeqMask) << kLeaseGateSeqShift) | low;
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

// The request the chain serves: kPending and kPendingEpoch change only at the next post, stream-ordered after the
// chain's last kernel.
SGL_DEVICE uint64_t pending_generation(const int32_t* state) {
  const uint32_t seq = static_cast<uint32_t>(state[kPending]);
  return (static_cast<uint64_t>(static_cast<uint32_t>(state[kPendingEpoch])) << 32) | seq;
}

SGL_DEVICE int64_t ring_index(uint32_t seq) {
  return static_cast<int64_t>((seq - 1u) % kDemandRecords);
}

// One request's typed lanes, as the post writes them into its record (lease_layout.h kRecLanes, kRecKinds).
struct TypedLanes {
  int32_t slot[kMaxIds];  // the RAM slot of a hit, the staging slot of a miss
  uint8_t kind[kMaxIds];  // kKind*
};

SGL_DEVICE bool is_cpu_kind(uint32_t kind) {
  return kind == kKindHitCpu || kind == kKindMissCpu;
}

// Seqlock writer: a record of SM hits only may lap the ring before the service reads it, so a half-rewritten record
// must never pass read_record's seq re-check. seq = 0, fence, payload, then seq with a release.
SGL_DEVICE void write_record(
    uint8_t* record, uint32_t seq, uint32_t epoch, int64_t row, uint32_t flags, uint64_t chain, const int32_t* protect,
    int protect_count, int64_t count, const int64_t* planned, const int32_t* dst, const float* weight,
    const TypedLanes& lanes) {
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint16_t>(record + kRecRow, static_cast<uint16_t>(row));
  st_relaxed_sys<uint16_t>(record + kRecCount, static_cast<uint16_t>(count));
  st_relaxed_sys<uint32_t>(record + kRecFlags, flags);
  st_relaxed_sys<uint32_t>(record + kRecChain, static_cast<uint32_t>(chain & 0xFFFFFFFFull));
  st_relaxed_sys<uint32_t>(record + kRecChainHi, static_cast<uint32_t>(chain >> 32));
  st_relaxed_sys<uint32_t>(record + kRecEpoch, epoch);
  st_relaxed_sys<uint16_t>(record + kRecProtectCount, static_cast<uint16_t>(protect_count));
  for (int i = 0; i < kMaxIds; ++i) {
    st_relaxed_sys<int32_t>(record + kRecProtect + 4 * i, i < protect_count ? protect[i] : -1);
    uint8_t* lane = record + kRecLanes + i * kLaneBytes;
    const bool used = i < count;
    st_relaxed_sys<int32_t>(lane + kLaneExpert, used ? static_cast<int32_t>(planned[i]) : -1);
    st_relaxed_sys<int32_t>(lane + kLaneSlot, used ? lanes.slot[i] : -1);
    st_relaxed_sys<int32_t>(lane + kLaneDst, used ? dst[i] : -1);
    st_relaxed_sys<uint32_t>(lane + kLaneWeight, __float_as_uint(used ? weight[i] : 0.0f));
    st_relaxed_sys<uint8_t>(record + kRecKinds + i, used ? lanes.kind[i] : static_cast<uint8_t>(0));
  }
  st_release_sys(record + kRecSeq, seq);
}

// The row's pending map delta (lease_layout.h, the delta block): wait, bounded by `deadline`, until the host has
// published the delta that follows the row's last map chain, then apply it once. The host publishes a chain's delta
// before it reads that chain's misses, so the wait is taken only when the host fell a whole token behind.
SGL_DEVICE void apply_map_delta(
    const uint8_t* delta, int32_t* ram_slot_row, int32_t* staging_row, int64_t* map_chain, int64_t* map_applied,
    int64_t row, int64_t experts, uint32_t row_capacity, uint64_t deadline) {
  const uint64_t want = static_cast<uint64_t>(map_chain[row]);
  while (ld_acquire_sys64(delta + kDeltaTag) != want) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0) __trap();  // the host never published it
    __nanosleep(256);
  }
  if (static_cast<uint64_t>(map_applied[row]) == want) return;
  const uint32_t n = ld_relaxed_sys<uint32_t>(delta + kDeltaCount);
  if (n > static_cast<uint32_t>(kDeltaMaxEntries)) __trap();
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t expert = ld_relaxed_sys<int32_t>(delta + kDeltaEntries + 8 * i);
    const int32_t slot = ld_relaxed_sys<int32_t>(delta + kDeltaEntries + 8 * i + 4);
    if (expert < 0 || expert >= experts || slot < -1 || slot >= static_cast<int32_t>(row_capacity)) __trap();
    ram_slot_row[expert] = slot;
  }
  for (int k = 0; k < kLeaseLanes; ++k)
    staging_row[k] = ld_relaxed_sys<int32_t>(delta + kDeltaStaging + 4 * k);
  map_applied[row] = static_cast<int64_t>(want);
}

// ram_slot_map.type_lanes, transcribed: each lane's kind and source slot. A hit takes its RAM slot, the m-th miss
// the m-th staging slot; the CPU takes the last split[n] of the n eligible lanes in plan order. Traps where the
// reference raises (a wider plan, a repeated expert, a miss with no staging slot, a split entry above n).
SGL_DEVICE void type_lanes(
    const int64_t* planned, int64_t count, int64_t experts, const int32_t* ram_slot_row, const int32_t* staging_row,
    const uint8_t* split, bool host_lanes, bool hit_copy_ce, bool cpu_on, bool cpu_misses, bool ce_ok, bool cpu_ok,
    const int32_t* dst, int32_t dst_rows, uint32_t row_capacity, TypedLanes& out) {
  if (count > kMaxIds) __trap();
  bool hit[kMaxIds];
  bool eligible[kMaxIds];
  int m = 0;
  int n = 0;
  for (int64_t j = 0; j < count; ++j) {
    const int64_t expert = planned[j];
    if (expert < 0 || expert >= experts) __trap();
    for (int64_t i = 0; i < j; ++i)
      if (planned[i] == expert) __trap();
    const int32_t slot = ram_slot_row[expert];
    hit[j] = slot >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(slot) >= row_capacity) __trap();
      out.slot[j] = slot;
    } else {
      if (m >= kLeaseLanes || staging_row[m] < 0) __trap();
      out.slot[j] = staging_row[m++];
    }
    eligible[j] = host_lanes && cpu_on && cpu_ok && (hit[j] || cpu_misses);
    n += eligible[j] ? 1 : 0;
  }
  int take = n > 0 ? ld_relaxed_sys<int32_t>(split + 4 * n) : 0;
  if (take < 0 || take > n) __trap();
  const bool copy_ok = host_lanes && hit_copy_ce && ce_ok;
  for (int64_t j = count - 1; j >= 0; --j) {
    const bool cpu = take > 0 && eligible[j];
    if (cpu) --take;
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? kKindHitCpu : kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < dst_rows ? kKindHitCopy : kKindHitSm;
    } else {
      kind = kKindMissGpu;
    }
    out.kind[j] = kind;
  }
}

}  // namespace device::expert_stream
}  // namespace sglang
