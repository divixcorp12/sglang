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

SGL_DEVICE void st_relaxed_sys_v2(uint8_t* address, uint32_t x, uint32_t y) {
  asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(address), "r"(x), "r"(y) : "memory");
}

SGL_DEVICE void st_relaxed_sys_v4(uint8_t* address, uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}

SGL_DEVICE uint4 ld_relaxed_sys_v4(const uint8_t* address) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(address) : "memory");
  return v;
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

// One request's typed lanes, as the post writes them into its record (lease_layout.h kRecLaneSlot, kRecKinds).
struct TypedLanes {
  int32_t slot[kMaxIds];  // the RAM slot of a hit, the staging slot of a miss
  uint8_t kind[kMaxIds];  // kKind*
};

SGL_DEVICE bool is_cpu_kind(uint32_t kind) {
  return kind == kKindHitCpu || kind == kKindMissCpu;
}

// One request's record as the post knows it; write_record narrows and packs it into the wire layout.
struct RecordFields {
  int64_t row;
  uint32_t flags;           // kRecFlag*
  uint64_t chain;           // the row's map-chain number, 0 when no lane misses
  uint32_t epoch;           // so G = epoch << 32 | seq
  const int32_t* protect;   // protect_count routed experts
  int protect_count;
  int64_t count;            // lanes
  const int64_t* planned;   // count lane experts
  const int32_t* dst;       // count VRAM destination slots
  const float* weight;      // count routing weights
  const TypedLanes* lanes;  // count kinds and source slots
};

// Two record ids as one i16 pair, -1 for none; traps on an id the record cannot carry rather than wrap it.
SGL_DEVICE uint32_t pack_ids(int64_t lo, int64_t hi) {
  if (lo < -1 || lo > kRecIdMax || hi < -1 || hi > kRecIdMax) __trap();
  return (static_cast<uint32_t>(lo) & 0xFFFFu) | (static_cast<uint32_t>(hi) << 16);
}

// Seqlock writer: a record of SM hits only may lap the ring before the service reads it, so a half-rewritten record
// must never pass read_record's seq re-check. seq = 0, fence, payload, then seq with a release.
SGL_DEVICE void write_record(uint8_t* record, uint32_t seq, const RecordFields& f) {
  if (f.count > kMaxIds || f.protect_count > kMaxIds) __trap();
  uint32_t protect_w[kMaxIds / 2], expert_w[kMaxIds / 2], slot_w[kMaxIds / 2], dst_w[kMaxIds / 2];
  uint32_t weight_w[kMaxIds];
  uint32_t kinds = 0;
#pragma unroll
  for (int i = 0; i < kMaxIds; i += 2) {
    const bool a = i < f.count, b = i + 1 < f.count;
    protect_w[i / 2] =
        pack_ids(i < f.protect_count ? f.protect[i] : -1, i + 1 < f.protect_count ? f.protect[i + 1] : -1);
    expert_w[i / 2] = pack_ids(a ? f.planned[i] : -1, b ? f.planned[i + 1] : -1);
    slot_w[i / 2] = pack_ids(a ? f.lanes->slot[i] : -1, b ? f.lanes->slot[i + 1] : -1);
    dst_w[i / 2] = pack_ids(a ? f.dst[i] : -1, b ? f.dst[i + 1] : -1);
  }
#pragma unroll
  for (int i = 0; i < kMaxIds; ++i) {
    const bool used = i < f.count;
    weight_w[i] = __float_as_uint(used ? f.weight[i] : 0.0f);
    kinds |= (used ? static_cast<uint32_t>(f.lanes->kind[i]) : 0u) << (4 * i);
  }
  const uint32_t counts = static_cast<uint32_t>(f.count) | static_cast<uint32_t>(f.protect_count) << 4;
  const uint32_t head = (static_cast<uint32_t>(f.row) & 0xFFFFu) | counts << 16 | (f.flags & 0xFFu) << 24;
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(record + kRecRow, head);
  st_relaxed_sys_v2(
      record + kRecChain, static_cast<uint32_t>(f.chain & 0xFFFFFFFFull), static_cast<uint32_t>(f.chain >> 32));
  st_relaxed_sys_v4(record + kRecEpoch, f.epoch, kinds, 0u, 0u);
  st_relaxed_sys_v4(record + kRecProtect, protect_w[0], protect_w[1], protect_w[2], protect_w[3]);
  st_relaxed_sys_v4(record + kRecLaneExpert, expert_w[0], expert_w[1], expert_w[2], expert_w[3]);
  st_relaxed_sys_v4(record + kRecLaneSlot, slot_w[0], slot_w[1], slot_w[2], slot_w[3]);
  st_relaxed_sys_v4(record + kRecLaneDst, dst_w[0], dst_w[1], dst_w[2], dst_w[3]);
  st_relaxed_sys_v4(record + kRecLaneWeight, weight_w[0], weight_w[1], weight_w[2], weight_w[3]);
  st_relaxed_sys_v4(record + kRecLaneWeight + 16, weight_w[4], weight_w[5], weight_w[6], weight_w[7]);
  st_release_sys(record + kRecSeq, seq);
}

// One row of the device's map bank (ExpertStreamDevice.map_bank), device memory.
struct RowMap {
  int32_t* ram_slot;     // [experts]: the expert's RAM slot, -1 when not resident
  int32_t* staging;      // [kLeaseLanes]: the row's staging slots
  int64_t* map_chain;    // the row's chain word
  int64_t* map_applied;  // the chain number of the row's last applied delta
  int64_t experts;
  uint32_t row_capacity;
};

// A row's map delta in registers: load_map_delta issues every load before it uses any, so their round trips overlap.
struct MapDelta {
  uint32_t count;
  int32_t staging[kLeaseLanes];
  int32_t expert[kDeltaMaxEntries];
  int32_t slot[kDeltaMaxEntries];
};

// Waits, bounded by `deadline`, until the host has published the delta that follows the row's last map chain; true
// when it is not applied yet. The host publishes a chain's delta before it reads that chain's misses, so the wait is
// taken only when the host fell a whole token behind.
SGL_DEVICE bool await_map_delta(const uint8_t* delta, const RowMap& map, uint64_t deadline) {
  const uint64_t want = static_cast<uint64_t>(*map.map_chain);
  while (ld_acquire_sys64(delta + kDeltaTag) != want) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0) __trap();  // the host never published it
    __nanosleep(256);
  }
  return static_cast<uint64_t>(*map.map_applied) != want;
}

// The delta's payload; only after await_map_delta's acquire of its tag.
SGL_DEVICE MapDelta load_map_delta(const uint8_t* delta) {
  static_assert(kDeltaStaging % 16 == 0 && kDeltaEntries % 16 == 0, "16-byte delta loads");
  static_assert(kLeaseLanes % 4 == 0 && kDeltaMaxEntries % 2 == 0, "whole 16-byte loads");
  constexpr int kStagingLoads = kLeaseLanes / 4;
  constexpr int kEntryLoads = kDeltaMaxEntries / 2;  // {expert, slot} pairs, two a load
  uint4 v[kStagingLoads + kEntryLoads];
  MapDelta d;
  d.count = ld_relaxed_sys<uint32_t>(delta + kDeltaCount);
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i)
    v[i] = ld_relaxed_sys_v4(delta + kDeltaStaging + 16 * i);
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i)
    v[kStagingLoads + i] = ld_relaxed_sys_v4(delta + kDeltaEntries + 16 * i);
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i) {
    d.staging[4 * i] = static_cast<int32_t>(v[i].x);
    d.staging[4 * i + 1] = static_cast<int32_t>(v[i].y);
    d.staging[4 * i + 2] = static_cast<int32_t>(v[i].z);
    d.staging[4 * i + 3] = static_cast<int32_t>(v[i].w);
  }
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i) {
    const uint4 e = v[kStagingLoads + i];
    d.expert[2 * i] = static_cast<int32_t>(e.x);
    d.slot[2 * i] = static_cast<int32_t>(e.y);
    d.expert[2 * i + 1] = static_cast<int32_t>(e.z);
    d.slot[2 * i + 1] = static_cast<int32_t>(e.w);
  }
  return d;
}

// Validates a loaded delta and applies it to the row once: map_applied takes the row's chain number.
SGL_DEVICE void apply_map_delta(const MapDelta& d, const RowMap& map) {
  if (d.count > static_cast<uint32_t>(kDeltaMaxEntries)) __trap();
#pragma unroll
  for (int i = 0; i < kDeltaMaxEntries; ++i) {
    if (static_cast<uint32_t>(i) < d.count) {
      const int32_t expert = d.expert[i];
      const int32_t slot = d.slot[i];
      if (expert < 0 || expert >= map.experts || slot < -1 || slot >= static_cast<int32_t>(map.row_capacity)) __trap();
      map.ram_slot[expert] = slot;
    }
  }
#pragma unroll
  for (int k = 0; k < kLeaseLanes; ++k)
    map.staging[k] = d.staging[k];
  *map.map_applied = *map.map_chain;
}

// The plan as the post reads it: count planned experts and their VRAM destination slots, device memory.
struct LanePlan {
  const int64_t* planned;
  const int32_t* dst;
  int64_t count;
};

// Everything besides the map that decides a lane's kind, loaded before type_lanes runs.
struct LanePolicy {
  bool host_lanes;   // a captured post while the copy engine is armed (kCopyArmed)
  bool hit_copy_ce;  // SGLANG_DSV41_RAM_HIT_COPY=ce
  bool cpu_on;
  bool cpu_misses;
  bool ce_ok;   // the row's copy table is set
  bool cpu_ok;  // the row's CPU layer is registered
  int32_t dst_rows;
  int32_t split[kLeaseLanes + 1];  // kSplit: CPU lanes per n eligible lanes
};

// kSplit's table in three 16-byte loads; the last three words loaded lie past the table and are dropped.
SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[kLeaseLanes + 1]) {
  static_assert(kSplit % 16 == 0 && kSplit + 48 <= kLeaseBlockBytes && kLeaseLanes + 1 <= 12, "three loads");
  uint4 v[3];
#pragma unroll
  for (int i = 0; i < 3; ++i)
    v[i] = ld_relaxed_sys_v4(split + 16 * i);
  const uint32_t words[12] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w,
                              v[2].x, v[2].y, v[2].z, v[2].w};
#pragma unroll
  for (int n = 0; n <= kLeaseLanes; ++n)
    out[n] = static_cast<int32_t>(words[n]);
}

// ram_slot_map.type_lanes, transcribed: each lane's kind and source slot. A hit takes its RAM slot, the m-th miss
// the m-th staging slot; the CPU takes the last split[n] of the n eligible lanes in plan order. Traps where the
// reference raises (a wider plan, a repeated expert, a miss with no staging slot, a split entry above n). It reads no
// host memory: the caller loads the split table into the policy.
SGL_DEVICE void type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
  if (plan.count > kMaxIds) __trap();
  int64_t expert[kMaxIds];
  int32_t dst[kMaxIds];
  int32_t ram[kMaxIds];
  int32_t staging[kLeaseLanes];
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j) {
    expert[j] = j < plan.count ? plan.planned[j] : 0;
    dst[j] = j < plan.count ? plan.dst[j] : -1;
  }
#pragma unroll
  for (int k = 0; k < kLeaseLanes; ++k)
    staging[k] = map.staging[k];
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j)
    if (j < plan.count && (expert[j] < 0 || expert[j] >= map.experts)) __trap();
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j)
    ram[j] = j < plan.count ? map.ram_slot[expert[j]] : -1;
  bool hit[kMaxIds];
  bool eligible[kMaxIds];
  int m = 0;
  int n = 0;
#pragma unroll
  for (int j = 0; j < kMaxIds; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else {
      if (m >= kLeaseLanes || staging[m] < 0) __trap();
      out.slot[j] = staging[m++];
    }
    eligible[j] = policy.host_lanes && policy.cpu_on && policy.cpu_ok && (hit[j] || policy.cpu_misses);
    n += eligible[j] ? 1 : 0;
  }
  int take = n > 0 ? policy.split[n] : 0;
  if (take < 0 || take > n) __trap();
  const bool copy_ok = policy.host_lanes && policy.hit_copy_ce && policy.ce_ok;
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const bool cpu = take > 0 && eligible[j];
    if (cpu) --take;
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? kKindHitCpu : kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < policy.dst_rows ? kKindHitCopy : kKindHitSm;
    } else {
      kind = kKindMissGpu;
    }
    out.kind[j] = kind;
  }
}

}  // namespace device::expert_stream
}  // namespace sglang
