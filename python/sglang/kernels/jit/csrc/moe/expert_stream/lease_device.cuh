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

SGL_DEVICE uint64_t tagged_word(uint64_t tag, uint64_t generation) {
  return (tag << 56) | generation;
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

SGL_DEVICE void write_record(
    uint8_t* record, uint32_t seq, int64_t row, const int32_t* protect, int protect_count, uint32_t armed) {
  // Seqlock writer: an unarmed record may lap the ring before the service reads it, so a half-rewritten record must
  // never pass read_record's seq re-check. seq = 0, fence, payload, then seq with a release.
  st_relaxed_sys<uint32_t>(record + kRecSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint16_t>(record + kRecRow, static_cast<uint16_t>(row));
  st_relaxed_sys<uint16_t>(record + kRecProtectCount, static_cast<uint16_t>(protect_count));
  st_relaxed_sys<uint16_t>(record + kRecArmed, static_cast<uint16_t>(armed));
  for (int i = 0; i < kMaxIds; ++i)
    st_relaxed_sys<int32_t>(record + kRecProtect + 4 * i, i < protect_count ? protect[i] : -1);
  st_release_sys(record + kRecSeq, seq);
}

// One lane's RowResult of generation G, read with the acquire that publishes it: `host_slot` is valid only when the
// tag is not zero. The host stores the payload before the ready word's release and rewrites a lane only for a
// request G + 16, which cannot be granted before G's Done retired its leases (LEASE_PROTOCOL.md, "Reuse"): so one
// acquire and a payload load after it are the whole read, with no seqlock re-check.
struct LaneRead {
  uint64_t tag;  // 0: not (yet) published for G
  int32_t host_slot;
};

SGL_DEVICE LaneRead lane_read(const uint8_t* result, uint64_t generation, uint32_t capacity) {
  const uint64_t ready = ld_acquire_sys64(result + kLeaseRrReady);
  if ((ready & kGenerationMask) != generation) return LaneRead{0, 0};
  const int32_t host_slot = ld_relaxed_sys<int32_t>(result + kLeaseRrHostSlot);  // ordered after the acquire
  if (host_slot < 0 || static_cast<uint32_t>(host_slot) >= capacity) __trap();  // a slot outside the row's slabs
  return LaneRead{ready >> 56, host_slot};
}

// A CPU lane is the copy thread's to complete as well: every stage treats it as COPYING.
SGL_DEVICE bool copy_owned(uint64_t tag) {
  return tag == kLeaseTagCopying || tag == kLeaseTagCpu;
}

}  // namespace device::expert_stream
}  // namespace sglang
