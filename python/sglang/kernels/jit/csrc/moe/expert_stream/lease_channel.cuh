// The device half of the lease channel (lease_channel_layout.h). Bodies moved from lease_kernels.cuh (advance,
// publish_head), lease_device.cuh (the record seqlock, the generation) and row_copy_kernels.cuh (close_gate,
// commit_or_trap).
#pragma once

#include <sgl_kernel/utils.cuh>

#include <cuda/atomic>

#include "lease_channel_layout.h"
#include "lease_primitives.cuh"
#include <cstdint>

namespace sglang {
namespace device::expert_stream::channel {

using namespace ::sglang::expert_stream::channel;

// The device's own words (`state`, int32, device memory): never on the wire. Every client's state starts with these.
constexpr int kPosted = 0;        // the last posted seq
constexpr int kPending = 1;       // the seq of the request the client's wait serves, 0 when nothing is pending
constexpr int kEpoch = 2;         // times seq32 wrapped; G = epoch << 32 | seq
constexpr int kPendingEpoch = 3;  // the epoch of the request kPending names

// The next seq, never 0; a wrap bumps the epoch, so G = epoch << 32 | seq never repeats.
SGL_DEVICE uint32_t advance(int32_t* state) {
  uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
  if (seq == 0) {
    seq = 1;
    state[kEpoch] += 1;
  }
  state[kPosted] = static_cast<int32_t>(seq);
  return seq;
}

SGL_DEVICE uint64_t generation(uint32_t seq, uint32_t epoch) {
  return (static_cast<uint64_t>(epoch) << 32) | seq;
}

template <class S>
SGL_DEVICE int64_t ring_index(uint32_t seq) {
  return static_cast<int64_t>((seq - 1u) % S::kRecords);
}

template <class S>
SGL_DEVICE uint8_t* record_at(uint8_t* page, uint32_t seq) {
  return page + S::kRing + ring_index<S>(seq) * S::kRecordBytes;
}

// The seqlock's open: seq = 0, then a release fence, so no payload store passes it. The seq word is the record's
// first.
template <class S>
SGL_DEVICE void begin_record(uint8_t* record) {
  st_relaxed_sys<uint32_t>(record, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
}

// The seqlock's close: the seq, stored last with a release.
template <class S>
SGL_DEVICE void end_record(uint8_t* record, uint32_t seq) {
  st_release_sys(record, seq);
}

// A release orders every earlier store of this thread: the record (and any client area) comes first.
template <class S>
SGL_DEVICE void publish_head(uint8_t* page, uint32_t seq) {
  st_release_sys(page + S::kHead, seq);
}

// Closes the gate for G, then re-checks done[G] after a system fence and opens the gate itself if the host already
// completed. Dekker with the host's complete() (done store, fence, gate load): this close is ordered before the done
// load, so one side always sees the other's store and opens the gate. Both open with the same word, so opening twice
// is harmless.
template <class S>
SGL_DEVICE void close_gate(uint8_t* lease, uint32_t seq, uint64_t gen) {
  uint8_t* const gate = lease + S::kGate;
  st_relaxed_sys<uint32_t>(gate, gate_word(seq, kGateClosed));
  __threadfence_system();
  if (ld_acquire_sys64(lease + S::kDone + ring_index<S>(seq) * S::kDoneBytes) == gen) {
    st_release_sys(gate, gate_word(seq, kGateOpen));
  }
}

// After the stream's wait: the gate is only the wake-up, done[G] is what commits. Its acquire orders the host's
// results before every later read of them. Only a teardown opens a gate without done[G]: the host is gone, and its
// results may not have landed.
template <class S>
SGL_DEVICE void commit_or_trap(const uint8_t* lease, uint32_t seq, uint64_t gen) {
  if (ld_acquire_sys64(lease + S::kDone + ring_index<S>(seq) * S::kDoneBytes) != gen) __trap();
}

}  // namespace device::expert_stream::channel
}  // namespace sglang
