// Device stamps (EXPERT_STREAM_DEVICE_STAMPS, the traced device module only): %globaltimer at fixed points of each
// record's chain, written into a pinned host ring the service dumps at shutdown. Without the define every stamp
// compiles to nothing and the module exports no setter.
#pragma once

#include "lease_device.cuh"
#include <cstdint>

namespace sglang {
namespace device::expert_stream {

// One record's row of the ring: its seq, then one %globaltimer ns per point (0: not reached).
enum StampPoint : int {
  kStampSeq = 0,
  kStampPostStart = 1,   // the post kernel began (the layer's MoE reached the lease)
  kStampPostEnd = 2,     // the post published the record
  kStampStreamStart = 3, // S began copying GPU misses' pieces
  kStampStreamEnd = 4,   // S copied the last piece
  kStampWaitStart = 5,   // CW closed the gate: the stream now waits for the copy thread
  kStampWaitEnd = 6,     // CC began: the stream saw the gate open
  kStampWords = 8,
};
// Rows in the ring, by seq; arbitrary, sized for a capture's few thousand records with room to spare.
constexpr uint32_t kStampRing = 65536;

#ifdef EXPERT_STREAM_DEVICE_STAMPS
// The ring, set once by set_device_stamps before any captured chain runs; null leaves every stamp a no-op.
static __device__ uint64_t* g_stamp_ring = nullptr;

SGL_DEVICE uint64_t* stamp_row(uint32_t seq) {
  uint64_t* const ring = g_stamp_ring;
  return ring == nullptr || seq == 0 ? nullptr : ring + static_cast<uint64_t>(seq % kStampRing) * kStampWords;
}

// The post opens the record's row: the seq, its two stamps, every later point cleared, so a lapped row never mixes.
SGL_DEVICE void stamp_post(uint32_t seq, uint64_t start) {
  uint64_t* const row = stamp_row(seq);
  if (row == nullptr) return;
  row[kStampSeq] = seq;
  row[kStampPostStart] = start;
  row[kStampPostEnd] = global_ns();
  for (int k = kStampStreamStart; k < kStampWords; ++k)
    row[k] = 0;
}

// A later point of the chain, kept only when the row is still this record's.
SGL_DEVICE void stamp_point(uint32_t seq, int point) {
  uint64_t* const row = stamp_row(seq);
  if (row == nullptr || row[kStampSeq] != seq) return;
  row[point] = global_ns();
}
#else
SGL_DEVICE void stamp_post(uint32_t, uint64_t) {}
SGL_DEVICE void stamp_point(uint32_t, int) {}
#endif

}  // namespace device::expert_stream
}  // namespace sglang
