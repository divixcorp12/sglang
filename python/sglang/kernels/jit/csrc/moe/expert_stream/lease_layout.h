// Wire layout shared by the device kernels and the host service: the request page, the completion block and the map
// delta block.
//
//   Request page      device-written, host-read: a ring of demand records, one per posted request
//   Completion block  host-written, device-read: PieceMask, CopyDone, the copy gate, the CPU split table
//   Map delta block   host-written, device-read: one delta record per row, after the completion block
//
// Every offset is a byte offset. LeaseLayout<NumLanes, NumNodes> computes them; python/sglang/kernels/ops/moe/
// expert_lease_block.py mirrors them as wire_layout, and test_expert_stream_lease_layout checks the two agree.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Wire (v2)".
#pragma once

#include "lease_channel_layout.h"
#include <cstdint>

// The lane count of this build: every JIT build of the device kernels and the host passes -DSGLANG_EXPERT_STREAM_LANES.
#ifndef SGLANG_EXPERT_STREAM_LANES
#define SGLANG_EXPERT_STREAM_LANES 8
#endif

// The NUMA node count of this build: every JIT build of the device kernels and the host passes
// -DSGLANG_EXPERT_STREAM_NODES alongside the lanes.
#ifndef SGLANG_EXPERT_STREAM_NODES
#define SGLANG_EXPERT_STREAM_NODES 1
#endif

namespace sglang::expert_stream::wire {

constexpr int64_t wire_round_up(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

template <int NumLanes, int NumNodes = 1>
struct LeaseLayout {
  static_assert(1 <= NumLanes && NumLanes <= 32, "a record carries 1..32 lanes");
  static_assert(NumNodes >= 1, "at least one NUMA node");
  // Lane arrays are i16, so a multiple of 8 lanes is whole 16-byte vector loads and stores.
  static constexpr int kLanes = static_cast<int>(wire_round_up(NumLanes, 8));
  static constexpr int kNodes = NumNodes;

  // The node whose group serves `expert`: its staging list, its CPU split and its slots. The one home rule, so a
  // popularity table can replace it here without touching a caller.
  static constexpr int home(int64_t expert) {
    return static_cast<int>(expert % kNodes);
  }

  // ---- Request page: device-written, host-read ----
  static constexpr int64_t kDemandHead = 0;    // u32: the last posted seq, stored with a release
  static constexpr int64_t kDemandRing = 128;  // a 128-byte block of its own
  static constexpr uint32_t kDemandRecords = 16;
  static constexpr int64_t kRecSeq = 0;    // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
  static constexpr int64_t kRecRow = 4;    // u16
  static constexpr int64_t kRecCounts = 6; // u8: see kPackedCounts
  static constexpr int64_t kRecFlags = 7;  // u8
  static constexpr uint32_t kRecFlagCaptured = 1;
  static constexpr int64_t kRecChain = 8;  // u64: the row's map-chain number, 0 when no lane misses
  static constexpr int64_t kRecEpoch = 16; // u32: the device's epoch, so G = epoch << 32 | seq
  static constexpr int64_t kRecKinds = 20; // u32[kKindWords]: lane j's kind in word j / 8, bits 4(j % 8)..+3
  static constexpr int kKindWords = kLanes / 8;
  // v2 packs lanes (bits 0-3) and protect ids (bits 4-7) into kRecCounts; wider records hold the lane count there and
  // the protect count in its own byte.
  static constexpr bool kPackedCounts = kLanes == 8;
  static constexpr int64_t kRecProtectCount = kRecKinds + 4 * kKindWords;  // u8, unpacked counts only
  static constexpr int64_t kRecHeaderBytes = wire_round_up(kRecProtectCount + (kPackedCounts ? 0 : 1), 16);
  static constexpr int64_t kRecProtect = kRecHeaderBytes;         // i16[kLanes]
  static constexpr int64_t kRecLaneExpert = kRecProtect + 2 * kLanes;   // i16[kLanes]
  // i16[kLanes]: a hit's RAM slot, a miss's staging slot
  static constexpr int64_t kRecLaneSlot = kRecLaneExpert + 2 * kLanes;
  static constexpr int64_t kRecLaneDst = kRecLaneSlot + 2 * kLanes;     // i16[kLanes]: the VRAM destination slot
  static constexpr int64_t kRecLaneWeight = kRecLaneDst + 2 * kLanes;   // f32[kLanes]
  static constexpr int64_t kRecPayloadEnd = kRecLaneWeight + 4 * kLanes;
  static constexpr int64_t kRecordBytes = wire_round_up(kRecPayloadEnd, 128);  // whole L2 adjacent-line pairs
  static constexpr int64_t kRecIdMax = 32767;
  static constexpr int64_t kPageBytes = kDemandRing + kDemandRecords * kRecordBytes;
  static constexpr uint32_t kKindHitCopy = 1;
  static constexpr uint32_t kKindHitSm = 2;
  static constexpr uint32_t kKindHitCpu = 3;
  static constexpr uint32_t kKindMissGpu = 4;
  static constexpr uint32_t kKindMissCpu = 5;
  static constexpr int64_t kHotHeaderBytes = 8;
  static constexpr int64_t kHotAlignment = 64;
  static constexpr uint32_t kHotRecords = kDemandRecords;

  // ---- Completion block: host-written, device-read, 4096-byte aligned ----
  static constexpr int64_t kLeaseBlockAlign = 4096;
  static constexpr int64_t kLeasePieceMask = 0;  // u64[kDemandRecords][kLanes], one 128-byte line each
  static constexpr int64_t kLeasePieceMaskLineBytes = 128;
  static constexpr int64_t kLeaseCopyDone = kLeasePieceMask + kDemandRecords * kLanes * kLeasePieceMaskLineBytes;
  static constexpr int64_t kLeaseCopyDoneBytes = 8;
  static constexpr int64_t kLeaseCopyGate = kLeaseCopyDone + 128;
  static constexpr uint32_t kLeaseGateClosed = channel::kGateClosed;  // the lease channel's gate encoding
  static constexpr uint32_t kLeaseGateOpen = channel::kGateOpen;
  static constexpr uint32_t kLeaseGateSeqShift = channel::kGateSeqShift;
  static constexpr uint32_t kLeaseGateSeqMask = channel::kGateSeqMask;
  static constexpr int64_t kCopyArmed = kLeaseCopyGate + 128;  // u32
  static constexpr int64_t kSplit = kCopyArmed + 128;  // i32[kNodes][kSplitStride / 4]: CPU lanes per n eligible lanes
  static constexpr int64_t kSplitStride = wire_round_up(4 * (kLanes + 1), 16);  // a node's table, in 16-byte loads
  static constexpr int64_t kLeaseBlockBytes = wire_round_up(kSplit + kNodes * kSplitStride, kLeaseBlockAlign);

  // ---- Map delta block: one record per row after the completion block ----
  static constexpr int64_t kDeltaBase = kLeaseBlockBytes;
  static constexpr int64_t kDeltaTag = 0;      // u64, stored last with a release
  static constexpr int64_t kDeltaCount = 8;    // u32
  static constexpr int64_t kDeltaStaging = 16; // i16[kNodes][kLanes], -1 past the list
  static constexpr int64_t kDeltaEntries = wire_round_up(kDeltaStaging + 2 * kNodes * kLanes, 16);
  static constexpr int64_t kDeltaMaxEntries = 2 * kLanes;  // an insert and an eviction per miss
  static constexpr int64_t kDeltaStride = wire_round_up(kDeltaEntries + 4 * kDeltaMaxEntries, 256);
  static_assert(kRecEpoch % 16 == 0 && kRecProtect % 16 == 0 && kRecLaneExpert % 16 == 0 && kRecLaneSlot % 16 == 0 &&
                    kRecLaneDst % 16 == 0 && kRecLaneWeight % 16 == 0 && kRecordBytes % 16 == 0,
                "the record's 16-byte stores are aligned");
  static_assert(kSplit + kNodes * kSplitStride <= kLeaseBlockBytes, "completion block");
  static_assert(kLeaseBlockBytes % kLeaseBlockAlign == 0, "the block is whole pages");
  static_assert(kDeltaEntries + 4 * kDeltaMaxEntries <= kDeltaStride, "delta record");
};

using Wire = LeaseLayout<SGLANG_EXPERT_STREAM_LANES, SGLANG_EXPERT_STREAM_NODES>;

// The target is the lease channel's first client (LEASE_PROTOCOL.md, "The lease channel"): its request page and its
// completion block's CopyDone words and gate.
template <class L>
using TargetChannelOf = channel::ChannelSpec<L::kDemandHead, L::kDemandRing, L::kDemandRecords, L::kRecordBytes,
                                             L::kLeaseCopyDone, L::kLeaseCopyGate>;
using TargetChannel = TargetChannelOf<Wire>;
static_assert(TargetChannel::kDone == Wire::kLeaseCopyDone && TargetChannel::kGate == Wire::kLeaseCopyGate, "channel");
static_assert(TargetChannel::kDoneBytes == Wire::kLeaseCopyDoneBytes && Wire::kRecSeq == 0, "the seq word is first");

using V2 = LeaseLayout<8, 1>;
static_assert(V2::kRecordBytes == 128 && V2::kPageBytes == 2176 && V2::kRecLaneWeight == 96, "v2 request page");
static_assert(V2::kLeaseCopyDone == 0x4000 && V2::kSplit == 16768 && V2::kLeaseBlockBytes == 20480, "v2 block");
static_assert(V2::kDeltaEntries == 32 && V2::kDeltaMaxEntries == 16 && V2::kDeltaStride == 256, "v2 delta");
static_assert(LeaseLayout<8, 2>::home(7) == 1 && V2::home(7) == 0, "home is expert % nodes");
static_assert(Wire::kSplit % 16 == 0 && Wire::kRecProtect % 16 == 0 && Wire::kRecordBytes % 128 == 0, "alignment");

}  // namespace sglang::expert_stream::wire
