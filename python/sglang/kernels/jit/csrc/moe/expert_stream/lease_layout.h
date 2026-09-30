// Wire layout of the expert-stream request page and lease block (analysis/dsv41-drive/LEASE_PROTOCOL.md).
// Mirrored by ops/moe/expert_stream_transport.py and ops/moe/expert_lease_block.py; test_exl3_ram_miss_device_args
// checks both. Only `constexpr <type> kName = <integer expression>;` lines: that test parses them.
#pragma once

#include <cstdint>

namespace sglang::expert_stream::wire {

// ---- Request page: the demand ring ----
constexpr int64_t kDemandHead = 0;  // u32, device: the last posted seq
constexpr int64_t kDemandDone = 4;  // u32, host: every request up to this seq is served (there is no failed state)
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kRecordBytes = 128;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;           // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
constexpr int64_t kRecRow = 4;           // u16
constexpr int64_t kRecProtectCount = 6;  // u16
constexpr int64_t kRecArmed = 8;         // u16: 1 when the request has lanes, so LaneRequest[idx] is its own
constexpr int64_t kRecProtect = 16;      // i32[kMaxIds]: the layer's routed experts
constexpr int64_t kPageBytes = kDemandRing + kDemandRecords * kRecordBytes;
// The hot page (GPU hot mode), a separate pinned page: per record {u32 seq; u32 reserved}, then the hot bitmap.
constexpr int64_t kHotHeaderBytes = 8;
constexpr int64_t kHotAlignment = 64;
constexpr uint32_t kHotRecords = kDemandRecords;

// ---- Lease block: fixed areas, 4096-byte aligned ----
constexpr int64_t kLeaseRing = 16;  // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;  // == kMaxIds
constexpr int64_t kLeaseBlockAlign = 4096;
// Area S, host-written: RowResult[kLeaseRing][kLeaseLanes]. The payload first, the ready word last with a release.
constexpr int64_t kLeaseRowResult = 0;
constexpr int64_t kLeaseRowResultBytes = 16;
constexpr int64_t kLeaseRrReady = 0;     // u64: tag << 56 | G, G the 56-bit request generation (epoch << 32 | seq)
constexpr int64_t kLeaseRrHostSlot = 8;  // i32: the leased source slot
constexpr uint64_t kLeaseTagReady = 1;    // every byte of the slot is final
constexpr uint64_t kLeaseTagLoading = 2;  // being read: the lane's PieceMask word says which pieces are final
constexpr uint64_t kLeaseTagCopying = 3;  // the copy thread writes the lane's destination slot (CopyDone)
constexpr uint64_t kLeaseTagCpu = 4;      // the CPU expert thread computes the lane; nothing writes its destination
// Area P, host-written: PieceMask[kLeaseRing][kLeaseLanes], u64 G << 8 | 8 piece bits, one 128-byte line each so a
// lane's poll never shares a line with another lane.
constexpr int64_t kLeasePieceMask = 4096;
constexpr int64_t kLeasePieceMaskLineBytes = 128;
// Area C, host-written: CopyDone[kLeaseRing] (u64 G: every COPYING and CPU lane of G completed), then the copy wait's
// gate on its own line: (seq & kLeaseGateSeqMask) << kLeaseGateSeqShift | 1, with bit 31 set while closed. CW
// closes it; CW or the copy thread opens it; the decode stream waits on it with cuStreamWaitValue32 GEQ open.
constexpr int64_t kLeaseCopyDone = kLeasePieceMask + kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
constexpr int64_t kLeaseCopyDoneBytes = 8;
constexpr int64_t kLeaseCopyGate = kLeaseCopyDone + 128;
constexpr uint32_t kLeaseGateClosed = 0x80000001u;  // (int32_t)(gate - 1) < 0: the cyclic GEQ blocks
constexpr uint32_t kLeaseGateOpen = 1;
constexpr uint32_t kLeaseGateSeqShift = 2;
constexpr uint32_t kLeaseGateSeqMask = 0x1FFFFFFF;
// Area D, device-written: LaneRequest[kLeaseRing], then Done[kLeaseRing].
constexpr int64_t kLeaseLaneRequest = 24576;
constexpr int64_t kLeaseLaneRequestBytes = 128;
constexpr int64_t kLeaseLrGen = 0;     // u64 G, stored last with a release; 0 while the payload is rewritten
constexpr int64_t kLeaseLrCount = 8;   // u32
constexpr int64_t kLeaseLrFlags = 12;  // u32
// Posted from a captured graph: the service may publish hit lanes COPYING, and CPU lanes when it has CPU experts
// (the post then staged the layer's input row for them).
constexpr uint32_t kLeaseLrFlagCaptured = 1;
constexpr int64_t kLeaseLrExpert = 16;  // i32[kLeaseLanes]
constexpr int64_t kLeaseLrDst = 48;     // i32[kLeaseLanes]: the plan's destination slot, -1 past the plan
constexpr int64_t kLeaseLrWeight = 80;  // f32[kLeaseLanes]: the lane expert's routing weight, 0 past the plan
// u64 G, written by CW: no kernel of G reads a leased slot of G after it.
constexpr int64_t kLeaseDone = kLeaseLaneRequest + kLeaseRing * kLeaseLaneRequestBytes;
constexpr int64_t kLeaseDoneBytes = 8;
constexpr int64_t kLeaseBlockBytes = 28672;

static_assert(kLeaseRowResult + kLeaseRing * kLeaseLanes * kLeaseRowResultBytes <= kLeasePieceMask, "area S");
static_assert(kLeaseCopyGate + 4 <= kLeaseLaneRequest, "area C");
static_assert(kLeaseLrWeight + 4 * kLeaseLanes <= kLeaseLaneRequestBytes, "LaneRequest: the payload fits one record");
static_assert(kLeaseDone + kLeaseRing * kLeaseDoneBytes <= kLeaseBlockBytes, "area D");
static_assert(kLeaseBlockBytes % kLeaseBlockAlign == 0, "the block is whole pages");

}  // namespace sglang::expert_stream::wire
