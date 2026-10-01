// Wire layout of the expert-stream request page, completion block and map delta block (analysis/dsv41-drive/LEASE_PROTOCOL.md).
// Mirrored by ops/moe/expert_stream_transport.py and ops/moe/expert_lease_block.py; test_exl3_ram_miss_device_args
// checks both. Only `constexpr <type> kName = <integer expression>;` lines: that test parses them.
#pragma once

#include <cstdint>

namespace sglang::expert_stream::wire {

// ---- Request page: device-written, host-read ----
constexpr int64_t kDemandHead = 0;  // u32: the last posted seq, stored with a release
constexpr int64_t kDemandRing = 64;
constexpr uint32_t kDemandRecords = 16;
constexpr int64_t kRecordBytes = 256;
constexpr int kMaxIds = 8;
constexpr int64_t kRecSeq = 0;            // u32 seqlock word: 0 while the payload is rewritten, the seq stored last
constexpr int64_t kRecRow = 4;            // u16
constexpr int64_t kRecCount = 6;          // u16: lanes
constexpr int64_t kRecFlags = 8;          // u32
constexpr uint32_t kRecFlagCaptured = 1;  // posted from a captured graph
constexpr int64_t kRecChain = 12;         // u32: low half of the row's map-chain number, 0 when no lane misses
constexpr int64_t kRecChainHi = 16;       // u32: high half
constexpr int64_t kRecEpoch = 24;         // u32: the device's epoch, so G = epoch << 32 | seq
constexpr int64_t kRecProtectCount = 20;  // u16
constexpr int64_t kRecProtect = 32;       // i32[kMaxIds]: every routed expert of the request
constexpr int64_t kRecLanes = 64;         // kMaxIds lanes of kLaneBytes
constexpr int64_t kLaneBytes = 16;
constexpr int64_t kLaneExpert = 0;        // i32
constexpr int64_t kLaneSlot = 4;          // i32: the RAM slot of a hit, the staging slot of a miss
constexpr int64_t kLaneDst = 8;           // i32: the VRAM destination slot
constexpr int64_t kLaneWeight = 12;       // f32: the lane expert's routing weight
constexpr int64_t kRecKinds = 192;        // u8[kMaxIds]: kKind*
constexpr int64_t kRecIdMax = 32767;      // the largest expert, slot or destination an i16 field carries
constexpr int64_t kPageBytes = kDemandRing + kDemandRecords * kRecordBytes;
// Lane kinds (ram_slot_map.LaneKind): what moves the bytes, and what the device waits on.
constexpr uint32_t kKindHitCopy = 1;  // the copy thread's DMA; CopyDone
constexpr uint32_t kKindHitSm = 2;    // C1; stream order
constexpr uint32_t kKindHitCpu = 3;   // the CPU, from the RAM slot; CopyDone
constexpr uint32_t kKindMissGpu = 4;  // NVMe into the staging slot, then S; PieceMask
constexpr uint32_t kKindMissCpu = 5;  // NVMe into the staging slot, then the CPU; CopyDone
// The hot page (GPU hot mode), a separate pinned page: per record {u32 seq; u32 reserved}, then the hot bitmap.
constexpr int64_t kHotHeaderBytes = 8;
constexpr int64_t kHotAlignment = 64;
constexpr uint32_t kHotRecords = kDemandRecords;

// ---- Completion block: host-written, device-read, 4096-byte aligned ----
constexpr int64_t kLeaseRing = 16;  // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;  // == kMaxIds
constexpr int64_t kLeaseBlockAlign = 4096;
// PieceMask[kLeaseRing][kLeaseLanes]: u64 G << 8 | 8 piece bits, one 128-byte line each so a lane's poll never shares
// a line with another lane.
constexpr int64_t kLeasePieceMask = 0;
constexpr int64_t kLeasePieceMaskLineBytes = 128;
// CopyDone[kLeaseRing] (u64 G: every kHitCopy, kHitCpu and kMissCpu lane of G completed), then the copy wait's gate on
// its own line: (seq & kLeaseGateSeqMask) << kLeaseGateSeqShift | 1, with bit 31 set while closed. CW closes it; CW or
// the copy thread opens it; the decode stream waits on it with cuStreamWaitValue32 GEQ open.
constexpr int64_t kLeaseCopyDone = kLeasePieceMask + kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
constexpr int64_t kLeaseCopyDoneBytes = 8;
constexpr int64_t kLeaseCopyGate = kLeaseCopyDone + 128;
constexpr uint32_t kLeaseGateClosed = 0x80000001u;  // (int32_t)(gate - 1) < 0: the cyclic GEQ blocks
constexpr uint32_t kLeaseGateOpen = 1;
constexpr uint32_t kLeaseGateSeqShift = 2;
constexpr uint32_t kLeaseGateSeqMask = 0x1FFFFFFF;
constexpr int64_t kCopyArmed = kLeaseCopyGate + 128;  // u32: 1 once the service armed its copy engine
constexpr int64_t kSplit = kCopyArmed + 128;          // i32[kLeaseLanes + 1]: CPU lanes per n eligible lanes
constexpr int64_t kLeaseBlockBytes = 20480;

// ---- Map delta block: host-written, device-read, one record per row, after the completion block in one allocation ----
constexpr int64_t kDeltaBase = kLeaseBlockBytes;
constexpr int64_t kDeltaStride = 256;
constexpr int64_t kDeltaTag = 0;       // u64: the map-chain number this delta follows, stored last with a release
constexpr int64_t kDeltaCount = 8;     // u32: entries used
constexpr int64_t kDeltaStaging = 16;  // i32[kLeaseLanes]: the row's staging slots after this delta, -1 past K
constexpr int64_t kDeltaEntries = 48;  // {i32 expert, i32 slot}[kDeltaMaxEntries]: ram_slot[expert] = slot, -1 unmaps
constexpr int64_t kDeltaMaxEntries = 16;

static_assert(kLeaseRing == kDemandRecords && kLeaseLanes == kMaxIds, "the completion block follows the ring");
static_assert(kRecLanes + kMaxIds * kLaneBytes <= kRecKinds, "record lanes");
static_assert(kRecKinds + kMaxIds <= kRecordBytes, "record");
static_assert(kSplit + 4 * (kLeaseLanes + 1) <= kLeaseBlockBytes, "completion block");
static_assert(kLeaseBlockBytes % kLeaseBlockAlign == 0, "the block is whole pages");
static_assert(kDeltaEntries + 8 * kDeltaMaxEntries <= kDeltaStride, "delta record");

}  // namespace sglang::expert_stream::wire
