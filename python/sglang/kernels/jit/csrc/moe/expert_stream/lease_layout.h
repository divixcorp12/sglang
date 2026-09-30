// Wire layout of the expert-stream request page, lease block and prefetch page (LEASE_PROTOCOL.md section 4).
// Mirrored by ops/moe/expert_stream_transport.py and ops/moe/expert_lease_block.py; test_exl3_ram_miss_device_args
// checks both. Only `constexpr <type> kName = <integer expression>;` lines: that test parses them.
#pragma once

#include <cstdint>

namespace sglang::expert_stream::wire {

// ---- Request page (plan D10) ----
constexpr int64_t kDemandHead = 0;
constexpr int64_t kDemandDone = 4;
constexpr int64_t kFatal = 8;
constexpr int64_t kAdviseHead = 16;
constexpr int64_t kAdviseDone = 20;
constexpr int64_t kBusySeq = 24;
constexpr int64_t kHeartbeat = 28;
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
// uint32: nonzero when the device waits on this demand record (need non-empty or advise on).
constexpr int64_t kRecArmed = 80;
// uint32: the layer's planned lane count as the device knew it when it posted (plan.count, RAM hits and
// misses together, not clamped to kMaxIds); an advisory carries the rows it asks for.
constexpr int64_t kRecLanes = 84;
constexpr uint16_t kServed = 1;
constexpr uint16_t kFailed = 2;

// ---- Lease block (LEASE_PROTOCOL.md section 4) ----
// The lease block beside the request page. Its layout is written here (the one home; exl3_ram_miss_host.cpp
// and lease_device.cuh both include this header) and mirrored in ops/moe/expert_lease_block.py;
// test_exl3_ram_miss_device_args checks they agree. The
// publication word (tag << 56 | generation) is built in code: the layout test parses these lines with + - * only.
constexpr int64_t kLeaseRing = 16;  // == kDemandRecords
constexpr int64_t kLeaseLanes = 8;  // == kMaxIds
constexpr int64_t kLeaseHeaderRing = 8;
constexpr int64_t kLeaseHeaderLanes = 12;
constexpr int64_t kLeaseHeaderShutdown = 20;
constexpr int64_t kLeaseHeaderSlotGenOffset = 32;
constexpr int64_t kLeaseHeaderDOffset = 36;
constexpr int64_t kLeaseHeaderPieceOffset = 40;
constexpr int64_t kLeaseHeaderCopyOffset = 44;
constexpr int64_t kLeaseRowTable = 128;
constexpr int64_t kLeaseRowResult = 4096;
constexpr int64_t kLeaseRowResultBytes = 32;
constexpr int64_t kLeaseRrReady = 0;
constexpr int64_t kLeaseRrSlotGeneration = 8;
constexpr int64_t kLeaseRrHostSlot = 12;
constexpr int64_t kLeaseRrExpert = 16;
constexpr int64_t kLeaseSlotGen = kLeaseRowResult + kLeaseRing * kLeaseLanes * kLeaseRowResultBytes;
constexpr int64_t kLeaseLaneRequest = 0;
constexpr int64_t kLeaseLaneRequestBytes = 128;
constexpr int64_t kLeaseLrGen = 0;
constexpr int64_t kLeaseLrCount = 8;
constexpr int64_t kLeaseLrRow = 12;
constexpr int64_t kLeaseLrExpert = 16;
constexpr int64_t kLeaseLrDst = 48;    // int32 per lane: the plan's destination slot, -1 past the plan
constexpr int64_t kLeaseLrFlags = 80;  // u32; bit kLeaseLrFlagCopyEngine lets the service copy this request's hits
constexpr uint32_t kLeaseLrFlagCopyEngine = 1;
// f32 per lane (CPU experts, plan 2026-09-29-dsv41-cpu-experts): the lane expert's routing weight, 0 past the plan.
constexpr int64_t kLeaseLrWeight = 84;
// u32 flag: the service may compute this request's resident lanes on the CPU (tag kLeaseTagCpu); needs the copy engine
// flag too, since the copy thread completes those lanes. The post also staged the layer's input row for it.
constexpr uint32_t kLeaseLrFlagCpuExperts = 2;
static_assert(kLeaseLrExpert + 4 * kLeaseLanes == kLeaseLrDst, "LaneRequest: dst_slot[] follows expert[]");
static_assert(kLeaseLrDst + 4 * kLeaseLanes == kLeaseLrFlags, "LaneRequest: flags follow dst_slot[]");
static_assert(kLeaseLrFlags + 4 == kLeaseLrWeight, "LaneRequest: weight[] follows flags");
static_assert(kLeaseLrWeight + 4 * kLeaseLanes <= kLeaseLaneRequestBytes, "LaneRequest: the payload fits one record");
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
// SmAck[kLeaseRing], device-written (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES): tagged(kLeaseTagSmAck, generation)
// once the copy wait has finished every SM read of that request's leased slots. The service releases a COPYING lease
// of a row with SM entries only after its DMA completed AND this word reached the request's generation.
constexpr int64_t kLeaseSmAck = kLeaseStreamProbe + kLeaseRing * kLeaseStreamProbeBytes;
constexpr int64_t kLeaseSmAckBytes = 8;
constexpr int64_t kLeaseRowTableBytes = 8;
// Area P, service-written, at kLeaseHeaderPieceOffset: PieceMask[kLeaseRing][kLeaseLanes], a per-lane
// generation-tagged 8-bit readiness bitmask (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment). Each word
// gets its own 128 B line, so the device's per-lane poll never shares a line with a lane it did not ask for.
// The stream kernel reads it (ld.acquire.sys) and the service's reader owner sets its bits.
constexpr int64_t kLeasePieceMaskLineBytes = 128;
constexpr int64_t kLeasePieceMaskBytes = 8;  // one uint64 per word
constexpr int64_t kLeaseAreaPieceMaskBytes = kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
// Area C, service-written, at kLeaseHeaderCopyOffset (copy-engine plan): CopyDone[kLeaseRing], {u32 lane mask;
// u32 reserved; u64 tagged(kLeaseTagCopied, generation)}, the tagged word stored last, after the service observed
// the completion of every copy-engine copy of that request's COPYING lanes.
constexpr int64_t kLeaseCopyDoneBytes = 16;
constexpr int64_t kLeaseCdMask = 0;
constexpr int64_t kLeaseCdGen = 8;
constexpr int64_t kLeaseAreaCopyDoneBytes = kLeaseRing * kLeaseCopyDoneBytes;
// Area C, after CopyDone[] (ABI 4, LEASE_PROTOCOL.md 7.6 "The stream-ordered copy wait"): the copy wait's gate, one
// u32 on its own line. CW closes it before it arms; it is opened, once per armed request, by CW itself when CopyDone
// already carries the request, else by a host releaser; the decode stream waits on it with cuStreamWaitValue32 (GEQ
// kLeaseGateOpen) in between. The word names the request: (seq & kLeaseGateSeqMask) << kLeaseGateSeqShift, OR'd with
// kLeaseGateClosed (bit 31 set: the cyclic GEQ blocks) or an outcome (1..3: it passes). Then CopyArm, u64
// tagged(kLeaseTagCopyArm, generation) on the next line, device-written after the gate closed.
constexpr int64_t kLeaseCopyGate = kLeaseAreaCopyDoneBytes;
constexpr int64_t kLeaseCopyArm = kLeaseCopyGate + 128;
constexpr int64_t kLeaseAreaCBytes = kLeaseCopyArm + 8;
constexpr uint32_t kLeaseGateClosed = 0x80000001u;  // with the request's seq field: (int32_t)(gate - 1) < 0
constexpr uint32_t kLeaseGateOpen = 1;               // CopyDone carries the armed generation
constexpr uint32_t kLeaseGateTimeout = 2;  // the host's copy-wait deadline passed; the page's fatal word is raised
constexpr uint32_t kLeaseGateAborted = 3;  // the fatal word or the header's shutdown word was raised
constexpr uint32_t kLeaseGateOutcomeMask = 3;
constexpr uint32_t kLeaseGateSeqShift = 2;
constexpr uint32_t kLeaseGateSeqMask = 0x1FFFFFFF;  // 29 bits of the request's seq
static_assert(kLeaseSmAck + kLeaseRing * kLeaseSmAckBytes <= 4096, "area D fits one page");
static_assert(kLeaseAreaCBytes <= 4096, "area C fits one page");

// Tags of the byte above the 56-bit request generation, and the reasons a Terminal record carries (section 4.3, 13).
constexpr uint64_t kLeaseTagDemand = 1;
constexpr uint64_t kLeaseTagReady = 1;
constexpr uint64_t kLeaseTagLoading = 2;  // RowResult.ready: leased, still loading (piece-streaming plan; task 1)
// RowResult.ready: leased; the service's copy engine writes this lane's destination slot, so no kernel copies it
// and nothing reads the slot before CopyDone carries the generation.
constexpr uint64_t kLeaseTagCopying = 3;
// RowResult.ready (CPU experts): leased; the service's CPU expert thread computes this lane's expert from its host slot
// and nothing copies its destination slot. The device treats it as COPYING (CW waits for its CopyDone) and leaves the
// route's slot out of the fused MoE, adding the CPU's partial sum instead.
constexpr uint64_t kLeaseTagCpu = 4;
constexpr uint64_t kLeaseTagCopied = 1;   // CopyDone
constexpr uint64_t kLeaseTagCopyArm = 1;  // CopyArm
constexpr uint64_t kLeaseTagConsumed = 1;
constexpr uint64_t kLeaseTagViolated = 2;
constexpr uint64_t kLeaseTagTerminal = 1;
constexpr uint64_t kLeaseTagStreamed = 1;  // StreamProbe
constexpr uint64_t kLeaseTagSmAck = 1;     // SmAck
constexpr uint32_t kLeaseReasonTimeout = 1;
constexpr uint32_t kLeaseReasonAborted = 2;
constexpr uint32_t kLeaseReasonFailed = 3;
constexpr uint32_t kLeaseReasonIdentity = 4;
constexpr uint32_t kLeaseReasonCount = 5;

// ---- Native prefetch page (plan 2026-09-25-dsv41-native-prefetch) ----
// A pinned 256-byte page beside the lease block. The device (the plan kernel) writes the request line, the service
// the done line. The device posts one request and waits for its done word before it posts the next, so one request
// line is enough; `gen` is the device's 56-bit prefetch counter, tagged kPfTagRequest.
constexpr int64_t kPfReqGen = 0;        // u64 tagged(kPfTagRequest, gen), stored last with a release
constexpr int64_t kPfReqRow = 8;        // i32 streamed row of the target layer
constexpr int64_t kPfReqExpert = 12;    // i32 expert
constexpr int64_t kPfReqDst = 16;       // i32 destination hot slot of the target layer
constexpr int64_t kPfDoneGen = 128;     // u64 tagged(kPfTagCopied | kPfTagSkipped, gen), service-written
constexpr int64_t kPfDoneReason = 136;  // u32, why a request was skipped (kPfSkip*), stored before kPfDoneGen
constexpr int64_t kPrefetchPageBytes = 256;
constexpr uint64_t kPfTagRequest = 1;
constexpr uint64_t kPfTagCopied = 1;
constexpr uint64_t kPfTagSkipped = 2;
constexpr uint32_t kPfSkipUnarmed = 1;
constexpr uint32_t kPfSkipNotReady = 2;
constexpr uint32_t kPfSkipInvalid = 3;

// The page and lease block back to back, and the lease block's own alignment.
constexpr int64_t kPageBytes = kAdviseRing + kAdviseRecords * kRecordBytes;
constexpr int64_t kLeaseBlockAlign = 4096;  // lease block, and each area offset inside it

}  // namespace sglang::expert_stream::wire
