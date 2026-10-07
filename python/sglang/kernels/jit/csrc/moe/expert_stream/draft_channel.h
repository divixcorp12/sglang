// The DSpark draft's lease channel (LEASE_PROTOCOL.md, "The lease channel"): the protocol's second client, after the
// target. One pinned buffer of kChannelBytes holds the page (head, a ring of four records) and the completion block
// (done[4], the gate). A record names one call's CPU share: the stage, its rows and routes per row; the inputs and the
// output live in the draft's own pinned areas (dspark_draft_cpu.DraftCpuAreas), [stages, kMaxRows, ...] each.
//
// The draft posts at most one record before its finish waits on the gate, so the ring never laps: a lapped or torn
// record is a protocol failure, and the host fail-stops on it.
#pragma once

#include <cstdint>

#include "lease_channel_layout.h"

namespace sglang::expert_stream::draft {

constexpr int kMaxRows = 16;  // DraftResidentMoe.TOKENS: the most tokens one call holds
constexpr int kMaxK = 8;      // routes per token

// Record (a 12-byte payload in a 128-byte slot): seq u32 @0 (the seqlock word), stage u16 @4, rows u8 @6, k u8 @7,
// epoch u32 @8.
constexpr int64_t kRecStage = 4;
constexpr int64_t kRecRows = 6;
constexpr int64_t kRecK = 7;
constexpr int64_t kRecEpoch = 8;
// Diagnostic device-clock marker before the record/head release. Zero in uninstrumented records.
constexpr int64_t kRecPublishNs = 16;

using DraftChannel = channel::ChannelSpec</*Head*/ 0, /*Ring*/ 128, /*Records*/ 4, /*RecordBytes*/ 128,
                                          /*Done*/ 640, /*Gate*/ 768>;
constexpr int64_t kChannelBytes = 4096;  // the page and the completion block in one pinned buffer

static_assert(DraftChannel::kRing + DraftChannel::kRecords * DraftChannel::kRecordBytes <= DraftChannel::kDone,
              "the ring ends before the done words");
static_assert(DraftChannel::kGate + 4 <= kChannelBytes, "the gate is inside the buffer");
static_assert(kRecEpoch + 4 <= DraftChannel::kRecordBytes, "the record fits its slot");

}  // namespace sglang::expert_stream::draft
