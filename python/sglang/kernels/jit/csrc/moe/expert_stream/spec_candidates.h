// The GPU scorer's candidate page (spec docs/superpowers/specs/2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md):
// pinned host memory the select kernel writes (spec_score.cuh) and the speculative threads read (host/ram_tier.h).
//
//   slot (seq - 1) % kCandRecords, kCandStride bytes: u32 seq (the seqlock word, stored last), u16 count, u16 flags,
//   then kMaxCandidates entries of u16 expert, u8 rank, a pad byte, f32 margin
//
// python/sglang/kernels/ops/moe/expert_stream_transport.py mirrors it as CAND_*; test_exl3_ram_miss_device_args checks.
#pragma once

#include "lease_layout.h"
#include <cstdint>

namespace sglang::expert_stream::wire {

struct SpecCandidates {
  static constexpr int kMaxCandidates = 8;
  static constexpr uint32_t kCandRecords = 16;
  static constexpr int64_t kCandSeq = 0;      // u32: 0 while the slot is rewritten, the record's seq stored last
  static constexpr int64_t kCandCount = 4;    // u16
  static constexpr int64_t kCandFlags = 6;    // u16: kCandFlag*
  static constexpr int64_t kCandEntries = 8;  // kMaxCandidates entries, best margin first
  static constexpr int64_t kCandEntryBytes = 8;
  static constexpr int64_t kCandExpert = 0;  // u16
  static constexpr int64_t kCandRank = 2;    // u8: the pick's position in the token that gave its best margin
  static constexpr int64_t kCandMargin = 4;  // f32: that token's score for it less its top_k-th
  static constexpr int64_t kCandPayloadBytes = kCandEntries + kMaxCandidates * kCandEntryBytes;
  static constexpr int64_t kCandStride = 128;
  static constexpr int64_t kCandPageBytes = kCandRecords * kCandStride;
  static constexpr uint32_t kCandFlagOversize = 1;  // the record's tokens are outside 1..tokens_max: count 0

  // Host only: the device computes the same expression inline.
  static constexpr int64_t slot_offset(uint32_t seq) {
    return static_cast<int64_t>((seq - 1u) % kCandRecords) * kCandStride;
  }
};

static_assert(SpecCandidates::kCandRecords == Wire::kDemandRecords, "one slot per demand record of the ring");
static_assert(SpecCandidates::kCandPayloadBytes == 72 && SpecCandidates::kCandPayloadBytes <= SpecCandidates::kCandStride,
              "a slot's payload fits its stride");

}  // namespace sglang::expert_stream::wire
