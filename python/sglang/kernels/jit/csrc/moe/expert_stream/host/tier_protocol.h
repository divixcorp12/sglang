// Page and lease-block constants, and the stage trace ring (StageRing).
#pragma once

#include "row_reader.h"

namespace sglang {
namespace exl3_ram_miss {


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
// The layout is written here, in exl3_ram_miss.cuh and in ops/moe/exl3_lease_block.py; the layout test checks
// that they agree, so these lines use only + - * over integers and known names. The service writes areas H and
// S (header, row table, row results, slot generations); the device writes area D; Python writes nothing.
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
constexpr int64_t kLeaseLrDst = 48;
constexpr int64_t kLeaseLrFlags = 80;
constexpr uint32_t kLeaseLrFlagCopyEngine = 1;
static_assert(kLeaseLrExpert + 4 * kLeaseLanes == kLeaseLrDst, "LaneRequest: dst_slot[] follows expert[]");
static_assert(kLeaseLrDst + 4 * kLeaseLanes == kLeaseLrFlags, "LaneRequest: flags follow dst_slot[]");
static_assert(kLeaseLrFlags + 4 <= kLeaseLaneRequestBytes, "LaneRequest: the payload fits one record");
constexpr int64_t kLeaseLaneAck = kLeaseLaneRequest + kLeaseRing * kLeaseLaneRequestBytes;
constexpr int64_t kLeaseLaneAckBytes = 8;
constexpr int64_t kLeaseTerminal = kLeaseLaneAck + kLeaseRing * kLeaseLanes * kLeaseLaneAckBytes;
constexpr int64_t kLeaseTerminalBytes = 16;
constexpr int64_t kLeaseTermSkippedMask = 0;
constexpr int64_t kLeaseTermReason = 4;
constexpr int64_t kLeaseTermGen = 8;
// StreamProbe[kLeaseRing], device-written: the stream kernel's tagged(1, generation) once it has copied a piece.
constexpr int64_t kLeaseStreamProbe = kLeaseTerminal + kLeaseRing * kLeaseTerminalBytes;
constexpr int64_t kLeaseStreamProbeBytes = 8;
// SmAck[kLeaseRing], device-written: the copy wait finished its SM reads of that request's leased slots.
constexpr int64_t kLeaseSmAck = kLeaseStreamProbe + kLeaseRing * kLeaseStreamProbeBytes;
constexpr int64_t kLeaseSmAckBytes = 8;
constexpr int64_t kLeaseRowTableBytes = 8;

// Area P, service-written, at a new header offset (kLeaseHeaderPieceOffset): PieceMask[kLeaseRing][kLeaseLanes],
// a per-lane generation-tagged 8-bit readiness bitmask (piece-streaming plan, LEASE_PROTOCOL.md E1 amendment).
// Each word gets its own 128 B line, so the device's per-lane poll never shares a line with a lane it did not
// ask for. Under piece streaming the tier stores `gen << 8` into each miss lane's word at reservation, and the
// reader's owner sets one bit per packed piece (publish_piece); with the flag off nothing writes it.
constexpr int64_t kLeasePieceMaskLineBytes = 128;
constexpr int64_t kLeasePieceMaskBytes = 8;  // one uint64 per word
constexpr int64_t kLeaseAreaPieceMaskBytes = kLeaseRing * kLeaseLanes * kLeasePieceMaskLineBytes;
static_assert(kPieceTargets >= kLeaseLanes, "a row's pieces are published to at most one word per lane");
// Area C, service-written, at kLeaseHeaderCopyOffset: CopyDone[kLeaseRing], {u32 lane mask; u32 reserved; u64
// tagged(kLeaseTagCopied, generation)}. Only the copy-engine thread writes it, after it observed the copies complete.
constexpr int64_t kLeaseCopyDoneBytes = 16;
constexpr int64_t kLeaseCdMask = 0;
constexpr int64_t kLeaseCdGen = 8;
constexpr int64_t kLeaseAreaCopyDoneBytes = kLeaseRing * kLeaseCopyDoneBytes;
static_assert(kLeaseSmAck + kLeaseRing * kLeaseSmAckBytes <= 4096, "area D fits one page");

// kQuarantine (piece streaming only): a slot whose read failed while a lane still leased it under tag LOADING. Its
// mapping is cleared on entry, it is never taken, evicted or counted as a victim, and it becomes kFree when its last
// lease is retired (retire_leases). A leased slot is never released: that is the S6 rule under piece streaming.
enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2, kQuarantine = 3 };

// RowResult.ready tags (LEASE_PROTOCOL.md 4.3): READY for a lane whose row is resident, LOADING (piece streaming) for
// a miss lane granted at reservation, whose pieces become readable bit by bit through its PieceMask word.
constexpr uint64_t kLeaseTagReady = 1;
constexpr uint64_t kLeaseTagLoading = 2;
// A hit lane the copy-engine thread copies into its destination slot; the device neither copies nor acknowledges it.
constexpr uint64_t kLeaseTagCopying = 3;
constexpr uint64_t kLeaseTagCopied = 1;  // CopyDone
constexpr uint64_t kLeaseTagSmAck = 1;   // SmAck

enum Counter : int {
  kServedRequests = 0,
  kTouchOnly,
  kRowsRead,
  kReadErrors,
  kEvictions,
  kOverruns,
  kAdvisories,
  kAdvisoriesSkipped,
  kAdvisoryRows,
  kLateAfterFatal,
  kNoVictim,
  kVersion,
  kRunning,
  kSpinCpu,
  kDeferred,  // demands held back because their only victims are leased: one per deferral, none evicted
  kLeasesGranted,     // one per lane of a served request in lease mode
  kLeasesAcked,       // released by the device's acknowledgement
  kLeasesVoided,      // released by a terminal record that named the lane
  kLeaseDoubleSignal, // a lane signalled by both, or twice: released once, counted here
  kLateAfterTerminal, // a request the device had already given up on: dropped without a lease
  kDeferredReuse,     // a demand held back because its request slot still holds an unretired lease row
  // S7. The hit-lane subset of kLeasesGranted: lanes granted BEFORE read() by V1's first phase. Separate because
  // kLeasesGranted cannot distinguish the groups, so a build that publishes nothing early -- falling through to
  // the batched grant -- would satisfy every timing assertion by accident. Zero on the single-phase path.
  kHitLeasesGranted,
  kPieceStreamRefused,   // requests refused because piece streaming is on without two-phase and lease mode
  kPiecePublishRefused,  // piece publishes a readiness word refused, over the reader's life (each failed its read)
  kSlotsQuarantined,     // piece streaming: leased slots of a failed read put in kQuarantine instead of released
  kLeasesCopied,         // copy engine: released on the service's own observation that the lane's copy completed
  kCopyJobs,             // copy engine: requests whose COPYING lanes were handed to the copy thread
  kCopyLanes,            // ... and their lanes
  kCopyBytes,            // bytes the copy thread issued
  kCopyIssueNs,          // host ns in the copy thread's CUDA calls that issue copies and record events, summed
  kCopyLatencyNs,        // submit (the grant) to completion observed, summed over jobs
  kCopyLatencyMaxNs,
  kCopyFallbacks,        // hit lanes published READY while the copy engine was armed (flag off, no table, bad slot)
  kCopyErrors,           // CUDA errors on the copy thread: the page is fatal and the leases stay held
  kCopyGenerationMismatches,  // a completed lane whose slot generation moved: fatal, CopyDone never published
  // Native prefetch (plan 2026-09-25-dsv41-native-prefetch): advisory next-layer copies through the copy engine.
  kPrefetchRequests,         // prefetch requests the device posted and the service read
  kPrefetchIssued,           // ... leased and handed to the copy thread
  kPrefetchCopied,           // ... whose copy completed and whose PrefetchDone said COPIED
  kPrefetchSkippedUnarmed,   // skipped: the copy engine was not armed (or the service is closing)
  kPrefetchSkippedNotReady,  // skipped: the row was not READY in the pinned tier when the service looked
  kPrefetchSkippedInvalid,   // skipped: bad row, expert or destination slot
  kPrefetchUsed,             // copied rows the target layer's next request routed
  kPrefetchWasted,           // copied rows it did not route
  kPrefetchHeld,             // prefetch jobs the copy thread held back behind a demand job
  kPrefetchLatencyNs,        // request read to completion observed, summed over copied prefetches
  kCounterCount,
};

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

// The lease block's publication words are 64 bits: a tag in the top byte over a 56-bit request generation
// (LEASE_PROTOCOL.md 4.2). Built in code, not in a k-constant: the layout test parses those with + - * only.
inline uint64_t load_acquire64(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release64(uint8_t* address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}

inline uint64_t generation_of(uint64_t word) {
  return word & ((uint64_t(1) << 56) - 1);
}

inline uint64_t tag_of(uint64_t word) {
  return word >> 56;
}

inline uint64_t tagged_word(uint64_t tag, uint64_t generation) {
  return (tag << 56) | generation;
}

// The device never posts sequence 0 (the post kernel and sim_post wrap 0xFFFFFFFF to 1), so a
// service that reaches 0 would spend an iteration on a record nobody posted and store a done word
// of 0.
inline uint32_t skip_zero(uint32_t seq) {
  return seq == 0 ? 1u : seq;
}

inline bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

inline int64_t record_offset(int64_t ring, uint32_t records, uint32_t seq) {
  return ring + static_cast<int64_t>((seq - 1u) % records) * kRecordBytes;
}

struct Request {
  uint32_t seq = 0;
  int64_t row = 0;
  uint32_t after = 0;
  bool armed = true;
  uint32_t lanes = 0;
  std::vector<int32_t> need;
  std::vector<int32_t> protect;
  std::vector<uint8_t> hot_bitmap;
  // Lease mode: the device's lane list and 56-bit request generation, from the lane request (not the record).
  uint64_t gen = 0;
  std::vector<int32_t> lane_experts;
  std::vector<int32_t> lane_dst;  // the plan's destination slot per lane, -1 unknown
  uint32_t lane_flags = 0;        // kLeaseLrFlag*
};

// The service's private account of one request's leases, by request slot (LEASE_PROTOCOL.md 5.2).
struct LaneLease {
  uint8_t state = 0;  // 0 none, 1 granted, 2 acknowledged (or its copy-engine copy completed), 3 voided by a terminal
  int32_t slot = -1;
  uint32_t slot_generation = 0;
  bool counted = false;  // a second signal for this lane was already counted
  // Published COPYING: only the copy thread's observed completion releases it; LaneAck and Terminal never do.
  bool copy_engine = false;
};

struct Outstanding {
  bool active = false;
  // A further lane group is still to be granted into this entry (V1 two-phase, S1/S4). While it is set the entry
  // counts as open even though no lane is in state 1 yet, so retire_leases cannot free the ring index out from
  // under a grant that has not run.
  bool grants_pending = false;
  uint64_t gen = 0;
  int64_t row = 0;
  uint32_t count = 0;
  LaneLease lane[kLeaseLanes];
};

// Seqlock read: the writer stores the payload, fences, then the seq word last, so a
// record whose seq reads `expected` both before and after the payload is whole.
inline bool read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return false;
  uint16_t row, need, protect;
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&need, record + kRecNeedCount, 2);
  std::memcpy(&protect, record + kRecProtectCount, 2);
  std::memcpy(&request->after, record + kRecAfter, 4);
  uint32_t armed;
  std::memcpy(&armed, record + kRecArmed, 4);
  std::memcpy(&request->lanes, record + kRecLanes, 4);
  request->armed = armed != 0;
  request->seq = expected;
  request->row = row;
  const auto* need_ids = reinterpret_cast<const int32_t*>(record + kRecNeed);
  const auto* protect_ids = reinterpret_cast<const int32_t*>(record + kRecProtect);
  request->need.assign(need_ids, need_ids + std::min<int>(need, kMaxIds));
  request->protect.assign(protect_ids, protect_ids + std::min<int>(protect, kMaxIds));
  std::atomic_thread_fence(std::memory_order_acquire);
  return load_acquire(record + kRecSeq) == expected;
}

inline void set_status(uint8_t* record, uint16_t status) {
  __atomic_store_n(reinterpret_cast<uint16_t*>(record + kRecStatus), status, __ATOMIC_RELEASE);
}

inline bool listed(const std::vector<int32_t>& ids, int32_t id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// Fixed-capacity single-producer single-consumer queue of stage records: the service thread
// pushes and drops (counted) when full, one Python caller at a time drains. Allocated once, when
// the trace is enabled; a push copies a record into a preallocated slot and allocates nothing.
class StageRing {
 public:
  explicit StageRing(size_t capacity) : slots_(capacity) {}

  void push(const StageRecord& record) {
    const uint64_t head = head_.load(std::memory_order_relaxed);
    if (head - tail_.load(std::memory_order_acquire) >= slots_.size()) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      ++unreported_;
      return;
    }
    StageRecord& slot = slots_[head % slots_.size()];
    slot = record;
    slot.dropped_before = unreported_;
    unreported_ = 0;
    head_.store(head + 1, std::memory_order_release);
  }

  int64_t drain(StageRecord* out, int64_t max) {
    const uint64_t head = head_.load(std::memory_order_acquire);
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    int64_t count = 0;
    while (tail < head && count < max)
      out[count++] = slots_[tail++ % slots_.size()];
    tail_.store(tail, std::memory_order_release);
    return count;
  }

  int64_t dropped() const {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  std::vector<StageRecord> slots_;
  std::atomic<uint64_t> head_{0};
  std::atomic<uint64_t> tail_{0};
  std::atomic<int64_t> dropped_{0};
  int64_t unreported_ = 0;  // producer only: drops since the last record that got in
};

// ---- Copy engine (LEASE_PROTOCOL.md 7.6) ----


}  // namespace exl3_ram_miss
}  // namespace sglang
