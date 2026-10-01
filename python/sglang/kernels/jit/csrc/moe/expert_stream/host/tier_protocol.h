// Page and block constants, the service's request records, and the stage trace ring (StageRing).
#pragma once

#include "../lease_layout.h"
#include "fixed_vec.h"
#include "row_reader.h"

namespace sglang {
namespace expert_stream {

using namespace ::sglang::expert_stream::wire;

// kStaging: one of the row's K staging slots, never mapped; an NVMe miss is read into it (LEASE_PROTOCOL.md).
enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2, kStaging = 3 };

static_assert(kPieceTargets >= kLeaseLanes, "a row's pieces are published to at most one word per lane");

enum Counter : int {
  kServedRequests = 0,
  kTouchOnly,
  kRowsRead,
  kReadErrors,
  kEvictions,
  kOverruns,
  kNoVictim,
  kVersion,
  kRunning,
  kSpinCpu,
  kRamInsertSkipped,     // a miss with no evictable slot: computed, not cached in RAM (its staging slot stays one)
  kPiecePublishRefused,  // piece publishes a readiness word refused, over the reader's life (each failed its read)
  kCopyJobs,             // copy engine: records with copy-engine or CPU lanes handed to the copy thread
  kCopyLanes,            // ... and their lanes
  kCopyBytes,            // bytes the copy thread issued
  kCopyIssueNs,          // host ns in the copy thread's CUDA calls that issue copies and record events, summed
  kCopyLatencyNs,        // submit (record time) to completion observed, summed over jobs
  kCopyLatencyMaxNs,
  // CPU experts (plan 2026-09-29-dsv41-cpu-experts): CPU jobs (one per part of a record) and their lanes.
  kCpuJobs,
  kCpuLanes,
  // The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Task 13): Python commands (set_hot, attach_row, the
  // snapshots) the tier's owner applied, queued through the command ring or run directly. A metric: tests only (F24).
  kCommandsApplied,
  kCounterCount,
};

// Counters the production build keeps (plan 2026-09-29-hotpath-zero-overhead D1): the shutdown line's served, rows and
// errors, the admission policy's outcomes, and the functional version.
constexpr bool is_core_counter(int k) {
  switch (k) {
    case kServedRequests: case kTouchOnly: case kRowsRead: case kReadErrors: case kEvictions: case kOverruns:
    case kNoVictim: case kVersion: case kRunning: case kSpinCpu: case kRamInsertSkipped:
      return true;
    default:
      return false;
  }
}
static_assert(is_core_counter(kVersion), "version is functional (Python's LRU view invalidates on it): never a metric");

inline uint32_t load_acquire(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

// The completion and delta blocks' 64-bit words: a 56-bit request generation G, a map-chain tag.
inline uint64_t load_acquire64(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(address), __ATOMIC_ACQUIRE);
}

inline void store_release64(uint8_t* address, uint64_t value) {
  __atomic_store_n(reinterpret_cast<uint64_t*>(address), value, __ATOMIC_RELEASE);
}

// The device never posts sequence 0 (the post kernel wraps 0xFFFFFFFF to 1), so a service that reaches 0 would spend
// an iteration on a record nobody posted and store a done word of 0.
inline uint32_t skip_zero(uint32_t seq) {
  return seq == 0 ? 1u : seq;
}

inline bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

inline int64_t record_offset(int64_t ring, uint32_t records, uint32_t seq) {
  return ring + static_cast<int64_t>((seq - 1u) % records) * kRecordBytes;
}

// A request's distinct experts: its protect ids and its lanes' experts (plan 2026-09-29-hotpath-zero-overhead Task 11).
// Every per-request list of the service is bounded by it, so none needs the heap.
constexpr size_t kWanted = kMaxIds + kLeaseLanes;

// One lane of a record: the device typed it from its slot map (ram_slot_map.type_lanes).
struct Lane {
  int32_t expert = -1;
  int32_t slot = -1;  // the RAM slot of a hit, the staging slot of a miss
  int32_t dst = -1;   // the VRAM destination slot
  float weight = 0.0f;
  uint8_t kind = 0;   // kKind*
};

inline bool is_miss(uint8_t kind) {
  return kind == kKindMissGpu || kind == kKindMissCpu;
}

// One demand record, as the service thread reads it: fixed-size, so reading one allocates nothing (spec A1-A3, A11).
struct Request {
  uint32_t seq = 0;
  uint64_t gen = 0;  // epoch << 32 | seq
  int64_t row = 0;
  bool captured = false;
  uint64_t chain = 0;  // the row's map-chain number when a lane misses, else 0
  FixedVec<int32_t, kMaxIds> protect;
  FixedVec<Lane, kLeaseLanes> lanes;
  const uint8_t* hot_bitmap = nullptr;  // GPU hot mode: RamTier::hot_scratch_, valid until the next record read
  // A lane the service itself works on, or that the device waits for: anything but an SM hit. A record with one is
  // waited on by the device, so it cannot lap the ring.
  bool host_work() const {
    for (const Lane& lane : lanes)
      if (lane.kind != kKindHitSm) return true;
    return false;
  }
};

// Seqlock read: the writer stores the payload, fences, then the seq word last, so a record whose seq reads `expected`
// both before and after the payload is whole. Records nothing waits for lap the ring unread, so a torn one must be
// detectable. A whole record whose count or kinds are out of range is malformed: the device never writes one.
enum class RecordRead { kOk, kTorn, kMalformed };

inline RecordRead read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return RecordRead::kTorn;
  uint16_t row, count, protect;
  uint32_t flags, chain_lo, chain_hi, epoch;
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&count, record + kRecCount, 2);
  std::memcpy(&flags, record + kRecFlags, 4);
  std::memcpy(&chain_lo, record + kRecChain, 4);
  std::memcpy(&chain_hi, record + kRecChainHi, 4);
  std::memcpy(&epoch, record + kRecEpoch, 4);
  std::memcpy(&protect, record + kRecProtectCount, 2);
  if (count > kLeaseLanes) count = kLeaseLanes + 1;  // judged once the seq re-check says the record is whole
  request->seq = expected;
  request->gen = static_cast<uint64_t>(epoch) << 32 | expected;
  request->row = row;
  request->captured = (flags & kRecFlagCaptured) != 0;
  request->chain = static_cast<uint64_t>(chain_hi) << 32 | chain_lo;
  const auto* protect_ids = reinterpret_cast<const int32_t*>(record + kRecProtect);
  request->protect.assign(protect_ids, protect_ids + std::min<int>(protect, kMaxIds));
  request->lanes.clear();
  for (int j = 0; j < std::min<int>(count, kLeaseLanes); ++j) {
    const uint8_t* lane = record + kRecLanes + j * kLaneBytes;
    Lane l;
    std::memcpy(&l.expert, lane + kLaneExpert, 4);
    std::memcpy(&l.slot, lane + kLaneSlot, 4);
    std::memcpy(&l.dst, lane + kLaneDst, 4);
    std::memcpy(&l.weight, lane + kLaneWeight, 4);
    l.kind = record[kRecKinds + j];
    request->lanes.push_back(l);
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  if (load_acquire(record + kRecSeq) != expected) return RecordRead::kTorn;
  if (count > kLeaseLanes) return RecordRead::kMalformed;
  for (const Lane& lane : request->lanes)
    if (lane.kind < kKindHitCopy || lane.kind > kKindMissCpu) return RecordRead::kMalformed;
  return RecordRead::kOk;
}

// Fail-stop: every host failure of the protocol ends the process here, with one line first. Nothing recovers, so
// nothing signals: a device waiting on the request traps at its deadline, or dies with the process.
[[noreturn]] inline void fail_stop(const std::string& message) {
  std::fprintf(stderr, "FATAL %s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
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

}  // namespace expert_stream
}  // namespace sglang
