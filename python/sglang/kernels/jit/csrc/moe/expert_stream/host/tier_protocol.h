// Page and lease-block constants, the service's request records, and the stage trace ring (StageRing).
#pragma once

#include "../lease_layout.h"
#include "fixed_vec.h"
#include "row_reader.h"

namespace sglang {
namespace expert_stream {

using namespace ::sglang::expert_stream::wire;

enum : uint8_t { kFree = 0, kLoading = 1, kReady = 2 };

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
  kDeferred,       // demands held back because their only victims are leased: one per deferral, none evicted
  kLeasesGranted,  // one per lane of a served request
  kLeasesAcked,    // released on the request's Done word
  kDeferredReuse,  // a demand held back because its request slot still holds an unretired lease row
  kPiecePublishRefused,  // piece publishes a readiness word refused, over the reader's life (each failed its read)
  kLeasesCopied,         // copy engine: released on the service's own observation that the lane's copy completed
  kCopyJobs,             // copy engine: requests whose COPYING lanes were handed to the copy thread
  kCopyLanes,            // ... and their lanes
  kCopyBytes,            // bytes the copy thread issued
  kCopyIssueNs,          // host ns in the copy thread's CUDA calls that issue copies and record events, summed
  kCopyLatencyNs,        // submit (the grant) to completion observed, summed over jobs
  kCopyLatencyMaxNs,
  kCopyFallbacks,  // hit lanes published READY while the copy engine was armed (flag off, no table, bad slot)
  // CPU experts (plan 2026-09-29-dsv41-cpu-experts): copy-engine requests some of whose lanes went to the CPU.
  kCpuJobs,
  kCpuLanes,  // ... and those lanes
  // The single-owner tier (plan 2026-09-29-hotpath-zero-overhead Task 13): Python commands (set_hot, inject_lease, the
  // snapshots) the tier's owner applied, queued through the command ring or run directly. A metric: tests only (F24).
  kCommandsApplied,
  kCounterCount,
};

// Counters the production build keeps (plan 2026-09-29-hotpath-zero-overhead D1): the shutdown line's served, rows and
// errors, the admission policy's outcomes, and the functional version.
constexpr bool is_core_counter(int k) {
  switch (k) {
    case kServedRequests: case kTouchOnly: case kRowsRead: case kReadErrors: case kEvictions: case kOverruns:
    case kNoVictim: case kVersion: case kRunning: case kSpinCpu: case kDeferred: case kDeferredReuse:
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

// The lease block's words are 64 bits: a 56-bit request generation G, under a tag byte in a RowResult.
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

// One demand record, as the service thread reads it: fixed-size, so reading one allocates nothing (spec A1-A3, A11).
struct Request {
  uint32_t seq = 0;
  int64_t row = 0;
  bool armed = true;
  FixedVec<int32_t, kMaxIds> protect;
  const uint8_t* hot_bitmap = nullptr;  // GPU hot mode: RamTier::hot_scratch_, valid until the next record read
  // An armed request's lanes and 56-bit generation, from its LaneRequest (not the record).
  uint64_t gen = 0;
  FixedVec<int32_t, kLeaseLanes> lane_experts;
  FixedVec<int32_t, kLeaseLanes> lane_dst;  // the plan's destination slot per lane, -1 unknown
  bool captured = false;                    // kLeaseLrFlagCaptured
  FixedVec<float, kLeaseLanes> lane_weight;  // the lane expert's routing weight (kLeaseLrWeight), for CPU experts
};

// The service's private account of one request's leases, by request slot.
struct LaneLease {
  uint8_t state = 0;  // 0 none, 1 granted, 2 released
  int32_t slot = -1;
  // Published COPYING or CPU: only the copy thread's observed completion releases it, never Done alone.
  bool copy_engine = false;
};

struct Outstanding {
  bool active = false;
  uint64_t gen = 0;
  int64_t row = 0;
  uint32_t count = 0;
  LaneLease lane[kLeaseLanes];
};

// Seqlock read: the writer stores the payload, fences, then the seq word last, so a record whose seq reads `expected`
// both before and after the payload is whole. Unarmed records lap the ring unread, so a torn one must be detectable.
inline bool read_record(const uint8_t* record, uint32_t expected, Request* request) {
  if (load_acquire(record + kRecSeq) != expected) return false;
  uint16_t row, protect, armed;
  std::memcpy(&row, record + kRecRow, 2);
  std::memcpy(&protect, record + kRecProtectCount, 2);
  std::memcpy(&armed, record + kRecArmed, 2);
  request->armed = armed != 0;
  request->seq = expected;
  request->row = row;
  const auto* protect_ids = reinterpret_cast<const int32_t*>(record + kRecProtect);
  request->protect.assign(protect_ids, protect_ids + std::min<int>(protect, kMaxIds));
  std::atomic_thread_fence(std::memory_order_acquire);
  return load_acquire(record + kRecSeq) == expected;
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
