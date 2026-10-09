// The service's side of the lease protocol: lane kinds and slot states, counters, request records and the stage ring.
//
//   kFree / kReady / kStaging / kSpec   slot states of a RAM-tier row
//   Counter / is_core_counter   the service's counters, and the subset ProdBuild keeps
//   load_acquire / store_release   word access to the lease blocks shared with the device
//   Lane / Request / read_record   one demand record as the service thread reads it, with a torn-read check
//   fail_stop                   the one way a host protocol failure ends the process
//   StageRing                   the SPSC queue of stage trace records
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md.
#pragma once

#include "../lease_layout.h"
#include "lease_channel.h"
#include "fixed_vec.h"
#include "row_reader.h"

namespace sglang {
namespace expert_stream {

using namespace ::sglang::expert_stream::wire;

// Slot states. kStaging is one of the row's K staging slots, never mapped; an NVMe miss is read into it
// (analysis/dsv41-drive/LEASE_PROTOCOL.md). kSpec is a slot of the speculative pool (ram_prefetch.h): never mapped,
// never a victim, never released.
enum : uint8_t { kFree = 0, kReady = 2, kStaging = 3, kSpec = 4 };

static_assert(kPieceTargets >= Wire::kLanes, "a row's pieces are published to at most one word per lane");

// Indices into the service's counter array. Only the is_core_counter subset is kept by ProdBuild; the rest are
// metrics.
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
  // CPU experts: CPU jobs (one per part of a record) and their lanes.
  kCpuJobs,
  kCpuLanes,
  // RAM prefetch (ram_prefetch.h): speculative reads started, landed; pool rows a forced miss swapped in, and of those
  // the ones whose read was still in flight (promoted); candidates dropped (stale, mapped or pooled since scoring, or
  // a full ring); reads failed; demand reads that waited at the reader's turn for a speculative read.
  kSpecIssued,
  kSpecLanded,
  kSpecUsed,
  kSpecPromoted,  // counts the waits, also one whose read then fails or is dropped as stale
  kSpecDropped,
  kSpecFailed,
  kSpecDelayed,
  // ... the scorer's records scored and their scoring time in ns (metrics).
  kSpecScored,
  kSpecScoreNs,
  kCounterCount,
};

// True for the counters the production build keeps: the shutdown line's served, rows and errors, the admission
// policy's outcomes, the RAM prefetch's outcomes (a counters-off A/B reports them), and the functional version.
constexpr bool is_core_counter(int k) {
  switch (k) {
    case kServedRequests:
    case kTouchOnly:
    case kRowsRead:
    case kReadErrors:
    case kEvictions:
    case kOverruns:
    case kNoVictim:
    case kVersion:
    case kRunning:
    case kSpinCpu:
    case kRamInsertSkipped:
    case kSpecIssued:
    case kSpecLanded:
    case kSpecUsed:
    case kSpecPromoted:
    case kSpecDropped:
    case kSpecFailed:
    case kSpecDelayed:
      return true;
    default:
      return false;
  }
}
static_assert(is_core_counter(kVersion), "version is functional (Python's LRU view invalidates on it): never a metric");

// Acquire load of the 32-bit word at `address` in a lease block shared with the device.
inline uint32_t load_acquire(const uint8_t* address) {
  return channel::load_acquire(address);
}

// Release store of the 32-bit word at `address` in a lease block shared with the device.
inline void store_release(uint8_t* address, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(address), value, __ATOMIC_RELEASE);
}

// Acquire load of a 64-bit word of the completion and delta blocks: a 56-bit request generation, a map-chain tag.
inline uint64_t load_acquire64(const uint8_t* address) {
  return __atomic_load_n(reinterpret_cast<const uint64_t*>(address), __ATOMIC_ACQUIRE);
}

// Release store of a 64-bit word of the completion and delta blocks.
inline void store_release64(uint8_t* address, uint64_t value) {
  channel::store_release64(address, value);
}

// Maps sequence 0 to 1. The device never posts sequence 0 (the post kernel wraps 0xFFFFFFFF to 1), so a service that
// reached 0 would spend an iteration on a record nobody posted and store a done word of 0.
inline uint32_t skip_zero(uint32_t seq) {
  return channel::skip_zero(seq);
}

// True when `observed` has reached `seq`, correct across the 2^32 wrap.
inline bool reached(uint32_t observed, uint32_t seq) {
  return channel::reached(observed, seq);
}

// Byte offset of the record for sequence `seq` in a ring of `records` records starting at `ring`.
inline int64_t record_offset(int64_t ring, uint32_t records, uint32_t seq) {
  return ring + static_cast<int64_t>((seq - 1u) % records) * Wire::kRecordBytes;
}

// The bound on a request's distinct experts: its protect ids and its lanes' experts. Every per-request list of the
// service is bounded by it, so none needs the heap.
constexpr size_t kWanted = Wire::kLanes + Wire::kLanes;

// One lane of a record: the device typed it from its slot map (ram_slot_map.type_lanes).
struct Lane {
  int32_t expert = -1;
  int32_t slot = -1;  // the RAM slot of a hit, the staging slot of a miss
  int32_t dst = -1;   // the VRAM destination slot
  float weight = 0.0f;
  uint8_t kind = 0;  // kKind*
};

inline bool is_miss(uint8_t kind) {
  return kind == Wire::kKindMissGpu || kind == Wire::kKindMissCpu;
}

// One demand record, as the service thread reads it. Fixed-size, so reading one allocates nothing.
struct Request {
  uint32_t seq = 0;
  uint64_t gen = 0;  // epoch << 32 | seq
  int64_t row = 0;
  bool captured = false;
  uint64_t chain = 0;  // the row's map-chain number when a lane misses, else 0
  FixedVec<int32_t, Wire::kLanes> protect;
  FixedVec<Lane, Wire::kLanes> lanes;
  const uint8_t* hot_bitmap = nullptr;  // GPU hot mode: RamTier::hot_scratch_, valid until the next record read
  // True when a lane needs the service itself or is one the device waits for: anything but an SM hit. The device
  // waits on such a record, so it cannot lap the ring.
  bool host_work() const {
    for (const Lane& lane : lanes)
      if (lane.kind != Wire::kKindHitSm) return true;
    return false;
  }
};

// The outcome of read_record: a whole record, a torn one, or one the device never writes.
enum class RecordRead { kOk, kTorn, kMalformed };

static_assert(Wire::kRecPayloadEnd <= Wire::kRecordBytes, "read_record copies the whole payload");

// Reads the record at `record` into `request` if its seq word is `expected`. Returns kTorn when the writer
// overwrote it during the read, and kMalformed for a whole record whose counts or kinds are out of range.
//
// A seqlock: the writer stores the payload, fences, then the seq word last, so a record whose seq reads `expected`
// both before and after the payload is whole. Records nothing waits for lap the ring unread, so a torn one must be
// detectable. The payload is copied once, at a constant size, between the two seq loads: every cache line's load
// issue together, before anything depends on the counts, and nothing after the second seq load reads the shared
// record.
inline RecordRead read_record(const uint8_t* record, uint32_t expected, Request* request) {
  alignas(64) uint8_t raw[Wire::kRecordBytes];
  if (!channel::read_seqlocked<TargetChannel>(record, expected, raw)) return RecordRead::kTorn;
  uint16_t row;
  uint8_t counts, flags;
  uint64_t chain;
  uint32_t epoch, kinds[Wire::kKindWords];
  int16_t protect_ids[Wire::kLanes], expert[Wire::kLanes], slot[Wire::kLanes], dst[Wire::kLanes];
  float weight[Wire::kLanes];
  std::memcpy(&row, raw + Wire::kRecRow, 2);
  std::memcpy(&counts, raw + Wire::kRecCounts, 1);
  std::memcpy(&flags, raw + Wire::kRecFlags, 1);
  std::memcpy(&chain, raw + Wire::kRecChain, 8);
  std::memcpy(&epoch, raw + Wire::kRecEpoch, 4);
  std::memcpy(kinds, raw + Wire::kRecKinds, sizeof(kinds));
  std::memcpy(protect_ids, raw + Wire::kRecProtect, sizeof(protect_ids));
  std::memcpy(expert, raw + Wire::kRecLaneExpert, sizeof(expert));
  std::memcpy(slot, raw + Wire::kRecLaneSlot, sizeof(slot));
  std::memcpy(dst, raw + Wire::kRecLaneDst, sizeof(dst));
  std::memcpy(weight, raw + Wire::kRecLaneWeight, sizeof(weight));
  const int count = Wire::kPackedCounts ? (counts & 0xF) : counts;
  const int protect = Wire::kPackedCounts ? (counts >> 4) : raw[Wire::kRecProtectCount];
  if (count > Wire::kLanes || protect > Wire::kLanes) return RecordRead::kMalformed;
  const auto kind_of = [&](int j) { return (kinds[j / 8] >> (4 * (j % 8))) & 0xFu; };
  bool bad_kind = false;
  for (int j = 0; j < count; ++j)
    bad_kind |= kind_of(j) < Wire::kKindHitCopy || kind_of(j) > Wire::kKindMissCpu;
  if (bad_kind) return RecordRead::kMalformed;
  request->seq = expected;
  request->gen = static_cast<uint64_t>(epoch) << 32 | expected;
  request->row = row;
  request->captured = (flags & Wire::kRecFlagCaptured) != 0;
  request->chain = chain;
  for (int i = 0; i < Wire::kLanes; ++i)
    request->protect[i] = protect_ids[i];
  request->protect.resize(protect);
  for (int j = 0; j < Wire::kLanes; ++j)
    request->lanes[j] = Lane{expert[j], slot[j], dst[j], weight[j], static_cast<uint8_t>(kind_of(j))};
  request->lanes.resize(count);
  return RecordRead::kOk;
}

// Ends the process with one line on stderr. Every host failure of the protocol comes here: nothing recovers, so
// nothing signals, and a device waiting on the request traps at its deadline or dies with the process.
[[noreturn]] inline void fail_stop(const std::string& message) {
  std::fprintf(stderr, "FATAL %s\n", message.c_str());
  std::fflush(stderr);
  std::abort();
}

// A fixed-capacity single-producer single-consumer queue of stage trace records.
//
// The service thread pushes and drops (counted) when full; one Python caller at a time drains. Allocated once, when
// the trace is enabled: a push copies a record into a preallocated slot and allocates nothing.
class StageRing {
 public:
  explicit StageRing(size_t capacity) : slots_(capacity) {}

  // Producer: appends `record`, or counts a drop when full. A record that gets in carries the drops before it.
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

  // Consumer: copies up to `max` records into `out`, oldest first. Returns how many.
  int64_t drain(StageRecord* out, int64_t max) {
    const uint64_t head = head_.load(std::memory_order_acquire);
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    int64_t count = 0;
    while (tail < head && count < max)
      out[count++] = slots_[tail++ % slots_.size()];
    tail_.store(tail, std::memory_order_release);
    return count;
  }

  // Records dropped over the ring's life.
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
