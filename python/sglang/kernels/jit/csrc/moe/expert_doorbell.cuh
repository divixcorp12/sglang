// Doorbell expert-row copies: a GPU kernel posts a copy plan into a pinned
// request page, a CPU spin thread copies the planned pinned rows on a CUDA
// stream, and GPU resolve kernels wait until the thread publishes completion
// and report per tag whether the plan was delivered.
//
// Request page (pinned host bytes):
//   [0]  u32 head      last posted sequence; release-stored LAST by the poster
//   [4]  u32 claimed   last sequence the thread committed to copy; written
//                      before its final abandoned/disabled check and before
//                      it queues any copy
//   [8]  u32 disabled  release-stored 1 by the first drain that runs out
//   [12] u32 reserved
//   [16] u32 abandoned[max_tags]  last sequence a resolve of that tag gave up on
//   [header_bytes] ring records, record k holds sequence (k + 1) mod ring:
//        {u32 seq, u32 count, u32 tag, u32 unserviced,
//         int64 source_rows[capacity], int32 destination_slots[capacity]}
//        the poster zeroes `unserviced`; the thread sets it before publishing
//        a request it did not copy
//
// Device state (int32 words): posted, disabled, degraded, timeouts, waits,
// last polls, record mismatches, resolved, drain timeouts, drains, drain
// pending, disabled posts, then one sequence per tag (0 = nothing posted).
// A separate int32 word per tag holds the delivered flag of its latest post.
// Completion lives in two device words {seq, unserviced} written only by the
// thread's eight-byte host-to-device copy, queued behind that request's row
// copies on the thread's stream, so the words reach a sequence only after
// the rows it covers have landed. Sequence 0 is never posted; it means
// "nothing posted" and "nothing abandoned".

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "expert_cache_transfer.cuh"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sglang {

namespace expert_doorbell {

constexpr int kBlockSize = 32;
constexpr int64_t kHeadOffset = 0;
constexpr int64_t kClaimedOffset = 4;
constexpr int64_t kDisabledOffset = 8;
constexpr int64_t kAbandonedBase = 16;
constexpr int64_t kRecordHeaderBytes = 16;
constexpr int64_t kRecordUnservicedOffset = 12;

constexpr int64_t kPosted = 0;
constexpr int64_t kDisabled = 1;
constexpr int64_t kDegraded = 2;
constexpr int64_t kTimeouts = 3;
constexpr int64_t kWaits = 4;
constexpr int64_t kLastPolls = 5;
constexpr int64_t kRecordMismatches = 6;
constexpr int64_t kResolved = 7;
constexpr int64_t kDrainTimeouts = 8;
constexpr int64_t kDrains = 9;
constexpr int64_t kDrainPending = 10;
constexpr int64_t kDisabledPosts = 11;
constexpr int64_t kTagBase = 12;

constexpr int64_t kFirstChunkPolls = 256;
constexpr int64_t kChunkGrowth = 4;
constexpr int64_t kFirstDrainPolls = 1 << 16;

constexpr int64_t kPollAcquire = 0;
constexpr int64_t kPollVolatile = 1;
constexpr int64_t kPollNoncoherent = 2;

constexpr int64_t kSegmentColumns = 5;
constexpr int64_t kTraceColumns = 8;
constexpr int64_t kTraceCapacity = 4096;

__device__ __host__ __forceinline__ int64_t record_bytes(int64_t capacity) {
  return (kRecordHeaderBytes + 12 * capacity + 7) / 8 * 8;
}

__device__ __host__ __forceinline__ int64_t header_bytes_for(int64_t max_tags) {
  return (kAbandonedBase + 4 * max_tags + 7) / 8 * 8;
}

__device__ __host__ __forceinline__ int64_t record_offset(uint32_t seq, int64_t capacity, int64_t ring, int64_t header_bytes) {
  return header_bytes + static_cast<int64_t>((seq - 1u) % static_cast<uint32_t>(ring)) * record_bytes(capacity);
}

constexpr int64_t kHeadStoreRelease = 0;
constexpr int64_t kHeadStoreVolatile = 1;

__device__ __forceinline__ void store_release_system(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void store_volatile_global(uint32_t* address, uint32_t value) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ void store_head(uint32_t* address, uint32_t value, int64_t head_store) {
  if (head_store == kHeadStoreVolatile) {
    store_volatile_global(address, value);
  } else {
    store_release_system(address, value);
  }
}

__device__ __forceinline__ uint32_t load_acquire_device(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ uint32_t load_volatile_device(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ uint32_t load_noncoherent_device(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ uint32_t load_completion(const uint32_t* done, int64_t poll_mode) {
  if (poll_mode == kPollVolatile) {
    return load_volatile_device(done);
  }
  if (poll_mode == kPollNoncoherent) {
    return load_noncoherent_device(done);
  }
  return load_acquire_device(done);
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __host__ __forceinline__ bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

// Polls the completion word for `seq` up to `limit` times (forever when
// `limit` is negative), confirms a completed poll with a device-scope
// acquire, and adds the polls to the last-polls word.
__device__ __forceinline__ uint32_t poll_completion(
    const uint32_t* done, uint32_t seq, int64_t limit, int64_t poll_mode, int32_t* state) {
  int64_t polls = 0;
  uint32_t observed;
  for (;;) {
    observed = load_completion(done, poll_mode);
    if (reached(observed, seq) || (limit >= 0 && polls >= limit)) {
      break;
    }
#if __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
    ++polls;
  }
  if (poll_mode != kPollAcquire && reached(observed, seq)) {
    observed = load_acquire_device(done);
  }
  state[kLastPolls] += static_cast<int32_t>(polls);
  return observed;
}

// Whether the thread copied the published `seq`. While the completion words
// still hold `seq`, their unserviced word answers with device loads, checked
// again after the load so a later publish cannot pair another request's flag
// with this sequence. Once a later request was published, the record's
// unserviced mark answers; an overwritten record counts as a mismatch and as
// not delivered.
__device__ __forceinline__ int32_t delivered_flag(
    const uint8_t* page, int32_t* state, const uint32_t* done, int64_t record, uint32_t seq) {
  if (load_acquire_device(done) == seq) {
    const uint32_t unserviced = load_acquire_device(done + 1);
    if (load_acquire_device(done) == seq) {
      return unserviced == 0 ? 1 : 0;
    }
  }
  const auto header = reinterpret_cast<const uint32_t*>(page + record);
  if (header[0] != seq) {
    state[kRecordMismatches] += 1;
    return 0;
  }
  const auto unserviced = reinterpret_cast<const uint32_t*>(page + record + kRecordUnservicedOffset);
  return load_acquire_system(unserviced) == 0 ? 1 : 0;
}

}  // namespace expert_doorbell

// Thread 0 claims the next sequence and fills the record header; every lane
// then copies its stride of the plan into the record; thread 0 finally
// release-stores the head, publishing the whole record to the host before
// the sequence that names it. The tag's delivered flag is cleared first.
// Once the copier is disabled nothing is posted: the tag's sequence is set
// to 0, so its resolve reports nothing delivered and the caller's residual
// copy serves the whole plan.
__global__ __launch_bounds__(expert_doorbell::kBlockSize, 1) void expert_doorbell_post_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    int32_t* __restrict__ delivered,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t head_store) {
  using namespace expert_doorbell;
  __shared__ uint32_t shared_seq;
  __shared__ int64_t shared_count;
  __shared__ int64_t shared_record;
  if (threadIdx.x == 0) {
    delivered[tag] = 0;
    if (state[kDisabled] != 0) {
      state[kTagBase + tag] = 0;
      state[kDisabledPosts] += 1;
      shared_seq = 0;
      shared_count = 0;
      shared_record = header_bytes;
    } else {
      const uint32_t next = static_cast<uint32_t>(state[kPosted]) + 1u;
      const uint32_t seq = next == 0 ? 1u : next;
      state[kPosted] = static_cast<int32_t>(seq);
      state[kTagBase + tag] = static_cast<int32_t>(seq);
      const int64_t requested = count[0];
      const int64_t active = requested < 0 ? 0 : (requested > capacity ? capacity : requested);
      const int64_t record = record_offset(seq, capacity, ring, header_bytes);
      auto header = reinterpret_cast<uint32_t*>(page + record);
      header[0] = seq;
      header[1] = static_cast<uint32_t>(active);
      header[2] = static_cast<uint32_t>(tag);
      header[3] = 0;
      shared_seq = seq;
      shared_count = active;
      shared_record = record;
    }
  }
  __syncthreads();
  auto record_rows = reinterpret_cast<int64_t*>(page + shared_record + kRecordHeaderBytes);
  auto record_slots = reinterpret_cast<int32_t*>(page + shared_record + kRecordHeaderBytes + 8 * capacity);
  for (int64_t entry = threadIdx.x; entry < shared_count; entry += kBlockSize) {
    record_rows[entry] = source_rows[entry];
    record_slots[entry] = destination_slots[entry];
  }
  __syncthreads();
  if (threadIdx.x == 0 && shared_seq != 0) {
    store_head(reinterpret_cast<uint32_t*>(page + kHeadOffset), shared_seq, head_store);
  }
}

// One chunk of a resolve. It polls the completion word with `poll_mode`
// loads and, once a poll shows the tag's sequence completed, confirms with
// one device-scope acquire so the row copies ordered before the completion
// write are visible to every later kernel, and sets the tag's delivered flag
// unless the thread marked the request unserviced. A running kernel holds
// back copies queued after it launched on a stream the thread created, so a
// resolve is a chain of launches with growing poll budgets; once a chunk
// resolves the sequence the remaining chunks return at once. If the final
// chunk also runs out, it marks the tag's sequence abandoned and the copier
// degraded, then reads the thread's claim. The thread claims a sequence
// before its final abandoned check and queues copies only after that check,
// so a sequence still unclaimed is never copied: it resolves undelivered at
// once. A claimed sequence may have copies queued and is left pending for the
// drain launches that follow. Only the resolving tag's sequence is waited
// for, so a later request for another tag never delays it.
__global__ __launch_bounds__(expert_doorbell::kBlockSize, 1) void expert_doorbell_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    int32_t* __restrict__ delivered,
    const uint32_t* __restrict__ done,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t chunk_timeout_polls,
    int64_t chunk_degraded_polls,
    int64_t chunk_index,
    int64_t final_chunk,
    int64_t poll_mode) {
  using namespace expert_doorbell;
  if (threadIdx.x != 0) {
    return;
  }
  const uint32_t seq = static_cast<uint32_t>(state[kTagBase + tag]);
  if (chunk_index == 0) {
    state[kLastPolls] = 0;
    state[kDrainPending] = 0;
    if (seq != 0) {
      state[kWaits] += 1;
    }
  }
  if (seq == 0 || static_cast<uint32_t>(state[kResolved]) == seq) {
    return;
  }
  const int64_t record = record_offset(seq, capacity, ring, header_bytes);
  const int64_t limit = state[kDegraded] != 0 ? chunk_degraded_polls : chunk_timeout_polls;
  const uint32_t observed = poll_completion(done, seq, limit, poll_mode, state);
  if (reached(observed, seq)) {
    state[kResolved] = static_cast<int32_t>(seq);
    state[kDegraded] = 0;
    delivered[tag] = delivered_flag(page, state, done, record, seq);
  } else if (final_chunk != 0) {
    state[kDegraded] = 1;
    state[kTimeouts] += 1;
    store_release_system(reinterpret_cast<uint32_t*>(page + kAbandonedBase + 4 * tag), seq);
    if (reached(load_acquire_system(reinterpret_cast<const uint32_t*>(page + kClaimedOffset)), seq)) {
      state[kDrains] += 1;
      state[kDrainPending] = static_cast<int32_t>(seq);
    } else {
      state[kResolved] = static_cast<int32_t>(seq);
    }
  }
}

// One chunk of a drain, which runs only for the sequence the final resolve
// chunk left pending: the thread had committed to copy it. It polls the
// completion word until the thread publishes the sequence, which the thread
// queues behind that request's copies, so they land before the resolve
// returns instead of after a later write to the same slots. If the final
// chunk runs out, the copier is disabled for good (the thread discards every
// request it has not committed to, and later posts post nothing) and the
// chunk keeps waiting without a bound for this committed request's copies:
// once queued on a live stream they complete, and returning earlier would
// let them land on rows a later forward writes.
__global__ __launch_bounds__(expert_doorbell::kBlockSize, 1) void expert_doorbell_drain_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    int32_t* __restrict__ delivered,
    const uint32_t* __restrict__ done,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t chunk_polls,
    int64_t final_chunk,
    int64_t poll_mode) {
  using namespace expert_doorbell;
  if (threadIdx.x != 0) {
    return;
  }
  const uint32_t seq = static_cast<uint32_t>(state[kTagBase + tag]);
  if (seq == 0 || state[kDrainPending] == 0 || static_cast<uint32_t>(state[kDrainPending]) != seq) {
    return;
  }
  const int64_t record = record_offset(seq, capacity, ring, header_bytes);
  uint32_t observed = poll_completion(done, seq, chunk_polls, poll_mode, state);
  if (!reached(observed, seq)) {
    if (final_chunk == 0) {
      return;
    }
    state[kDrainTimeouts] += 1;
    if (state[kDisabled] == 0) {
      state[kDisabled] = 1;
      store_release_system(reinterpret_cast<uint32_t*>(page + kDisabledOffset), 1u);
    }
    observed = poll_completion(done, seq, -1, poll_mode, state);
  }
  state[kDrainPending] = 0;
  state[kResolved] = static_cast<int32_t>(seq);
  state[kDegraded] = 0;
  delivered[tag] = delivered_flag(page, state, done, record, seq);
}

void expert_doorbell_post(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView delivered,
    tvm::ffi::TensorView source_rows,
    tvm::ffi::TensorView destination_slots,
    tvm::ffi::TensorView count,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t head_store) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, expert_doorbell::kBlockSize, stream)(
      expert_doorbell_post_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<int32_t*>(delivered.data_ptr()),
      static_cast<const int64_t*>(source_rows.data_ptr()),
      static_cast<const int32_t*>(destination_slots.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      tag,
      capacity,
      ring,
      header_bytes,
      head_store);
}

void expert_doorbell_resolve(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView delivered,
    tvm::ffi::TensorView done,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t timeout_polls,
    int64_t degraded_polls,
    int64_t drain_polls,
    int64_t poll_mode) {
  using namespace expert_doorbell;
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  const auto page_data = static_cast<uint8_t*>(page.data_ptr());
  const auto state_data = static_cast<int32_t*>(state.data_ptr());
  const auto delivered_data = static_cast<int32_t*>(delivered.data_ptr());
  const auto done_data = static_cast<const uint32_t*>(done.data_ptr());
  int64_t timeout_left = timeout_polls;
  int64_t degraded_left = degraded_polls;
  int64_t budget = kFirstChunkPolls;
  for (int64_t chunk = 0;; ++chunk) {
    const int64_t chunk_timeout = budget < timeout_left ? budget : timeout_left;
    const int64_t chunk_degraded = budget < degraded_left ? budget : degraded_left;
    timeout_left -= chunk_timeout;
    degraded_left -= chunk_degraded;
    const bool final_chunk = timeout_left == 0 && degraded_left == 0;
    host::LaunchKernel(1, kBlockSize, stream)(
        expert_doorbell_wait_kernel,
        page_data,
        state_data,
        delivered_data,
        done_data,
        tag,
        capacity,
        ring,
        header_bytes,
        chunk_timeout,
        chunk_degraded,
        chunk,
        static_cast<int64_t>(final_chunk),
        poll_mode);
    if (final_chunk) {
      break;
    }
    budget *= kChunkGrowth;
  }
  int64_t drain_left = drain_polls;
  for (int64_t drain_chunk = 0;; ++drain_chunk) {
    const int64_t chunk_polls =
        drain_chunk == 0 && kFirstDrainPolls < drain_left ? kFirstDrainPolls : drain_left;
    drain_left -= chunk_polls;
    const bool final_chunk = drain_left == 0;
    host::LaunchKernel(1, kBlockSize, stream)(
        expert_doorbell_drain_kernel,
        page_data,
        state_data,
        delivered_data,
        done_data,
        tag,
        capacity,
        ring,
        header_bytes,
        chunk_polls,
        static_cast<int64_t>(final_chunk),
        poll_mode);
    if (final_chunk) {
      break;
    }
  }
}

namespace expert_doorbell {

enum class RequestStatus : int64_t {
  kPending = 0,
  kServiced = 1,
  kSkippedAbandoned = 2,
  kSkippedOverrun = 3,
  kInvalidRecord = 4,
  kCopyFailed = 5,
  kDiscardedDisabled = 6,
};

enum Counter : int64_t {
  kServiced = 0,
  kSkippedAbandonedCount,
  kSkippedOverrunCount,
  kInvalidRecords,
  kCopyErrors,
  kRowsCopied,
  kBytesCopied,
  kTraceCount,
  kRunning,
  kSpinCpu,
  kLastSeen,
  kLastCopyError,
  kCopyApi,
  kSrcAccessOrder,
  kExternalStream,
  kLateCompletions,
  kDiscardedDisabledCount,
  kCounterCount,
};

enum CopyApi : int64_t {
  kBatchCopy = 0,
  kPerSegmentCopy = 1,
};

struct TraceEntry {
  int64_t seq = 0;
  int64_t count = 0;
  int64_t seen_ns = 0;
  int64_t enqueued_ns = 0;
  int64_t complete_ns = 0;
  int64_t bytes = 0;
  int64_t status = 0;
  int64_t publish_slot = -1;
};

struct Segment {
  uint64_t source;
  uint64_t destination;
  uint64_t row_bytes;
  int64_t source_rows;
  int64_t destination_rows;
};

inline int64_t monotonic_ns() {
  timespec now{};
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<int64_t>(now.tv_sec) * 1000000000LL + now.tv_nsec;
}

inline uint32_t read_host_word(const uint8_t* address) {
  return *reinterpret_cast<const volatile uint32_t*>(address);
}

inline void write_host_word(uint8_t* address, uint32_t value) {
  *reinterpret_cast<volatile uint32_t*>(address) = value;
}

inline size_t widest_segment_set(const std::vector<int64_t>& set_offsets) {
  size_t widest = 0;
  for (size_t index = 1; index < set_offsets.size(); ++index) {
    const size_t span = static_cast<size_t>(set_offsets[index] - set_offsets[index - 1]);
    widest = span > widest ? span : widest;
  }
  return widest;
}

/// A CPU spin thread that services requests posted to one request page.
///
/// It copies each request's rows on a CUDA stream, either one it creates or one
/// the caller passes in, with one batched copy (``copy_api`` kBatchCopy, using
/// ``src_access_order`` for every copy) or one copy per segment row
/// (kPerSegmentCopy), and then queues a four-byte host-to-device copy of the
/// request's sequence into the device completion word. Segments are grouped
/// into sets by ``set_offsets``: with one set every request copies it,
/// otherwise a request copies the set its tag names and a tag without a set is
/// an invalid record. Before queuing any copy the thread writes the claimed
/// word, then checks the disabled word and the tag's abandoned word; it copies
/// only if neither applies. Every request ends in a publish; one the thread did
/// not copy is first marked unserviced in its record. ``late_completions``
/// counts serviced requests at or before their tag's abandoned sequence:
/// copies committed before their resolve timed out, which that resolve's
/// drain waited for. Every publish reads its own pinned word, which is never
/// rewritten while an earlier publish from it is still queued.
class DoorbellThread {
 public:
  DoorbellThread(
      uint8_t* page,
      int64_t capacity,
      int64_t ring,
      int64_t header_bytes,
      int64_t max_tags,
      std::vector<Segment> segments,
      std::vector<int64_t> set_offsets,
      uint32_t* done,
      int32_t* publish_words,
      int64_t publish_ring,
      int device,
      int cpu_core,
      bool prefer_overlap,
      int64_t copy_api,
      int64_t src_access_order,
      ::cudaStream_t external_stream)
      : page_(page),
        capacity_(capacity),
        ring_(ring),
        header_bytes_(header_bytes),
        max_tags_(max_tags),
        segments_(std::move(segments)),
        set_offsets_(std::move(set_offsets)),
        widest_set_(widest_segment_set(set_offsets_)),
        done_(done),
        publish_words_(publish_words),
        publish_ring_(publish_ring),
        device_(device),
        cpu_core_(cpu_core),
        prefer_overlap_(prefer_overlap),
        copy_api_(copy_api),
        src_access_order_(src_access_order),
        owns_stream_(external_stream == nullptr),
        rows_(capacity),
        slots_(capacity),
        sources_(capacity * widest_set_),
        destinations_(capacity * widest_set_),
        sizes_(capacity * widest_set_),
        publish_events_(publish_ring),
        publish_trace_(publish_ring, -1),
        publish_serviced_(publish_ring, 0),
        publish_tags_(publish_ring, 0),
        stream_(external_stream),
        trace_(kTraceCapacity) {
    for (auto& counter : counters_) {
      counter.store(0);
    }
    counters_[kCopyApi].store(copy_api);
    counters_[kSrcAccessOrder].store(src_access_order);
    counters_[kExternalStream].store(owns_stream_ ? 0 : 1);
  }

  bool start() {
    thread_ = std::thread([this] { run(); });
    while (!started_.load() && !failed_.load()) {
      std::this_thread::yield();
    }
    if (failed_.load()) {
      thread_.join();
      return false;
    }
    return true;
  }

  void stop() {
    stop_.store(true);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void pause(bool paused) {
    paused_.store(paused);
  }

  /// Fault injection: sleep `service_delay_ns` after committing to copy a
  /// request and before queuing its copies, and report every copy as failed
  /// without issuing it when `fail_copies` is set.
  void inject(int64_t service_delay_ns, bool fail_copies) {
    service_delay_ns_.store(service_delay_ns);
    fail_copies_.store(fail_copies);
  }

  void counters(int64_t* out) const {
    for (int64_t index = 0; index < kCounterCount; ++index) {
      out[index] = counters_[index].load();
    }
  }

  int64_t trace(int64_t* out, int64_t rows) {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    const int64_t total = counters_[kTraceCount].load();
    const int64_t available = total < kTraceCapacity ? total : kTraceCapacity;
    const int64_t written = available < rows ? available : rows;
    for (int64_t index = 0; index < written; ++index) {
      const TraceEntry& entry = trace_[(total - written + index) % kTraceCapacity];
      const int64_t values[kTraceColumns] = {
          entry.seq,
          entry.count,
          entry.seen_ns,
          entry.enqueued_ns,
          entry.complete_ns,
          entry.bytes,
          entry.status,
          entry.publish_slot};
      std::memcpy(out + index * kTraceColumns, values, sizeof(values));
    }
    return written;
  }

 private:
  void run() {
    if (::cudaSetDevice(device_) != ::cudaSuccess || (owns_stream_ && ::cudaStreamCreate(&stream_) != ::cudaSuccess)) {
      failed_.store(true);
      return;
    }
    if (cpu_core_ >= 0) {
      cpu_set_t cpus;
      CPU_ZERO(&cpus);
      CPU_SET(cpu_core_, &cpus);
      pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus);
    }
    for (auto& event : publish_events_) {
      if (::cudaEventCreate(&event) != ::cudaSuccess) {
        failed_.store(true);
        return;
      }
    }
    counters_[kSpinCpu].store(sched_getcpu());
    counters_[kRunning].store(1);
    uint32_t seen = read_host_word(page_ + kHeadOffset);
    counters_[kLastSeen].store(seen);
    started_.store(true);
    for (;;) {
      const bool stopping = stop_.load(std::memory_order_relaxed);
      poll_completions();
      if (paused_.load(std::memory_order_relaxed) && !stopping) {
        __builtin_ia32_pause();
        continue;
      }
      const uint32_t head = read_host_word(page_ + kHeadOffset);
      if (head != seen) {
        const int64_t seen_ns = monotonic_ns();
        for (uint32_t seq = seen + 1u; seq != head + 1u; ++seq) {
          service(seq, head, seen_ns);
        }
        seen = head;
        counters_[kLastSeen].store(seen);
        continue;
      }
      if (stopping) {
        break;
      }
      __builtin_ia32_pause();
    }
    ::cudaStreamSynchronize(stream_);
    poll_completions();
    for (auto event : publish_events_) {
      ::cudaEventDestroy(event);
    }
    if (owns_stream_) {
      ::cudaStreamDestroy(stream_);
    }
    counters_[kRunning].store(0);
  }

  /// Service one request. The thread writes the claimed word, then reads the
  /// disabled word and the tag's abandoned word, and queues copies only after
  /// that read: a resolve that stores its abandoned word and then finds the
  /// request unclaimed knows no copy of it will ever be queued, and a drain
  /// that stores disabled knows the thread copies nothing it had not already
  /// committed to. Every request ends in a publish.
  void service(uint32_t seq, uint32_t head, int64_t seen_ns) {
    TraceEntry entry;
    entry.seq = seq;
    entry.seen_ns = seen_ns;
    uint8_t* const record = page_ + record_offset(seq, capacity_, ring_, header_bytes_);
    const uint32_t tag = read_host_word(record + 8);
    if (static_cast<int64_t>(head - seq) >= ring_ - 1) {
      counters_[kSkippedOverrunCount].fetch_add(1);
      finish_unserviced(seq, 0, entry, record, RequestStatus::kSkippedOverrun);
      return;
    }
    const uint32_t count = read_host_word(record + 4);
    const size_t sets = set_offsets_.size() - 1;
    const size_t set = sets == 1 ? 0 : static_cast<size_t>(tag);
    if (read_host_word(record) != seq || count > static_cast<uint32_t>(capacity_) || set >= sets ||
        static_cast<int64_t>(tag) >= max_tags_) {
      counters_[kInvalidRecords].fetch_add(1);
      finish_unserviced(seq, 0, entry, record, RequestStatus::kInvalidRecord);
      return;
    }
    std::memcpy(rows_.data(), record + kRecordHeaderBytes, 8 * count);
    std::memcpy(slots_.data(), record + kRecordHeaderBytes + 8 * capacity_, 4 * count);
    if (read_host_word(record) != seq) {
      counters_[kInvalidRecords].fetch_add(1);
      finish_unserviced(seq, tag, entry, record, RequestStatus::kInvalidRecord);
      return;
    }
    entry.count = count;

    size_t copies = 0;
    int64_t rows_copied = 0;
    const auto first = segments_.begin() + set_offsets_[set];
    const auto last = segments_.begin() + set_offsets_[set + 1];
    for (uint32_t index = 0; index < count; ++index) {
      const int64_t row = rows_[index];
      const int64_t slot = slots_[index];
      for (auto segment = first; segment != last; ++segment) {
        if (row < 0 || row >= segment->source_rows || slot < 0 || slot >= segment->destination_rows) {
          counters_[kCopyErrors].fetch_add(1);
          finish_unserviced(seq, tag, entry, record, RequestStatus::kInvalidRecord);
          return;
        }
      }
    }
    for (uint32_t index = 0; index < count; ++index) {
      const int64_t row = rows_[index];
      const int64_t slot = slots_[index];
      for (auto it = first; it != last; ++it) {
        const Segment& segment = *it;
        sources_[copies] = reinterpret_cast<void*>(segment.source + static_cast<uint64_t>(row) * segment.row_bytes);
        destinations_[copies] =
            reinterpret_cast<void*>(segment.destination + static_cast<uint64_t>(slot) * segment.row_bytes);
        sizes_[copies] = segment.row_bytes;
        entry.bytes += static_cast<int64_t>(segment.row_bytes);
        ++copies;
      }
      ++rows_copied;
    }

    write_host_word(page_ + kClaimedOffset, seq);
    _mm_mfence();
    if (read_host_word(page_ + kDisabledOffset) != 0) {
      counters_[kDiscardedDisabledCount].fetch_add(1);
      finish_unserviced(seq, tag, entry, record, RequestStatus::kDiscardedDisabled);
      return;
    }
    const uint32_t abandoned = read_host_word(page_ + kAbandonedBase + 4 * static_cast<int64_t>(tag));
    if (abandoned != 0 && static_cast<int32_t>(seq - abandoned) <= 0) {
      counters_[kSkippedAbandonedCount].fetch_add(1);
      finish_unserviced(seq, tag, entry, record, RequestStatus::kSkippedAbandoned);
      return;
    }

    const int64_t service_delay_ns = service_delay_ns_.load(std::memory_order_relaxed);
    if (service_delay_ns > 0) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(service_delay_ns));
    }
    const ::cudaError_t copy_status = fail_copies_.load(std::memory_order_relaxed) ? ::cudaErrorUnknown
                                      : copies == 0                                ? ::cudaSuccess
                                      : copy_api_ == kPerSegmentCopy               ? copy_per_segment(copies)
                                                                                   : copy_batch(copies);
    if (copy_status != ::cudaSuccess) {
      counters_[kCopyErrors].fetch_add(1);
      counters_[kLastCopyError].store(static_cast<int64_t>(copy_status));
      finish_unserviced(seq, tag, entry, record, RequestStatus::kCopyFailed);
      return;
    }

    entry.status = static_cast<int64_t>(RequestStatus::kServiced);
    counters_[kRowsCopied].fetch_add(rows_copied);
    counters_[kBytesCopied].fetch_add(entry.bytes);
    counters_[kServiced].fetch_add(1);
    publish(seq, tag, entry, true);
  }

  /// Mark a request the thread did not copy as unserviced in its record, when
  /// the record still holds it, then publish it, so its resolve reports it
  /// undelivered. A partial copy queued before a failure lands before that
  /// publish.
  void finish_unserviced(uint32_t seq, uint32_t tag, TraceEntry& entry, uint8_t* record, RequestStatus status) {
    entry.status = static_cast<int64_t>(status);
    if (read_host_word(record) == seq) {
      write_host_word(record + kRecordUnservicedOffset, 1);
      _mm_mfence();
    }
    publish(seq, tag, entry, false);
  }

  /// Queue the eight-byte publish of `{seq, unserviced}` into the completion
  /// words behind everything already queued on the stream.
  void publish(uint32_t seq, uint32_t tag, TraceEntry& entry, bool serviced) {
    const int64_t slot = publish_cursor_++ % publish_ring_;
    if (publish_trace_[slot] >= 0) {
      ::cudaEventSynchronize(publish_events_[slot]);
      poll_completions();
    }
    publish_words_[2 * slot] = static_cast<int32_t>(seq);
    publish_words_[2 * slot + 1] = serviced ? 0 : 1;
    publish_serviced_[slot] = serviced ? 1 : 0;
    publish_tags_[slot] = static_cast<int64_t>(tag) < max_tags_ ? tag : 0;
    ::cudaMemcpyAsync(done_, publish_words_ + 2 * slot, 2 * sizeof(uint32_t), ::cudaMemcpyHostToDevice, stream_);
    ::cudaEventRecord(publish_events_[slot], stream_);
    entry.enqueued_ns = monotonic_ns();
    entry.publish_slot = slot;
    publish_trace_[slot] = push_trace(entry);
    pending_.push_back(slot);
  }

  ::cudaError_t copy_batch(size_t copies) {
    ::cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = static_cast<decltype(attributes.srcAccessOrder)>(src_access_order_);
    attributes.flags = prefer_overlap_ ? ::cudaMemcpyFlagPreferOverlapWithCompute : ::cudaMemcpyFlagDefault;
    size_t attribute_index = 0;
    return ::cudaMemcpyBatchAsync(
        destinations_.data(),
        const_cast<const void* const*>(sources_.data()),
        sizes_.data(),
        copies,
        &attributes,
        &attribute_index,
        1,
        stream_);
  }

  ::cudaError_t copy_per_segment(size_t copies) {
    for (size_t index = 0; index < copies; ++index) {
      const ::cudaError_t status = ::cudaMemcpyAsync(
          destinations_[index], sources_[index], sizes_[index], ::cudaMemcpyHostToDevice, stream_);
      if (status != ::cudaSuccess) {
        return status;
      }
    }
    return ::cudaSuccess;
  }

  void poll_completions() {
    while (pending_head_ < pending_.size()) {
      const int64_t slot = pending_[pending_head_];
      if (::cudaEventQuery(publish_events_[slot]) != ::cudaSuccess) {
        break;
      }
      const int64_t complete_ns = monotonic_ns();
      const uint32_t seq = static_cast<uint32_t>(publish_words_[2 * slot]);
      const uint32_t abandoned =
          read_host_word(page_ + kAbandonedBase + 4 * static_cast<int64_t>(publish_tags_[slot]));
      if (publish_serviced_[slot] != 0 && abandoned != 0 && static_cast<int32_t>(seq - abandoned) <= 0) {
        counters_[kLateCompletions].fetch_add(1);
      }
      {
        std::lock_guard<std::mutex> guard(trace_mutex_);
        const int64_t trace_index = publish_trace_[slot];
        if (trace_index >= 0 && counters_[kTraceCount].load() - trace_index <= kTraceCapacity) {
          trace_[trace_index % kTraceCapacity].complete_ns = complete_ns;
        }
      }
      publish_trace_[slot] = -1;
      ++pending_head_;
    }
    if (pending_head_ == pending_.size()) {
      pending_.clear();
      pending_head_ = 0;
    }
  }

  int64_t push_trace(const TraceEntry& entry) {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    const int64_t index = counters_[kTraceCount].load();
    trace_[index % kTraceCapacity] = entry;
    counters_[kTraceCount].store(index + 1);
    return index;
  }

  uint8_t* page_;
  int64_t capacity_;
  int64_t ring_;
  int64_t header_bytes_;
  int64_t max_tags_;
  std::vector<Segment> segments_;
  std::vector<int64_t> set_offsets_;
  size_t widest_set_;
  uint32_t* done_;
  int32_t* publish_words_;
  int64_t publish_ring_;
  int device_;
  int cpu_core_;
  bool prefer_overlap_;
  int64_t copy_api_;
  int64_t src_access_order_;
  bool owns_stream_;
  std::vector<int64_t> rows_;
  std::vector<int32_t> slots_;
  std::vector<void*> sources_;
  std::vector<void*> destinations_;
  std::vector<size_t> sizes_;
  std::vector<::cudaEvent_t> publish_events_;
  std::vector<int64_t> publish_trace_;
  std::vector<char> publish_serviced_;
  std::vector<uint32_t> publish_tags_;
  std::vector<int64_t> pending_;
  size_t pending_head_ = 0;
  int64_t publish_cursor_ = 0;
  ::cudaStream_t stream_ = nullptr;
  std::thread thread_;
  std::atomic<bool> started_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> stop_{false};
  std::atomic<bool> paused_{false};
  std::atomic<int64_t> service_delay_ns_{0};
  std::atomic<bool> fail_copies_{false};
  std::atomic<int64_t> counters_[kCounterCount];
  std::mutex trace_mutex_;
  std::vector<TraceEntry> trace_;
};

inline std::mutex& registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline std::unordered_map<int64_t, std::unique_ptr<DoorbellThread>>& registry() {
  static std::unordered_map<int64_t, std::unique_ptr<DoorbellThread>> threads;
  return threads;
}

inline DoorbellThread* find_thread(int64_t handle) {
  std::lock_guard<std::mutex> guard(registry_mutex());
  const auto found = registry().find(handle);
  return found == registry().end() ? nullptr : found->second.get();
}

}  // namespace expert_doorbell

int64_t expert_doorbell_start(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView segment_table,
    tvm::ffi::TensorView set_offsets,
    tvm::ffi::TensorView done,
    tvm::ffi::TensorView publish_words,
    int64_t capacity,
    int64_t ring,
    int64_t header_bytes,
    int64_t max_tags,
    int64_t device,
    int64_t cpu_core,
    int64_t prefer_overlap,
    int64_t copy_api,
    int64_t src_access_order,
    int64_t external_stream) {
  using namespace expert_doorbell;
  const auto table = static_cast<const int64_t*>(segment_table.data_ptr());
  std::vector<Segment> segments(segment_table.size(0));
  for (size_t index = 0; index < segments.size(); ++index) {
    const int64_t* entry = table + index * kSegmentColumns;
    segments[index] = Segment{
        static_cast<uint64_t>(entry[0]), static_cast<uint64_t>(entry[1]), static_cast<uint64_t>(entry[2]), entry[3],
        entry[4]};
  }
  const auto offsets = static_cast<const int64_t*>(set_offsets.data_ptr());
  std::vector<int64_t> set_bounds(offsets, offsets + set_offsets.size(0));
  auto thread = std::make_unique<DoorbellThread>(
      static_cast<uint8_t*>(page.data_ptr()),
      capacity,
      ring,
      header_bytes,
      max_tags,
      std::move(segments),
      std::move(set_bounds),
      static_cast<uint32_t*>(done.data_ptr()),
      static_cast<int32_t*>(publish_words.data_ptr()),
      publish_words.size(0) / 2,
      static_cast<int>(device),
      static_cast<int>(cpu_core),
      prefer_overlap != 0,
      copy_api,
      src_access_order,
      reinterpret_cast<::cudaStream_t>(static_cast<intptr_t>(external_stream)));
  if (!thread->start()) {
    return -1;
  }
  std::lock_guard<std::mutex> guard(registry_mutex());
  static int64_t next_handle = 1;
  const int64_t handle = next_handle++;
  registry().emplace(handle, std::move(thread));
  return handle;
}

void expert_doorbell_stop(int64_t handle) {
  using namespace expert_doorbell;
  std::unique_ptr<DoorbellThread> thread;
  {
    std::lock_guard<std::mutex> guard(registry_mutex());
    const auto found = registry().find(handle);
    if (found == registry().end()) {
      return;
    }
    thread = std::move(found->second);
    registry().erase(found);
  }
  thread->stop();
}

void expert_doorbell_pause(int64_t handle, int64_t paused) {
  if (auto thread = expert_doorbell::find_thread(handle)) {
    thread->pause(paused != 0);
  }
}

void expert_doorbell_inject(int64_t handle, int64_t service_delay_ns, int64_t fail_copies) {
  if (auto thread = expert_doorbell::find_thread(handle)) {
    thread->inject(service_delay_ns, fail_copies != 0);
  }
}

int64_t expert_doorbell_counters(int64_t handle, tvm::ffi::TensorView out) {
  if (auto thread = expert_doorbell::find_thread(handle)) {
    thread->counters(static_cast<int64_t*>(out.data_ptr()));
    return 1;
  }
  return 0;
}

int64_t expert_doorbell_trace(int64_t handle, tvm::ffi::TensorView out) {
  if (auto thread = expert_doorbell::find_thread(handle)) {
    return thread->trace(static_cast<int64_t*>(out.data_ptr()), out.size(0));
  }
  return 0;
}

}  // namespace sglang
