// Doorbell expert-row copies: a GPU kernel posts a copy plan into a pinned
// request page, a CPU spin thread copies the planned pinned rows on its own
// CUDA stream, and a GPU waiter blocks until the thread publishes completion.
//
// Request page (pinned host bytes, GPU-written, CPU-read):
//   [0]  u32 head      last posted sequence; release-stored LAST by the poster
//   [4]  u32 abandoned last sequence a waiter gave up on
//   [16] ring records, record k holds sequence (k + 1) mod ring:
//        {u32 seq, u32 count, u32 tag, u32 reserved,
//         int64 source_rows[capacity], int32 destination_slots[capacity]}
//
// Device state (int32 words): posted, fallback count, degraded, timeouts,
// waits, last polls, record mismatches, resolved, then one sequence per tag.
// Completion lives in a separate device word written only by the thread's
// four-byte host-to-device copy, queued behind that request's row copies on
// the thread's stream, so the word reaches a sequence only after the rows it
// covers have landed.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "expert_cache_transfer.cuh"

#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <atomic>
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
constexpr int64_t kAbandonedOffset = 4;
constexpr int64_t kPageHeaderBytes = 16;
constexpr int64_t kRecordHeaderBytes = 16;

constexpr int64_t kPosted = 0;
constexpr int64_t kFallbackCount = 1;
constexpr int64_t kDegraded = 2;
constexpr int64_t kTimeouts = 3;
constexpr int64_t kWaits = 4;
constexpr int64_t kLastPolls = 5;
constexpr int64_t kRecordMismatches = 6;
constexpr int64_t kResolved = 7;
constexpr int64_t kTagBase = 8;

constexpr int64_t kFirstChunkPolls = 256;
constexpr int64_t kChunkGrowth = 4;

constexpr int64_t kPollAcquire = 0;
constexpr int64_t kPollVolatile = 1;
constexpr int64_t kPollNoncoherent = 2;

constexpr int64_t kSegmentColumns = 5;
constexpr int64_t kTraceColumns = 8;
constexpr int64_t kTraceCapacity = 4096;

__device__ __host__ __forceinline__ int64_t record_bytes(int64_t capacity) {
  return (kRecordHeaderBytes + 12 * capacity + 7) / 8 * 8;
}

__device__ __host__ __forceinline__ int64_t record_offset(uint32_t seq, int64_t capacity, int64_t ring) {
  return kPageHeaderBytes + static_cast<int64_t>((seq - 1u) % static_cast<uint32_t>(ring)) * record_bytes(capacity);
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

}  // namespace expert_doorbell

// Thread 0 claims the next sequence and fills the record header; every lane
// then copies its stride of the plan into the record; thread 0 finally
// release-stores the head, publishing the whole record to the host before
// the sequence that names it.
__global__ __launch_bounds__(expert_doorbell::kBlockSize, 1) void expert_doorbell_post_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const int64_t* __restrict__ source_rows,
    const int32_t* __restrict__ destination_slots,
    const int32_t* __restrict__ count,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t head_store) {
  using namespace expert_doorbell;
  __shared__ uint32_t shared_seq;
  __shared__ int64_t shared_count;
  __shared__ int64_t shared_record;
  if (threadIdx.x == 0) {
    const uint32_t seq = static_cast<uint32_t>(state[kPosted]) + 1u;
    state[kPosted] = static_cast<int32_t>(seq);
    state[kTagBase + tag] = static_cast<int32_t>(seq);
    const int64_t requested = count[0];
    const int64_t active = requested < 0 ? 0 : (requested > capacity ? capacity : requested);
    const int64_t record = record_offset(seq, capacity, ring);
    auto header = reinterpret_cast<uint32_t*>(page + record);
    header[0] = seq;
    header[1] = static_cast<uint32_t>(active);
    header[2] = static_cast<uint32_t>(tag);
    header[3] = 0;
    shared_seq = seq;
    shared_count = active;
    shared_record = record;
  }
  __syncthreads();
  auto record_rows = reinterpret_cast<int64_t*>(page + shared_record + kRecordHeaderBytes);
  auto record_slots = reinterpret_cast<int32_t*>(page + shared_record + kRecordHeaderBytes + 8 * capacity);
  for (int64_t entry = threadIdx.x; entry < shared_count; entry += kBlockSize) {
    record_rows[entry] = source_rows[entry];
    record_slots[entry] = destination_slots[entry];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    store_head(reinterpret_cast<uint32_t*>(page + kHeadOffset), shared_seq, head_store);
  }
}

// One chunk of a wait. It polls the completion word with `poll_mode` loads
// and, once a poll shows the tag's sequence completed, confirms with one
// device-scope acquire so the row copies ordered before the completion write
// are visible to every later kernel. A running kernel holds back copies the
// thread queues after it launched, so a wait is a chain of launches with
// growing poll budgets and those copies run between chunks; once a chunk
// resolves the sequence the remaining chunks return at once. If the final
// chunk also runs out, it marks the sequence abandoned and the copier
// degraded, and loads the request's plan back from its record so the
// fallback launch that follows copies it; otherwise the fallback count stays
// zero.
__global__ __launch_bounds__(expert_doorbell::kBlockSize, 1) void expert_doorbell_wait_kernel(
    uint8_t* __restrict__ page,
    int32_t* __restrict__ state,
    const uint32_t* __restrict__ done,
    int64_t* __restrict__ fallback_rows,
    int32_t* __restrict__ fallback_slots,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t chunk_timeout_polls,
    int64_t chunk_degraded_polls,
    int64_t chunk_index,
    int64_t final_chunk,
    int64_t poll_mode) {
  using namespace expert_doorbell;
  __shared__ int64_t shared_fallback;
  __shared__ int64_t shared_record;
  if (threadIdx.x == 0) {
    const uint32_t seq = static_cast<uint32_t>(state[kTagBase + tag]);
    const int64_t record = record_offset(seq, capacity, ring);
    shared_record = record;
    shared_fallback = 0;
    if (chunk_index == 0) {
      state[kWaits] += 1;
      state[kLastPolls] = 0;
      state[kFallbackCount] = 0;
    }
    if (static_cast<uint32_t>(state[kResolved]) != seq) {
      const int64_t limit = state[kDegraded] != 0 ? chunk_degraded_polls : chunk_timeout_polls;
      int64_t polls = 0;
      uint32_t observed;
      for (;;) {
        observed = load_completion(done, poll_mode);
        if (static_cast<int32_t>(observed - seq) >= 0 || polls >= limit) {
          break;
        }
#if __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
        ++polls;
      }
      if (poll_mode != kPollAcquire && static_cast<int32_t>(observed - seq) >= 0) {
        observed = load_acquire_device(done);
      }
      state[kLastPolls] += static_cast<int32_t>(polls);
      if (static_cast<int32_t>(observed - seq) >= 0) {
        state[kResolved] = static_cast<int32_t>(seq);
        state[kDegraded] = 0;
      } else if (final_chunk != 0) {
        state[kDegraded] = 1;
        state[kTimeouts] += 1;
        store_release_system(reinterpret_cast<uint32_t*>(page + kAbandonedOffset), seq);
        const auto header = reinterpret_cast<const uint32_t*>(page + record);
        if (header[0] == seq) {
          shared_fallback = static_cast<int64_t>(header[1]);
        } else {
          state[kRecordMismatches] += 1;
        }
        state[kFallbackCount] = static_cast<int32_t>(shared_fallback);
      }
    }
  }
  __syncthreads();
  const auto record_rows = reinterpret_cast<const int64_t*>(page + shared_record + kRecordHeaderBytes);
  const auto record_slots = reinterpret_cast<const int32_t*>(page + shared_record + kRecordHeaderBytes + 8 * capacity);
  for (int64_t entry = threadIdx.x; entry < shared_fallback; entry += kBlockSize) {
    fallback_rows[entry] = record_rows[entry];
    fallback_slots[entry] = record_slots[entry];
  }
}

void expert_doorbell_post(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView source_rows,
    tvm::ffi::TensorView destination_slots,
    tvm::ffi::TensorView count,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t head_store) {
  const auto stream = host::LaunchKernel::resolve_device(state.device());
  host::LaunchKernel(1, expert_doorbell::kBlockSize, stream)(
      expert_doorbell_post_kernel,
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<int32_t*>(state.data_ptr()),
      static_cast<const int64_t*>(source_rows.data_ptr()),
      static_cast<const int32_t*>(destination_slots.data_ptr()),
      static_cast<const int32_t*>(count.data_ptr()),
      tag,
      capacity,
      ring,
      head_store);
}

void expert_doorbell_wait(
    tvm::ffi::TensorView page,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView done,
    tvm::ffi::TensorView fallback_rows,
    tvm::ffi::TensorView fallback_slots,
    tvm::ffi::TensorView segments,
    int64_t tag,
    int64_t capacity,
    int64_t ring,
    int64_t timeout_polls,
    int64_t degraded_polls,
    int64_t poll_mode) {
  using namespace expert_doorbell;
  const auto stream = host::LaunchKernel::resolve_device(state.device());
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
        static_cast<uint8_t*>(page.data_ptr()),
        static_cast<int32_t*>(state.data_ptr()),
        static_cast<const uint32_t*>(done.data_ptr()),
        static_cast<int64_t*>(fallback_rows.data_ptr()),
        static_cast<int32_t*>(fallback_slots.data_ptr()),
        tag,
        capacity,
        ring,
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
  host::LaunchKernel(kExpertTransferGridSize, kExpertTransferBlockSize, stream)(
      copy_expert_row_segments_gpu_kernel,
      static_cast<const int64_t*>(segments.data_ptr()),
      static_cast<int64_t>(segments.size(0)),
      static_cast<const int64_t*>(fallback_rows.data_ptr()),
      static_cast<const int32_t*>(fallback_slots.data_ptr()),
      static_cast<const int32_t*>(state.data_ptr()) + expert_doorbell::kFallbackCount);
}

namespace expert_doorbell {

enum class RequestStatus : int64_t {
  kPending = 0,
  kServiced = 1,
  kSkippedAbandoned = 2,
  kSkippedOverrun = 3,
  kInvalidRecord = 4,
  kCopyFailed = 5,
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
  kCounterCount,
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

/// A CPU spin thread that services requests posted to one request page.
///
/// It copies each request's rows with one batched copy on its own CUDA stream
/// and then queues a four-byte host-to-device copy of the request's sequence
/// into the device completion word, so completion becomes visible to the GPU
/// no earlier than the row copies it covers. Every publish reads its own
/// pinned word, which is never rewritten while an earlier publish from it is
/// still queued.
class DoorbellThread {
 public:
  DoorbellThread(
      uint8_t* page,
      int64_t capacity,
      int64_t ring,
      std::vector<Segment> segments,
      uint32_t* done,
      int32_t* publish_words,
      int64_t publish_ring,
      int device,
      int cpu_core,
      bool prefer_overlap)
      : page_(page),
        capacity_(capacity),
        ring_(ring),
        segments_(std::move(segments)),
        done_(done),
        publish_words_(publish_words),
        publish_ring_(publish_ring),
        device_(device),
        cpu_core_(cpu_core),
        prefer_overlap_(prefer_overlap),
        rows_(capacity),
        slots_(capacity),
        sources_(capacity * segments_.size()),
        destinations_(capacity * segments_.size()),
        sizes_(capacity * segments_.size()),
        publish_events_(publish_ring),
        publish_trace_(publish_ring, -1),
        trace_(kTraceCapacity) {
    for (auto& counter : counters_) {
      counter.store(0);
    }
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
    if (::cudaSetDevice(device_) != ::cudaSuccess || ::cudaStreamCreate(&stream_) != ::cudaSuccess) {
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
    ::cudaStreamDestroy(stream_);
    counters_[kRunning].store(0);
  }

  void service(uint32_t seq, uint32_t head, int64_t seen_ns) {
    TraceEntry entry;
    entry.seq = seq;
    entry.seen_ns = seen_ns;
    const uint32_t abandoned = read_host_word(page_ + kAbandonedOffset);
    if (abandoned != 0 && static_cast<int32_t>(seq - abandoned) <= 0) {
      entry.status = static_cast<int64_t>(RequestStatus::kSkippedAbandoned);
      counters_[kSkippedAbandonedCount].fetch_add(1);
      push_trace(entry);
      return;
    }
    if (static_cast<int64_t>(head - seq) >= ring_ - 1) {
      entry.status = static_cast<int64_t>(RequestStatus::kSkippedOverrun);
      counters_[kSkippedOverrunCount].fetch_add(1);
      push_trace(entry);
      return;
    }
    const uint8_t* record = page_ + record_offset(seq, capacity_, ring_);
    const uint32_t count = read_host_word(record + 4);
    if (read_host_word(record) != seq || count > static_cast<uint32_t>(capacity_)) {
      entry.status = static_cast<int64_t>(RequestStatus::kInvalidRecord);
      counters_[kInvalidRecords].fetch_add(1);
      push_trace(entry);
      return;
    }
    std::memcpy(rows_.data(), record + kRecordHeaderBytes, 8 * count);
    std::memcpy(slots_.data(), record + kRecordHeaderBytes + 8 * capacity_, 4 * count);
    if (read_host_word(record) != seq) {
      entry.status = static_cast<int64_t>(RequestStatus::kInvalidRecord);
      counters_[kInvalidRecords].fetch_add(1);
      push_trace(entry);
      return;
    }
    entry.count = count;

    size_t copies = 0;
    int64_t rows_copied = 0;
    for (uint32_t index = 0; index < count; ++index) {
      const int64_t row = rows_[index];
      const int64_t slot = slots_[index];
      bool valid = true;
      for (const Segment& segment : segments_) {
        valid = valid && row >= 0 && row < segment.source_rows && slot >= 0 && slot < segment.destination_rows;
      }
      if (!valid) {
        counters_[kCopyErrors].fetch_add(1);
        continue;
      }
      for (const Segment& segment : segments_) {
        sources_[copies] = reinterpret_cast<void*>(segment.source + static_cast<uint64_t>(row) * segment.row_bytes);
        destinations_[copies] =
            reinterpret_cast<void*>(segment.destination + static_cast<uint64_t>(slot) * segment.row_bytes);
        sizes_[copies] = segment.row_bytes;
        entry.bytes += static_cast<int64_t>(segment.row_bytes);
        ++copies;
      }
      ++rows_copied;
    }

    if (copies > 0 && !copy_batch(copies)) {
      entry.status = static_cast<int64_t>(RequestStatus::kCopyFailed);
      counters_[kCopyErrors].fetch_add(1);
      push_trace(entry);
      return;
    }

    const int64_t slot = publish_cursor_++ % publish_ring_;
    if (publish_trace_[slot] >= 0) {
      ::cudaEventSynchronize(publish_events_[slot]);
      poll_completions();
    }
    publish_words_[slot] = static_cast<int32_t>(seq);
    ::cudaMemcpyAsync(done_, publish_words_ + slot, sizeof(uint32_t), ::cudaMemcpyHostToDevice, stream_);
    ::cudaEventRecord(publish_events_[slot], stream_);
    entry.enqueued_ns = monotonic_ns();
    entry.status = static_cast<int64_t>(RequestStatus::kServiced);
    entry.publish_slot = slot;
    counters_[kRowsCopied].fetch_add(rows_copied);
    counters_[kBytesCopied].fetch_add(entry.bytes);
    counters_[kServiced].fetch_add(1);
    publish_trace_[slot] = push_trace(entry);
    pending_.push_back(slot);
  }

  bool copy_batch(size_t copies) {
    ::cudaMemcpyAttributes attributes{};
    attributes.srcAccessOrder = ::cudaMemcpySrcAccessOrderStream;
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
               stream_) == ::cudaSuccess;
  }

  void poll_completions() {
    while (pending_head_ < pending_.size()) {
      const int64_t slot = pending_[pending_head_];
      if (::cudaEventQuery(publish_events_[slot]) != ::cudaSuccess) {
        break;
      }
      const int64_t complete_ns = monotonic_ns();
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
  std::vector<Segment> segments_;
  uint32_t* done_;
  int32_t* publish_words_;
  int64_t publish_ring_;
  int device_;
  int cpu_core_;
  bool prefer_overlap_;
  std::vector<int64_t> rows_;
  std::vector<int32_t> slots_;
  std::vector<void*> sources_;
  std::vector<void*> destinations_;
  std::vector<size_t> sizes_;
  std::vector<::cudaEvent_t> publish_events_;
  std::vector<int64_t> publish_trace_;
  std::vector<int64_t> pending_;
  size_t pending_head_ = 0;
  int64_t publish_cursor_ = 0;
  ::cudaStream_t stream_ = nullptr;
  std::thread thread_;
  std::atomic<bool> started_{false};
  std::atomic<bool> failed_{false};
  std::atomic<bool> stop_{false};
  std::atomic<bool> paused_{false};
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
    tvm::ffi::TensorView done,
    tvm::ffi::TensorView publish_words,
    int64_t capacity,
    int64_t ring,
    int64_t device,
    int64_t cpu_core,
    int64_t prefer_overlap) {
  using namespace expert_doorbell;
  const auto table = static_cast<const int64_t*>(segment_table.data_ptr());
  std::vector<Segment> segments(segment_table.size(0));
  for (size_t index = 0; index < segments.size(); ++index) {
    const int64_t* entry = table + index * kSegmentColumns;
    segments[index] = Segment{
        static_cast<uint64_t>(entry[0]), static_cast<uint64_t>(entry[1]), static_cast<uint64_t>(entry[2]), entry[3],
        entry[4]};
  }
  auto thread = std::make_unique<DoorbellThread>(
      static_cast<uint8_t*>(page.data_ptr()),
      capacity,
      ring,
      std::move(segments),
      static_cast<uint32_t*>(done.data_ptr()),
      static_cast<int32_t*>(publish_words.data_ptr()),
      publish_words.size(0),
      static_cast<int>(device),
      static_cast<int>(cpu_core),
      prefer_overlap != 0);
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
