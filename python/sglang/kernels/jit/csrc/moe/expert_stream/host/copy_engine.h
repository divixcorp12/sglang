// The copy engine: CopyLane through CopyEngine (LEASE_PROTOCOL.md 7.6).
#pragma once

#include "../lease_layout.h"
#include "tier_protocol.h"

namespace sglang {
namespace expert_stream {

using namespace ::sglang::expert_stream::wire;

// ---- Copy engine (LEASE_PROTOCOL.md 7.6) ----

struct CopyLane {
  int32_t lane = 0;
  int32_t host_slot = 0;
  int32_t dst_slot = 0;
  uint32_t slot_generation = 0;
};

// One request's COPYING lanes, handed from the grant to the copy thread.
struct CopyJob {
  uint64_t gen = 0;
  int64_t idx = 0;
  int64_t row = 0;
  uint32_t mask = 0;
  int count = 0;
  CopyLane lanes[kLeaseLanes];
  int64_t submit_ns = 0;
  int64_t token = -1;  // the backend's completion marker, recorded after the job's last copy
  bool prefetch = false;  // a native-prefetch job: issued only behind demand jobs, completed into PrefetchDone
  // Its row has SM entries (set at issue): the copy wait reads them from the leased slots, so the leases are released
  // only once it acknowledged its reads (SmAck), not on the DMA's completion. Never a prefetch job: no kernel reads
  // a prefetched row's slot, so a prefetch copies every entry itself.
  bool sm = false;
};

// One entry of a row's copy table: C1's (source slab, destination tensor, row bytes); lane rows index both.
struct CopyEntry {
  uint64_t src = 0;
  uint64_t dst = 0;
  int64_t bytes = 0;
  bool sm = false;  // SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES: the copy wait reads it, not the DMA
};

// What the copy thread drives: issue copies in order, mark the point after them, and ask whether a mark is complete.
// Every call is made on the copy thread only.
class CopyBackend {
 public:
  static constexpr int kDone = 0;
  static constexpr int kPending = 1;
  virtual ~CopyBackend() = default;
  virtual std::string init() = 0;  // empty on success
  virtual int issue(uint64_t dst, uint64_t src, int64_t bytes) = 0;  // 0 or an error code
  virtual int mark(int64_t* token) = 0;
  virtual int query(int64_t token) = 0;  // kDone, kPending, or an error code (negative for the host backend)
  virtual void shutdown(bool idle) = 0;
};

// The CUDA driver, resolved from libcuda.so.1 when the copy engine is enabled, so this module still builds and loads
// without a CUDA toolkit. It calls cuMemcpyAsync, cuEventRecord and cuEventQuery on its own non-blocking stream, and
// nothing that waits on another stream, a graph launch or the device (no synchronize, no module load, no allocation).
class CudaCopyBackend : public CopyBackend {
 public:
  explicit CudaCopyBackend(int device) : device_(device) {}

  std::string init() override {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (lib == nullptr) return "cannot open libcuda.so.1";
    bool ok = true;
    auto get = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
      ok = ok && fn != nullptr;
    };
    get(cu_init_, "cuInit");
    get(cu_device_get_, "cuDeviceGet");
    get(cu_primary_retain_, "cuDevicePrimaryCtxRetain");
    get(cu_primary_release_, "cuDevicePrimaryCtxRelease_v2");
    get(cu_ctx_set_current_, "cuCtxSetCurrent");
    get(cu_stream_create_, "cuStreamCreateWithPriority");
    get(cu_priority_range_, "cuCtxGetStreamPriorityRange");
    get(cu_stream_destroy_, "cuStreamDestroy_v2");
    get(cu_event_create_, "cuEventCreate");
    get(cu_event_destroy_, "cuEventDestroy_v2");
    get(cu_event_record_, "cuEventRecord");
    get(cu_event_query_, "cuEventQuery");
    get(cu_memcpy_async_, "cuMemcpyAsync");
    if (!ok) return "libcuda.so.1 lacks a driver entry point the copy engine needs";
    if (int r = cu_init_(0)) return "cuInit failed: " + std::to_string(r);
    if (int r = cu_device_get_(&cu_device_, device_)) return "cuDeviceGet failed: " + std::to_string(r);
    // The primary context: the one PyTorch uses, so the slabs' registrations and the destinations are valid here.
    if (int r = cu_primary_retain_(&context_, cu_device_)) return "cuDevicePrimaryCtxRetain failed: " + std::to_string(r);
    retained_ = true;
    if (int r = cu_ctx_set_current_(context_)) return "cuCtxSetCurrent failed: " + std::to_string(r);
    constexpr unsigned kNonBlocking = 1;  // CU_STREAM_NON_BLOCKING: no implicit sync with the legacy stream
    // The greatest priority, for a hardware queue of its own. Streams share CUDA_DEVICE_MAX_CONNECTIONS queues (the
    // server sets 8) and a queue runs in order, so a copy queued behind a stream whose head waits on the decode graph
    // cannot start until CW gives up: a deadlock only the device deadline breaks. hol_probe.py (raw streams): a fresh
    // default-priority stream waited 596 ms behind 8 blocked ones; greatest-priority kernels and copies never waited,
    // with up to 64 blocked. sglang creates no other greatest-priority stream.
    int least = 0;
    int greatest = 0;
    if (int r = cu_priority_range_(&least, &greatest)) return "cuCtxGetStreamPriorityRange failed: " + std::to_string(r);
    if (int r = cu_stream_create_(&stream_, kNonBlocking, greatest)) {
      return "cuStreamCreateWithPriority failed: " + std::to_string(r);
    }
    constexpr unsigned kDisableTiming = 2;  // CU_EVENT_DISABLE_TIMING
    for (auto& event : events_) {
      if (int r = cu_event_create_(&event, kDisableTiming)) return "cuEventCreate failed: " + std::to_string(r);
    }
    return "";
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    return cu_memcpy_async_(dst, src, static_cast<size_t>(bytes), stream_);
  }

  // Events are reused round robin: there are more of them than jobs can be outstanding (one per request slot).
  int mark(int64_t* token) override {
    const int64_t index = next_event_++ % static_cast<int64_t>(events_.size());
    *token = index;
    return cu_event_record_(events_[index], stream_);
  }

  int query(int64_t token) override {
    constexpr int kNotReady = 600;  // CUDA_ERROR_NOT_READY
    const int r = cu_event_query_(events_[token]);
    return r == 0 ? kDone : r == kNotReady ? kPending : r;
  }

  // After an error, or with copies in flight, keep the stream, events and context: a copy may still read a slab.
  void shutdown(bool idle) override {
    if (!idle) return;
    for (auto& event : events_) {
      if (event != nullptr) cu_event_destroy_(event);
    }
    if (stream_ != nullptr) cu_stream_destroy_(stream_);
    if (retained_) cu_primary_release_(cu_device_);
  }

 private:
  int device_;
  int cu_device_ = 0;
  void* context_ = nullptr;
  void* stream_ = nullptr;
  bool retained_ = false;
  std::array<void*, 2 * kDemandRecords> events_{};
  int64_t next_event_ = 0;
  int (*cu_init_)(unsigned) = nullptr;
  int (*cu_device_get_)(int*, int) = nullptr;
  int (*cu_primary_retain_)(void**, int) = nullptr;
  int (*cu_primary_release_)(int) = nullptr;
  int (*cu_ctx_set_current_)(void*) = nullptr;
  int (*cu_stream_create_)(void**, unsigned, int) = nullptr;
  int (*cu_priority_range_)(int*, int*) = nullptr;
  int (*cu_stream_destroy_)(void*) = nullptr;
  int (*cu_event_create_)(void**, unsigned) = nullptr;
  int (*cu_event_destroy_)(void*) = nullptr;
  int (*cu_event_record_)(void*, void*) = nullptr;
  int (*cu_event_query_)(void*) = nullptr;
  int (*cu_memcpy_async_)(uint64_t, uint64_t, size_t, void*) = nullptr;
};

// Test only (CPU): "copies" between host buffers. A mark completes only once the test has released it, and its
// copies land then, so a CopyDone published before its release is visible as bytes that are not there yet.
class HostCopyBackend : public CopyBackend {
 public:
  std::string init() override {
    return "";
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fail_issue_) return -1;
    pending_.push_back(CopyEntry{src, dst, bytes});
    return 0;
  }

  int mark(int64_t* token) override {
    std::lock_guard<std::mutex> guard(mutex_);
    marks_.push_back(std::move(pending_));
    pending_.clear();
    *token = static_cast<int64_t>(marks_.size()) - 1;
    return 0;
  }

  int query(int64_t token) override {
    std::lock_guard<std::mutex> guard(mutex_);
    if (fail_query_) return -2;
    if (token < completed_) return kDone;
    if (token != completed_ || released_ <= completed_) return kPending;  // one stream: marks complete in order
    for (const CopyEntry& copy : marks_[token]) {
      std::memcpy(reinterpret_cast<void*>(copy.dst), reinterpret_cast<const void*>(copy.src), copy.bytes);
    }
    ++completed_;
    return kDone;
  }

  void shutdown(bool) override {}

  // Lets `marks` more marks complete; negative: every mark, from now on.
  void release(int64_t marks) {
    std::lock_guard<std::mutex> guard(mutex_);
    released_ = marks < 0 ? INT64_MAX : released_ + marks;
  }

  void fail(bool issue, bool query) {
    std::lock_guard<std::mutex> guard(mutex_);
    fail_issue_ = issue;
    fail_query_ = query;
  }

  int64_t marked() {
    std::lock_guard<std::mutex> guard(mutex_);
    return static_cast<int64_t>(marks_.size());
  }

 private:
  std::mutex mutex_;
  std::vector<CopyEntry> pending_;
  std::vector<std::vector<CopyEntry>> marks_;
  int64_t completed_ = 0;
  int64_t released_ = 0;
  bool fail_issue_ = false;
  bool fail_query_ = false;
};

// The copy thread: issues each job's copies on the backend's stream, records one mark after them, polls marks in
// order and hands each completed job to `complete`. A backend error hands every job it still holds to `fail` and
// stops issuing: nothing it issued may be assumed complete, so none of those leases is released (E5).
class CopyEngine {
 public:
  // complete: the job's copies completed; false when it still waits for the copy wait's SmAck, which `acked` then
  // polls (true once it released the job's leases).
  using Handler = std::function<bool(const CopyJob&)>;
  using Failure = std::function<void(const CopyJob&, int)>;

  CopyEngine(std::unique_ptr<CopyBackend> backend, int64_t rows, int64_t spin_ns, std::atomic<int64_t>* counters,
             Handler complete, Handler acked, Failure fail, std::string prefix)
      : backend_(std::move(backend)),
        tables_(static_cast<size_t>(rows)),
        spin_ns_(spin_ns),
        counters_(counters),
        complete_(std::move(complete)),
        acked_(std::move(acked)),
        fail_(std::move(fail)),
        prefix_(std::move(prefix)) {}

  ~CopyEngine() {
    stop(5'000'000'000LL);
  }

  // Starts the thread and returns once the backend is initialised on it; throws with the backend's error.
  void start() {
    thread_ = std::thread([this] { run(); });
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [this] { return started_; });
    if (!init_error_.empty()) {
      lock.unlock();
      stop(0);
      throw std::runtime_error(prefix_ + "copy engine: " + init_error_);
    }
  }

  // Row `row`'s copy table and the number of destination rows each entry's tensor holds. Once per row.
  void set_table(int64_t row, std::vector<CopyEntry> entries, int64_t dst_rows) {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size())) throw std::runtime_error("copy table row out of range");
    Table& table = tables_[row];
    if (table.ready.load(std::memory_order_acquire)) throw std::runtime_error("copy table already set for this row");
    table.sm = std::any_of(entries.begin(), entries.end(), [](const CopyEntry& e) { return e.sm; });
    table.entries = std::move(entries);
    table.dst_rows = dst_rows;
    table.ready.store(true, std::memory_order_release);
  }

  bool eligible(int64_t row, int32_t dst_slot) const {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size())) return false;
    const Table& table = tables_[row];
    return table.ready.load(std::memory_order_acquire) && !table.entries.empty() && dst_slot >= 0 &&
           dst_slot < table.dst_rows;
  }

  void submit(const CopyJob& job) {
    {
      std::lock_guard<std::mutex> guard(mutex_);
      queue_.push_back(job);
      ++outstanding_;
    }
    work_cv_.notify_one();
  }

  // No job queued, in flight, awaiting its SmAck or still being handed to `complete`.
  bool idle() {
    std::lock_guard<std::mutex> guard(mutex_);
    return outstanding_ == 0;
  }

  bool wait_idle(int64_t deadline_ns) {
    while (!idle()) {
      if (now_ns() > deadline_ns) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  // Joins the thread once it has seen every in-flight job complete, or `drain_ns` passed.
  void stop(int64_t drain_ns) {
    if (!thread_.joinable()) return;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      stop_ = true;
      drain_deadline_ = now_ns() + drain_ns;
    }
    work_cv_.notify_one();
    thread_.join();
  }

  HostCopyBackend* host_backend() {
    return dynamic_cast<HostCopyBackend*>(backend_.get());
  }

  // Test only: one more copy of `bytes` issued ahead of every job's own, so each job completes that much later.
  void set_ballast(uint64_t dst, uint64_t src, int64_t bytes) {
    ballast_dst_.store(dst);
    ballast_src_.store(src);
    ballast_bytes_.store(bytes);
  }

 private:
  struct Table {
    std::vector<CopyEntry> entries;
    bool sm = false;  // an entry is left to the copy wait's SM reads
    int64_t dst_rows = 0;
    std::atomic<bool> ready{false};
  };

  void run() {
    pthread_setname_np(pthread_self(), "exl3-copy-eng");
    const std::string error = backend_->init();
    {
      std::lock_guard<std::mutex> guard(mutex_);
      init_error_ = error;
      started_ = true;
    }
    ready_cv_.notify_all();
    if (!error.empty()) return;
    std::deque<CopyJob> in_flight;
    std::deque<CopyJob> held;  // prefetch jobs not yet issued: demand goes first on the link
    std::deque<CopyJob> acking;  // copies completed, CopyDone published; the leases wait for the copy wait's SmAck
    int64_t last_active = now_ns();
    while (true) {
      std::deque<CopyJob> fresh;
      bool stopping = false;
      int64_t drain_deadline = 0;
      {
        std::lock_guard<std::mutex> guard(mutex_);
        fresh.swap(queue_);
        stopping = stop_;
        drain_deadline = drain_deadline_;
      }
      bool demand_fresh = false;
      for (CopyJob& job : fresh) {
        if (job.prefetch) {
          held.push_back(job);
          continue;
        }
        demand_fresh = true;
        issue_or_fail(job, in_flight);
      }
      // Demand before prefetch: a prefetch is issued only when no demand job arrived this pass and none is in flight,
      // so a mispredicted row never sits on the copy stream ahead of a demand row. Once issued it cannot be preempted;
      // the device never posts a demand while its own layer's prefetch is outstanding, so none can queue behind it.
      bool demand_in_flight = false;
      for (const CopyJob& job : in_flight) demand_in_flight = demand_in_flight || !job.prefetch;
      if (!held.empty() && (demand_fresh || demand_in_flight)) {
        if (!held_counted_) counters_[kPrefetchHeld].fetch_add(1);
        held_counted_ = true;
      }
      while (!held.empty() && ((!demand_fresh && !demand_in_flight) || stopping)) {
        held_counted_ = false;
        CopyJob job = held.front();
        held.pop_front();
        issue_or_fail(job, in_flight);
      }
      bool progressed = !fresh.empty();
      while (!in_flight.empty() && broken_ == 0) {
        const int state = backend_->query(in_flight.front().token);
        if (state == CopyBackend::kPending) break;
        if (state != CopyBackend::kDone) {
          broken_ = state;
          break;
        }
        const CopyJob job = in_flight.front();
        in_flight.pop_front();
        record_latency(job);
        if (complete_(job)) {
          finish();
        } else {
          acking.push_back(job);
        }
        progressed = true;
      }
      for (auto it = acking.begin(); it != acking.end();) {
        if (acked_(*it)) {
          finish();
          progressed = true;
          it = acking.erase(it);
        } else {
          ++it;
        }
      }
      if (broken_ != 0) {
        while (!in_flight.empty()) {
          finish_failed(in_flight.front(), broken_);
          in_flight.pop_front();
        }
      }
      if (stopping && held.empty() && ((in_flight.empty() && acking.empty()) || now_ns() > drain_deadline)) break;
      if (progressed) {
        last_active = now_ns();
      } else if (!in_flight.empty() || !held.empty() || !acking.empty() || now_ns() - last_active < spin_ns_) {
        _mm_pause();
      } else {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait_for(lock, std::chrono::milliseconds(1), [this] { return !queue_.empty() || stop_; });
      }
    }
    backend_->shutdown(in_flight.empty() && broken_ == 0);
  }

  void issue_or_fail(CopyJob& job, std::deque<CopyJob>& in_flight) {
    if (broken_ != 0) {
      finish_failed(job, broken_);
      return;
    }
    const int error_code = issue(job);
    if (error_code != 0) {
      broken_ = error_code;
      finish_failed(job, error_code);
      return;
    }
    in_flight.push_back(job);
  }

  int issue(CopyJob& job) {
    const Table& table = tables_[job.row];
    const int64_t start = now_ns();
    int64_t bytes = 0;
    if (const int64_t ballast = ballast_bytes_.load(); ballast > 0) {
      if (const int r = backend_->issue(ballast_dst_.load(), ballast_src_.load(), ballast)) return r;
    }
    job.sm = table.sm && !job.prefetch;
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      for (const CopyEntry& entry : table.entries) {
        if (entry.sm && job.sm) continue;
        const uint64_t dst = entry.dst + static_cast<uint64_t>(lane.dst_slot) * static_cast<uint64_t>(entry.bytes);
        const uint64_t src = entry.src + static_cast<uint64_t>(lane.host_slot) * static_cast<uint64_t>(entry.bytes);
        if (const int r = backend_->issue(dst, src, entry.bytes)) return r;
        bytes += entry.bytes;
      }
    }
    const int r = backend_->mark(&job.token);
    counters_[kCopyIssueNs].fetch_add(now_ns() - start);
    counters_[kCopyBytes].fetch_add(bytes);
    return r;
  }

  void record_latency(const CopyJob& job) {
    const int64_t latency = now_ns() - job.submit_ns;
    counters_[kCopyLatencyNs].fetch_add(latency);
    int64_t seen = counters_[kCopyLatencyMaxNs].load();
    while (latency > seen && !counters_[kCopyLatencyMaxNs].compare_exchange_weak(seen, latency)) {
    }
  }

  void finish_failed(const CopyJob& job, int error_code) {
    counters_[kCopyErrors].fetch_add(1);
    fail_(job, error_code);
    finish();
  }

  void finish() {
    std::lock_guard<std::mutex> guard(mutex_);
    --outstanding_;
  }

  std::unique_ptr<CopyBackend> backend_;
  std::vector<Table> tables_;
  int64_t spin_ns_;
  std::atomic<int64_t>* counters_;
  Handler complete_;
  Handler acked_;
  Failure fail_;
  std::string prefix_;
  std::thread thread_;
  std::mutex mutex_;  // guards queue_, outstanding_, stop_, drain_deadline_, started_ and init_error_
  std::condition_variable work_cv_;
  std::condition_variable ready_cv_;
  std::deque<CopyJob> queue_;
  int64_t outstanding_ = 0;
  bool stop_ = false;
  int64_t drain_deadline_ = 0;
  bool started_ = false;
  std::string init_error_;
  int broken_ = 0;  // copy thread only: the first backend error
  bool held_counted_ = false;  // copy thread only: the current hold was counted in kPrefetchHeld
  std::atomic<uint64_t> ballast_dst_{0};
  std::atomic<uint64_t> ballast_src_{0};
  std::atomic<int64_t> ballast_bytes_{0};
};


}  // namespace expert_stream
}  // namespace sglang
