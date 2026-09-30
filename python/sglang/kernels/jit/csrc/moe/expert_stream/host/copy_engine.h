// The copy engine: CopyLane through CopyEngine (LEASE_PROTOCOL.md 7.6).
#pragma once

#include "../lease_layout.h"
#include "build_policy.h"
#include "completion_word.h"
#include "cpu_experts.h"
#include "spsc_ring.h"
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
  float weight = 0.0f;  // the lane's routing weight: read only for a CPU lane
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
  int64_t token = -1;     // the backend's completion marker after the job's last copy (CUDA: the job's sequence)
  bool prefetch = false;  // a native-prefetch job: issued only behind demand jobs, completed into PrefetchDone
  // Its row has SM entries (set at issue): the copy wait reads them from the leased slots, so the leases are released
  // only once it acknowledged its reads (SmAck), not on the DMA's completion. Never a prefetch job: no kernel reads
  // a prefetched row's slot, so a prefetch copies every entry itself.
  bool sm = false;
  // CPU experts (tag kLeaseTagCpu): the lanes of `mask` the CPU expert thread computes instead of copying, and that
  // engine's job sequence for them (set at issue, 0 without CPU lanes). The job completes when its copies AND that
  // CPU job have; CopyDone then carries the whole mask. `token` is kNoToken when no lane was copied.
  uint32_t cpu_mask = 0;
  uint32_t cpu_seq = 0;
};

// CopyJob::token of a job that issued no copy (every lane on the CPU): nothing to query.
constexpr int64_t kNoToken = -1;

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
  virtual std::string init() = 0;                                    // empty on success
  virtual int issue(uint64_t dst, uint64_t src, int64_t bytes) = 0;  // 0 or an error code
  virtual int mark(int64_t* token) = 0;
  virtual int query(int64_t token) = 0;  // kDone, kPending, or an error code (negative for the host backend)
  virtual void shutdown(bool idle) = 0;
};
static_assert(CompletionWord::kDone == CopyBackend::kDone && CompletionWord::kPending == CopyBackend::kPending);

// The CUDA driver, resolved from libcuda.so.1 when the copy engine is enabled, so this module still builds and loads
// without a CUDA toolkit. Per job it calls cuMemcpyAsync per copy and one cuStreamWriteValue32_v2 of the job's
// sequence number into the completion word (completion_word.h) on its own non-blocking stream; it polls with plain
// acquire loads of that word, plus one cuStreamQuery per CompletionWord budget while a job is pending. Nothing waits
// on another stream, a graph launch or the device (no synchronize, no module load, no allocation per job).
//
// No fallback: the production recipe needs the copy engine, so init() refuses (and the copy engine's start() throws)
// when the v2 write-value op cannot be resolved, errors, or its first write does not reach the word; and when the v2
// wait-value op, which the decode stream's copy wait needs on host-mapped memory (LEASE_PROTOCOL.md 7.6), cannot be
// resolved, errors, does not hold its stream, or is not released by a host store.
class CudaCopyBackend : public CopyBackend {
 public:
  CudaCopyBackend(int device, std::string prefix) : device_(device), prefix_(std::move(prefix)) {}

  std::string init() override {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (lib == nullptr) return "cannot open libcuda.so.1";
    std::string missing;
    auto get = [&](auto& fn, const char* name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
      if (fn == nullptr && missing.empty()) missing = name;
    };
    get(cu_init_, "cuInit");
    get(cu_device_get_, "cuDeviceGet");
    get(cu_primary_retain_, "cuDevicePrimaryCtxRetain");
    get(cu_primary_release_, "cuDevicePrimaryCtxRelease_v2");
    get(cu_ctx_set_current_, "cuCtxSetCurrent");
    get(cu_stream_create_, "cuStreamCreateWithPriority");
    get(cu_priority_range_, "cuCtxGetStreamPriorityRange");
    get(cu_stream_destroy_, "cuStreamDestroy_v2");
    get(cu_stream_query_, "cuStreamQuery");
    get(cu_memcpy_async_, "cuMemcpyAsync");
    get(cu_mem_host_alloc_, "cuMemHostAlloc");
    get(cu_mem_host_device_ptr_, "cuMemHostGetDevicePointer_v2");
    get(cu_mem_free_host_, "cuMemFreeHost");
    // The v2 stream memory op, never the plain name: that is the v1 API, gated by NVreg_EnableStreamMemOPs (its _V1
    // device attribute reads 0 on divix01, while the v2 ops work; results.md 9b).
    get(cu_write_value32_, "cuStreamWriteValue32_v2");
    get(cu_wait_value32_, "cuStreamWaitValue32_v2");
    if (!missing.empty()) return "libcuda.so.1 lacks " + missing + ", which the copy engine needs";
    if (int r = cu_init_(0)) return "cuInit failed: " + std::to_string(r);
    if (int r = cu_device_get_(&cu_device_, device_)) return "cuDeviceGet failed: " + std::to_string(r);
    // The primary context: the one PyTorch uses, so the slabs' registrations and the destinations are valid here.
    if (int r = cu_primary_retain_(&context_, cu_device_))
      return "cuDevicePrimaryCtxRetain failed: " + std::to_string(r);
    retained_ = true;
    if (int r = cu_ctx_set_current_(context_)) return "cuCtxSetCurrent failed: " + std::to_string(r);
    constexpr unsigned kNonBlocking = 1;  // CU_STREAM_NON_BLOCKING: no implicit sync with the legacy stream
    // The greatest priority, for a hardware queue of its own. Streams share CUDA_DEVICE_MAX_CONNECTIONS queues (the
    // server sets 8) and a queue runs in order, so a copy queued behind a stream whose head waits on the decode graph
    // cannot start until CW gives up: a deadlock only the copy-wait timeout breaks. hol_probe.py (raw streams): a fresh
    // default-priority stream waited 596 ms behind 8 blocked ones; greatest-priority kernels and copies never waited,
    // with up to 64 blocked. sglang creates no other greatest-priority stream.
    int least = 0;
    int greatest = 0;
    if (int r = cu_priority_range_(&least, &greatest))
      return "cuCtxGetStreamPriorityRange failed: " + std::to_string(r);
    if (int r = cu_stream_create_(&stream_, kNonBlocking, greatest)) {
      return "cuStreamCreateWithPriority failed: " + std::to_string(r);
    }
    if (std::string error = init_word(); !error.empty()) return error;
    return probe_stream_wait();
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    return cu_memcpy_async_(dst, src, static_cast<size_t>(bytes), stream_);
  }

  // The job's token is its sequence number, written into the word after the job's copies (stream order).
  int mark(int64_t* token) override {
    const uint32_t seq = ++marked_;
    *token = seq;
    return cu_write_value32_(stream_, word_dev_, seq, 0);  // default flags: after the prior copies, fenced
  }

  int query(int64_t token) override {
    return word_.poll(static_cast<uint32_t>(token), [this] { return cu_stream_query_(stream_); });
  }

  // The word's final value is the exact number of jobs the stream completed (the Task P2 job count). After an error,
  // or with copies in flight, keep the stream, word and context: a copy may still read a slab, the stream may still
  // write the word.
  void shutdown(bool idle) override {
    if (word_host_ != nullptr) {
      std::fprintf(
          stderr,
          "%scompletion word %u at shutdown, %u jobs marked\n",
          prefix_.c_str(),
          static_cast<unsigned>(word_.load()),
          static_cast<unsigned>(marked_));
      std::fflush(stderr);
    }
    if (!idle) return;
    if (word_host_ != nullptr) cu_mem_free_host_(word_host_);
    if (stream_ != nullptr) cu_stream_destroy_(stream_);
    if (retained_) cu_primary_release_(cu_device_);
  }

 private:
  // One host-mapped pinned word, then a first write through the stream, waited for here (start-up only; a bounded
  // poll with a sleep, no clock): an op the driver rejects, or a write that never lands, refuses the start.
  std::string init_word() {
    constexpr unsigned kDeviceMap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP
    void* host = nullptr;
    if (int r = cu_mem_host_alloc_(&host, sizeof(uint32_t), kDeviceMap))
      return "cuMemHostAlloc of the completion word failed: " + std::to_string(r);
    word_host_ = static_cast<uint32_t*>(host);
    __atomic_store_n(word_host_, 0xFFFFFFFFu, __ATOMIC_RELEASE);  // not 0: the first write below must change it
    if (int r = cu_mem_host_device_ptr_(&word_dev_, host, 0))
      return "cuMemHostGetDevicePointer of the completion word failed: " + std::to_string(r);
    if (int r = cu_write_value32_(stream_, word_dev_, 0, 0)) {
      return "cuStreamWriteValue32_v2 failed (" + std::to_string(r) +
             "): the copy engine needs the v2 stream memory operations for its completion word";
    }
    constexpr int kPolls = 100'000;  // x >= 10 us: at least a second
    for (int i = 0;; ++i) {
      const int r = cu_stream_query_(stream_);
      if (r == 0) break;
      if (r != CompletionWord::kNotReady)
        return "the completion word's first write failed: cuStreamQuery " + std::to_string(r);
      if (i == kPolls) return "the completion word's first write did not complete within a second";
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    word_.bind(word_host_);
    if (word_.load() != 0) return "the completion word's first write completed but did not reach host memory";
    return "";
  }

  // The copy wait's primitive, tried once on this stream at start-up (a bounded poll, as init_word): a wait on a
  // host-mapped word must hold the stream until a host store satisfies it, then complete. The decode stream's gate
  // is the same kind of memory and the same op, so a device or driver without it is refused here, not found by a
  // decode that never ends.
  std::string probe_stream_wait() {
    constexpr unsigned kDeviceMap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP
    constexpr unsigned kGeq = 0;        // CU_STREAM_WAIT_VALUE_GEQ
    void* host = nullptr;
    if (int r = cu_mem_host_alloc_(&host, sizeof(uint32_t), kDeviceMap))
      return "cuMemHostAlloc of the stream-wait probe word failed: " + std::to_string(r);
    auto* word = static_cast<uint32_t*>(host);
    // The production encoding (lease_layout.h): a closed gate word (bit 31 set) that an open word releases.
    const uint32_t closed = kLeaseGateClosed;
    const uint32_t open_word = kLeaseGateOpen;
    __atomic_store_n(word, closed, __ATOMIC_RELEASE);
    uint64_t device_word = 0;
    std::string error;
    if (int r = cu_mem_host_device_ptr_(&device_word, host, 0)) {
      error = "cuMemHostGetDevicePointer of the stream-wait probe word failed: " + std::to_string(r);
    } else if (int r = cu_wait_value32_(stream_, device_word, kLeaseGateOpen, kGeq)) {
      error = "cuStreamWaitValue32_v2 failed (" + std::to_string(r) +
              "): the copy wait needs the v2 stream wait on host-mapped memory";
    } else {
      constexpr int kPolls = 100'000;  // x >= 10 us: at least a second
      int q = cu_stream_query_(stream_);
      if (q == 0) {
        error = "cuStreamWaitValue32_v2 did not hold its stream: the copy wait could not order the decode stream";
      } else if (q != CompletionWord::kNotReady) {
        error = "the stream-wait probe failed: cuStreamQuery " + std::to_string(q);
      }
      __atomic_store_n(word, open_word, __ATOMIC_RELEASE);  // releases the wait; also on an error, so nothing stays queued
      for (int i = 0; error.empty(); ++i) {
        q = cu_stream_query_(stream_);
        if (q == 0) break;
        if (q != CompletionWord::kNotReady) error = "the stream-wait probe failed: cuStreamQuery " + std::to_string(q);
        if (i == kPolls) error = "cuStreamWaitValue32_v2 was not released by a host store within a second";
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    // Freed only once the stream no longer waits on it; a probe that never completed leaks its 4 bytes instead.
    if (error.empty() || cu_stream_query_(stream_) == 0) cu_mem_free_host_(host);
    return error;
  }

  int device_;
  std::string prefix_;
  int cu_device_ = 0;
  void* context_ = nullptr;
  void* stream_ = nullptr;
  bool retained_ = false;
  uint32_t* word_host_ = nullptr;  // host-mapped pinned (cuMemHostAlloc DEVICEMAP); the stream writes it
  uint64_t word_dev_ = 0;          // its device address
  CompletionWord word_;            // copy thread only
  uint32_t marked_ = 0;            // copy thread only: the last job sequence written (wraps at 2^32)
  int (*cu_init_)(unsigned) = nullptr;
  int (*cu_device_get_)(int*, int) = nullptr;
  int (*cu_primary_retain_)(void**, int) = nullptr;
  int (*cu_primary_release_)(int) = nullptr;
  int (*cu_ctx_set_current_)(void*) = nullptr;
  int (*cu_stream_create_)(void**, unsigned, int) = nullptr;
  int (*cu_priority_range_)(int*, int*) = nullptr;
  int (*cu_stream_destroy_)(void*) = nullptr;
  int (*cu_stream_query_)(void*) = nullptr;
  int (*cu_memcpy_async_)(uint64_t, uint64_t, size_t, void*) = nullptr;
  int (*cu_mem_host_alloc_)(void**, size_t, unsigned) = nullptr;
  int (*cu_mem_host_device_ptr_)(uint64_t*, void*, unsigned) = nullptr;
  int (*cu_mem_free_host_)(void*) = nullptr;
  int (*cu_write_value32_)(void*, uint64_t, uint32_t, unsigned) = nullptr;
  int (*cu_wait_value32_)(void*, uint64_t, uint32_t, unsigned) = nullptr;
};

// The copy engine's queues (spec A7/A8): the service's job ring and the copy thread's in-flight, held and acking
// FIFOs. At most kDemandRecords + 1 jobs are outstanding at once (one per request slot, whose lease entry stays open
// until its COPYING leases are released, plus the one native-prefetch lease), so none of them can fill.
constexpr size_t kCopyRing = 32;
static_assert(kCopyRing > kDemandRecords + 1, "the copy engine's rings hold every job that can be outstanding");

// Test only (CPU): "copies" between host buffers. A mark completes only once the test has released it, and its
// copies land then (in query(), on the copy thread, in mark order: one stream), so a CopyDone published before its
// release is visible as bytes that are not there yet. issue, mark and query run on the copy thread; release, fail and
// marked on a test thread, through atomics only: no lock, and no allocation after construction (plan Task 12, F13).
class HostCopyBackend : public CopyBackend {
 public:
  // A mark's slot is reused kMarks marks later. A mark is open or in flight only while its job is (kCopyRing at most),
  // so the slot's earlier mark completed long before.
  static constexpr int kMarks = 2 * static_cast<int>(kCopyRing);
  // One mark's copies: every lane's copy of every layout name (row_layout.h caps a layout at 32), plus a ballast copy.
  static constexpr int kEntries = kLeaseLanes * 32 + 1;
  static constexpr int kIssueFailed = -1;  // fail(issue = true)
  static constexpr int kQueryFailed = -2;  // fail(query = true)
  static constexpr int kMarkFull = -3;     // more copies in one mark than kEntries: impossible under the layout cap

  std::string init() override {
    return "";
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    if (fail_issue_.load(std::memory_order_acquire)) return kIssueFailed;
    Mark& mark = marks_[open_ % kMarks];
    if (mark.count == kEntries) return kMarkFull;
    mark.entries[mark.count++] = CopyEntry{src, dst, bytes};
    return 0;
  }

  // Closes the open mark (its token is the number of marks closed before it) and opens the next.
  int mark(int64_t* token) override {
    *token = open_++;
    marks_[open_ % kMarks].count = 0;
    marked_.store(open_, std::memory_order_release);
    return 0;
  }

  int query(int64_t token) override {
    if (fail_query_.load(std::memory_order_acquire)) return kQueryFailed;
    if (token < completed_) return kDone;
    if (token != completed_ || released_.load(std::memory_order_acquire) <= completed_) return kPending;
    const Mark& mark = marks_[token % kMarks];
    for (int i = 0; i < mark.count; ++i) {
      const CopyEntry& copy = mark.entries[i];
      std::memcpy(
          reinterpret_cast<void*>(copy.dst), reinterpret_cast<const void*>(copy.src), static_cast<size_t>(copy.bytes));
    }
    ++completed_;
    return kDone;
  }

  void shutdown(bool) override {}

  // Lets `marks` more marks complete; negative: every mark, from now on (a standing budget, so a release made before
  // the copy thread has marked still counts).
  void release(int64_t marks) {
    int64_t seen = released_.load(std::memory_order_relaxed);
    int64_t next = 0;
    do {
      next = marks < 0 || seen > INT64_MAX - marks ? INT64_MAX : seen + marks;
    } while (!released_.compare_exchange_weak(seen, next, std::memory_order_release, std::memory_order_relaxed));
  }

  void fail(bool issue, bool query) {
    fail_issue_.store(issue, std::memory_order_release);
    fail_query_.store(query, std::memory_order_release);
  }

  // Marks closed so far: one per job issued.
  int64_t marked() const {
    return marked_.load(std::memory_order_acquire);
  }

 private:
  struct Mark {
    std::array<CopyEntry, kEntries> entries{};
    int count = 0;
  };
  std::array<Mark, kMarks> marks_{};  // about 0.5 MB, allocated once with the backend
  int64_t open_ = 0;                  // copy thread: the open mark's token
  int64_t completed_ = 0;             // copy thread: marks whose copies landed
  std::atomic<int64_t> marked_{0};    // written by the copy thread
  std::atomic<int64_t> released_{0};  // written by the test thread
  std::atomic<bool> fail_issue_{false};
  std::atomic<bool> fail_query_{false};
};

// The copy thread: issues each job's copies on the backend's stream, records one mark after them, polls marks in
// order and hands each completed job to its owner. A backend error hands every job it still holds to the owner's
// copy_failed and stops issuing: nothing it issued may be assumed complete, so none of those leases is released (E5).
//
// `Build`: the tier's build policy (build_policy.h). `Owner`: the RamTier, which befriends this class. Its hooks,
// all called on the copy thread (copy_failed also on the submitting thread, for a ring overflow):
//   - bool copy_completed(const CopyJob&): the job's copies completed; false when it still waits for the copy wait's
//     SmAck, which copy_acked then polls;
//   - bool copy_acked(const CopyJob&): true once it released the job's leases;
//   - void copy_failed(const CopyJob&, int error): completion cannot be established; fail stop;
//   - owner->template copy_count<K>(n): the copy thread's counters.
//
// Service to copy thread (spec 6.3 item 5): an SPSC job ring. submit() takes no lock and never blocks. The copy thread
// polls the ring; after spin_ns of idle (nothing popped, in flight, held or acking) it sleeps on a futex, and submit()
// makes the wake syscall only when that thread is asleep. No wake is lost:
//   - copy thread (idle): seen = wake_; sleeping_ = true; fence(seq_cst); if the ring is empty and no stop was asked
//     for, futex_wait(&wake_, seen);
//   - submit: push (release); fence(seq_cst); if sleeping_, ++wake_ and futex_wake.
// The two seq_cst fences are totally ordered. If submit's fence comes first, the copy thread's ring check after its
// fence sees the push and it does not wait. If the copy thread's fence comes first, submit's load after its fence
// sees sleeping_ and it wakes: either before futex_wait (then wake_ != seen, and the kernel returns at once, since it
// compares the word under its own lock) or after it (then the wake ends the wait). The wait's 1 ms cap is a backstop
// only, not part of the argument.
template <class Build, class Owner>
class CopyEngine {
  static_assert(BuildPolicy<Build>);

 public:
  // `thread_name` is the copy thread's pthread name (e.g. Layout::kName + "-copy-eng"), truncated to 15 bytes
  // (pthread_setname_np's limit).
  CopyEngine(
      std::unique_ptr<CopyBackend> backend,
      int64_t rows,
      int64_t spin_ns,
      Owner* owner,
      std::string prefix,
      std::string thread_name)
      : backend_(std::move(backend)),
        tables_(static_cast<size_t>(rows)),
        spin_ns_(spin_ns),
        owner_(owner),
        prefix_(std::move(prefix)),
        thread_name_(thread_name.substr(0, 15)) {}

  ~CopyEngine() {
    stop(5'000'000'000LL);
  }

  // Starts the thread and returns once the backend is initialised on it; throws with the backend's error.
  void start() {
    spin_iters_ = idle_budget(spin_ns_);  // on the caller's thread: the copy thread never reads the clock to pace
    thread_ = std::thread([this] { run(); });
    std::unique_lock<std::mutex> lock(start_mutex_);
    ready_cv_.wait(lock, [this] { return started_; });
    if (!init_error_.empty()) {
      lock.unlock();
      stop(0);
      throw std::runtime_error(prefix_ + init_error_);
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

  // The tier's owner only (the service thread, or the caller of pump()). Takes no lock and never blocks: at most
  // kDemandRecords + 1 jobs are outstanding, and the ring holds kCopyRing, so a full ring is an internal error, which
  // fails stop. The wake is a syscall only when the copy thread has gone to sleep (idle past spin_ns).
  void submit(const CopyJob& job) {
    if (!jobs_.push(job)) {
      owner_->copy_failed(job, kRingOverflow);
      return;
    }
    submitted_.store(submitted_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
  }

  // Every submitted job has completed (or failed) and been handed back: submitted_ is the submitter's, finished_ the
  // copy thread's, each written by one thread.
  bool idle() const {
    return finished_.load(std::memory_order_acquire) == submitted_.load(std::memory_order_acquire);
  }

  bool wait_idle(int64_t deadline_ns) {  // a paused caller or a test: not the hot path
    while (!idle()) {
      if (now_ns() > deadline_ns) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  // Joins the thread once it has seen every in-flight job complete, or `drain_ns` passed.
  void stop(int64_t drain_ns) {
    if (!thread_.joinable()) return;
    drain_deadline_.store(now_ns() + drain_ns, std::memory_order_relaxed);
    stop_.store(true, std::memory_order_release);
    // Release, so a copy thread whose acquire load of wake_ (sleep_until_submit's `seen`) reads this bump also sees
    // stop_ and does not sleep; one that reads the value before it fails futex_wait's compare or gets this wake.
    wake_.fetch_add(1, std::memory_order_release);
    futex_wake(&wake_);
    thread_.join();
  }

  HostCopyBackend* host_backend() {
    return dynamic_cast<HostCopyBackend*>(backend_.get());
  }

  // CPU experts: the engine a job's CPU lanes go to. Set before any grant publishes a CPU lane (the service sets it
  // before its thread starts); the release pairs with the copy thread's acquire at issue.
  void set_cpu(CpuExpertEngine* cpu) {
    cpu_.store(cpu, std::memory_order_release);
  }

  // Test only: one more copy of `bytes` issued ahead of every job's own, so each job completes that much later.
  // InstrBuild only.
  void set_ballast(uint64_t dst, uint64_t src, int64_t bytes)
    requires(Build::kFaults)
  {
    ballast_.dst.store(dst);
    ballast_.src.store(src);
    ballast_.bytes.store(bytes);
  }

 private:
  static constexpr int kRingOverflow = -1000;
  static constexpr int kNoCpuEngine = -1001;   // a job carries CPU lanes but no CPU expert engine is set
  static constexpr int kCpuRingOverflow = -1002;
  static constexpr int kCpuFailed = -1003;     // the CPU expert thread's forward failed
  using Queue = FixedDeque<CopyJob, kCopyRing>;

  struct Table {
    std::vector<CopyEntry> entries;
    bool sm = false;  // an entry is left to the copy wait's SM reads
    int64_t dst_rows = 0;
    std::atomic<bool> ready{false};
  };

  void run() {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    const std::string error = backend_->init();
    {
      std::lock_guard<std::mutex> guard(start_mutex_);  // the start handshake: setup, not the hot path
      init_error_ = error;
      started_ = true;
    }
    ready_cv_.notify_all();
    if (!error.empty()) return;
    Queue in_flight;
    Queue held;    // prefetch jobs not yet issued: demand goes first on the link
    Queue acking;  // copies completed, CopyDone published; the leases wait for the copy wait's SmAck
    uint64_t idle = 0;  // empty polls since the last progress: the spin budget counts these (spec M8)
    while (true) {
      // Read before the ring: stop() is asked only after the last submit, so a stop seen here has every job in the
      // ring.
      const bool stopping = stop_.load(std::memory_order_acquire);
      bool demand_fresh = false;
      bool progressed = false;
      CopyJob job;
      while (jobs_.pop(&job)) {
        progressed = true;
        if (job.prefetch) {
          push_or_fail(held, job);
          continue;
        }
        demand_fresh = true;
        issue_or_fail(job, in_flight);
      }
      // Demand before prefetch: a prefetch is issued only when no demand job arrived this pass and none is in flight,
      // so a mispredicted row never sits on the copy stream ahead of a demand row. Once issued it cannot be preempted;
      // the device never posts a demand while its own layer's prefetch is outstanding, so none can queue behind it.
      bool demand_in_flight = false;
      for (size_t i = 0; i < in_flight.size(); ++i)
        demand_in_flight = demand_in_flight || !in_flight[i].prefetch;
      if (!held.empty() && (demand_fresh || demand_in_flight)) {
        if (!held_counted_) count<kPrefetchHeld>();
        held_counted_ = true;
      }
      while (!held.empty() && ((!demand_fresh && !demand_in_flight) || stopping)) {
        held_counted_ = false;
        CopyJob next = held.front();
        held.pop_front();
        issue_or_fail(next, in_flight);
      }
      // The head job is polled every turn, oldest first. CudaCopyBackend's poll is one acquire load of the completion
      // word, no driver call (the cuEventQuery it replaces took a libcuda mutex per call: ~2,300-3,200 per job over
      // arm CS3, results.md 9a), and one cuStreamQuery per CompletionWord budget while the head stays pending.
      while (!in_flight.empty() && broken_ == 0) {
        const CopyJob& head = in_flight.front();
        const int state = head.token == kNoToken ? CopyBackend::kDone : backend_->query(head.token);
        if (state == CopyBackend::kPending) break;
        if (state != CopyBackend::kDone) {
          broken_ = state;
          break;
        }
        if (head.cpu_seq != 0) {
          const CpuExpertEngine* cpu = cpu_.load(std::memory_order_relaxed);  // set before the job was issued
          if (!cpu->done(head.cpu_seq)) {
            if (cpu->broken() != 0) broken_ = kCpuFailed;  // never done: fail the job, its leases stay held (E5)
            break;
          }
        }
        const CopyJob done = in_flight.front();
        in_flight.pop_front();
        record_latency(done);
        if (owner_->copy_completed(done)) {
          finish();
        } else {
          push_or_fail(acking, done);
        }
        progressed = true;
      }
      const size_t acking_before = acking.size();
      acking.erase_if([this](CopyJob& waiting) {
        if (!owner_->copy_acked(waiting)) return false;
        finish();
        return true;
      });
      progressed = progressed || acking.size() != acking_before;
      if (broken_ != 0) {
        while (!in_flight.empty()) {
          finish_failed(in_flight.front(), broken_);
          in_flight.pop_front();
        }
      }
      // The drain deadline is read only once a stop was asked for (short-circuit): never while serving.
      if (stopping && held.empty() && ((in_flight.empty() && acking.empty()) || now_ns() > drain_deadline())) break;
      if (progressed) {
        idle = 0;
      } else if (!in_flight.empty() || !held.empty() || !acking.empty() || ++idle < spin_iters_) {
        _mm_pause();
      } else {
        sleep_until_submit();  // the idle path: nothing to poll, and spin_ns of nothing arriving
        idle = 0;
      }
    }
    backend_->shutdown(in_flight.empty() && broken_ == 0);
  }

  int64_t drain_deadline() const {  // written by stop() before its release of stop_, read after its acquire
    return drain_deadline_.load(std::memory_order_relaxed);
  }

  // The class comment's wake protocol, copy side.
  void sleep_until_submit() {
    const uint32_t seen = wake_.load(std::memory_order_acquire);
    sleeping_.store(true, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with submit()'s push and sleeping_ load
    if (jobs_.empty() && !stop_.load(std::memory_order_acquire)) futex_wait(&wake_, seen, 1'000'000);  // 1 ms cap
    sleeping_.store(false, std::memory_order_relaxed);
  }

  // A full queue is impossible under the outstanding-job bound; if it ever happens, fail stop (E5: keep the leases).
  void push_or_fail(Queue& queue, const CopyJob& job) {
    if (!queue.push_back(job)) finish_failed(job, kRingOverflow);
  }

  void issue_or_fail(CopyJob& job, Queue& in_flight) {
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
    push_or_fail(in_flight, job);
  }

  int issue(CopyJob& job) {
    const Table& table = tables_[job.row];
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();  // copy_issue_ns, a metric
    int64_t bytes = 0;
    if constexpr (Build::kFaults) {
      if (const int64_t ballast = ballast_.bytes.load(); ballast > 0) {
        if (const int r = backend_->issue(ballast_.dst.load(), ballast_.src.load(), ballast)) return r;
      }
    }
    job.sm = table.sm && !job.prefetch;
    if (job.cpu_mask != 0) {
      // First, so the CPU starts while the copies below are issued.
      CpuExpertEngine* cpu = cpu_.load(std::memory_order_acquire);
      if (cpu == nullptr) return kNoCpuEngine;
      CpuJob cpu_job;
      cpu_job.row = job.row;
      for (int i = 0; i < job.count; ++i) {
        const CopyLane& lane = job.lanes[i];
        if ((job.cpu_mask >> lane.lane & 1u) == 0) continue;
        cpu_job.slots[cpu_job.k] = lane.host_slot;
        cpu_job.weights[cpu_job.k] = lane.weight;
        ++cpu_job.k;
      }
      job.cpu_seq = cpu->submit(cpu_job);
      if (job.cpu_seq == 0) return kCpuRingOverflow;
    }
    bool copied = false;
    if constexpr (Build::kFaults) copied = ballast_.bytes.load() > 0;
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      if ((job.cpu_mask >> lane.lane & 1u) != 0) continue;  // computed on the CPU: nothing copies its slot
      copied = true;
      for (const CopyEntry& entry : table.entries) {
        if (entry.sm && job.sm) continue;
        const uint64_t dst = entry.dst + static_cast<uint64_t>(lane.dst_slot) * static_cast<uint64_t>(entry.bytes);
        const uint64_t src = entry.src + static_cast<uint64_t>(lane.host_slot) * static_cast<uint64_t>(entry.bytes);
        if (const int r = backend_->issue(dst, src, entry.bytes)) return r;
        bytes += entry.bytes;
      }
    }
    if (!copied) {
      job.token = kNoToken;
      return 0;
    }
    const int r = backend_->mark(&job.token);
    if constexpr (Build::kMetrics) {
      count<kCopyIssueNs>(now_ns() - start);
      count<kCopyBytes>(bytes);
    }
    return r;
  }

  // copy_latency_ns and its max: metrics, so ProdBuild reads no clock here.
  void record_latency(const CopyJob& job) {
    if constexpr (Build::kMetrics) {
      const int64_t latency = now_ns() - job.submit_ns;
      count<kCopyLatencyNs>(latency);
      std::atomic<int64_t>& max = owner_->stats_.v[kCopyLatencyMaxNs];
      int64_t seen = max.load(std::memory_order_relaxed);
      while (latency > seen && !max.compare_exchange_weak(seen, latency, std::memory_order_relaxed)) {
      }
    } else {
      (void)job;
    }
  }

  // The copy thread's counters: the owner's copy_count, a core counter into its copy-thread block, a metric into the
  // shared stats (InstrBuild only).
  template <Counter K>
  void count(int64_t n = 1) {
    owner_->template copy_count<K>(n);
  }

  void finish_failed(const CopyJob& job, int error_code) {
    count<kCopyErrors>();
    owner_->copy_failed(job, error_code);
    finish();
  }

  void finish() {  // copy thread
    finished_.store(finished_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  }

  std::unique_ptr<CopyBackend> backend_;
  std::atomic<CpuExpertEngine*> cpu_{nullptr};
  std::vector<Table> tables_;
  int64_t spin_ns_;
  uint64_t spin_iters_ = 1;  // idle polls before the futex sleep: idle_budget(spin_ns_), set in start()
  Owner* owner_;
  std::string prefix_;
  std::string thread_name_;
  std::thread thread_;
  SpscRing<CopyJob, kCopyRing> jobs_;   // the submitter pushes, the copy thread pops
  std::atomic<uint64_t> submitted_{0};  // written by the submitter only
  std::atomic<uint64_t> finished_{0};   // written by the copy thread only
  std::atomic<bool> sleeping_{false};   // the copy thread is (about to be) in futex_wait
  std::atomic<uint32_t> wake_{0};       // the futex word: bumped by every wake
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> drain_deadline_{0};
  std::mutex start_mutex_;  // start()'s handshake only: started_ and init_error_
  std::condition_variable ready_cv_;
  bool started_ = false;
  std::string init_error_;
  int broken_ = 0;             // copy thread only: the first backend error
  bool held_counted_ = false;  // copy thread only: the current hold was counted in kPrefetchHeld
  struct Ballast {  // set_ballast: InstrBuild only
    std::atomic<uint64_t> dst{0};
    std::atomic<uint64_t> src{0};
    std::atomic<int64_t> bytes{0};
  };
  struct NoBallast {};
  [[no_unique_address]] std::conditional_t<Build::kFaults, Ballast, NoBallast> ballast_;
};

}  // namespace expert_stream
}  // namespace sglang
