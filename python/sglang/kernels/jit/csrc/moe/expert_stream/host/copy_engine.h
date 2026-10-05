// The copy engine: moves host-resident expert rows into device slots with DMA, off the decode stream.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include <algorithm>
#include <array>
#include <future>
#include <optional>
#include <sched.h>

#include "../lease_layout.h"
#include "build_policy.h"
#include "completion_word.h"
#include "cpu_experts.h"
#include "spsc_ring.h"
#include "tier_protocol.h"

namespace sglang {
namespace expert_stream {

using namespace ::sglang::expert_stream::wire;

/// One host lane of a record: copy host slot `host_slot` into device slot `dst_slot`, or, for a CPU lane, compute it.
struct CopyLane {
  int32_t lane = 0;
  int32_t host_slot = 0;
  int32_t dst_slot = 0;
  float weight = 0.0f;  // read only for a CPU lane
};

/// CopyJob::token of a job that issued no copy (every lane on the CPU): there is nothing to query.
constexpr int64_t kNoToken = -1;

/// One NUMA group's host lanes of one record, submitted by the service when it serves the record.
///
/// A job carries three kinds of lane: copy-engine hits, copied by DMA; CPU hits (`cpu_mask`, Wire::kKindHitCpu),
/// computed by the group's CPU expert engine as one job, sequence `cpu_seq`; and CPU misses (`late_cpu`,
/// Wire::kKindMissCpu), computed as their rows land, the last as sequence `late_seq`. The record's CopyDone is
/// published once every group in `groups` has sent its part and each part's copies and CPU jobs are done.
struct CopyJob {
  uint64_t gen = 0;
  int64_t idx = 0;
  int64_t row = 0;
  uint32_t mask = 0;
  int count = 0;
  CopyLane lanes[Wire::kLanes];
  int64_t submit_ns = 0;
  int64_t token = kNoToken;  // the backend's mark after the job's last copy
  uint32_t cpu_mask = 0;
  uint32_t cpu_seq = 0;
  int late_cpu = 0;
  uint32_t late_seq = 0;
  int group = 0;
  uint32_t groups = 1;  // bit g: group g sends a part of this record
};

/// One entry of a row's copy table: a source slab, a destination tensor and the bytes of one expert row in each.
/// A lane's host slot indexes the source and its device slot the destination.
struct CopyEntry {
  uint64_t src = 0;
  uint64_t dst = 0;
  int64_t bytes = 0;
  bool sm = false;  // copied by the copy wait's SM reads, not the DMA (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES)
};

/// The copy primitive the copy thread drives: issue copies in order, mark the point after them, and ask whether a
/// mark has completed. Every call is made on the copy thread.
class CopyBackend {
 public:
  static constexpr int kDone = 0;
  static constexpr int kPending = 1;
  virtual ~CopyBackend() = default;
  /// Prepares the backend on the copy thread; "" on success, else the reason it refused.
  virtual std::string init() = 0;
  /// Queues one copy after every earlier one; 0 or an error code.
  virtual int issue(uint64_t dst, uint64_t src, int64_t bytes) = 0;
  /// Marks the point after every copy issued so far and stores its token; 0 or an error code.
  virtual int mark(int64_t* token) = 0;
  /// kDone, kPending, or an error code (negative for the host backend).
  virtual int query(int64_t token) = 0;
  /// Releases the backend's resources if `idle`; otherwise keeps them, since a copy may still be in flight.
  virtual void shutdown(bool idle) = 0;
};
static_assert(CompletionWord::kDone == CopyBackend::kDone && CompletionWord::kPending == CopyBackend::kPending);

/// The production backend: CUDA driver copies on a private stream, completion tracked by a host-mapped word.
///
/// Per job it issues one cuMemcpyAsync per copy, then one cuStreamWriteValue32_v2 of the job's sequence number into
/// the completion word (completion_word.h). query() is an acquire load of that word, plus one cuStreamQuery per
/// CompletionWord budget while a job stays pending: no synchronize, module load or allocation per job.
///
/// - The driver is resolved from libcuda.so.1 at init(), so the module builds and loads without a CUDA toolkit.
/// - Only the v2 stream memory ops: the plain names are the v1 API, gated by NVreg_EnableStreamMemOPs, whose device
///   attribute reads 0 on the reference machine while the v2 ops work (analysis/dsv41-drive/hotpath/results.md 9b).
/// - The stream has the greatest priority, for a hardware queue of its own. Streams share CUDA_DEVICE_MAX_CONNECTIONS
///   queues (the server sets 8) and a queue runs in order, so a copy behind a stream waiting on the decode graph could
///   not start until the copy wait timed out. analysis/dsv41-drive/copy-engine/hol_probe.py: a default-priority stream
///   waited 596 ms behind 8 blocked ones; greatest-priority copies never waited, with up to 64 blocked.
/// - No fallback: init() refuses when the write-value op is missing, errors or never reaches the word, and when the
///   wait-value op the decode stream's copy wait uses on host-mapped memory is missing, errors, does not hold its
///   stream, or is not released by a host store.
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
    get(cu_write_value32_, "cuStreamWriteValue32_v2");
    get(cu_wait_value32_, "cuStreamWaitValue32_v2");
    if (!missing.empty()) return "libcuda.so.1 lacks " + missing + ", which the copy engine needs";
    if (int r = cu_init_(0)) return "cuInit failed: " + std::to_string(r);
    if (int r = cu_device_get_(&cu_device_, device_)) return "cuDeviceGet failed: " + std::to_string(r);
    // PyTorch's primary context, so the slabs' registrations and the destination tensors are valid here.
    if (int r = cu_primary_retain_(&context_, cu_device_))
      return "cuDevicePrimaryCtxRetain failed: " + std::to_string(r);
    retained_ = true;
    if (int r = cu_ctx_set_current_(context_)) return "cuCtxSetCurrent failed: " + std::to_string(r);
    int least = 0;
    int greatest = 0;
    if (int r = cu_priority_range_(&least, &greatest))
      return "cuCtxGetStreamPriorityRange failed: " + std::to_string(r);
    if (int r = cu_stream_create_(&stream_, kNonBlocking, greatest))
      return "cuStreamCreateWithPriority failed: " + std::to_string(r);
    if (std::string error = init_word(); !error.empty()) return error;
    return probe_stream_wait();
  }

  int issue(uint64_t dst, uint64_t src, int64_t bytes) override {
    return cu_memcpy_async_(dst, src, static_cast<size_t>(bytes), stream_);
  }

  /// The token is the job's sequence number, written into the completion word after the job's copies.
  int mark(int64_t* token) override {
    const uint32_t seq = ++marked_;
    *token = seq;
    return cu_write_value32_(stream_, word_dev_, seq, 0);  // default flags: after the prior copies, fenced
  }

  int query(int64_t token) override {
    return word_.poll(static_cast<uint32_t>(token), [this] { return cu_stream_query_(stream_); });
  }

  /// Logs the completion word's final value (the exact number of jobs the stream completed). Frees the stream, word and
  /// context only if `idle`: after an error, or with copies in flight, a copy may still read a slab and the stream may
  /// still write the word.
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
  static constexpr unsigned kNonBlocking = 1;  // CU_STREAM_NON_BLOCKING
  static constexpr unsigned kDeviceMap = 2;    // CU_MEMHOSTALLOC_DEVICEMAP
  static constexpr unsigned kGeq = 0;          // CU_STREAM_WAIT_VALUE_GEQ

  /// Queries the stream every 10 us until it completes or about a second passes; returns the last query.
  int wait_stream() {
    constexpr int kPolls = 100'000;
    int r = cu_stream_query_(stream_);
    for (int i = 0; i < kPolls && r == CompletionWord::kNotReady; ++i) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
      r = cu_stream_query_(stream_);
    }
    return r;
  }

  /// Allocates the host-mapped completion word and checks that a first write through the stream lands in it.
  std::string init_word() {
    void* host = nullptr;
    if (int r = cu_mem_host_alloc_(&host, sizeof(uint32_t), kDeviceMap))
      return "cuMemHostAlloc of the completion word failed: " + std::to_string(r);
    word_host_ = static_cast<uint32_t*>(host);
    __atomic_store_n(word_host_, 0xFFFFFFFFu, __ATOMIC_RELEASE);  // not 0: the first write below must change it
    if (int r = cu_mem_host_device_ptr_(&word_dev_, host, 0))
      return "cuMemHostGetDevicePointer of the completion word failed: " + std::to_string(r);
    if (int r = cu_write_value32_(stream_, word_dev_, 0, 0))
      return "cuStreamWriteValue32_v2 failed (" + std::to_string(r) +
             "): the copy engine needs the v2 stream memory operations for its completion word";
    if (const int r = wait_stream(); r != 0)
      return r == CompletionWord::kNotReady
                 ? "the completion word's first write did not complete within a second"
                 : "the completion word's first write failed: cuStreamQuery " + std::to_string(r);
    word_.bind(word_host_);
    if (word_.load() != 0) return "the completion word's first write completed but did not reach host memory";
    return "";
  }

  /// Tries the copy wait's primitive once on this stream: a wait on a host-mapped word must hold the stream until a
  /// host store satisfies it. The decode stream's gate uses the same op on the same kind of memory, so a driver
  /// without it is refused here instead of hanging the first decode.
  std::string probe_stream_wait() {
    void* host = nullptr;
    if (int r = cu_mem_host_alloc_(&host, sizeof(uint32_t), kDeviceMap))
      return "cuMemHostAlloc of the stream-wait probe word failed: " + std::to_string(r);
    const std::string error = probe_stream_wait_on(static_cast<uint32_t*>(host));
    // Freed only once the stream no longer waits on it; a probe that never completed leaks its 4 bytes instead.
    if (error.empty() || cu_stream_query_(stream_) == 0) cu_mem_free_host_(host);
    return error;
  }

  std::string probe_stream_wait_on(uint32_t* word) {
    __atomic_store_n(word, Wire::kLeaseGateClosed, __ATOMIC_RELEASE);  // the production gate encoding
    uint64_t device_word = 0;
    if (int r = cu_mem_host_device_ptr_(&device_word, word, 0))
      return "cuMemHostGetDevicePointer of the stream-wait probe word failed: " + std::to_string(r);
    if (int r = cu_wait_value32_(stream_, device_word, Wire::kLeaseGateOpen, kGeq))
      return "cuStreamWaitValue32_v2 failed (" + std::to_string(r) +
             "): the copy wait needs the v2 stream wait on host-mapped memory";
    const int held = cu_stream_query_(stream_);
    __atomic_store_n(word, Wire::kLeaseGateOpen, __ATOMIC_RELEASE);  // on every path, so nothing stays queued
    if (held == 0)
      return "cuStreamWaitValue32_v2 did not hold its stream: the copy wait could not order the decode stream";
    if (held != CompletionWord::kNotReady) return "the stream-wait probe failed: cuStreamQuery " + std::to_string(held);
    if (const int r = wait_stream(); r != 0)
      return r == CompletionWord::kNotReady ? "cuStreamWaitValue32_v2 was not released by a host store within a second"
                                            : "the stream-wait probe failed: cuStreamQuery " + std::to_string(r);
    return "";
  }

  int device_;
  std::string prefix_;
  int cu_device_ = 0;
  void* context_ = nullptr;
  void* stream_ = nullptr;
  bool retained_ = false;
  uint32_t* word_host_ = nullptr;  // host-mapped pinned memory the stream writes
  uint64_t word_dev_ = 0;
  CompletionWord word_;
  uint32_t marked_ = 0;  // the last job sequence written (wraps at 2^32)
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

/// Capacity of each group's job ring and of the copy thread's in-flight FIFO. The device waits on every record with
/// host lanes, so at most one job per demand record and group is outstanding.
constexpr size_t kCopyRing = 32;
static_assert(
    kCopyRing >= static_cast<size_t>(Wire::kDemandRecords) * Wire::kNodes,
    "the copy engine's queues hold every job that can be outstanding, across all groups");

/// The copy thread: issues each job's copies, marks the point after them, polls marks oldest first and hands each
/// completed record back to its owner.
///
/// `Build` is the tier's build policy (build_policy.h): metrics and fault injection compile in only where it says.
/// `Owner` is the RamTier, which befriends this class and provides, all called on the copy thread:
///   void copy_completed(const CopyJob&)   copies and CPU jobs done: publish CopyDone and open the gate;
///   void copy_failed(const CopyJob&, int)  completion cannot be established: fail stop (also called on the
///                                          submitting thread for a ring overflow);
///   owner->template copy_count<K>(n)       the copy thread's counters.
///
/// Each group's service thread pushes to its own SPSC ring; submit() takes no lock, never blocks and makes no syscall.
/// The copy thread never sleeps: it polls the rings and its in-flight marks with PAUSE between polls, on a core of
/// its own (ThreadingConfig.copy_cpus).
/// After the first backend error nothing the backend issued can be assumed complete, so that job and every job behind
/// it go to copy_failed.
template <class Build, class Owner>
class CopyEngine {
  static_assert(BuildPolicy<Build>);

 public:
  /// Why CopyDone is not stored, for the watchdog: seq << 32 | group << 8 | reason (kStallCpu: group's CPU job `seq`
  /// is not done; kStallNoPart: group's part of the record never came); 0 when nothing waits.
  static constexpr uint32_t kStallCpu = 1;
  static constexpr uint32_t kStallNoPart = 2;

  /// `thread_name` is truncated to 15 bytes (pthread_setname_np's limit). `cpus` is the thread's affinity
  /// (ThreadingConfig.copy_cpus, the GPU's node); empty inherits the starter's.
  CopyEngine(
      std::unique_ptr<CopyBackend> backend,
      int64_t rows,
      int groups,
      Owner* owner,
      std::string prefix,
      std::string thread_name,
      std::vector<int> cpus)
      : backend_(std::move(backend)),
        tables_(static_cast<size_t>(rows)),
        owner_(owner),
        prefix_(std::move(prefix)),
        thread_name_(thread_name.substr(0, 15)),
        cpus_(std::move(cpus)) {
    for (int g = 0; g < groups; ++g)
      jobs_.push_back(std::make_unique<SpscRing<CopyJob, kCopyRing>>());
  }

  ~CopyEngine() {
    stop(kDestructorDrainNs);
  }

  /// Starts the thread and returns once the backend is initialised on it; throws with the backend's error.
  void start() {
    std::future<std::string> started = started_.get_future();
    thread_ = std::thread([this] { run(); });
    if (const std::string error = started.get(); !error.empty()) {
      stop(0);
      throw std::runtime_error(prefix_ + error);
    }
  }

  /// Row `row`'s copy table and the number of destination rows each entry's tensor holds. Once per row.
  void set_table(int64_t row, std::vector<CopyEntry> entries, int64_t dst_rows) {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size())) throw std::runtime_error("copy table row out of range");
    Table& table = tables_[row];
    if (table.ready.load(std::memory_order_acquire)) throw std::runtime_error("copy table already set for this row");
    for (const CopyEntry& entry : entries)
      if (!entry.sm) table.dma.push_back(entry);
    table.dst_rows = entries.empty() ? 0 : dst_rows;
    table.ready.store(true, std::memory_order_release);
  }

  bool eligible(int64_t row, int32_t dst_slot) const {
    const Table* table = ready_table(row);
    return table != nullptr && dst_slot >= 0 && dst_slot < table->dst_rows;
  }

  /// The entries of row `row` that the DMA copies, for the split calibration, which must time exactly the bytes a
  /// decode moves. Throws if the row has no table.
  std::vector<CopyEntry> dma_entries(int64_t row) const {
    const Table* table = ready_table(row);
    if (table == nullptr) throw std::runtime_error("no copy table for row " + std::to_string(row));
    return table->dma;
  }

  /// Hands group `group`'s job to the copy thread; called only by that group's tier owner. A full ring is impossible
  /// under the outstanding-job bound (kCopyRing), so it fails stop.
  void submit(int group, const CopyJob& job) {
    if (!jobs_[group]->push(job)) {
      owner_->copy_failed(job, kRingOverflow);
      return;
    }
    submitted_[group].store(submitted_[group].load(std::memory_order_relaxed) + 1, std::memory_order_release);
  }

  /// True when every submitted job has completed or failed.
  bool idle() const {
    uint64_t submitted = 0;
    for (const auto& count : submitted_)
      submitted += count.load(std::memory_order_acquire);
    return finished_.load(std::memory_order_acquire) == submitted;
  }

  /// Polls until idle() or `deadline_ns` (now_ns() clock); whether it went idle. For a paused caller or a test.
  bool wait_idle(int64_t deadline_ns) {
    while (!idle()) {
      if (now_ns() > deadline_ns) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    return true;
  }

  /// Joins the thread once it has seen every in-flight job complete, or `drain_ns` passed.
  void stop(int64_t drain_ns) {
    if (!thread_.joinable()) return;
    drain_deadline_.store(now_ns() + drain_ns, std::memory_order_relaxed);
    stop_.store(true, std::memory_order_release);
    thread_.join();
  }

  CopyBackend* backend() {
    return backend_.get();
  }

  /// Sets the CPU expert engine whose done() completes group `group`'s CPU lanes, before any record carries one.
  void set_cpu(int group, CpuExpertEngine* cpu) {
    cpu_[group].store(cpu, std::memory_order_release);
  }

  uint64_t stall() const {
    return stall_.load(std::memory_order_relaxed);
  }

  /// Test only (InstrBuild): one extra copy of `bytes` ahead of every job's own, so each job completes that much later.
  void set_ballast(uint64_t dst, uint64_t src, int64_t bytes)
    requires(Build::kFaults)
  {
    ballast_.dst.store(dst);
    ballast_.src.store(src);
    ballast_.bytes.store(bytes);
  }

 private:
  static constexpr int kRingOverflow = -1000;
  static constexpr int64_t kDestructorDrainNs = 5'000'000'000;
  using Queue = FixedDeque<CopyJob, kCopyRing>;

  struct Table {
    std::vector<CopyEntry> dma;  // the entries the DMA copies; SM entries are the copy wait's
    int64_t dst_rows = 0;        // 0 for an empty table: no slot is eligible
    std::atomic<bool> ready{false};
  };

  /// The parts of one demand ring index's record seen so far.
  struct Assembly {
    uint64_t gen = 0;
    uint32_t got = 0;
  };

  struct Ballast {
    std::atomic<uint64_t> dst{0};
    std::atomic<uint64_t> src{0};
    std::atomic<int64_t> bytes{0};
  };
  struct NoBallast {};

  const Table* ready_table(int64_t row) const {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size())) return nullptr;
    const Table& table = tables_[row];
    return table.ready.load(std::memory_order_acquire) ? &table : nullptr;
  }

  /// Names and pins the thread, then initialises the backend on it; returns why it could not, else "".
  std::string init_thread() {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    if (!cpus_.empty()) {
      cpu_set_t set;
      CPU_ZERO(&set);
      for (const int cpu : cpus_)
        CPU_SET(cpu, &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) return "cannot pin the copy thread to its cores";
    }
    return backend_->init();
  }

  void run() {
    const std::string error = init_thread();
    started_.set_value(error);
    if (!error.empty()) return;
    Queue in_flight;
    while (true) {
      // Read before the rings: stop() comes after the last submit, so a stop seen here has every job in them.
      const bool stopping = stop_.load(std::memory_order_acquire);
      bool progressed = issue_submitted(in_flight);
      progressed |= retire_completed(in_flight);
      // The drain deadline is read only once a stop was asked for: never while serving.
      if (stopping && (in_flight.empty() || now_ns() > drain_deadline_.load(std::memory_order_relaxed))) break;
      if (!progressed) _mm_pause();
    }
    backend_->shutdown(in_flight.empty() && broken_ == 0);
  }

  /// Pops every group's submitted jobs and issues them; whether any came.
  bool issue_submitted(Queue& in_flight) {
    bool popped = false;
    CopyJob job;
    for (auto& ring : jobs_) {
      while (ring->pop(&job)) {
        popped = true;
        if (broken_ == 0) broken_ = issue(job);
        if (broken_ != 0)
          finish_failed(job, broken_);
        else if (!in_flight.push_back(job))
          finish_failed(job, kRingOverflow);
      }
    }
    return popped;
  }

  /// Retires in-flight jobs oldest first while their copies and CPU jobs are done; after a backend error, fails every
  /// one. Whether any completed. For CudaCopyBackend a poll is one acquire load, not a driver call: the cuEventQuery it
  /// replaced took a libcuda mutex per call, ~2,300-3,200 times per job (analysis/dsv41-drive/hotpath/results.md 9a).
  bool retire_completed(Queue& in_flight) {
    bool completed = false;
    while (!in_flight.empty() && broken_ == 0) {
      const CopyJob& head = in_flight.front();
      const int state = head.token == kNoToken ? CopyBackend::kDone : backend_->query(head.token);
      if (state == CopyBackend::kPending) break;
      if (state != CopyBackend::kDone) {
        broken_ = state;
        break;
      }
      if (const std::optional<uint32_t> seq = cpu_pending(head)) {
        set_stall(*seq, head.group, kStallCpu);
        break;
      }
      complete(head);
      in_flight.pop_front();
      completed = true;
    }
    if (broken_ != 0) {
      while (!in_flight.empty()) {
        finish_failed(in_flight.front(), broken_);
        in_flight.pop_front();
      }
    }
    return completed;
  }

  /// The sequence of `job`'s first CPU job that is not done yet, if any.
  std::optional<uint32_t> cpu_pending(const CopyJob& job) const {
    if (job.cpu_mask == 0 && job.late_cpu == 0) return std::nullopt;
    const CpuExpertEngine* cpu = cpu_[job.group].load(std::memory_order_relaxed);
    if (job.cpu_mask != 0 && !cpu->done(job.cpu_seq)) return job.cpu_seq;
    if (job.late_cpu > 0 && !cpu->done(job.late_seq)) return job.late_seq;
    return std::nullopt;
  }

  /// Records one group's part of a record as done, and hands the record to the owner once every part is.
  void complete(const CopyJob& job) {
    record_latency(job);
    Assembly& record = assembly_[job.idx];
    if (record.gen != job.gen) record = Assembly{job.gen, 0};
    record.got |= 1u << job.group;
    if (record.got == job.groups) {
      owner_->copy_completed(job);
      stall_.store(0, std::memory_order_relaxed);
    } else {
      set_stall(0, __builtin_ctz(job.groups & ~record.got), kStallNoPart);
    }
    finish();
  }

  void set_stall(uint32_t seq, int group, uint32_t reason) {
    stall_.store(
        static_cast<uint64_t>(seq) << 32 | static_cast<uint64_t>(group) << 8 | reason, std::memory_order_relaxed);
  }

  /// Issues `job`'s copies and marks the point after them; 0 or the backend's error.
  int issue(CopyJob& job) {
    const Table& table = tables_[job.row];
    int64_t start = 0;
    if constexpr (Build::kMetrics) start = now_ns();
    bool copied = false;
    if constexpr (Build::kFaults) {
      if (const int64_t ballast = ballast_.bytes.load(); ballast > 0) {
        if (const int r = backend_->issue(ballast_.dst.load(), ballast_.src.load(), ballast)) return r;
        copied = true;
      }
    }
    if ((job.cpu_mask != 0 || job.late_cpu > 0) && cpu_[job.group].load(std::memory_order_relaxed) == nullptr)
      fail_stop("copy job " + std::to_string(job.gen) + " carries CPU lanes with no CPU expert engine");
    int64_t bytes = 0;
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      if ((job.cpu_mask >> lane.lane & 1u) != 0) continue;
      copied = true;
      for (const CopyEntry& entry : table.dma) {
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

  /// Records copy_latency_ns and its max. Metrics only, so ProdBuild reads no clock here.
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

  template <Counter K>
  void count(int64_t n = 1) {
    owner_->template copy_count<K>(n);
  }

  void finish_failed(const CopyJob& job, int error_code) {
    owner_->copy_failed(job, error_code);
    finish();
  }

  void finish() {
    finished_.store(finished_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  }

  std::unique_ptr<CopyBackend> backend_;
  std::array<std::atomic<CpuExpertEngine*>, Wire::kNodes> cpu_{};
  std::vector<Table> tables_;
  Owner* owner_;
  std::string prefix_;
  std::string thread_name_;
  std::vector<int> cpus_;
  std::thread thread_;
  std::promise<std::string> started_;  // the thread's init result, for start()
  std::vector<std::unique_ptr<SpscRing<CopyJob, kCopyRing>>> jobs_;  // one per group
  std::array<std::atomic<uint64_t>, Wire::kNodes> submitted_{};      // each written by its group's thread only
  std::array<Assembly, Wire::kDemandRecords> assembly_{};            // copy thread only
  std::atomic<uint64_t> stall_{0};     // written by the copy thread, read by the watchdog
  std::atomic<uint64_t> finished_{0};  // written by the copy thread only
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> drain_deadline_{0};  // written by stop() before its release of stop_
  int broken_ = 0;                          // copy thread only: the first backend error
  [[no_unique_address]] std::conditional_t<Build::kFaults, Ballast, NoBallast> ballast_;
};

}  // namespace expert_stream
}  // namespace sglang
