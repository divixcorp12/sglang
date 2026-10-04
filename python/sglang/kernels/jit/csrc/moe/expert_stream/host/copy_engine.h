// The copy engine: moves host-resident expert rows into device slots with DMA, off the decode stream.
//
// The RAM-miss service turns each lease record's host lanes into one CopyJob and submits it to CopyEngine, whose copy
// thread issues the job's copies through a CopyBackend, marks the point after them, and reports completion back to
// the service (which publishes CopyDone and opens the decode stream's copy gate). Lanes computed by the CPU expert
// engine instead of copied ride in the same job, so one completion covers every host lane of a record.
//
//   CopyLane / CopyJob   one record's host lanes, service -> copy thread
//   CopyEntry            one (source slab, destination tensor, bytes) entry of a row's copy table
//   CopyBackend          the copy primitive: CudaCopyBackend in production, HostCopyBackend in CPU tests
//   CopyEngine           the copy thread, its job ring and its in-flight FIFO
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include <algorithm>
#include <array>
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

// One host lane of a record: copy host slot `host_slot` into device slot `dst_slot`, or, for a CPU lane, compute it.
struct CopyLane {
  int32_t lane = 0;
  int32_t host_slot = 0;
  int32_t dst_slot = 0;
  float weight = 0.0f;  // routing weight; read only for a CPU lane
};

// One record's host lanes, submitted by the service to the copy thread when the record is served.
//
// A job carries three kinds of lane:
//   - copy-engine hits: copied by DMA;
//   - CPU hits (`cpu_mask`, Wire::kKindHitCpu): computed by the CPU expert engine as one job, sequence `cpu_seq`;
//   - CPU misses (`late_cpu`, Wire::kKindMissCpu): computed as their rows land; `late_seq` is the last such CPU job.
// The job completes when its copies and every CPU job of the record are done, so a single CopyDone covers every host
// lane. `cpu_seq` is read only when `cpu_mask` is set, `late_seq` only when `late_cpu` is. A record's host lanes arrive
// as one job per NUMA group that has any (`groups`); CopyDone covers every host lane once every part is done.
struct CopyJob {
  uint64_t gen = 0;
  int64_t idx = 0;
  int64_t row = 0;
  uint32_t mask = 0;
  int count = 0;
  CopyLane lanes[Wire::kLanes];
  int64_t submit_ns = 0;
  int64_t token = -1;  // backend marker after the last copy; kNoToken if none
  bool sm = false;     // the row has SM entries: the copy wait copies those, so the DMA skips them
  uint32_t cpu_mask = 0;
  uint32_t cpu_seq = 0;
  int late_cpu = 0;
  uint32_t late_seq = 0;
  int group = 0;        // the NUMA group whose lanes these are: its CPU engine's done() completes cpu_seq/late_seq
  uint32_t groups = 1;  // bit g: group g sends a part of this record; CopyDone waits for every one
};

// CopyJob::token of a job that issued no copy (every lane on the CPU): there is nothing to query.
constexpr int64_t kNoToken = -1;

// One entry of a row's copy table: a source slab, a destination tensor and the bytes of one expert row in each.
// A lane's host slot indexes the source and its device slot the destination.
struct CopyEntry {
  uint64_t src = 0;
  uint64_t dst = 0;
  int64_t bytes = 0;
  bool sm = false;  // copied by the copy wait's SM reads, not the DMA (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES)
};

// The copy primitive the copy thread drives: issue copies in order, mark the point after them, and ask whether a
// mark has completed. Every call is made on the copy thread.
class CopyBackend {
 public:
  static constexpr int kDone = 0;
  static constexpr int kPending = 1;
  virtual ~CopyBackend() = default;
  // Prepares the backend on the copy thread. Returns an empty string on success, else the reason it refused.
  virtual std::string init() = 0;
  // Queues one copy after every earlier one. Returns 0 or an error code.
  virtual int issue(uint64_t dst, uint64_t src, int64_t bytes) = 0;
  // Marks the point after every copy issued so far and stores its token. Returns 0 or an error code.
  virtual int mark(int64_t* token) = 0;
  // Returns kDone, kPending, or an error code (negative for the host backend).
  virtual int query(int64_t token) = 0;
  // Releases the backend's resources if `idle`; otherwise keeps them, since a copy may still be in flight.
  virtual void shutdown(bool idle) = 0;
};
static_assert(CompletionWord::kDone == CopyBackend::kDone && CompletionWord::kPending == CopyBackend::kPending);

// The production backend: CUDA driver copies on a private stream, completion tracked by a host-mapped word.
//
// Per job it issues one cuMemcpyAsync per copy, then one cuStreamWriteValue32_v2 of the job's sequence number into
// the completion word (completion_word.h). query() is a plain acquire load of that word, plus one cuStreamQuery per
// CompletionWord budget while a job stays pending. Nothing waits on another stream, a graph launch or the device: no
// synchronize, no module load and no allocation per job.
//
// Design choices:
//   - The driver is resolved from libcuda.so.1 at init(), so the module builds and loads without a CUDA toolkit.
//   - The v2 stream memory ops, never the plain names: the plain names are the v1 API, gated by
//     NVreg_EnableStreamMemOPs (its device attribute reads 0 on the reference machine while the v2 ops work;
//     analysis/dsv41-drive/hotpath/results.md section 9b).
//   - The stream has the greatest priority, for a hardware queue of its own. Streams share
//     CUDA_DEVICE_MAX_CONNECTIONS queues (the server sets 8) and a queue runs in order, so a copy queued behind a
//     stream whose head waits on the decode graph could not start until the copy wait gave up: a deadlock only its
//     timeout breaks. Measured with analysis/dsv41-drive/copy-engine/hol_probe.py: a fresh default-priority stream
//     waited 596 ms behind 8 blocked ones; greatest-priority copies never waited, with up to 64 blocked.
//   - No fallback. The production recipe needs the copy engine, so init() refuses when the write-value op is missing,
//     errors or never reaches the word, and when the wait-value op (which the decode stream's copy wait uses on
//     host-mapped memory) is missing, errors, does not hold its stream, or is not released by a host store.
//
// The class is used only on the copy thread.
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
    get(cu_write_value32_, "cuStreamWriteValue32_v2");  // v2 only: see the class comment
    get(cu_wait_value32_, "cuStreamWaitValue32_v2");
    if (!missing.empty()) return "libcuda.so.1 lacks " + missing + ", which the copy engine needs";
    if (int r = cu_init_(0)) return "cuInit failed: " + std::to_string(r);
    if (int r = cu_device_get_(&cu_device_, device_)) return "cuDeviceGet failed: " + std::to_string(r);
    // PyTorch's primary context, so the slabs' registrations and the destination tensors are valid here.
    if (int r = cu_primary_retain_(&context_, cu_device_))
      return "cuDevicePrimaryCtxRetain failed: " + std::to_string(r);
    retained_ = true;
    if (int r = cu_ctx_set_current_(context_)) return "cuCtxSetCurrent failed: " + std::to_string(r);
    constexpr unsigned kNonBlocking = 1;  // CU_STREAM_NON_BLOCKING: no implicit sync with the legacy stream
    // Greatest priority, for a hardware queue of its own: see the class comment.
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

  // The token is the job's sequence number, written into the completion word after the job's copies.
  int mark(int64_t* token) override {
    const uint32_t seq = ++marked_;
    *token = seq;
    return cu_write_value32_(stream_, word_dev_, seq, 0);  // default flags: after the prior copies, fenced
  }

  int query(int64_t token) override {
    return word_.poll(static_cast<uint32_t>(token), [this] { return cu_stream_query_(stream_); });
  }

  // Logs the completion word's final value, which is the exact number of jobs the stream completed. Frees the stream,
  // word and context only if `idle`: after an error, or with copies in flight, a copy may still read a slab and the
  // stream may still write the word.
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
  // Allocates the host-mapped completion word and waits (up to about a second) for a first write through the stream
  // to land in it. Returns the reason if the driver rejects the op or the write never arrives. Start-up only.
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

  // Tries the copy wait's primitive once on this stream: a wait on a host-mapped word must hold the stream until a
  // host store satisfies it, then complete. The decode stream's gate uses the same op on the same kind of memory, so
  // a driver without it is refused here instead of hanging the first decode. Start-up only.
  std::string probe_stream_wait() {
    constexpr unsigned kDeviceMap = 2;  // CU_MEMHOSTALLOC_DEVICEMAP
    constexpr unsigned kGeq = 0;        // CU_STREAM_WAIT_VALUE_GEQ
    void* host = nullptr;
    if (int r = cu_mem_host_alloc_(&host, sizeof(uint32_t), kDeviceMap))
      return "cuMemHostAlloc of the stream-wait probe word failed: " + std::to_string(r);
    auto* word = static_cast<uint32_t*>(host);
    // The production encoding (lease_layout.h): a closed gate word (bit 31 set) that an open word releases.
    const uint32_t closed = Wire::kLeaseGateClosed;
    const uint32_t open_word = Wire::kLeaseGateOpen;
    __atomic_store_n(word, closed, __ATOMIC_RELEASE);
    uint64_t device_word = 0;
    std::string error;
    if (int r = cu_mem_host_device_ptr_(&device_word, host, 0)) {
      error = "cuMemHostGetDevicePointer of the stream-wait probe word failed: " + std::to_string(r);
    } else if (int r = cu_wait_value32_(stream_, device_word, Wire::kLeaseGateOpen, kGeq)) {
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
      __atomic_store_n(
          word, open_word, __ATOMIC_RELEASE);  // releases the wait; also on an error, so nothing stays queued
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

// Capacity of the copy engine's two queues: the service's job ring and the copy thread's in-flight FIFO. The device
// waits on every record with host lanes, so at most one job per demand record is outstanding and neither queue fills.
constexpr size_t kCopyRing = 32;
static_assert(kCopyRing > Wire::kDemandRecords, "the copy engine's queues hold every job that can be outstanding");

// A CPU-only test backend: "copies" between host buffers, completed only when the test releases them.
//
// A mark's copies land when it is released, in query(), in mark order (one stream). A CopyDone published before its
// release therefore shows up as bytes that are not there yet, which is what the ordering tests look for.
//
// issue, mark and query run on the copy thread; release, fail and marked on a test thread. The two sides share only
// atomics: no lock, and no allocation after construction, so the backend cannot hide a race the CUDA one would show.
class HostCopyBackend : public CopyBackend {
 public:
  // A mark's slot is reused kMarks marks later. A mark is open or in flight only while its job is (kCopyRing at most),
  // so the slot's earlier mark completed long before.
  static constexpr int kMarks = 2 * static_cast<int>(kCopyRing);
  // One mark's copies: every lane's copy of every layout name (row_layout.h caps a layout at 32), plus a ballast copy.
  static constexpr int kEntries = Wire::kLanes * 32 + 1;
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

  // Lets `marks` more marks complete, or every mark from now on if negative. The budget is standing, so a release
  // made before the copy thread marks still counts.
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

// The copy thread: issues each job's copies, marks the point after them, polls marks oldest first and hands each
// completed job back to its owner.
//
// Template parameters:
//   - `Build`: the tier's build policy (build_policy.h); metrics and fault injection compile in only where it says.
//   - `Owner`: the RamTier, which befriends this class and provides, all called on the copy thread:
//       void copy_completed(const CopyJob&)   copies and CPU jobs done: publish CopyDone and open the gate;
//       void copy_failed(const CopyJob&, int)  completion cannot be established: fail stop (also called on the
//                                              submitting thread for a ring overflow);
//       owner->template copy_count<K>(n)       the copy thread's counters.
//
// Errors: after the first backend error nothing the backend issued can be assumed complete, so that job and every job
// behind it go to copy_failed.
//
// Service to copy thread: an SPSC job ring. submit() takes no lock and never blocks. The copy thread polls the ring
// and, after spin_ns with nothing popped or in flight, sleeps on a futex; submit() makes the wake syscall only when
// the copy thread is asleep. No wake is lost:
//   - copy thread (idle): seen = wake_; sleeping_ = true; fence(seq_cst);
//                         if the ring is empty and no stop was asked for, futex_wait(&wake_, seen).
//   - submit:             push (release); fence(seq_cst); if sleeping_, ++wake_ and futex_wake.
// The two seq_cst fences are totally ordered. If submit's fence is first, the copy thread's ring check after its own
// fence sees the push and it does not wait. If the copy thread's fence is first, submit sees sleeping_ and wakes it:
// before futex_wait (wake_ != seen, so the kernel returns at once; it compares the word under its own lock) or after
// (the wake ends the wait). The wait's 1 ms cap is a backstop, not part of the argument.
template <class Build, class Owner>
class CopyEngine {
  static_assert(BuildPolicy<Build>);

 public:
  // `thread_name` is the copy thread's pthread name (e.g. Layout::kName + "-copy-eng"), truncated to 15 bytes
  // (pthread_setname_np's limit). `cpus` is the thread's affinity; empty inherits the starter's.
  CopyEngine(
      std::unique_ptr<CopyBackend> backend,
      int64_t rows,
      int groups,
      int64_t spin_ns,
      Owner* owner,
      std::string prefix,
      std::string thread_name,
      std::vector<int> cpus)
      : backend_(std::move(backend)),
        tables_(static_cast<size_t>(rows)),
        spin_ns_(spin_ns),
        owner_(owner),
        prefix_(std::move(prefix)),
        thread_name_(thread_name.substr(0, 15)),
        cpus_(std::move(cpus)) {
    for (int g = 0; g < groups; ++g) jobs_.push_back(std::make_unique<SpscRing<CopyJob, kCopyRing>>());
  }

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

  // The entries of row `row` that the DMA copies (SM entries belong to the copy wait). Used by the split calibration,
  // which must time exactly the bytes a decode moves. Throws if the row has no table.
  std::vector<CopyEntry> dma_entries(int64_t row) const {
    if (row < 0 || row >= static_cast<int64_t>(tables_.size()) || !tables_[row].ready.load(std::memory_order_acquire))
      throw std::runtime_error("no copy table for row " + std::to_string(row));
    std::vector<CopyEntry> entries;
    for (const CopyEntry& entry : tables_[row].entries)
      if (!entry.sm) entries.push_back(entry);
    return entries;
  }

  // Hands a job to the copy thread. Called only by the tier's owner (the service thread, or the caller of pump()).
  // Takes no lock and never blocks; makes a syscall only to wake a sleeping copy thread. Each group has its own ring
  // and pushes only to it. At most Wire::kDemandRecords jobs per group are outstanding and a ring holds kCopyRing, so
  // a full ring is an internal error and fails stop.
  void submit(int group, const CopyJob& job) {
    if (!jobs_[group]->push(job)) {
      owner_->copy_failed(job, kRingOverflow);
      return;
    }
    submitted_[group].store(submitted_[group].load(std::memory_order_relaxed) + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
  }

  // True when every submitted job has completed or failed.
  bool idle() const {
    uint64_t submitted = 0;
    for (const auto& count : submitted_) submitted += count.load(std::memory_order_acquire);
    return finished_.load(std::memory_order_acquire) == submitted;
  }

  // Polls until idle() or `deadline_ns` (now_ns() clock); returns whether it went idle. For a paused caller or a
  // test, not the hot path.
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

  // Sets the CPU expert engine whose done() completes group `group`'s CPU lanes. The service sets it before its
  // thread starts, so before any record carries a CPU lane.
  void set_cpu(int group, CpuExpertEngine* cpu) {
    cpu_[group].store(cpu, std::memory_order_release);
  }

  // Why CopyDone is not stored, for the watchdog: seq << 32 | group << 8 | reason (kStallCpu: group's CPU job `seq`
  // is not done; kStallNoPart: group's part of the record never came); 0 when nothing waits.
  static constexpr uint32_t kStallCpu = 1;
  static constexpr uint32_t kStallNoPart = 2;
  uint64_t stall() const {
    return stall_.load(std::memory_order_relaxed);
  }

  // Test only (InstrBuild): issues one extra copy of `bytes` ahead of every job's own, so each job completes that much
  // later.
  void set_ballast(uint64_t dst, uint64_t src, int64_t bytes)
    requires(Build::kFaults)
  {
    ballast_.dst.store(dst);
    ballast_.src.store(src);
    ballast_.bytes.store(bytes);
  }

 private:
  static constexpr int kRingOverflow = -1000;
  using Queue = FixedDeque<CopyJob, kCopyRing>;

  struct Table {
    std::vector<CopyEntry> entries;
    bool sm = false;  // an entry is left to the copy wait's SM reads
    int64_t dst_rows = 0;
    std::atomic<bool> ready{false};
  };

  void run() {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    std::string error;
    if (!cpus_.empty()) {
      // ThreadingConfig.copy_cpus: the GPU's node. A thread created later inherits the enabling caller's affinity.
      cpu_set_t set;
      CPU_ZERO(&set);
      for (const int cpu : cpus_)
        CPU_SET(cpu, &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) error = "cannot pin the copy thread to its cores";
    }
    if (error.empty()) error = backend_->init();
    {
      std::lock_guard<std::mutex> guard(start_mutex_);  // the start handshake: setup, not the hot path
      init_error_ = error;
      started_ = true;
    }
    ready_cv_.notify_all();
    if (!error.empty()) return;
    Queue in_flight;
    uint64_t idle = 0;  // empty polls since the last progress, counted against spin_iters_
    while (true) {
      // Read before the ring: stop() comes after the last submit, so a stop seen here has every job in the ring.
      const bool stopping = stop_.load(std::memory_order_acquire);
      bool progressed = false;
      CopyJob job;
      for (auto& ring : jobs_) {
        while (ring->pop(&job)) {
          progressed = true;
          issue_or_fail(job, in_flight);
        }
      }
      // Poll the head job every turn. For CudaCopyBackend that is one acquire load, not a driver call: the
      // cuEventQuery it replaced took a libcuda mutex per call, ~2,300-3,200 times per job
      // (analysis/dsv41-drive/hotpath/results.md section 9a).
      while (!in_flight.empty() && broken_ == 0) {
        const CopyJob& head = in_flight.front();
        const int state = head.token == kNoToken ? CopyBackend::kDone : backend_->query(head.token);
        if (state == CopyBackend::kPending) break;
        if (state != CopyBackend::kDone) {
          broken_ = state;
          break;
        }
        // Set before the service threads started.
        const CpuExpertEngine* cpu = cpu_[head.group].load(std::memory_order_relaxed);
        const uint32_t pending = head.cpu_mask != 0 && !cpu->done(head.cpu_seq) ? head.cpu_seq
                                 : head.late_cpu > 0 && !cpu->done(head.late_seq) ? head.late_seq
                                                                                   : 0;
        if (pending != 0) {
          stall_.store(static_cast<uint64_t>(pending) << 32 | static_cast<uint64_t>(head.group) << 8 | kStallCpu,
                       std::memory_order_relaxed);
          break;
        }
        const CopyJob done = in_flight.front();
        in_flight.pop_front();
        record_latency(done);
        Assembly& record = assembly_[done.idx];
        if (record.gen != done.gen) record = Assembly{done.gen, 0};
        record.got |= 1u << done.group;
        if (record.got == done.groups) {
          owner_->copy_completed(done);
          stall_.store(0, std::memory_order_relaxed);
        } else {
          const int missing = __builtin_ctz(done.groups & ~record.got);
          stall_.store(static_cast<uint64_t>(missing) << 8 | kStallNoPart, std::memory_order_relaxed);
        }
        finish();
        progressed = true;
      }
      if (broken_ != 0) {
        while (!in_flight.empty()) {
          finish_failed(in_flight.front(), broken_);
          in_flight.pop_front();
        }
      }
      // The drain deadline is read only once a stop was asked for (short-circuit): never while serving.
      if (stopping && (in_flight.empty() || now_ns() > drain_deadline())) break;
      if (progressed) {
        idle = 0;
      } else if (!in_flight.empty() || ++idle < spin_iters_) {
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
    if (rings_empty() && !stop_.load(std::memory_order_acquire)) futex_wait(&wake_, seen, 1'000'000);
    sleeping_.store(false, std::memory_order_relaxed);
  }

  bool rings_empty() const {
    return std::all_of(jobs_.begin(), jobs_.end(), [](const auto& ring) { return ring->empty(); });
  }

  // A full queue is impossible under the outstanding-job bound; if it ever happens, fail stop.
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
    job.sm = table.sm;
    if ((job.cpu_mask != 0 || job.late_cpu > 0) && cpu_[job.group].load(std::memory_order_relaxed) == nullptr)
      fail_stop("copy job " + std::to_string(job.gen) + " carries CPU lanes with no CPU expert engine");
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

  // Records copy_latency_ns and its max. Metrics only, so ProdBuild reads no clock here.
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

  // Adds to a copy-thread counter through the owner: a core counter goes to its copy-thread block, a metric to the
  // shared stats (InstrBuild only).
  template <Counter K>
  void count(int64_t n = 1) {
    owner_->template copy_count<K>(n);
  }

  void finish_failed(const CopyJob& job, int error_code) {
    owner_->copy_failed(job, error_code);
    finish();
  }

  void finish() {  // copy thread
    finished_.store(finished_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  }

  std::unique_ptr<CopyBackend> backend_;
  std::array<std::atomic<CpuExpertEngine*>, Wire::kNodes> cpu_{};
  std::vector<Table> tables_;
  int64_t spin_ns_;
  uint64_t spin_iters_ = 1;  // idle polls before the futex sleep: idle_budget(spin_ns_), set in start()
  Owner* owner_;
  std::string prefix_;
  std::string thread_name_;
  std::vector<int> cpus_;  // the copy thread's affinity; empty inherits the starter's
  std::thread thread_;
  std::vector<std::unique_ptr<SpscRing<CopyJob, kCopyRing>>> jobs_;  // one per group: each group pushes its own
  std::array<std::atomic<uint64_t>, Wire::kNodes> submitted_{};      // each written by its group's thread only
  // The parts of each ring index's record seen so far: the copy thread's only.
  struct Assembly {
    uint64_t gen = 0;
    uint32_t got = 0;
  };
  std::array<Assembly, Wire::kDemandRecords> assembly_{};
  std::atomic<uint64_t> stall_{0};  // the copy thread's, read relaxed by the watchdog
  std::atomic<uint64_t> finished_{0};   // written by the copy thread only
  std::atomic<bool> sleeping_{false};   // the copy thread is (about to be) in futex_wait
  std::atomic<uint32_t> wake_{0};       // the futex word: bumped by every wake
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> drain_deadline_{0};
  std::mutex start_mutex_;  // start()'s handshake only: started_ and init_error_
  std::condition_variable ready_cv_;
  bool started_ = false;
  std::string init_error_;
  int broken_ = 0;  // copy thread only: the first backend error
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
