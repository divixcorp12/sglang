// The CPU expert engine: computes a record's CPU lanes on a host thread instead of copying their rows to the device.
//
// A CPU lane is a RAM-tier expert the device typed with kLeaseTagCpu. The engine runs it from its host slot through a
// format's C ABI forward (CpuExpertForward) and writes the result into the row's pinned output, which the device folds
// into its fused MoE.
//
//   CpuExpertForward   the format's kernel, as a C ABI
//   CpuExpertKeepWarm  the format's idle loop, as a C ABI
//   CpuJob             one forward: up to Wire::kLanes lanes of one row, writing one output part
//   CpuExpertConfig    the pinned input/output tables, the thread count and the cores
//   CpuExpertEngine    the thread, its job ring and its done word
//
// The tier's owner (the service thread) is the engine's only client. It submits a record's CPU hits before the record's
// copy job, and each CPU miss once its row has landed. The copy thread only reads done(): it treats a copy job as
// complete once every CPU job of the record is, so CopyDone and the copy wait's gate keep their single publisher. A
// failed forward fails stop.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include "../lease_layout.h"
#include "cpu_experts_abi.h"
#include "reader_base.h"
#include "spsc_ring.h"
#include "tier_protocol.h"
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <vector>

namespace sglang::expert_stream {

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
              "the keep-warm reads kick_ as a plain uint32_t");

// A format's CPU expert kernel as a C ABI: the native half of CpuExpertQuantTrait
// (python/sglang/srt/layers/moe/cpu_experts/pool.py). `call` is cpu_experts_abi.h's contract; the engine passes
// one row (rows 1) whose slots index the layer's pinned host tier. Returns 0 on success. Called from the CPU expert
// thread only.
using CpuExpertForward = int (*)(const SglangCpuExpertsForward* call);

// A format's keep-warm, as a C ABI: runs register-only work of the kernel's vector width on `threads` workers, the
// calling thread counted as one, until *word != seen or CLOCK_MONOTONIC reaches deadline_ns, so the cores keep the
// kernel's frequency license through an idle gap instead of ramping back on the next job. Returns 0 on success.
// Called from the CPU expert thread only, between forwards.
using CpuExpertKeepWarm = int (*)(int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns);

// One forward over up to Wire::kLanes lanes of one row.
//
// A record produces at most one job for its CPU hits (output part 0) and one job per batch of CPU misses that landed
// together (part 1). Every miss job after the first adds into the part, in landing order, so the part's fp32 sum order
// varies from run to run.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;         // the output part: 0 the CPU hits' partial sum, 1 the CPU misses'
  bool accumulate = false;  // add into the part rather than overwrite it
  uint32_t seq = 0;         // from claim(); done() compares against it
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
};

// The pinned tables the engine reads and writes, and how it runs. Validated by the CpuExpertEngine constructor.
struct CpuExpertConfig {
  CpuExpertForward forward = nullptr;
  int64_t rows = 0;  // streamed rows; each gets its layer handle later (set_layer), until then the CPU skips it
  const uint8_t* x_base = nullptr;  // row r's input at x_base + r * x_stride, written by the post kernel
  int64_t x_stride = 0;
  uint8_t* out_base = nullptr;  // row r's part p fp32 output at out_base + r * out_stride + p * out_part_stride
  int64_t out_stride = 0;
  int64_t out_part_stride = 0;  // 0: one part, so CPU misses are refused (RamTier::serve_record)
  int64_t hidden = 0;
  int threads = 1;
  std::vector<int> cores;  // worker 0 uses the first CPU; the kernel pins each helper to its assigned CPU
  int64_t spin_ns = 50'000'000;
  CpuExpertKeepWarm keep_warm = nullptr;  // run while idle for keep_warm_ns after each job; nullptr or 0 ns: off
  int64_t keep_warm_ns = 0;
};

// The CPU expert thread, its job ring and its done word.
//
// One thread consumes a lock-free SPSC ring fed by the tier's owner. Jobs run in sequence order, and done_ only
// increases, so done(seq) is a single compare. The owner claims sequences with claim() and submits jobs with submit();
// any thread may read done(). The idle protocol is the copy engine's (copy_engine.h): spin for spin_ns, then a futex
// sleep whose wake the submitter makes only when this thread has gone to sleep (a Dekker pair of seq_cst fences
// between submit() and sleep_until_submit()). With a keep-warm, the thread first hands its workers to it for
// keep_warm_ns after each job; every submit and stop() bumps kick_, which ends it.
//
// The done_ store is a release after the job's output rows are written, and done() is an acquire: a CopyDone published
// after observing a job done orders the output before the device's read of it.
class CpuExpertEngine {
 public:
  static constexpr size_t kRing =
      std::bit_ceil(static_cast<size_t>(wire::Wire::kDemandRecords) * (wire::Wire::kLanes + 1));
  // A record has at most one CPU-hit job and one job per CPU miss, so the ring holds every job that can be outstanding.
  static_assert(
      kRing >= wire::Wire::kDemandRecords * (wire::Wire::kLanes + 1),
      "the ring holds every job that can be outstanding");

  CpuExpertEngine(CpuExpertConfig config, std::string prefix, std::string thread_name)
      : config_(std::move(config)), prefix_(std::move(prefix)), thread_name_(thread_name.substr(0, 15)) {
    if (config_.forward == nullptr) throw std::runtime_error(prefix_ + "no CPU expert forward");
    if (config_.x_base == nullptr || config_.out_base == nullptr || config_.x_stride <= 0 || config_.out_stride <= 0)
      throw std::runtime_error(prefix_ + "the CPU expert input and output rows are required");
    if (config_.hidden <= 0 || config_.out_stride < config_.hidden * static_cast<int64_t>(sizeof(float)))
      throw std::runtime_error(prefix_ + "the CPU expert output rows are smaller than the hidden size");
    if (config_.threads < 1) throw std::runtime_error(prefix_ + "the CPU expert pool needs at least one thread");
    if (config_.rows < 1) throw std::runtime_error(prefix_ + "the CPU expert engine needs its row count");
    handles_ = std::make_unique<std::atomic<int64_t>[]>(static_cast<size_t>(config_.rows));
    for (int64_t r = 0; r < config_.rows; ++r)
      handles_[r].store(-1, std::memory_order_relaxed);
  }

  // Registers the trait's layer handle for `row`, once per row, from any thread. Throws on a bad row or a second call.
  // The release pairs with eligible()'s acquire on the service thread, so a grant that sends the row to the CPU is
  // ordered after the registration it relies on.
  void set_layer(int64_t row, int64_t handle) {
    if (row < 0 || row >= config_.rows || handle < 0) throw std::runtime_error(prefix_ + "bad CPU expert layer");
    int64_t unset = -1;
    if (!handles_[row].compare_exchange_strong(unset, handle, std::memory_order_acq_rel))
      throw std::runtime_error(prefix_ + "a CPU expert layer is registered once");
  }

  ~CpuExpertEngine() {
    stop();
  }

  // Starts the thread and waits until it has pinned itself to its core. Throws, after joining it, if it cannot.
  void start() {
    spin_iters_ = idle_budget(config_.spin_ns);
    thread_ = std::thread([this] { run(); });
    std::unique_lock<std::mutex> lock(start_mutex_);
    ready_cv_.wait(lock, [this] { return started_; });
    if (!init_error_.empty()) {
      lock.unlock();
      stop();
      throw std::runtime_error(prefix_ + init_error_);
    }
  }

  // Output parts per row: 2 when the rows hold a CPU-hit and a CPU-miss partial sum each.
  int parts() const {
    return config_.out_part_stride > 0 ? 2 : 1;
  }

  const std::vector<int>& cores() const {
    return config_.cores;
  }

  // True for a row the device may send to the CPU: inside the tables and registered.
  bool eligible(int64_t row) const {
    return row >= 0 && row < config_.rows && handles_[row].load(std::memory_order_acquire) >= 0;
  }

  // Tier owner only. Returns the first of `n` consecutive sequences for jobs it will submit, in order. A sequence left
  // unsubmitted is skipped: done() of a later one covers it, so a record can claim one per possible job.
  uint32_t claim(int n) {
    const uint32_t first = claimed_ + 1;
    claimed_ += static_cast<uint32_t>(n);
    return first;
  }

  // Tier owner only, with job.seq claimed and above every earlier submit's. Returns false when the ring is full, which
  // kRing's bound rules out: the caller fails stop.
  bool submit(const CpuJob& job) {
    if (!jobs_.push(job)) return false;
    kick_.fetch_add(1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
    return true;
  }

  // True once the job with sequence `seq`, or a later one, has finished. Any thread.
  bool done(uint32_t seq) const {
    return static_cast<int32_t>(done_.load(std::memory_order_acquire) - seq) >= 0;
  }

  // Metrics for the service's counters: jobs finished, lanes computed, and forward time in ns.
  int64_t jobs() const {
    return jobs_done_.load(std::memory_order_relaxed);
  }
  int64_t lanes() const {
    return lanes_done_.load(std::memory_order_relaxed);
  }
  int64_t compute_ns() const {
    return compute_ns_.load(std::memory_order_relaxed);
  }

  // Stops and joins the thread; idempotent.
  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    kick_.fetch_add(1, std::memory_order_release);
    wake_.fetch_add(1, std::memory_order_release);
    futex_wake(&wake_);
    thread_.join();
  }

 private:
  // The thread body: pins to the caller core, reports readiness to start(), then pops and runs jobs.
  void run() {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    std::string error;
    if (!config_.cores.empty()) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(config_.cores.front(), &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) error = "cannot pin the CPU expert thread to its caller core";
    }
    {
      std::lock_guard<std::mutex> guard(start_mutex_);
      init_error_ = error;
      started_ = true;
    }
    ready_cv_.notify_all();
    if (!error.empty()) return;
    uint64_t idle = 0;
    const bool warm = config_.keep_warm != nullptr && config_.keep_warm_ns > 0;
    int64_t warm_until = 0;  // the keep-warm window after the last job; 0 before the first
    while (!stop_.load(std::memory_order_acquire)) {
      // Read before the pop: a submit after an empty pop bumps kick_ past `kick` and so ends the keep-warm.
      const uint32_t kick = kick_.load(std::memory_order_acquire);
      CpuJob job;
      if (!jobs_.pop(&job)) {
        if (warm && now_ns() < warm_until) {
          const int result = config_.keep_warm(
              config_.threads, reinterpret_cast<const uint32_t*>(&kick_), kick, warm_until);
          if (result != 0) fail_stop(prefix_ + "CPU expert keep-warm failed (" + std::to_string(result) + ")");
          continue;
        }
        if (++idle < spin_iters_) {
          _mm_pause();
        } else {
          sleep_until_submit();
          idle = 0;
        }
        continue;
      }
      idle = 0;
      const int64_t start = now_ns();
      SglangCpuExpertsForward call{};
      call.abi_version = SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION;
      call.rows = 1;
      call.layer = handles_[job.row].load(std::memory_order_acquire);
      call.x = config_.x_base + job.row * config_.x_stride;
      call.slots = job.slots;
      call.weights = job.weights;
      call.out =
          reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
      call.k = job.k;
      call.threads = config_.threads;
      call.accumulate = job.accumulate ? 1 : 0;
      const int result = config_.forward(&call);
      if (result != 0)
        fail_stop(
            prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed (" + std::to_string(result) +
            ")");
      compute_ns_.fetch_add(now_ns() - start, std::memory_order_relaxed);
      jobs_done_.fetch_add(1, std::memory_order_relaxed);
      lanes_done_.fetch_add(job.k, std::memory_order_relaxed);
      done_.store(job.seq, std::memory_order_release);  // after the output rows: done() is an acquire
      if (warm) warm_until = now_ns() + config_.keep_warm_ns;
    }
  }

  // Sleeps on wake_ until a submit or stop() bumps it, for at most 1 ms.
  void sleep_until_submit() {
    const uint32_t seen = wake_.load(std::memory_order_acquire);
    sleeping_.store(true, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with submit()'s push and sleeping_ load
    if (jobs_.empty() && !stop_.load(std::memory_order_acquire)) futex_wait(&wake_, seen, 1'000'000);  // 1 ms cap
    sleeping_.store(false, std::memory_order_relaxed);
  }

  CpuExpertConfig config_;
  std::unique_ptr<std::atomic<int64_t>[]> handles_;
  std::string prefix_;
  std::string thread_name_;
  std::thread thread_;
  SpscRing<CpuJob, kRing> jobs_;
  uint32_t claimed_ = 0;  // the tier owner's, the last sequence claimed
  std::atomic<uint32_t> done_{0};
  std::atomic<int64_t> jobs_done_{0};
  std::atomic<int64_t> lanes_done_{0};
  std::atomic<int64_t> compute_ns_{0};
  std::atomic<bool> sleeping_{false};
  std::atomic<uint32_t> wake_{0};
  std::atomic<uint32_t> kick_{0};  // bumped by every submit and by stop(); the keep-warm returns once it moves
  std::atomic<bool> stop_{false};
  uint64_t spin_iters_ = 1;
  std::mutex start_mutex_;
  std::condition_variable ready_cv_;
  bool started_ = false;
  std::string init_error_;
};

}  // namespace sglang::expert_stream
