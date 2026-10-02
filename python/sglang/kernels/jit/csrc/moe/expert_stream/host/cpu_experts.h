// The CPU expert thread (plan docs/superpowers/plans/2026-09-29-dsv41-cpu-experts.md, section 4 and "Step B"): it
// computes a request's CPU lanes, RAM-tier experts the grant published with tag kLeaseTagCpu, from their leased host
// slots through a format's C ABI forward, into the row's pinned output that the device folds into its fused MoE.
//
// The tier's owner (the service thread) is its only client: it submits a record's CPU hits before the record's copy
// job, and each CPU miss once its row landed. The copy thread only reads done() and treats the copy job as complete
// once every CPU job of the record is. So CopyDone and the copy wait's gate keep their one publisher
// (LEASE_PROTOCOL.md, "Copy engine"). A failed forward fails stop here.
#pragma once

#include "../lease_layout.h"
#include "reader_base.h"
#include "spsc_ring.h"
#include "tier_protocol.h"
#include <atomic>
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

// A format's CPU expert kernel as a C ABI, the native half of CpuExpertQuantTrait (python/sglang/srt/layers/moe/
// cpu_experts/pool.py): overwrite out[0, hidden) with sum over i < k of weights[i] * expert(slots[i])(x), or add that
// sum to it when `accumulate` is nonzero, where slots[i] indexes the layer's pinned host tier and x is one input row in
// the trait's x format. `layer` is the handle the trait registered for that layer. Returns 0 on success. Called from
// the CPU expert thread only, with `threads` the kernel's worker count, the calling thread counted as one of them.
using CpuExpertForward = int (*)(
    int64_t layer,
    const void* x,
    const int32_t* slots,
    const float* weights,
    int32_t k,
    float* out,
    int32_t threads,
    int32_t accumulate);

struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;  // the output part: 0 the CPU hits' partial sum, 1 the CPU misses'
  // Add into the part rather than overwrite it: a record's later CPU-miss jobs, in landing order, so the part's fp32
  // sum order varies from run to run.
  bool accumulate = false;
  uint32_t seq = 0;  // from claim(); done() compares against it
  int32_t k = 0;
  int32_t slots[wire::kLeaseLanes] = {};
  float weights[wire::kLeaseLanes] = {};
};

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
};

// One thread, a lock-free SPSC job ring from the tier's owner, and a monotonically increasing done word. Jobs run in
// sequence order, so done() is a single compare. The idle protocol is the copy engine's (copy_engine.h): spin for
// spin_ns, then a futex sleep whose wake the submitter makes only when this thread has gone to sleep.
class CpuExpertEngine {
 public:
  static constexpr size_t kRing = 256;
  // A record has at most one CPU-hit job and one job per CPU miss.
  static_assert(
      kRing >= wire::kDemandRecords * (wire::kLeaseLanes + 1), "the ring holds every job that can be outstanding");

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

  // Any thread, once per row: the trait's layer handle for `row`. The release pairs with eligible()'s acquire on the
  // service thread, so a grant that sends the row to the CPU is ordered after the registration it relies on.
  void set_layer(int64_t row, int64_t handle) {
    if (row < 0 || row >= config_.rows || handle < 0) throw std::runtime_error(prefix_ + "bad CPU expert layer");
    int64_t unset = -1;
    if (!handles_[row].compare_exchange_strong(unset, handle, std::memory_order_acq_rel))
      throw std::runtime_error(prefix_ + "a CPU expert layer is registered once");
  }

  ~CpuExpertEngine() {
    stop();
  }

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

  // A row the device may send to the CPU: registered, and inside the tables.
  bool eligible(int64_t row) const {
    return row >= 0 && row < config_.rows && handles_[row].load(std::memory_order_acquire) >= 0;
  }

  // The tier's owner only. The first of `n` consecutive sequences for jobs it will submit, in order. A sequence left
  // unsubmitted is skipped: done() of a later one covers it, so a record can claim one per possible job.
  uint32_t claim(int n) {
    const uint32_t first = claimed_ + 1;
    claimed_ += static_cast<uint32_t>(n);
    return first;
  }

  // The tier's owner only, with job.seq claimed and above every earlier submit's. False when the ring is full (an
  // internal error under kRing's bound: the caller fails stop).
  bool submit(const CpuJob& job) {
    if (!jobs_.push(job)) return false;
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
    return true;
  }

  // Any thread. Acquire: a job seen done has its output written, so a CopyDone published after this observation orders
  // the output before the device's read of it.
  bool done(uint32_t seq) const {
    return static_cast<int32_t>(done_.load(std::memory_order_acquire) - seq) >= 0;
  }

  // Metrics, for the service's counters: jobs finished, lanes computed, and forward time in ns.
  int64_t jobs() const {
    return jobs_done_.load(std::memory_order_relaxed);
  }
  int64_t lanes() const {
    return lanes_done_.load(std::memory_order_relaxed);
  }
  int64_t compute_ns() const {
    return compute_ns_.load(std::memory_order_relaxed);
  }

  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    wake_.fetch_add(1, std::memory_order_release);
    futex_wake(&wake_);
    thread_.join();
  }

 private:
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
    while (!stop_.load(std::memory_order_acquire)) {
      CpuJob job;
      if (!jobs_.pop(&job)) {
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
      const int result = config_.forward(
          handles_[job.row].load(std::memory_order_acquire),
          config_.x_base + job.row * config_.x_stride,
          job.slots,
          job.weights,
          job.k,
          reinterpret_cast<float*>(
              config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride),
          config_.threads,
          job.accumulate ? 1 : 0);
      if (result != 0)
        fail_stop(
            prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed (" + std::to_string(result) +
            ")");
      compute_ns_.fetch_add(now_ns() - start, std::memory_order_relaxed);
      jobs_done_.fetch_add(1, std::memory_order_relaxed);
      lanes_done_.fetch_add(job.k, std::memory_order_relaxed);
      done_.store(job.seq, std::memory_order_release);  // after the output rows: done() is an acquire
    }
  }

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
  uint32_t claimed_ = 0;  // the tier owner's
  std::atomic<uint32_t> done_{0};
  std::atomic<int64_t> jobs_done_{0};
  std::atomic<int64_t> lanes_done_{0};
  std::atomic<int64_t> compute_ns_{0};
  std::atomic<bool> sleeping_{false};
  std::atomic<uint32_t> wake_{0};
  std::atomic<bool> stop_{false};
  uint64_t spin_iters_ = 1;
  std::mutex start_mutex_;
  std::condition_variable ready_cv_;
  bool started_ = false;
  std::string init_error_;
};

}  // namespace sglang::expert_stream
