// The CPU expert thread (plan docs/superpowers/plans/2026-09-29-dsv41-cpu-experts.md, section 4 and "Step B"): it
// computes a request's CPU lanes, RAM-tier experts the grant published with tag kLeaseTagCpu, from their leased host
// slots through a format's C ABI forward, into the row's pinned output that the device folds into its fused MoE.
//
// The copy thread is its only client. It submits a job when it issues the request's copy job, and treats the copy
// job as complete only once this thread has also finished that job. So CopyDone, the lease release and the copy wait's
// gate keep their one publisher (LEASE_PROTOCOL.md, "Copy engine"). A failed forward is never marked done: this
// thread reports the error, and the copy thread fails the job, which fails the process stop.
#pragma once

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../lease_layout.h"
#include "reader_base.h"
#include "spsc_ring.h"

namespace sglang::expert_stream {

// A format's CPU expert kernel as a C ABI, the native half of CpuExpertQuantTrait (python/sglang/srt/layers/moe/
// cpu_experts/pool.py): overwrite out[0, hidden) with sum over i < k of weights[i] * expert(slots[i])(x), where
// slots[i] indexes the layer's pinned host tier and x is one input row in the trait's x format. `layer` is the handle
// the trait registered for that layer. Returns 0 on success. Called from the CPU expert thread only, with `threads`
// the kernel's worker count, the calling thread counted as one of them.
using CpuExpertForward =
    int (*)(int64_t layer, const void* x, const int32_t* slots, const float* weights, int32_t k, float* out,
            int32_t threads);

struct CpuJob {
  int64_t row = 0;
  uint32_t seq = 0;  // the engine's job sequence, from 1; done() compares against it
  int32_t k = 0;
  int32_t slots[wire::kLeaseLanes] = {};
  float weights[wire::kLeaseLanes] = {};
};

struct CpuExpertConfig {
  CpuExpertForward forward = nullptr;
  int64_t rows = 0;  // streamed rows; each gets its layer handle later (set_layer), until then the CPU skips it
  const uint8_t* x_base = nullptr;  // row r's input at x_base + r * x_stride, written by the post kernel
  int64_t x_stride = 0;
  uint8_t* out_base = nullptr;  // row r's fp32 output at out_base + r * out_stride, read by the device
  int64_t out_stride = 0;
  int64_t hidden = 0;
  int threads = 1;
  std::vector<int> cores;  // this thread's affinity, which the kernel's own workers may inherit
  int64_t spin_ns = 50'000'000;
};

// One thread, a lock-free SPSC job ring from the copy thread, and a monotonically increasing done word. Jobs run in
// order, so done() is a single compare. The idle protocol is the copy engine's (copy_engine.h): spin for spin_ns,
// then a futex sleep whose wake the submitter makes only when this thread has gone to sleep.
class CpuExpertEngine {
 public:
  static constexpr size_t kRing = 32;
  static_assert(kRing > wire::kDemandRecords + 1, "the ring holds every job that can be outstanding");

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

  // A row the grant may send to the CPU: registered, and inside the tables.
  bool eligible(int64_t row) const {
    return row >= 0 && row < config_.rows && handles_[row].load(std::memory_order_acquire) >= 0;
  }

  // Copy thread only. The job's sequence, or 0 when the ring is full (an internal error: at most kDemandRecords + 1
  // copy jobs are outstanding, so the copy thread fails the job).
  uint32_t submit(CpuJob job) {
    uint32_t seq = submitted_ + 1;
    if (seq == 0) seq = 1;  // 0 means "no job" to the copy thread
    job.seq = seq;
    if (!jobs_.push(job)) return 0;
    submitted_ = seq;
    std::atomic_thread_fence(std::memory_order_seq_cst);  // Dekker with run()'s sleeping_ store and ring re-check
    if (sleeping_.load(std::memory_order_relaxed)) {
      wake_.fetch_add(1, std::memory_order_relaxed);
      futex_wake(&wake_);
    }
    return seq;
  }

  // Any thread. Acquire: a job seen done has its output written, so a CopyDone published after this observation orders
  // the output before the device's read of it.
  bool done(uint32_t seq) const {
    return static_cast<int32_t>(done_.load(std::memory_order_acquire) - seq) >= 0;
  }

  // The first forward error, or 0. Sticky: a broken engine runs no further job.
  int broken() const {
    return broken_.load(std::memory_order_acquire);
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
      for (int core : config_.cores)
        CPU_SET(core, &set);
      if (sched_setaffinity(0, sizeof(set), &set) != 0) error = "cannot pin the CPU expert thread to its cores";
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
      if (broken_.load(std::memory_order_relaxed) != 0) continue;  // never marked done: its leases stay held
      const int64_t start = now_ns();
      const int result = config_.forward(
          handles_[job.row].load(std::memory_order_acquire),
          config_.x_base + job.row * config_.x_stride,
          job.slots,
          job.weights,
          job.k,
          reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride),
          config_.threads);
      if (result != 0) {
        std::fprintf(
            stderr, "ERROR %sforward of row %lld failed (%d); leases held\n", prefix_.c_str(),
            static_cast<long long>(job.row), result);
        std::fflush(stderr);
        broken_.store(result, std::memory_order_release);
        continue;
      }
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
  uint32_t submitted_ = 0;  // the copy thread's
  std::atomic<uint32_t> done_{0};
  std::atomic<int> broken_{0};
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
