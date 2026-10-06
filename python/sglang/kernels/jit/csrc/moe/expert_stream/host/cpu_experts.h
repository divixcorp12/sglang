// The CPU expert engine: computes a record's CPU lanes on a host thread instead of copying their rows to the device.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include "../lease_layout.h"
#include "cpu_experts/kernel.hpp"
#include "reader_base.h"
#include "spsc_ring.h"
#include "tier_protocol.h"
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sglang::expert_stream {

/// Every row's CPU expert layer, one per RamTier and shared by every group's engine (a layer addresses its row's whole
/// slab). Set once per row from any thread, read lock-free by the engines, and destroyed with the tier after every
/// engine has stopped (RamTier's member order).
class CpuExpertLayers {
 public:
  explicit CpuExpertLayers(int64_t rows)
      : rows_(rows),
        layers_(std::make_unique<cpu_experts::ExpertLayer[]>(static_cast<size_t>(rows))),
        state_(std::make_unique<std::atomic<uint8_t>[]>(static_cast<size_t>(rows))) {
    for (int64_t r = 0; r < rows_; ++r)
      state_[r].store(kEmpty, std::memory_order_relaxed);
  }
  CpuExpertLayers(const CpuExpertLayers&) = delete;
  CpuExpertLayers& operator=(const CpuExpertLayers&) = delete;

  int64_t rows() const {
    return rows_;
  }

  /// Installs `layer` as `row`'s (the caller checks both); false when the row already has one. The release pairs with
  /// get()'s acquire, so an engine that sees the layer sees all of it.
  bool set(int64_t row, const cpu_experts::ExpertLayer& layer) {
    uint8_t empty = kEmpty;
    if (!state_[row].compare_exchange_strong(empty, kWriting, std::memory_order_relaxed)) return false;
    layers_[row] = layer;
    state_[row].store(kReady, std::memory_order_release);
    return true;
  }

  /// `row`'s layer; nullptr outside [0, rows) or before set() completes.
  const cpu_experts::ExpertLayer* get(int64_t row) const {
    if (row < 0 || row >= rows_ || state_[row].load(std::memory_order_acquire) != kReady) return nullptr;
    return &layers_[row];
  }

 private:
  static constexpr uint8_t kEmpty = 0, kWriting = 1, kReady = 2;
  int64_t rows_;
  std::unique_ptr<cpu_experts::ExpertLayer[]> layers_;
  std::unique_ptr<std::atomic<uint8_t>[]> state_;
};

/// One forward over up to Wire::kLanes lanes of one row.
///
/// A record produces at most one job for its CPU hits (part 0) and one per batch of CPU misses that landed together
/// (part 1). Every miss job after the first adds into part 1 in landing order, so that part's fp32 sum order varies
/// from run to run.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;
  bool accumulate = false;  // add into the part rather than overwrite it
  uint32_t seq = 0;         // from claim()
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
};

/// The pinned tables a CpuExpertEngine reads and writes, and how it runs. Row r's input is at x_base + r * x_stride
/// (written by the post kernel); its part p fp32 output at out_base + r * out_stride + p * out_part_stride.
struct CpuExpertConfig {
  const cpu_experts::CpuExpertKernel* kernel = nullptr;
  const CpuExpertLayers* layers = nullptr;  // RamTier sets it
  const uint8_t* x_base = nullptr;
  int64_t x_stride = 0;
  uint8_t* out_base = nullptr;
  int64_t out_stride = 0;
  int64_t out_part_stride = 0;  // 0: one part, so CPU misses are refused (RamTier::serve_record)
  int64_t hidden = 0;
  int threads = 1;
  std::vector<int> cores;  // worker i runs on cores[i]; the engine's own thread on cores[0]
  int64_t keep_warm_ns = 0;  // how long after each job the held team runs register work instead of PAUSE
  bool check_calls = false;  // run the kernel's check() before every forward (the instr build)
};

/// The CPU expert thread, its job ring and its done word.
///
/// A CPU lane is a RAM-tier expert the device typed with kLeaseTagCpu: the engine runs it from its host slot through
/// the format's CpuExpertKernel and writes the result into the row's pinned output, which the device folds into its
/// fused MoE. A failed forward fails stop.
///
/// The tier's owner (the service thread) is the only client: it claims sequences with claim() and submits jobs with
/// submit(), a record's CPU hits before its copy job and each CPU miss once its row has landed. Jobs run in sequence
/// order and done_ only increases, so done(seq) is a single compare any thread may make. The copy thread reads only
/// done(), so CopyDone and the copy wait's gate keep their single publisher.
///
/// Idle, nothing sleeps: the thread holds its team in the kernel's keep_warm, which runs register work at the forward's
/// width for keep_warm_ns after each job and then PAUSE, until a submit or stop() moves kick_. A job therefore never
/// waits for a worker to wake, and only a long gap pays the vector-frequency ramp.
class CpuExpertEngine {
 public:
  /// A record has at most one CPU-hit job and one job per CPU miss: the ring holds every job that can be outstanding.
  static constexpr size_t kRing =
      std::bit_ceil(static_cast<size_t>(wire::Wire::kDemandRecords) * (wire::Wire::kLanes + 1));

  CpuExpertEngine(CpuExpertConfig config, std::string prefix, std::string thread_name)
      : config_(std::move(config)), prefix_(std::move(prefix)), thread_name_(thread_name.substr(0, 15)) {
    validate();
  }

  ~CpuExpertEngine() {
    stop();
  }

  /// Starts the thread and waits until it has pinned itself to its core. Throws, after joining it, if it cannot.
  void start() {
    std::future<std::string> started = started_.get_future();
    thread_ = std::thread([this] { run(); });
    if (const std::string error = started.get(); !error.empty()) {
      stop();
      throw std::runtime_error(prefix_ + error);
    }
  }

  /// Output parts per row: 2 when the rows hold a CPU-hit and a CPU-miss partial sum each.
  int parts() const {
    return config_.out_part_stride > 0 ? 2 : 1;
  }

  const std::vector<int>& cores() const {
    return config_.cores;
  }

  /// True for a row the device may send to the CPU: inside the tables and its layer made. The acquire in get() orders a
  /// grant that sends the row to the CPU after the layer it relies on.
  bool eligible(int64_t row) const {
    return config_.layers->get(row) != nullptr;
  }

  /// Tier owner only: the first of `n` consecutive sequences for jobs it will submit, in order. An unsubmitted sequence
  /// is skipped (done() of a later one covers it), so a record can claim one per possible job.
  uint32_t claim(int n) {
    const uint32_t first = claimed_ + 1;
    claimed_ += static_cast<uint32_t>(n);
    return first;
  }

  /// Tier owner only, with job.seq claimed and above every earlier submit's. False when the ring is full, which kRing
  /// rules out: the caller fails stop.
  bool submit(const CpuJob& job) {
    if (!jobs_.push(job)) return false;
    kick_.fetch_add(1, std::memory_order_release);
    return true;
  }

  /// True once the job with sequence `seq`, or a later one, has finished. The acquire pairs with the release after the
  /// job's output rows, so a CopyDone published after it orders the output before the device reads it.
  bool done(uint32_t seq) const {
    return static_cast<int32_t>(done_.load(std::memory_order_acquire) - seq) >= 0;
  }

  /// Jobs finished, lanes computed and forward time in ns, for the service's counters.
  int64_t jobs() const {
    return jobs_done_.load(std::memory_order_relaxed);
  }
  int64_t lanes() const {
    return lanes_done_.load(std::memory_order_relaxed);
  }
  int64_t compute_ns() const {
    return compute_ns_.load(std::memory_order_relaxed);
  }

  /// Stops and joins the thread; idempotent.
  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    kick_.fetch_add(1, std::memory_order_release);
    thread_.join();
  }

 private:
  static_assert(
      sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
      "the keep-warm reads kick_ as a plain uint32_t");

  /// Refuses a config the forwards would refuse later, since a refused forward aborts the process.
  void validate() const {
    const CpuExpertConfig& c = config_;
    if (c.kernel == nullptr) throw std::runtime_error(prefix_ + "no CPU expert kernel");
    if (c.kernel->max_routes() < wire::Wire::kLanes || c.kernel->max_rows() < 1)
      throw std::runtime_error(
          prefix_ + "kernel " + c.kernel->name() + " takes " + std::to_string(c.kernel->max_routes()) +
          " routes per forward, a job up to " + std::to_string(wire::Wire::kLanes));
    if (c.layers == nullptr) throw std::runtime_error(prefix_ + "no CPU expert layers");
    if (c.x_base == nullptr || c.out_base == nullptr || c.x_stride <= 0 || c.out_stride <= 0)
      throw std::runtime_error(prefix_ + "the CPU expert input and output rows are required");
    if (c.hidden <= 0 || c.out_stride < c.hidden * static_cast<int64_t>(sizeof(float)))
      throw std::runtime_error(prefix_ + "the CPU expert output rows are smaller than the hidden size");
    if (c.threads < 1) throw std::runtime_error(prefix_ + "the CPU expert pool needs at least one thread");
    for (size_t i = 0; i < c.cores.size(); ++i) {
      if (c.cores[i] < 0 || c.cores[i] >= CPU_SETSIZE)
        throw std::runtime_error(
            prefix_ + "CPU expert core " + std::to_string(c.cores[i]) + " is outside [0, CPU_SETSIZE)");
      for (size_t j = 0; j < i; ++j)
        if (c.cores[j] == c.cores[i])
          throw std::runtime_error(prefix_ + "CPU expert core " + std::to_string(c.cores[i]) + " repeats");
    }
    if (!c.cores.empty() && static_cast<size_t>(c.threads) > c.cores.size())
      throw std::runtime_error(
          prefix_ + std::to_string(c.threads) + " workers on " + std::to_string(c.cores.size()) + " cores");
  }

  /// Names the thread and pins it to cores[0]; returns why it could not, else "".
  std::string pin() const {
    pthread_setname_np(pthread_self(), thread_name_.c_str());
    if (config_.cores.empty()) return "";
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(config_.cores.front(), &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0 ? "" : "cannot pin the CPU expert thread to its caller core";
  }

  void run() {
    const std::string error = pin();
    started_.set_value(error);
    if (!error.empty()) return;
    int64_t warm_until = 0;  // the register-work window after the last job; 0 before the first
    while (!stop_.load(std::memory_order_acquire)) {
      // Read before the pop: a submit after an empty pop moves kick_ past `kick` and so ends the hold.
      const uint32_t kick = kick_.load(std::memory_order_acquire);
      CpuJob job;
      if (jobs_.pop(&job))
        warm_until = run_job(job) + config_.keep_warm_ns;
      else
        hold(kick, warm_until);
    }
  }

  /// Runs one forward and publishes it in done_; returns the clock at its end.
  int64_t run_job(const CpuJob& job) {
    const int64_t start = now_ns();
    cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = job.k;
    call.threads = config_.threads;
    call.x = config_.x_base + job.row * config_.x_stride;
    call.slots = job.slots;
    call.weights = job.weights;
    call.out =
        reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
    call.accumulate = job.accumulate;
    call.cores = config_.cores;
    try {
      const cpu_experts::ExpertLayer* layer = config_.layers->get(job.row);
      if (layer == nullptr) throw std::invalid_argument("the row has no registered layer");
      if (config_.check_calls) config_.kernel->check(*layer, call);
      config_.kernel->forward(*layer, call);
    } catch (const std::exception& e) {
      fail_stop(prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed: " + e.what());
    }
    const int64_t end = now_ns();
    add(compute_ns_, end - start);
    add(jobs_done_, 1);
    add(lanes_done_, job.k);
    done_.store(job.seq, std::memory_order_release);
    return end;
  }

  /// Holds the team until kick_ moves past `kick`.
  void hold(uint32_t kick, int64_t warm_until) {
    try {
      config_.kernel->keep_warm(
          config_.cores, config_.threads, reinterpret_cast<const uint32_t*>(&kick_), kick, warm_until);
    } catch (const std::exception& e) {
      fail_stop(prefix_ + "CPU expert keep-warm failed: " + e.what());
    }
  }

  /// A counter only this thread writes: no locked add needed.
  static void add(std::atomic<int64_t>& counter, int64_t n) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  CpuExpertConfig config_;
  std::string prefix_;
  std::string thread_name_;
  std::thread thread_;
  std::promise<std::string> started_;  // the thread's pin result, for start()
  SpscRing<CpuJob, kRing> jobs_;
  std::atomic<uint32_t> kick_{0};  // moved by every submit and by stop(); ends the hold
  uint32_t claimed_ = 0;  // tier owner only
  std::atomic<uint32_t> done_{0};
  std::atomic<int64_t> jobs_done_{0};
  std::atomic<int64_t> lanes_done_{0};
  std::atomic<int64_t> compute_ns_{0};
  std::atomic<bool> stop_{false};
};

}  // namespace sglang::expert_stream
