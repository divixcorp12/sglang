// The CPU expert engine: computes a record's CPU lanes on a host thread instead of copying their rows to the device.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include "../cpu_token_table.h"
#include "../lease_layout.h"
#include "cpu_experts/kernel.hpp"
#include "reader_base.h"
#include "spsc_ring.h"
#include "tier_protocol.h"
#include <array>
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

/// Refuses a CPU expert team its forwards would refuse later: fewer than one thread, a core outside [0, CPU_SETSIZE)
/// or repeated, or more threads than cores. Every CPU expert thread (CpuExpertEngine, DraftCpuThread) checks its own.
inline void check_cpu_expert_team(const std::string& prefix, const std::vector<int>& cores, int threads) {
  if (threads < 1) throw std::runtime_error(prefix + "the CPU expert pool needs at least one thread");
  for (size_t i = 0; i < cores.size(); ++i) {
    if (cores[i] < 0 || cores[i] >= CPU_SETSIZE)
      throw std::runtime_error(prefix + "CPU expert core " + std::to_string(cores[i]) + " is outside [0, CPU_SETSIZE)");
    for (size_t j = 0; j < i; ++j)
      if (cores[j] == cores[i])
        throw std::runtime_error(prefix + "CPU expert core " + std::to_string(cores[i]) + " repeats");
  }
  if (!cores.empty() && static_cast<size_t>(threads) > cores.size())
    throw std::runtime_error(prefix + std::to_string(threads) + " workers on " + std::to_string(cores.size()) + " cores");
}

/// One forward over up to Wire::kLanes lanes of one row.
///
/// A record produces at most one job for its CPU hits (part 0) and one per batch of CPU misses that landed together
/// (part 1). Every miss job after the first adds into part 1 in landing order, so that part's fp32 sum order varies
/// from run to run. A record's job is per_token: on a multi-token row (CpuExpertConfig::tokens > 1) the row's token
/// table gives each token its own slots and weights, `lanes` naming each job lane's column there.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;
  bool accumulate = false;  // add into the part rather than overwrite it
  bool per_token = false;   // a record's job: a multi-token row's table gives each token its weights
  uint32_t seq = 0;         // from claim()
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
  int32_t lanes[wire::Wire::kLanes] = {};  // each job lane's record lane
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
  int64_t tokens = 1;         // tokens a row holds; above 1 a token table follows the inputs (cpu_token_table.h)
  int64_t x_token_bytes = 0;  // bytes between two tokens' staged inputs
  int threads = 1;
  std::vector<int> cores;  // worker i runs on cores[i]; the engine's own thread on cores[0]
  int64_t keep_warm_ns = 0;  // how long after each job the held team runs register work instead of PAUSE
  int64_t spin_ns = -1;      // how long the team is held in PAUSE after that before the engine sleeps; < 0: never
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
/// Idle: the thread holds its team in the kernel's keep_warm, which runs register work at the forward's width for
/// keep_warm_ns after each job and then PAUSE, until a submit or stop() rings the doorbell, so a job inside the hold
/// never waits for a worker to wake. spin_ns after the warm window (and spin_ns after start, before the first job) the
/// hold releases the team to OpenMP's idle wait and the thread sleeps on the doorbell; the next submit wakes it. A
/// negative spin_ns holds the team until the next submit, however long.
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
    doorbell_.ring();
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
    doorbell_.ring();
    thread_.join();
  }

 private:
  static_assert(
      sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
      "the keep-warm reads the doorbell's word as a plain uint32_t");

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
    if (c.tokens < 1 || c.tokens > CpuTokenTable::kMaxTokens || (c.tokens > 1 && c.x_token_bytes < 2 * c.hidden))
      throw std::runtime_error(prefix_ + "the CPU expert rows hold 1-32 tokens of the hidden size");
    check_cpu_expert_team(prefix_, c.cores, c.threads);
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
    constexpr int64_t kNever = INT64_MAX;
    int64_t warm_until = 0;  // the register-work window after the last job; 0 before the first
    // When the hold releases the team; 0 once it has and the thread sleeps between submits.
    int64_t release_at = config_.spin_ns < 0 ? kNever : now_ns() + config_.spin_ns;
    while (!stop_.load(std::memory_order_acquire)) {
      // Read before the pop: a submit after an empty pop moves the word past `kick` and so ends the hold.
      const uint32_t kick = doorbell_.word().load(std::memory_order_acquire);
      CpuJob job;
      if (jobs_.pop(&job)) {
        warm_until = run_job(job) + config_.keep_warm_ns;
        release_at = config_.spin_ns < 0 ? kNever : warm_until + config_.spin_ns;
      } else if (release_at != 0) {
        hold(kick, warm_until, release_at);
        if (doorbell_.word().load(std::memory_order_acquire) == kick) release_at = 0;  // ran out: sleep from now on
      } else {
        doorbell_.sleep_unless([this] { return !jobs_.empty() || stop_.load(std::memory_order_acquire); });
      }
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
    if (job.per_token && config_.tokens > 1) {
      call.rows = static_cast<int32_t>(expand_tokens(job, call.x));
      call.slots = token_slots_.data();
      call.weights = token_weights_.data();
      // A token that routes none of the lanes must read 0, whatever an earlier record left: zero the rows here rather
      // than trust the kernel's overwrite, then accumulate.
      if (!job.accumulate) std::memset(call.out, 0, static_cast<size_t>(call.rows) * config_.hidden * sizeof(float));
      call.accumulate = true;
    }
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

  /// A record's job on a multi-token row: each token's slots and weights from the row's token table
  /// (cpu_token_table.h), -1 where the token does not route the lane's expert (the kernel skips only slot -1, never a
  /// zero weight). Returns the record's tokens; fails stop on a count the row cannot hold.
  uint32_t expand_tokens(const CpuJob& job, const uint8_t* x) {
    constexpr int64_t kLanes = wire::Wire::kLanes;
    const uint8_t* table = x + config_.tokens * config_.x_token_bytes;
    const uint32_t tokens = load_u32(table);
    if (tokens < 1 || tokens > static_cast<uint32_t>(config_.tokens)) {
      fail_stop(prefix_ + "row " + std::to_string(job.row) + "'s token table holds " + std::to_string(tokens) +
                " tokens");
      return 1;
    }
    for (uint32_t t = 0; t < tokens; ++t)
      for (int32_t i = 0; i < job.k; ++i) {
        const int32_t lane = job.lanes[i];
        const bool routed = (load_u32(table + CpuTokenTable::kHeaderBytes + 4 * lane) >> t & 1u) != 0;
        const uint32_t bits = load_u32(table + CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane));
        token_slots_[t * job.k + i] = routed ? job.slots[i] : -1;
        token_weights_[t * job.k + i] = routed ? std::bit_cast<float>(bits) : 0.0f;
      }
    return tokens;
  }

  static uint32_t load_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
  }

  /// Holds the team until the doorbell's word moves past `kick` or the clock reaches release_at.
  void hold(uint32_t kick, int64_t warm_until, int64_t release_at) {
    try {
      config_.kernel->keep_warm(
          config_.cores,
          config_.threads,
          reinterpret_cast<const uint32_t*>(&doorbell_.word()),
          kick,
          warm_until,
          release_at);
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
  Doorbell doorbell_;  // rung by every submit and by stop(); its word ends the hold
  uint32_t claimed_ = 0;  // tier owner only
  std::atomic<uint32_t> done_{0};
  std::atomic<int64_t> jobs_done_{0};
  std::atomic<int64_t> lanes_done_{0};
  std::atomic<int64_t> compute_ns_{0};
  // A per-token job's expanded slots and weights, [tokens][k]; this thread only.
  std::array<int32_t, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_slots_{};
  std::array<float, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> token_weights_{};
  std::atomic<bool> stop_{false};
};

}  // namespace sglang::expert_stream
