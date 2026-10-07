// The CPU expert engine: computes a record's CPU lanes on a host thread instead of copying their rows to the device.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include "../cpu_token_table.h"
#include "../draft_channel.h"
#include "../lease_layout.h"
#include "cpu_experts/kernel.hpp"
#include "lease_channel.h"
#include "job_trace.h"
#include "reader_base.h"
#include "spsc_ring.h"
#include "tier_protocol.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sglang::expert_stream {

namespace draft {

/// Instr build only (DraftSource::test_hooks): how long the draft poll path sleeps between its loads, to make the
/// teardown interleaving deterministic (draft_test_poll_pause).
inline std::atomic<int64_t> g_test_poll_pause_us{0};

/// The call's live routes (slot >= 0) past each slot's first. Outside the timed forward; at most kMaxRows * kMaxK.
inline int count_shared_routes(const int32_t* slots, int n) {
  int32_t live[kMaxRows * kMaxK];
  int count = 0;
  for (int i = 0; i < n; ++i)
    if (slots[i] >= 0) live[count++] = slots[i];
  std::sort(live, live + count);
  int shared = 0;
  for (int i = 1; i < count; ++i) shared += live[i] == live[i - 1];
  return shared;
}

}  // namespace draft

using draft::DraftChannel;

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
/// or repeated, or more threads than cores. Every CPU expert thread (CpuExpertEngine) checks its own.
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

/// The DSpark draft channel (draft_channel.h) as a CpuExpertEngine's second job source (plan 2026-10-06 Task 11): each
/// posted record is a stage's M-row forward over the stage's pinned areas, completed through the lease channel (done,
/// then the Dekker open of the gate). The stages' layers read the draft's own slabs (pageable host memory).
struct DraftSource {
  uint8_t* channel = nullptr;      // draft::kChannelBytes, pinned: the page and the completion block
  const uint8_t* x = nullptr;      // fp16 [stages, kMaxRows, hidden]
  const int32_t* slots = nullptr;  // [stages, kMaxRows, kMaxK]
  const float* weights = nullptr;  // [stages, kMaxRows, kMaxK]
  float* out = nullptr;            // [stages, kMaxRows, hidden]
  int64_t hidden = 0;
  std::vector<cpu_experts::ExpertLayer> layers;  // one per stage, each on the engine's kernel
  bool test_hooks = false;                       // the instr build's: honour draft::g_test_poll_pause_us
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
  bool draft_only = false;   // a DSpark draft-only launch: no rows, layers or ring jobs; the draft source is the only one
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
///
/// With a DSpark draft source (attach_draft) the thread serves the draft channel too, one job at a time with the
/// target's; its idle hold watches the doorbell and the channel head, and its poll sleeps 50 us at a time since the GPU
/// cannot ring the futex. A draft-only engine (CpuExpertConfig::draft_only) has only that source.
template <BuildPolicy Build>
class BasicCpuExpertEngine {
 public:
  /// A record has at most one CPU-hit job and one job per CPU miss: the ring holds every job that can be outstanding.
  static constexpr size_t kRing =
      std::bit_ceil(static_cast<size_t>(wire::Wire::kDemandRecords) * (wire::Wire::kLanes + 1));

  BasicCpuExpertEngine(CpuExpertConfig config, std::string prefix, std::string thread_name)
      : config_(std::move(config)), prefix_(std::move(prefix)), thread_name_(thread_name.substr(0, 15)), trace_(thread_name_) {
    validate();
  }

  ~BasicCpuExpertEngine() {
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

  /// Tokens a row holds (CpuExpertConfig::tokens).
  int64_t tokens() const {
    return config_.tokens;
  }

  /// Calibration only (the tier's owner, copy engine not armed, so no record writes or reads the row): makes `row`'s
  /// token table that of a verify of `tokens` tokens that routes every job lane with weight 1, so a per_token CpuJob
  /// whose lanes[i] = i runs `tokens` rows of all its k lanes. The row's staged inputs are left as they are: the
  /// timing does not depend on them. The tables are the pinned rows the Python service owns and the post kernel writes
  /// (cpu_token_table.h); the const is only the engine's own read-only view of them.
  void write_calibration_table(int64_t row, int64_t tokens) const {
    constexpr int64_t kLanes = wire::Wire::kLanes;
    if (tokens < 1 || tokens > config_.tokens)
      throw std::runtime_error(
          prefix_ + "calibration tokens " + std::to_string(tokens) + " is not within the rows' 1.." +
          std::to_string(config_.tokens));
    if (tokens == 1) return;
    uint8_t* table = const_cast<uint8_t*>(config_.x_base) + row * config_.x_stride + config_.tokens * config_.x_token_bytes;
    const auto store = [](uint8_t* at, uint32_t v) { std::memcpy(at, &v, sizeof v); };
    store(table, static_cast<uint32_t>(tokens));
    const uint32_t mask = tokens == 32 ? ~0u : (1u << tokens) - 1u;
    for (int64_t lane = 0; lane < kLanes; ++lane)
      store(table + CpuTokenTable::kHeaderBytes + 4 * lane, mask);
    for (int64_t t = 0; t < tokens; ++t)
      for (int64_t lane = 0; lane < kLanes; ++lane)
        store(table + CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane), std::bit_cast<uint32_t>(1.0f));
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
    if constexpr (Build::kMetrics) trace_.emit("cpu_submit", job.row, 0, job.seq, -1, job.part, job.k);
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

  static constexpr const char* kDraftPrefix = "DSpark draft CPU experts: ";

  /// Makes the DSpark draft channel this engine's second job source: from then on the thread also serves each posted
  /// record, one job at a time with the target's, never two at once. Starts the watchdog, which fail-stops a job of
  /// either kind that runs past fatal_wait_ns and a posted record left unserved that long. After start(); once.
  void attach_draft(std::unique_ptr<DraftSource> source, int64_t fatal_wait_ns) {
    if (draft_owned_ != nullptr) throw std::runtime_error(std::string(kDraftPrefix) + "a draft source is attached already");
    if (!thread_.joinable()) throw std::runtime_error(std::string(kDraftPrefix) + "attach after the engine started");
    if (fatal_wait_ns <= 0) throw std::runtime_error(std::string(kDraftPrefix) + "the fatal wait must be positive");
    const DraftSource& d = *source;
    if (!d.channel || !d.x || !d.slots || !d.weights || !d.out || d.hidden <= 0 || d.layers.empty())
      throw std::runtime_error(std::string(kDraftPrefix) + "the channel, the stage areas and a layer per stage are required");
    for (size_t s = 0; s < d.layers.size(); ++s) {
      if (d.layers[s].kernel != config_.kernel)
        throw std::runtime_error(std::string(kDraftPrefix) + "stage " + std::to_string(s) +
                                 " runs on another kernel than the team's");
      if (d.layers[s].hidden != d.hidden)
        throw std::runtime_error(std::string(kDraftPrefix) + "stage " + std::to_string(s) + "'s layer has hidden " +
                                 std::to_string(d.layers[s].hidden) + ", the areas " + std::to_string(d.hidden));
    }
    if (config_.kernel->max_routes() < draft::kMaxK || config_.kernel->max_rows() < draft::kMaxRows)
      throw std::runtime_error(std::string(kDraftPrefix) + "kernel " + config_.kernel->name() +
                               " takes fewer rows or routes than a draft call");
    const uint32_t head = channel::head<DraftChannel>(d.channel);
    draft_completed_.store(head, std::memory_order_relaxed);
    draft_next_ = channel::skip_zero(head + 1u);  // the run thread reads it after acquiring draft_
    fatal_wait_ns_ = fatal_wait_ns;
    draft_owned_ = std::move(source);
    draft_.store(draft_owned_.get(), std::memory_order_release);
    doorbell_.ring();  // a sleeping thread starts watching the channel
    watchdog_ = std::thread([this] { watch(); });
  }

  /// Stops serving the draft channel: the thread finishes a draft job in progress and drops the source, then the gate
  /// a wait still holds closed is opened (no completer is left). The watchdog runs on, so a hung job still fail-stops.
  /// Idempotent; stop() opens the gate too.
  void detach_draft() {
    DraftSource* d = draft_.load(std::memory_order_acquire);
    if (d == nullptr) return;
    draft_detach_.store(true, std::memory_order_seq_cst);
    doorbell_.ring();
    while (draft_.load(std::memory_order_acquire) != nullptr && !exited_.load(std::memory_order_acquire))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    channel::open_closed_gate<DraftChannel>(d->channel);
  }

  struct DraftStats {
    int64_t jobs, rows, forward_ns, holds, collided_jobs, shared_routes, collided_forward_ns;
  };
  DraftStats draft_stats() const {
    return {draft_jobs_.load(std::memory_order_relaxed), draft_rows_.load(std::memory_order_relaxed),
            draft_forward_ns_.load(std::memory_order_relaxed), draft_holds_.load(std::memory_order_relaxed),
            draft_collided_jobs_.load(std::memory_order_relaxed), draft_shared_routes_.load(std::memory_order_relaxed),
            draft_collided_forward_ns_.load(std::memory_order_relaxed)};
  }

  /// Stops and joins the thread, then the watchdog (a hung forward still fail-stops meanwhile), then opens a draft gate
  /// a wait still holds closed; idempotent.
  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    doorbell_.ring();
    thread_.join();
    watchdog_stop_.store(true, std::memory_order_seq_cst);
    if (watchdog_.joinable()) watchdog_.join();
    if (draft_owned_ != nullptr) channel::open_closed_gate<DraftChannel>(draft_owned_->channel);
  }

 private:
  static_assert(
      sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
      "the keep-warm reads the doorbell's word as a plain uint32_t");

  /// Refuses a config the forwards would refuse later, since a refused forward aborts the process.
  void validate() const {
    const CpuExpertConfig& c = config_;
    if (c.kernel == nullptr) throw std::runtime_error(prefix_ + "no CPU expert kernel");
    if (!c.draft_only) {
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
      // A per_token job runs `tokens` rows through one forward: a kernel with fewer would fail at the first verify.
      if (c.tokens > c.kernel->max_rows())
        throw std::runtime_error(
            prefix_ + "kernel " + c.kernel->name() + " takes " + std::to_string(c.kernel->max_rows()) +
            " rows per forward, a job up to " + std::to_string(c.tokens));
    }
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
    // When the hold releases the team; 0 once it has and the thread sleeps between jobs.
    int64_t release_at = config_.spin_ns < 0 ? kNever : now_ns() + config_.spin_ns;
    const uint64_t spin_iters = idle_budget(config_.spin_ns);
    uint64_t idle = 0;
    while (!stop_.load(std::memory_order_acquire)) {
      DraftSource* draft = draft_.load(std::memory_order_acquire);
      if (draft != nullptr && draft_detach_.load(std::memory_order_acquire)) {
        draft_.store(nullptr, std::memory_order_release);  // detach_draft waits for this
        draft = nullptr;
      }
      if (draft != nullptr && draft->test_hooks)
        if (const int64_t pause = draft::g_test_poll_pause_us.load(std::memory_order_relaxed); pause > 0)
          std::this_thread::sleep_for(std::chrono::microseconds(pause));
      // Read before the pop and the head load: a submit or a post after them moves a word the hold watches.
      const uint32_t kick = doorbell_.word().load(std::memory_order_acquire);
      const uint32_t head = draft != nullptr ? channel::head<DraftChannel>(draft->channel) : 0u;
      CpuJob job;
      if (jobs_.pop(&job)) {
        warm_until = run_job(job) + config_.keep_warm_ns;
        release_at = config_.spin_ns < 0 ? kNever : warm_until + config_.spin_ns;
        idle = 0;
      } else if (draft != nullptr && head != 0 && channel::reached(head, draft_next_)) {
        if (head != draft_next_)
          fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(draft_next_) + " lapped (head " +
                    std::to_string(head) + "); the device posts one record per wait");
        warm_until = serve_draft(*draft, draft_next_) + config_.keep_warm_ns;
        release_at = config_.spin_ns < 0 ? kNever : warm_until + config_.spin_ns;
        draft_next_ = channel::skip_zero(draft_next_ + 1u);
        idle = 0;
      } else if (release_at != 0) {
        hold(kick, draft, head, warm_until, release_at);
        const bool moved = doorbell_.word().load(std::memory_order_acquire) != kick ||
                           (draft != nullptr && channel::head<DraftChannel>(draft->channel) != head);
        if (!moved) release_at = 0;  // ran out: sleep (or, with a draft source, poll) from now on
      } else if (draft != nullptr) {
        // The GPU cannot ring the futex: spin the idle budget, then sleep 50 us at a time; a submit, an attach or
        // stop() cuts a sleep short.
        if (++idle < spin_iters) {
          _mm_pause();
        } else {
          if constexpr (Build::kMetrics) trace_.emit("cpu_wait_begin", -1, 0, kick, -1, 50'000, head);
          doorbell_.sleep_unless([this] { return !jobs_.empty() || stop_.load(std::memory_order_acquire); }, 50'000);
          if constexpr (Build::kMetrics) trace_.emit("cpu_wait_end", -1, 0, kick, -1, 50'000, head);
        }
      } else {
        if constexpr (Build::kMetrics) trace_.emit("cpu_wait_begin", -1, 0, kick, -1, 1'000'000, head);
        doorbell_.sleep_unless([this] {
          return !jobs_.empty() || stop_.load(std::memory_order_acquire) ||
                 draft_.load(std::memory_order_acquire) != nullptr;
        });
        if constexpr (Build::kMetrics) trace_.emit("cpu_wait_end", -1, 0, kick, -1, 1'000'000, head);
      }
    }
    exited_.store(true, std::memory_order_release);
  }

  /// Runs one forward and publishes it in done_; returns the clock at its end.
  int64_t run_job(const CpuJob& job) {
    const int64_t start = now_ns();
    if constexpr (Build::kMetrics) trace_.emit("cpu_start", job.row, 0, job.seq, -1, job.part, job.k);
    if constexpr (Build::kMetrics) trace_.resources("cpu_faults_start", "cpu_switches_start", job.row, 0, job.seq);
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
      call.rows = static_cast<int32_t>(expand_tokens(job, static_cast<const uint8_t*>(call.x)));
      call.slots = token_slots_.data();
      call.weights = token_weights_.data();
      // A token that routes none of the lanes must read 0, whatever an earlier record left: zero the rows here rather
      // than trust the kernel's overwrite, then accumulate.
      if (!job.accumulate) std::memset(call.out, 0, static_cast<size_t>(call.rows) * config_.hidden * sizeof(float));
      call.accumulate = true;
    }
    if constexpr (Build::kMetrics) {
      if (trace_.enabled()) {
        int live = 0;
        for (int i = 0; i < call.rows * call.k; ++i) live += call.slots[i] >= 0;
        trace_.emit("cpu_shape", job.row, 0, job.seq, -1, call.rows, call.k, live);
      }
    }
    busy(kTargetJob, job.row);
    try {
      const cpu_experts::ExpertLayer* layer = config_.layers->get(job.row);
      if (layer == nullptr) throw std::invalid_argument("the row has no registered layer");
      if (config_.check_calls) config_.kernel->check(*layer, call);
      config_.kernel->forward(*layer, call);
    } catch (const std::exception& e) {
      fail_stop(prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed: " + e.what());
    }
    job_started_ns_.store(0, std::memory_order_release);
    const int64_t end = now_ns();
    add(compute_ns_, end - start);
    add(jobs_done_, 1);
    add(lanes_done_, job.k);
    if constexpr (Build::kMetrics) trace_.resources("cpu_faults_end", "cpu_switches_end", job.row, 0, job.seq);
    if constexpr (Build::kMetrics) trace_.emit("cpu_end", job.row, 0, job.seq, -1, job.part, job.k);
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

  /// Holds the team until the doorbell's word moves past `kick`, the draft head (with a draft source) past `head`, or
  /// the clock reaches release_at. Without a draft source this is today's one-word hold.
  void hold(uint32_t kick, const DraftSource* draft, uint32_t head, int64_t warm_until, int64_t release_at) {
    if constexpr (Build::kMetrics) trace_.emit("cpu_hold_begin", -1, 0, kick, -1, head, warm_until, release_at);
    try {
      if (draft == nullptr) {
        config_.kernel->keep_warm(
            config_.cores,
            config_.threads,
            reinterpret_cast<const uint32_t*>(&doorbell_.word()),
            kick,
            warm_until,
            release_at);
      } else {
        add(draft_holds_, 1);
        config_.kernel->keep_warm_either(
            config_.cores,
            config_.threads,
            reinterpret_cast<const uint32_t*>(&doorbell_.word()),
            kick,
            reinterpret_cast<const uint32_t*>(draft->channel + DraftChannel::kHead),
            head,
            warm_until,
            release_at);
      }
    } catch (const std::exception& e) {
      fail_stop(prefix_ + "CPU expert keep-warm failed: " + e.what());
    }
    if constexpr (Build::kMetrics) trace_.emit("cpu_hold_end", -1, 0, kick, -1, head);
  }

  static constexpr int32_t kTargetJob = 1, kDraftJob = 2;

  /// Marks the job the thread starts now for the watchdog: its kind and its id (a target job's row, a draft record's
  /// sequence). The release orders the id and kind before the start time the watchdog reads first.
  void busy(int32_t kind, int64_t id) {
    job_kind_.store(kind, std::memory_order_relaxed);
    job_id_.store(id, std::memory_order_relaxed);
    job_started_ns_.store(now_ns(), std::memory_order_release);
  }

  /// Reads draft record `seq`, runs its forward on the team, completes it through the channel; returns the forward's
  /// end. A torn or malformed record, or a refused forward, fail-stops.
  int64_t serve_draft(const DraftSource& d, uint32_t seq) {
    alignas(64) uint8_t raw[DraftChannel::kRecordBytes];
    const uint8_t* rec = channel::record_at<DraftChannel>(d.channel, seq);
    if (!channel::read_seqlocked<DraftChannel>(rec, seq, raw))
      fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(seq) + " torn");
    uint32_t word, epoch;
    std::memcpy(&word, raw + draft::kRecStage, 4);
    std::memcpy(&epoch, raw + draft::kRecEpoch, 4);
    const int stage = static_cast<int>(word & 0xFFFFu), rows = static_cast<int>((word >> 16) & 0xFFu),
              k = static_cast<int>(word >> 24);
    if (stage >= static_cast<int>(d.layers.size()) || rows < 1 || rows > draft::kMaxRows || k < 1 || k > draft::kMaxK)
      fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(seq) + " malformed (stage " +
                std::to_string(stage) + ", rows " + std::to_string(rows) + ", k " + std::to_string(k) + ")");
    // The slot and weight areas are kMaxK wide per token; the kernel reads [rows, k] contiguous, so compact them.
    int32_t slots[draft::kMaxRows * draft::kMaxK];
    float weights[draft::kMaxRows * draft::kMaxK];
    const int32_t* s = d.slots + static_cast<int64_t>(stage) * draft::kMaxRows * draft::kMaxK;
    const float* w = d.weights + static_cast<int64_t>(stage) * draft::kMaxRows * draft::kMaxK;
    for (int t = 0; t < rows; ++t)
      for (int i = 0; i < k; ++i) {
        slots[t * k + i] = s[t * draft::kMaxK + i];
        weights[t * k + i] = w[t * draft::kMaxK + i];
      }
    const int shared = draft::count_shared_routes(slots, rows * k);
    const cpu_experts::ExpertLayer& layer = d.layers[stage];
    cpu_experts::ForwardCall call;
    call.rows = rows;
    call.k = k;
    call.threads = config_.threads;
    call.cores = config_.cores;
    call.x = d.x + static_cast<int64_t>(stage) * draft::kMaxRows * d.hidden * 2;
    call.slots = slots;
    call.weights = weights;
    call.out = d.out + static_cast<int64_t>(stage) * draft::kMaxRows * d.hidden;
    busy(kDraftJob, seq);
    const int64_t start = now_ns();
    if constexpr (Build::kMetrics) trace_.emit("draft_start", stage, epoch, seq, -1, rows, k, shared);
    if constexpr (Build::kMetrics) trace_.resources("draft_faults_start", "draft_switches_start", stage, epoch, seq);
    try {
      layer.kernel->forward(layer, call);
    } catch (const std::exception& e) {
      fail_stop(std::string(kDraftPrefix) + "forward of record " + std::to_string(seq) + " (stage " +
                std::to_string(stage) + ") failed: " + e.what());
    }
    const int64_t end = now_ns();
    job_started_ns_.store(0, std::memory_order_release);
    add(draft_forward_ns_, end - start);
    add(draft_jobs_, 1);
    add(draft_rows_, rows);
    if (shared > 0) {
      add(draft_collided_jobs_, 1);
      add(draft_shared_routes_, shared);
      add(draft_collided_forward_ns_, end - start);
    }
    if constexpr (Build::kMetrics) trace_.resources("draft_faults_end", "draft_switches_end", stage, epoch, seq);
    if constexpr (Build::kMetrics) trace_.emit("draft_end", stage, epoch, seq, -1, rows, k, shared);
    channel::complete<DraftChannel>(d.channel, seq, static_cast<uint64_t>(epoch) << 32 | seq);
    draft_completed_.store(seq, std::memory_order_release);
    return end;
  }

  /// Every 20 ms, once a draft source is attached: fail-stops a job of either kind that has run for fatal_wait_ns, and
  /// a posted draft record that has stood unserved that long.
  void watch() {
    uint32_t watched = 0;
    int64_t since = 0;
    const std::string wait = std::to_string(static_cast<double>(fatal_wait_ns_) * 1e-9);
    while (!watchdog_stop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (watchdog_stop_.load(std::memory_order_acquire)) return;
      const int64_t now = now_ns();
      const int64_t started = job_started_ns_.load(std::memory_order_acquire);
      if (started != 0 && now - started >= fatal_wait_ns_) {
        const int64_t id = job_id_.load(std::memory_order_relaxed);
        if (job_kind_.load(std::memory_order_relaxed) == kDraftJob)
          fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(id) + " incomplete after " + wait +
                    " s (fatal wait)");
        fail_stop(prefix_ + "the CPU job of row " + std::to_string(id) + " incomplete after " + wait + " s (fatal wait)");
      }
      const DraftSource* d = draft_.load(std::memory_order_acquire);
      const uint32_t head = d != nullptr ? channel::head<DraftChannel>(d->channel) : 0u;
      if (head == 0 || head == draft_completed_.load(std::memory_order_acquire)) {
        watched = 0;
        continue;
      }
      if (head != watched) {
        watched = head;
        since = now;
      } else if (now - since >= fatal_wait_ns_) {
        fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(head) + " incomplete after " + wait +
                  " s (fatal wait)");
      }
    }
  }

  /// A counter only this thread writes: no locked add needed.
  static void add(std::atomic<int64_t>& counter, int64_t n) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  CpuExpertConfig config_;
  std::string prefix_;
  std::string thread_name_;
  [[no_unique_address]] JobTrace<Build::kMetrics> trace_;
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
  // The DSpark draft source (attach_draft): owned here, published to the run thread through draft_ (null: none, or
  // detached). draft_next_ is the run thread's after the publication.
  std::unique_ptr<DraftSource> draft_owned_;
  std::atomic<DraftSource*> draft_{nullptr};
  std::atomic<bool> draft_detach_{false}, watchdog_stop_{false}, exited_{false};
  uint32_t draft_next_ = 0;
  std::atomic<uint32_t> draft_completed_{0};
  int64_t fatal_wait_ns_ = 0;
  std::thread watchdog_;
  // The job running now, for the watchdog (busy()): its start (0: none), kind and id.
  std::atomic<int64_t> job_started_ns_{0}, job_id_{0};
  std::atomic<int32_t> job_kind_{0};
  std::atomic<int64_t> draft_jobs_{0}, draft_rows_{0}, draft_forward_ns_{0}, draft_holds_{0}, draft_collided_jobs_{0},
      draft_shared_routes_{0}, draft_collided_forward_ns_{0};
};

using CpuExpertEngine = BasicCpuExpertEngine<ProdBuild>;

}  // namespace sglang::expert_stream
