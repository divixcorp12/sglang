// The CPU expert engine: computes a record's CPU lanes on a host thread instead of copying their rows to the device.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "Copy engine".
#pragma once

#include "../cpu_token_table.h"
#include "../lease_layout.h"
#include "cpu_experts/job_window.hpp"
#include "cpu_experts/kernel.hpp"
#include "cpu_experts/team.hpp"
#include "draft_experts.h"
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
#include <cstdio>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <memory>
#include <optional>
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
template <class Layer>
class CpuExpertLayers {
 public:
  explicit CpuExpertLayers(int64_t rows)
      : rows_(rows),
        layers_(std::make_unique<Layer[]>(static_cast<size_t>(rows))),
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
  bool set(int64_t row, const Layer& layer) {
    uint8_t empty = kEmpty;
    if (!state_[row].compare_exchange_strong(empty, kWriting, std::memory_order_relaxed)) return false;
    layers_[row] = layer;
    state_[row].store(kReady, std::memory_order_release);
    return true;
  }

  /// `row`'s layer; nullptr outside [0, rows) or before set() completes.
  const Layer* get(int64_t row) const {
    if (row < 0 || row >= rows_ || state_[row].load(std::memory_order_acquire) != kReady) return nullptr;
    return &layers_[row];
  }

 private:
  static constexpr uint8_t kEmpty = 0, kWriting = 1, kReady = 2;
  int64_t rows_;
  std::unique_ptr<Layer[]> layers_;
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
///
/// A staged job (CpuExpertConfig::two_stage) is a CPU miss job sent once its rows' first stage (w13) has landed: its
/// gate/up phases run, then the window holds the rest of the job until the tier calls open_stage_two(seq), once every
/// row is whole; other jobs run meanwhile.
struct CpuJob {
  int64_t row = 0;
  int32_t part = 0;
  bool accumulate = false;  // add into the part rather than overwrite it
  bool per_token = false;   // a record's job: a multi-token row's table gives each token its weights
  bool staged = false;      // its rows' second stage may still be landing: wait for open_stage_two(seq)
  uint32_t seq = 0;         // from claim()
  int32_t k = 0;
  int32_t slots[wire::Wire::kLanes] = {};
  float weights[wire::Wire::kLanes] = {};
  int32_t lanes[wire::Wire::kLanes] = {};  // each job lane's record lane
};

/// The pinned tables a CpuExpertEngine reads and writes, and how it runs. Row r's input is at x_base + r * x_stride
/// (written by the post kernel); its part p fp32 output at out_base + r * out_stride + p * out_part_stride.
template <class Layer>
struct CpuExpertConfig {
  const CpuExpertLayers<Layer>* layers = nullptr;  // RamTier sets it
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
  int64_t keep_warm_ns = 0;  // how long after its last task the idle team runs register work instead of PAUSE
  bool check_calls = false;  // run the kernel's check() before admitting every job (the instr build)
  bool draft_only = false;   // a DSpark draft-only launch: no rows, layers or ring jobs; the draft source is the only one
  bool two_stage = false;    // SGLANG_EXPERT_STREAM_CPU_TWO_STAGE: the tier sends each CPU miss as a staged job
                             // (CpuJob::staged)
  // SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT: a job's overwrite is the kernel's store-first overwrite
  // (OutMode::kOverwriteByStore) rather than its zero-first one (OutMode::kOverwrite).
  bool store_out = false;
  // SGLANG_EXPERT_STREAM_CPU_TWO_STAGE w13_weights / w2_weights: a weight per mirror root for each span of a two-span row
  // (empty: the mirror's shares). The tier hands them to its readers (RamTier::enable_cpu_experts).
  std::vector<double> span_weights[2];
};

/// The CPU expert thread, its job ring and its done word.
///
/// A CPU lane is a RAM-tier expert the device typed with kLeaseTagCpu: the engine runs it from its host slot through
/// the format's kernel (Build::CpuKernel) and writes the result into the row's pinned output, which the device folds into its
/// fused MoE. A failed forward fails stop.
///
/// The tier's owner (the service thread) is the only client: it claims sequences with claim() and submits jobs with
/// submit(), a record's CPU hits before its copy job and each CPU miss once its row has landed. Jobs are admitted in
/// sequence order and done_ only increases, so done(seq) is a single compare any thread may make: it passes a sequence
/// only once every job admitted before it has finished, whatever order they finish in. The copy thread reads only
/// done(), so CopyDone and the copy wait's gate keep their single publisher.
///
/// The jobs run in a window (cpu_experts/job_window.hpp) of kWindowSlots slots over the team (cpu_experts/team.hpp),
/// the thread's own for its lifetime: threads - 1 workers pinned to cores[1..], itself worker 0 on cores[0] and the
/// window's only admitter. Between claims of tasks the thread pops the ring, then the draft channel, does the job's
/// serial setup into a free slot and admits it at once (no batching delay); a full window stops admission and the
/// thread runs tasks. Nothing here sleeps. Between tasks the workers run the kernel's register work for keep_warm_ns
/// and then PAUSE, watching the window's word; the thread does the same, a quantum (kIdleQuantumNs) at a time, re-reading
/// the submit word, the draft head, stop() and detach_draft() between quanta, since no host thread can ring those. So
/// a job never waits for a worker to wake, and no state read before a wait can go stale across it.
///
/// With a DSpark draft source (attach_draft, draft_experts.h) the thread serves the draft channel too, the target's
/// ring first (a counter records the admissions that found both pending). A draft-only engine
/// (CpuExpertConfig::draft_only) has only that source.
// The int64 moments per output part of the engine's live fit (BasicCpuExpertEngine::fit_moments).
constexpr int kCpuFitMoments = 9;

template <BuildPolicy Build>
class BasicCpuExpertEngine {
 public:
  /// The format's kernel type (cpu_experts/binding.hpp), its layers and the engine's inputs on them.
  using Kernel = typename Build::CpuKernel;
  using Layer = typename Kernel::Layer;
  using Layers = CpuExpertLayers<Layer>;
  using Config = CpuExpertConfig<Layer>;
  using DraftSource = DraftSourceOf<Layer>;

  /// A record has at most one CPU-hit job and one job per CPU miss: the ring holds every job that can be outstanding.
  static constexpr size_t kRing =
      std::bit_ceil(static_cast<size_t>(wire::Wire::kDemandRecords) * (wire::Wire::kLanes + 1));

  BasicCpuExpertEngine(Config config, std::string prefix, std::string thread_name)
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
  /// The tables and geometry it was enabled with: the RAM prefetch's scorer reads the staged inputs through them.
  const Config& config() const {
    return config_;
  }

  /// The tokens that route calibration lane `lane` of a `tokens`-token verify: token lane % tokens, and for every
  /// fourth lane the next token too. 1.25 tokens per lane is the 1.23 the target's CPU jobs measure (cpu_shape events,
  /// DSpark production, 2026-10-07): a lane routed by all 6 tokens is three kernel chunks instead of one, and priced
  /// the CPU at 3x.
  static uint32_t calibration_lane_mask(int64_t lane, int64_t tokens) {
    uint32_t mask = 1u << (lane % tokens);
    if (lane % 4 == 3) mask |= 1u << ((lane + 1) % tokens);
    return mask;
  }

  /// Calibration only (the tier's owner, copy engine not armed, so no record writes or reads the row): makes `row`'s
  /// token table that of a verify of `tokens` tokens routing lane j by calibration_lane_mask(j) with weight 1, so a
  /// per_token CpuJob whose lanes[i] = i runs `tokens` rows of its k lanes at a served verify's cost. The row's staged
  /// inputs are left as they are: the timing does not depend on them. The tables are the pinned rows the Python
  /// service owns and the post kernel writes (cpu_token_table.h); the const is only the engine's own read-only view of
  /// them.
  ///
  /// `per_lane` above 0 routes each lane by that many consecutive tokens from token lane % tokens instead (the split
  /// model's calibration, split_calibration.h kSplitCellHit): a job of k lanes then runs k * per_lane (token, expert)
  /// pairs.
  void write_calibration_table(int64_t row, int64_t tokens, int64_t per_lane = 0) const {
    constexpr int64_t kLanes = wire::Wire::kLanes;
    if (tokens < 1 || tokens > config_.tokens)
      throw std::runtime_error(
          prefix_ + "calibration tokens " + std::to_string(tokens) + " is not within the rows' 1.." +
          std::to_string(config_.tokens));
    if (per_lane < 0 || per_lane > tokens)
      throw std::runtime_error(
          prefix_ + "calibration tokens per lane " + std::to_string(per_lane) + " is not within 0.." +
          std::to_string(tokens));
    if (tokens == 1) return;
    uint8_t* table = const_cast<uint8_t*>(config_.x_base) + row * config_.x_stride + config_.tokens * config_.x_token_bytes;
    const auto store = [](uint8_t* at, uint32_t v) { std::memcpy(at, &v, sizeof v); };
    const auto mask = [&](int64_t lane) {
      if (per_lane == 0) return calibration_lane_mask(lane, tokens);
      uint32_t bits = 0;
      for (int64_t t = 0; t < per_lane; ++t)
        bits |= 1u << ((lane + t) % tokens);
      return bits;
    };
    store(table, static_cast<uint32_t>(tokens));
    store(table + CpuTokenTable::kLiveTokens, static_cast<uint32_t>(tokens));
    for (int64_t lane = 0; lane < kLanes; ++lane)
      store(table + CpuTokenTable::kHeaderBytes + 4 * lane, mask(lane));
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
    submitted_.store(job.seq, std::memory_order_relaxed);
    kick_.fetch_add(1, std::memory_order_release);
    return true;
  }

  /// For the copy-wait abort, on the watchdog's thread: why job `seq` is not done. Whether the tier owner submitted it
  /// (an unsubmitted one is a CPU miss whose row has not landed, since a record's hit job is submitted before its copy
  /// job), the job `seq` is, if it is in the window (its row, the phase it is in and its age), else the oldest job in
  /// the window, how many jobs the window holds, and the last job done. Reads the clock once.
  std::string stall_state(uint32_t seq) const {
    const bool submitted = static_cast<int32_t>(submitted_.load(std::memory_order_relaxed) - seq) >= 0;
    std::string out = submitted ? " (submitted; the engine is "
                                : " (not submitted: a CPU miss whose row has not landed; the engine is ";
    const Window* window = live_.load(std::memory_order_acquire);
    const WindowSlot* mine = nullptr;
    const WindowSlot* oldest = nullptr;
    int held = 0;
    if (window != nullptr) {
      for (int i = 0; i < kWindowSlots; ++i) {
        const WindowSlot& s = window->slot(i);
        if (!s.active()) continue;
        ++held;
        if (oldest == nullptr || s.admit_ns() < oldest->admit_ns()) oldest = &s;
        if (s.kind() == kTargetJob && s.seq() == seq) mine = &s;
      }
    }
    const WindowSlot* shown = mine != nullptr ? mine : oldest;
    if (shown == nullptr) {
      out += "idle";
    } else {
      const int64_t started = shown->admit_ns();
      char ms[32];
      std::snprintf(ms, sizeof(ms), "%.1f", static_cast<double>(now_ns() - started) / 1e6);
      out += std::string("running ") +
             (shown->kind() == kDraftJob ? "a draft job (record " : "a target job (row ") +
             std::to_string(shown->id()) + ") for " + ms + " ms, in phase " + std::to_string(shown->phase()) + " of " +
             std::to_string(Kernel::kPhases) + (mine != nullptr ? "" : "; job " + std::to_string(seq) + " is not in the window");
    }
    return out + ", the window holds " + std::to_string(held) + " of " + std::to_string(kWindowSlots) +
           ", done through job " + std::to_string(done_.load(std::memory_order_acquire)) + ")";
  }

  /// True once the job with sequence `seq`, or a later one, has finished. The acquire pairs with the release after the
  /// job's output rows, so a CopyDone published after it orders the output before the device reads it.
  bool done(uint32_t seq) const {
    return static_cast<int32_t>(done_.load(std::memory_order_acquire) - seq) >= 0;
  }

  /// Tier owner only: the staged job `seq` may read its rows' second stage. The release pairs with the forward's acquire
  /// before it does, so the rows' bytes the caller fenced before this are seen.
  void open_stage_two(uint32_t seq) {
    stage_gates_[seq % kRing].store(seq, std::memory_order_release);
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
  /// Staged jobs run, those whose second stage had not landed when their gate/up GEMVs were done, and the ns they
  /// waited for it (the instr build only; a prod gate reads no clock, so its wait ns stay 0).
  int64_t staged_jobs() const {
    return staged_jobs_.load(std::memory_order_relaxed);
  }
  int64_t stage_two_waits() const {
    return stage_two_waits_.load(std::memory_order_relaxed);
  }
  int64_t stage_two_wait_ns() const {
    return stage_two_wait_ns_.load(std::memory_order_relaxed);
  }
  /// The window's always-on counters: jobs admitted, spells with work pending and no free slot, and admissions that
  /// found the target ring and the draft channel both pending (the target's went first). Zeros before start().
  cpu_experts::WindowCounters window_counters() const {
    const Window* window = live_.load(std::memory_order_acquire);
    return window != nullptr ? window->counters() : cpu_experts::WindowCounters{};
  }

  /// The split model's live fit (split_model_calibration.py, SplitModelRefiner): per output part (0 hit jobs, 1 miss
  /// jobs), the least-squares moments of every unstaged target job since start, kFitMoments int64 each: jobs, then the
  /// sums of k, pairs, k*k, k*pairs, pairs*pairs, ns, k*ns and pairs*ns, where k is the job's lanes, pairs its (token,
  /// expert) pairs and ns its forward time. Any thread: a consistent snapshot (a seqlock against the engine's update).
  static constexpr int kFitMoments = kCpuFitMoments;
  void fit_moments(int64_t* out) const {
    for (;;) {
      const uint32_t before = fit_version_.load(std::memory_order_acquire);
      if (before & 1u) {
        _mm_pause();
        continue;
      }
      for (int part = 0; part < 2; ++part)
        for (int i = 0; i < kFitMoments; ++i)
          out[part * kFitMoments + i] = fit_[part][i].load(std::memory_order_relaxed);
      std::atomic_thread_fence(std::memory_order_acquire);
      if (fit_version_.load(std::memory_order_relaxed) == before) return;
    }
  }

  static constexpr const char* kDraftPrefix = DraftExperts<Build>::kPrefix;
  using DraftStats = typename DraftExperts<Build>::Stats;

  /// Makes the DSpark draft channel this engine's second job source (draft_experts.h): from then on the thread also
  /// serves each posted record, one job at a time with the target's, never two at once. Starts the watchdog, which
  /// fail-stops a job of either kind that runs past fatal_wait_ns and a posted record left unserved that long. After
  /// start(); once.
  void attach_draft(std::unique_ptr<DraftSource> source, int64_t fatal_wait_ns) {
    if (!thread_.joinable()) throw std::runtime_error(std::string(kDraftPrefix) + "attach after the engine started");
    if (fatal_wait_ns <= 0) throw std::runtime_error(std::string(kDraftPrefix) + "the fatal wait must be positive");
    draft_.attach(std::move(source));  // the thread sees it within a quantum
    fatal_wait_ns_ = fatal_wait_ns;
    watchdog_ = std::thread([this] { watch(); });
  }

  /// Stops serving the draft channel: the thread finishes a draft job in progress and lets go of the source, then the
  /// gate a wait still holds closed is opened (no completer is left). The watchdog runs on, so a hung job still
  /// fail-stops. Idempotent; stop() opens the gate too.
  void detach_draft() {
    draft_.detach();
  }

  DraftStats draft_stats() const {
    return draft_.stats();
  }

  /// Stops and joins the thread, which admits nothing more and drains the window (a gate still shut opens), then the
  /// watchdog (a hung job still fail-stops meanwhile), then the window and the team, and opens a draft gate a wait still
  /// holds closed; idempotent.
  void stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_release);
    thread_.join();
    watchdog_stop_.store(true, std::memory_order_seq_cst);
    if (watchdog_.joinable()) watchdog_.join();
    live_.store(nullptr, std::memory_order_release);
    window_.reset();
    team_.reset();
    draft_.open_gate();
  }

 private:
  /// Refuses a config the forwards would refuse later, since a refused forward aborts the process.
  void validate() const {
    const Config& c = config_;
    if constexpr (!Kernel::usable()) throw std::runtime_error(prefix_ + "no CPU expert kernel");
    if (!c.draft_only) {
      if (Kernel::max_routes() < wire::Wire::kLanes || Kernel::max_rows() < 1)
        throw std::runtime_error(
            prefix_ + "kernel " + Kernel::name() + " takes " + std::to_string(Kernel::max_routes()) +
            " routes per forward, a job up to " + std::to_string(wire::Wire::kLanes));
      if (c.layers == nullptr) throw std::runtime_error(prefix_ + "no CPU expert layers");
      if (c.x_base == nullptr || c.out_base == nullptr || c.x_stride <= 0 || c.out_stride <= 0)
        throw std::runtime_error(prefix_ + "the CPU expert input and output rows are required");
      if (c.hidden <= 0 || c.out_stride < c.hidden * static_cast<int64_t>(sizeof(float)))
        throw std::runtime_error(prefix_ + "the CPU expert output rows are smaller than the hidden size");
      if (c.tokens < 1 || c.tokens > CpuTokenTable::kMaxTokens || (c.tokens > 1 && c.x_token_bytes < 2 * c.hidden))
        throw std::runtime_error(prefix_ + "the CPU expert rows hold 1-32 tokens of the hidden size");
      // A per_token job runs `tokens` rows through one forward: a kernel with fewer would fail at the first verify.
      if (c.tokens > Kernel::max_rows())
        throw std::runtime_error(
            prefix_ + "kernel " + Kernel::name() + " takes " + std::to_string(Kernel::max_rows()) +
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

  /// Jobs in the window (the design's D7): a ring job's wait for a second, overlapping one is bounded by the oldest
  /// job's remaining phases, and a wider window only holds more jobs between whose phases the workers split.
  static constexpr int kWindowSlots = 4;
  static constexpr int32_t kTargetJob = 1, kDraftJob = 2;

  /// The window's hooks: the engine's completion and, in the instr build, the per-run trace.
  struct Hooks {
    BasicCpuExpertEngine* engine;
    template <class S>
    void on_complete(S& s) noexcept {
      engine->complete(s);
    }
    template <class S>
    void on_run(S& s, int phase, int t0, int t1, int worker, uint64_t cycles) noexcept
      requires(Build::kMetrics)
    {
      engine->ran(s, phase, t0, t1, worker, cycles);
    }
  };
  using Window = cpu_experts::JobWindow<Kernel, kWindowSlots, Hooks>;
  using WindowSlot = typename Window::Slot;

  /// The storage one window slot's job reads while it runs, and what its completion needs; indexed by the slot.
  struct SlotAux {
    CpuJob job;           // a ring job: the call's slots and weights for a one-token row point into it
    bool draft_job = false;
    int64_t pairs = 0;    // the ring job's (token, expert) pairs
    // A per-token job's expanded slots and weights, [tokens][k].
    std::array<int32_t, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> slots{};
    std::array<float, CpuTokenTable::kMaxTokens * wire::Wire::kLanes> weights{};
    typename DraftExperts<Build>::Prepared draft;  // a draft record's call and what its completion needs
  };

  /// How long one idle wait lasts before the thread re-reads the submit word, the draft head, stop() and
  /// detach_draft(), which no host thread rings (or, with jobs in the window, rings the window's word): the bound on
  /// noticing any of them. Short against a job (0.5 ms and up), long against the clock read that ends it.
  static constexpr int64_t kIdleQuantumNs = 2'000;

  void run() {
    std::string error = pin();
    if (error.empty()) {
      try {
        team_.emplace(config_.cores, config_.threads, &Kernel::warm, config_.keep_warm_ns, thread_name_.c_str());
        window_.emplace(*team_, Hooks{this});
        window_->publish_done_to(&done_, kTargetJob);
        tsc_ns_ = cpu_experts::tsc_ns_per_cycle();
      } catch (const std::exception& e) {
        error = std::string("CPU expert team: ") + e.what();
      }
    }
    started_.set_value(error);
    if (!error.empty()) return;
    live_.store(&*window_, std::memory_order_release);
    window_->start();
    int64_t warm_until = 0;  // the register-work window after the last task; 0 before the first
    bool was_running = false;
    while (!stop_.load(std::memory_order_acquire)) {
      drain_ended();
      DraftSource* draft = draft_.source();
      // Read before the pops and the step: a submit or a phase change after it moves the word the idle wait watches.
      const uint32_t kick = kick_.load(std::memory_order_acquire);
      const uint32_t seen = window_->word();
      const uint32_t record = draft != nullptr ? draft_.pending(*draft) : 0u;
      const bool admitted = admit_pending(draft, record);
      const cpu_experts::WindowStep step = window_->step(0);
      if (admitted || step == cpu_experts::WindowStep::kRan) {
        was_running = true;
        continue;
      }
      if (was_running) {
        warm_until = cpu_experts::team_now_ns() + config_.keep_warm_ns;
        was_running = false;
      }
      // Idle, as the workers are on the window's word: register work until warm_until, then PAUSE, one quantum at a
      // time. With jobs in the window it is their word, which their phases move; empty, the submit word.
      if (window_->any_active())
        window_->idle_wait(seen, true, warm_until);
      else
        team_->wait(reinterpret_cast<const uint32_t*>(&kick_), kick, warm_until, now_ns() + kIdleQuantumNs);
    }
    window_->close();
    window_->drain();
    window_->join();
    drain_ended();
    draft_.leave();
  }

  /// Admits one job into a free slot: the target ring's oldest, else the draft channel's record. True when it did.
  bool admit_pending(DraftSource* draft, uint32_t record) {
    const int slot = window_->reserve();
    if (slot < 0) {
      // One count per spell, not per spin of the loop.
      if (!full_noted_ && (!jobs_.empty() || record != 0)) {
        window_->note_full();
        full_noted_ = true;
      }
      return false;
    }
    full_noted_ = false;
    CpuJob job;
    if (jobs_.pop(&job)) {
      if (record != 0) window_->note_both_pending();
      admit_target(slot, job);
      return true;
    }
    if (record == 0) return false;
    admit_draft(slot, *draft, record);
    return true;
  }

  /// Snapshots the faults and switches of jobs that have finished, on this thread: they are its own counters. Before a
  /// slot's storage is reused, and the loop's top: a slot is Free only after its job's bit is set.
  void drain_ended() {
    if constexpr (Build::kMetrics) {
      uint32_t ended = ended_.exchange(0, std::memory_order_acquire);
      for (; ended != 0; ended &= ended - 1) {
        const SlotAux& a = aux_[static_cast<size_t>(std::countr_zero(ended))];
        if (a.draft_job) {
          draft_.resources_end(a.draft);
        } else {
          trace_.resources("cpu_faults_end", "cpu_switches_end", a.job.row, 0, a.job.seq);
        }
      }
    }
  }

  /// Builds a ring job's call in its slot's storage and admits it: the token expansion, the instr build's check and
  /// the kernel's prepare. A job the kernel refuses fails stop.
  void admit_target(int slot, const CpuJob& job) {
    drain_ended();
    SlotAux& a = aux_[static_cast<size_t>(slot)];
    a.job = job;
    a.draft_job = false;
    if constexpr (Build::kMetrics) trace_.emit("cpu_start", job.row, 0, job.seq, -1, job.part, job.k);
    if constexpr (Build::kMetrics) trace_.resources("cpu_faults_start", "cpu_switches_start", job.row, 0, job.seq);
    using cpu_experts::OutMode;
    cpu_experts::ForwardCall call;
    call.rows = 1;
    call.k = job.k;
    call.x = config_.x_base + job.row * config_.x_stride;
    call.slots = a.job.slots;
    call.weights = a.job.weights;
    call.out =
        reinterpret_cast<float*>(config_.out_base + job.row * config_.out_stride + job.part * config_.out_part_stride);
    call.out_mode = job.accumulate ? OutMode::kAccumulate
                    : config_.store_out ? OutMode::kOverwriteByStore
                                        : OutMode::kOverwrite;
    // A staged job's gate phase waits, once its first stage is done, until the tier opens the job's gate; the window
    // holds the job meanwhile and runs the others. The wait is the job's own time, so the watchdog still times it;
    // stop() alone ends it early, at shutdown.
    cpu_experts::StageGate gate{&stage_gates_[job.seq % kRing], job.seq, &stop_};
    if constexpr (Build::kMetrics) gate.timed = true;
    a.pairs = job.k;  // the job's (token, expert) pairs, for the split model's fit: one a lane on a one-token row
    if (job.per_token && config_.tokens > 1) {
      call.rows = static_cast<int32_t>(expand_tokens(job, static_cast<const uint8_t*>(call.x), a));
      call.slots = a.slots.data();
      call.weights = a.weights.data();
      a.pairs = 0;
      for (int i = 0; i < call.rows * call.k; ++i)
        a.pairs += call.slots[i] >= 0;
    }
    if constexpr (Build::kMetrics) {
      if (trace_.enabled()) {
        int live = 0;
        for (int i = 0; i < call.rows * call.k; ++i) live += call.slots[i] >= 0;
        trace_.emit("cpu_shape", job.row, 0, job.seq, -1, call.rows, call.k, live);
      }
    }
    try {
      const Layer* layer = config_.layers->get(job.row);
      if (layer == nullptr) throw std::invalid_argument("the row has no registered layer");
      if (config_.check_calls) Kernel::check(*layer, call);
      // A job writes (row, part); one still in the window on the same output goes first (an accumulating miss job adds
      // to what the one before it left).
      const uint64_t out_key = static_cast<uint64_t>(job.row) * 2 + static_cast<uint64_t>(job.part != 0);
      if (!window_->admit_at(slot, job.seq, kTargetJob, *layer, call, out_key, job.staged ? &gate : nullptr, job.row))
        throw std::logic_error("the window refused the job");
    } catch (const std::exception& e) {
      fail_stop(prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed: " + e.what());
    }
  }

  /// Builds the draft record's call in its slot's storage and admits it.
  void admit_draft(int slot, const DraftSource& d, uint32_t record) {
    drain_ended();
    SlotAux& a = aux_[static_cast<size_t>(slot)];
    a.draft_job = true;
    draft_.prepare(d, record, a.draft);
    draft_.admitting(a.draft);
    try {
      if (!window_->admit_at(slot, record, kDraftJob, *a.draft.layer, a.draft.call, cpu_experts::kNoOutKey, nullptr,
                             record))
        throw std::logic_error("the window refused the record");
    } catch (const std::exception& e) {
      fail_stop(std::string(kDraftPrefix) + "forward of record " + std::to_string(record) + " (stage " +
                std::to_string(a.draft.stage) + ") failed: " + e.what());
    }
  }

  struct SpinGuard {
    explicit SpinGuard(std::atomic_flag& flag) : flag_(flag) {
      while (flag_.test_and_set(std::memory_order_acquire)) _mm_pause();
    }
    ~SpinGuard() { flag_.clear(std::memory_order_release); }
    std::atomic_flag& flag_;
  };

  /// The window's completion hook: a job has run all its phases, on whichever worker finished it, before done_ can
  /// pass it. Completions run one at a time (their counters and the fit's seqlock have a single writer at once). A
  /// job the kernel refused (status 2) or failed fails stop, naming its row.
  void complete(const WindowSlot& s) noexcept {
    SpinGuard guard(complete_lock_);
    const SlotAux& a = aux_[static_cast<size_t>(s.index())];
    const int status = s.status();
    const int64_t elapsed = cpu_experts::team_now_ns() - s.admit_ns();
    if (s.kind() == kDraftJob) {
      if (status != 0)
        fail_stop(std::string(kDraftPrefix) + "forward of record " + std::to_string(s.id()) + " (stage " +
                  std::to_string(a.draft.stage) + ") failed: " + status_text(status));
      draft_.finish(a.draft, elapsed);
    } else {
      const CpuJob& job = a.job;
      if (status != 0)
        fail_stop(prefix_ + "CPU expert forward of row " + std::to_string(job.row) + " failed: " + status_text(status));
      if (job.staged) record_stage_two(job, s);
      add(compute_ns_, elapsed);
      add(jobs_done_, 1);
      add(lanes_done_, job.k);
      // A staged job's time includes its wait for the second stage's read: not a job's cost. The rest is the
      // workers' time on its runs, over the team: what it would take with the team to itself.
      if (!job.staged) {
        const int64_t worker_ns = static_cast<int64_t>(static_cast<double>(s.worker_tsc()) * tsc_ns_);
        record_fit(job.part == 0 ? 0 : 1, job.k, a.pairs, worker_ns / window_->workers());
      }
      if constexpr (Build::kMetrics) trace_.emit("cpu_end", job.row, 0, job.seq, -1, job.part, job.k);
    }
    if constexpr (Build::kMetrics) ended_.fetch_or(1u << s.index(), std::memory_order_release);
  }

  /// Instr builds, per run of tasks: which slot ran which phase on which worker, for how many TSC cycles.
  void ran(const WindowSlot& s, int phase, int t0, int t1, int worker, uint64_t cycles) noexcept {
    trace_.emit("cpu_run", s.id(), static_cast<uint64_t>(s.index()), s.seq(), worker, phase, t1 - t0,
                static_cast<int64_t>(cycles));
  }

  static const char* status_text(int status) {
    return status == 2 ? "the job refused its input (status 2)" : "the job failed";
  }

  /// A record's job on a multi-token row: each token's slots and weights from the row's token table
  /// (cpu_token_table.h), -1 where the token does not route the lane's expert (the kernel skips only slot -1, never a
  /// zero weight), into the slot's table. Returns the record's tokens; fails stop on a count the row cannot hold.
  uint32_t expand_tokens(const CpuJob& job, const uint8_t* x, SlotAux& a) {
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
        a.slots[t * job.k + i] = routed ? job.slots[i] : -1;
        a.weights[t * job.k + i] = routed ? std::bit_cast<float>(bits) : 0.0f;
      }
    return tokens;
  }

  /// A staged job's gate after its last phase: the window's stats for it and the cpu_stage_two event, stamped when the
  /// gate opened. A kernel with no gate phase never waited, and would have run Down on rows still landing: its output
  /// is refused.
  void record_stage_two(const CpuJob& job, const WindowSlot& gate) {
    if (gate.gate_calls != 1) {
      fail_stop(prefix_ + "kernel " + Kernel::name() + " waited " + std::to_string(gate.gate_calls) + " times for row " +
                std::to_string(job.row) + "'s second stage, not once");
      return;
    }
    add(staged_jobs_, 1);
    if (gate.gate_spun) {
      add(stage_two_waits_, 1);
      add(stage_two_wait_ns_, gate.gate_waited_ns);
    }
    if constexpr (Build::kMetrics)
      trace_.emit_at("cpu_stage_two", gate.gate_called_ns + gate.gate_waited_ns, job.row, 0, job.seq, -1,
                     gate.gate_waited_ns, gate.gate_spun ? 1 : 0);
  }

  static uint32_t load_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
  }

  /// Every 20 ms, once a draft source is attached: fail-stops a job of either kind in the window that has been there
  /// for fatal_wait_ns, and a posted draft record that has stood unserved that long.
  void watch() {
    uint32_t watched = 0;
    int64_t since = 0;
    const std::string wait = std::to_string(static_cast<double>(fatal_wait_ns_) * 1e-9);
    while (!watchdog_stop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      if (watchdog_stop_.load(std::memory_order_acquire)) return;
      const int64_t now = now_ns();
      if (const Window* window = live_.load(std::memory_order_acquire); window != nullptr) {
        for (int i = 0; i < kWindowSlots; ++i) {
          const WindowSlot& s = window->slot(i);
          // The state first: an Active slot's admit time is its job's or a later one's, never an earlier one's.
          if (!s.active()) continue;
          const int64_t started = s.admit_ns();
          if (now - started < fatal_wait_ns_) continue;
          if (s.kind() == kDraftJob)
            fail_stop(std::string(kDraftPrefix) + "record " + std::to_string(s.id()) + " incomplete after " + wait +
                      " s (fatal wait)");
          fail_stop(prefix_ + "the CPU job of row " + std::to_string(s.id()) + " incomplete after " + wait +
                    " s (fatal wait)");
        }
      }
      const uint32_t head = draft_.unserved_head();
      if (head == 0) {
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

  /// A counter only one thread writes at a time (the completion lock's holder): no locked add needed.
  static void add(std::atomic<int64_t>& counter, int64_t n) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  /// Adds one job to part's fit moments (fit_moments), inside the seqlock's odd version.
  void record_fit(int part, int64_t k, int64_t pairs, int64_t ns) {
    const uint32_t version = fit_version_.load(std::memory_order_relaxed);
    fit_version_.store(version + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    const int64_t terms[kFitMoments] = {1, k, pairs, k * k, k * pairs, pairs * pairs, ns, k * ns, pairs * ns};
    for (int i = 0; i < kFitMoments; ++i)
      add(fit_[part][i], terms[i]);
    fit_version_.store(version + 2, std::memory_order_release);
  }

  Config config_;
  std::string prefix_;
  std::string thread_name_;
  [[no_unique_address]] JobTrace<Build::kMetrics> trace_;
  [[no_unique_address]] DraftDelayTrigger<kDraftDelayTriggerOnly && !Build::kMetrics> delay_trigger_{thread_name_};
  std::thread thread_;
  std::promise<std::string> started_;  // the thread's pin result, for start()
  SpscRing<CpuJob, kRing> jobs_;
  alignas(64) std::atomic<uint32_t> kick_{0};  // moved by every submit; the thread's idle wait watches it
  // The run thread's team and window, made at its start and kept until stop(): the watchdog and stall_state() read the
  // window after the thread exits, through live_. The window is declared after the team, so it goes first.
  std::optional<cpu_experts::Team> team_;
  std::optional<Window> window_;
  std::atomic<const Window*> live_{nullptr};
  double tsc_ns_ = 0;                          // ns per TSC cycle, for the workers' time in a job's fit
  std::atomic_flag complete_lock_;             // one completion at a time
  std::atomic<uint32_t> ended_{0};             // instr build: slots whose job finished, for the thread's end snapshots
  bool full_noted_ = false;                    // this thread: the window's full spell is counted
  std::array<SlotAux, kWindowSlots> aux_{};
  uint32_t claimed_ = 0;  // tier owner only
  std::atomic<uint32_t> submitted_{0};  // the last job submitted (the tier owner writes), for stall_state()
  std::atomic<uint32_t> done_{0};
  std::atomic<int64_t> jobs_done_{0};
  std::atomic<int64_t> lanes_done_{0};
  std::atomic<int64_t> compute_ns_{0};
  std::atomic<int64_t> staged_jobs_{0}, stage_two_waits_{0}, stage_two_wait_ns_{0};
  // The split model's fit moments per part (fit_moments) and their seqlock version, odd while the thread updates them.
  std::array<std::array<std::atomic<int64_t>, kFitMoments>, 2> fit_{};
  std::atomic<uint32_t> fit_version_{0};
  // Staged jobs: each one's gate word, at its sequence modulo the ring (the tier owner stores, the thread reads).
  std::array<std::atomic<uint32_t>, kRing> stage_gates_{};
  std::atomic<bool> stop_{false};
  // The DSpark draft source and its read of the channel (attach_draft): the team's threads and cores, and the
  // engine's trace and trigger, are its.
  DraftExperts<Build> draft_{trace_, delay_trigger_};
  std::atomic<bool> watchdog_stop_{false};
  int64_t fatal_wait_ns_ = 0;
  std::thread watchdog_;
};

}  // namespace sglang::expert_stream
