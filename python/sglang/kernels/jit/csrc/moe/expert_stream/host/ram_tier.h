// Tier/RamTier: slot state and admission.
#pragma once

#include "../row_layout.h"
#include "copy_engine.h"

namespace sglang {
namespace expert_stream {

struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (LOADING or READY) or -1
  std::vector<uint8_t> hot;
  std::vector<uint32_t> leases;  // GPU-reader leases per slot; 0 frees a slot for eviction
  std::vector<uint8_t> filling;      // a prefill fill is still writing this kReady slot: never a victim, never released
  // Prefill share (plan 2026-09-25-dsv41-prefill-eviction): 1 while a row a prefill admitted has not been used by
  // decode. `owned` counts them. Only take_admit_slot_locked sets it, so it stays all zero with the share off.
  std::vector<uint8_t> prefill_owned;
  int64_t owned = 0;
  // Written by the owner's serve() only (a relaxed store through std::atomic_ref), read relaxed by layer_rows from
  // any thread.
  int64_t rows_demand = 0;
};

// What a request could take from a tier, counted without taking anything.
struct VictimCensus {
  int64_t free = 0;       // FREE slots
  int64_t evictable = 0;  // READY, not hot, not requested, not leased
  int64_t leased = 0;     // as evictable, but leased: they would be victims if the leases retired
};

// The pinned-slot bookkeeping of every streamed layer (plan D12) and the service of one
// request at a time. pump_demand is called by one caller at a time: a test's pump(), or the Task 12 thread. The Python-facing methods follow the single-owner rule (plan
// 2026-09-29-hotpath-zero-overhead Task 13), stated at "Python-facing bookkeeping" below.
template <class Source>
class RamTier {
 public:
  using Layout = typename Source::LayoutType;
  using Build = typename Source::BuildType;  // ProdBuild or InstrBuild (build_policy.h)
  using Engine = CopyEngine<Build, RamTier>;
  // The copy engine calls copy_completed, copy_acked and copy_failed, and reads stats_ for its latency max.
  friend Engine;

  RamTier(
      uint8_t* page,
      int32_t* slot_map,
      uint8_t* lease,
      int64_t lease_bytes,
      Tables tables,
      std::vector<int64_t> capacity,
      bool direct,
      uint8_t* hot_page,
      int64_t hot_bytes)
      : page_(page),
        map_(slot_map),
        lease_(lease),
        hot_page_(hot_page),
        layers_(tables.layers),
        experts_(tables.experts),
        reader_(std::move(tables), direct),
        tiers_(static_cast<size_t>(layers_)) {
    hot_stride_ = ((kHotHeaderBytes + (experts_ + 7) / 8 + kHotAlignment - 1) / kHotAlignment) * kHotAlignment;
    if (hot_page_ != nullptr && hot_bytes != kHotRecords * hot_stride_)
      throw std::runtime_error(error_prefix<Layout>() + "hot bitmap sidecar size disagrees with expert count");
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      tier.capacity = capacity[row];
      tier.slot_to_expert.assign(tier.capacity, -1);
      tier.state.assign(tier.capacity, kFree);
      tier.stamp.assign(tier.capacity, 0);
      tier.expert_slot.assign(experts_, -1);
      tier.hot.assign(experts_, 0);
      tier.leases.assign(tier.capacity, 0);
      tier.filling.assign(tier.capacity, 0);
      tier.prefill_owned.assign(tier.capacity, 0);
    }
    init_lease_block(lease_bytes);
    // The service grants every miss lane LOADING and streams it piece by piece: the reader must publish pieces.
    reader_.set_piece_stream(true);
    // The request path's buffers, sized once (spec A2, A10): nothing on it grows after construction.
    hot_scratch_.assign(static_cast<size_t>((experts_ + 7) / 8), 0);
    packed_.reserve(kWanted);
    piece_targets_.reserve(kWanted);
    int64_t widest = 0;
    for (int64_t c : capacity)
      widest = std::max(widest, c);
    fill_packed_.reserve(static_cast<size_t>(widest));
  }

  // The copy thread's callbacks use the lease block, copy_done_ and the counter blocks, which are destroyed before
  // copy_engine_ would be.
  ~RamTier() {
    fill_join();  // the fill thread reads through reader_ into the slabs; nothing else holds the tier by now
    if (copy_engine_ != nullptr) copy_engine_->stop(5'000'000'000LL);
    if (cpu_ != nullptr) cpu_->stop();  // after the copy thread, its only client
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = load_acquire(page_ + kDemandDone) + 1u;
    if (next_demand_ == 0) next_demand_ = 1;
    return true;
  }

  uint8_t* page() const {
    return page_;
  }
  // The watchdog's hung-request marker (D6): nonzero while a demand or a fill is in service, a new value per episode. The watchdog thread times how long one value persists; the service reads no clock for it.
  uint64_t busy_episode() const {
    return busy_.load(std::memory_order_acquire);
  }
  // The service thread's set-once words (kRunning, kSpinCpu), written by that thread only.
  void set_counter(int index, int64_t value) {
    core_.set(index, value);
  }

  // Counters (plan 2026-09-29-hotpath-zero-overhead D1). A core counter (is_core_counter) lives in the writing thread's
  // own line-private block: count() is the tier's owner's (the service thread, or the caller owning the tier while it
  // is paused or pumped: one writer at a time, handed over by the same edges as the tier), copy_count() the copy
  // thread's. The prefill fill thread counts nothing: its epilogue runs on the owner (finish_fill_owned, Task 15).
  // Every other counter is a metric: InstrBuild keeps it as a shared relaxed atomic, ProdBuild has none.
  template <Counter K>
  void count(int64_t n = 1) {
    count_into<K>(core_, n);
  }
  template <Counter K>
  void copy_count(int64_t n = 1) {
    count_into<K>(copy_core_, n);
  }
  bool threaded() const {
    return threaded_.load();
  }
  // RamThread::start before the thread exists, and RamThread::stop after its join (plan F14): the join is what makes
  // every tier write of the service thread visible, and this release hands that on to a caller's caller_owns().
  void set_threaded(bool threaded) {
    threaded_.store(threaded, std::memory_order_release);
  }

  // ---- The single owner (plan 2026-09-29-hotpath-zero-overhead Task 13; see "Python-facing bookkeeping") ----

  static constexpr int kMaxHotWords = 16;  // set_hot's bitmap: 1024 experts

  // A Python-side call for the tier's owner. A kSnapshot runs `snapshot` there and hands back its value through
  // `result`; kInjectLease answers 0, or -1 for an underflow. `done` (the waiting caller's) becomes 1 once applied, 2
  // if the owner's body threw: a command never throws on the service thread.
  struct Command {
    enum Kind : uint8_t { kSetHot, kSnapshot, kInjectLease };
    Kind kind = kSetHot;
    int64_t row = 0;    // kSnapshot: the snapshot's own argument (lease_entry: the request slot)
    int64_t arg = 0;    // kInjectLease: the slot
    int64_t delta = 0;  // kInjectLease
    uint64_t hot[kMaxHotWords] = {};
    int64_t (*snapshot)(RamTier*, const Command&) = nullptr;
    int64_t* out = nullptr;
    const void* input = nullptr;  // kSnapshot: an argument the caller keeps alive (victim_census's wanted list)
    int64_t* result = nullptr;    // kSnapshot, kInjectLease: the answer
    std::atomic<uint32_t>* done = nullptr;
  };

  // The tier's owner is the service thread while it runs, else the caller: a paused service (parked_, set by the
  // pausing caller once the service has parked) or no thread at all (pump mode, or after stop()'s join).
  bool caller_owns() const {
    return !threaded_.load(std::memory_order_acquire) || parked_.load(std::memory_order_acquire);
  }
  // RamThread::pause (true, after the service parked) and resume (false, before the service may run), both under
  // caller_mutex_: a caller that takes caller_mutex_ afterwards sees who owns the tier.
  void set_parked(bool parked) {
    parked_.store(parked, std::memory_order_release);
  }
  std::mutex& caller_mutex() {
    return caller_mutex_;
  }

  // The owner, between requests: apply every queued Python command, in order. The ring's one consumer is always the
  // current owner: the service thread, or a caller once the service has parked or joined, each handoff ordered by
  // the pause handshake's release/acquire or the join (RamThread). An empty ring costs one relaxed load and a compare.
  void drain_commands() {
    Command command;
    while (commands_.pop(&command))
      apply_command(command);
  }

  // A Python-side call that needs the tier. It runs here when this caller owns the tier (after anything queued before
  // the handoff); otherwise it is queued for the service (drain_commands), and every kind but kSetHot waits for its
  // answer. The ring's producer is whoever holds caller_mutex_. If the service stops before it drains the queue, the
  // caller then owns the tier and drains it itself (stop() clears threaded_ after the join without caller_mutex_, plan
  // F14): nothing queued is left unanswered. A service alive but hung in a read is aborted by the watchdog (D6).
  // The service cannot park meanwhile: a pause takes caller_mutex_, which this caller holds.
  void run_as_owner(Command command) {
    std::atomic<uint32_t> done{0};
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (!caller_owns()) {
      const bool wait = command.kind != Command::kSetHot;
      if (wait) command.done = &done;
      bool queued = false;
      while (!caller_owns()) {
        if (commands_.push(command)) {
          queued = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(20));  // a full ring: wait for room, never drop
      }
      if (queued) {
        if (!wait) return;
        while (done.load(std::memory_order_acquire) == 0) {
          if (caller_owns()) {
            drain_commands();
          } else {
            std::this_thread::sleep_for(std::chrono::microseconds(20));
          }
        }
        command_outcome(done.load(std::memory_order_acquire));
        return;
      }
    }
    drain_commands();  // anything queued before the handoff comes first
    command.done = &done;
    apply_command(command);
    command_outcome(done.load(std::memory_order_acquire));
  }

  // Serve the next posted demand record, if any. True when it handled one.
  bool pump_demand() {
    const uint32_t head = load_acquire(page_ + kDemandHead);
    const bool posted = head != 0 && reached(head, next_demand_);
    // Commands a caller queued before it posted this demand are applied before it is served: the demand head's
    // acquire above made the caller's earlier push visible. The loop drains between requests too (RamThread::run).
    // Pump mode queues nothing (its caller owns the tier), so only the service thread looks.
    if (posted) drain_commands_on_service();
    drain_copy_completions();  // D7: the COPYING leases the copy thread handed back, released before the pass below
    retire_leases();           // first, so that an idle pump still retires
    if (admission_closed_.load()) return false;
    if (!posted) return false;
    // A deferred demand is not looked at again until a lease retires: no stage record, no clock read per poll.
    if (deferred_seq_ == next_demand_ && lease_changes_ == deferred_stamp_) return false;
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq. Only unarmed records
      // lap: the device posts an armed one only after the previous armed one's chain ended, which needs it served.
      count<kOverruns>(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    // A torn record or lane request was overwritten by a later post, so nothing waits on it: skipped, and counted.
    if (!read_record(record, next_demand_, &request) || (request.armed && !read_lane_request(next_demand_, &request))) {
      count<kOverruns>();
    } else {
      if (gpu_hot_mode_.load() && request.armed) {
        if (!(request.row >= 0 && request.row < layers_ && read_gpu_hot(next_demand_, &request) &&
              load_acquire(record + kRecSeq) == next_demand_)) {
          fail_stop(error_prefix<Layout>() + "request " + std::to_string(next_demand_) + ": no hot set for it");
        }
        apply_gpu_hot(request);
      }
      const Defer reason = request.armed ? defers(request) : Defer::kNone;
      if (reason != Defer::kNone) {
        // Held back, not served: return before handle_demand (so the busy episode stays untouched, or the watchdog
        // would count the wait as a hung read) and before the tail (no demand_done, no advance). No stage record is
        // pushed; the first observation time is kept for the one written when it is served.
        if (deferred_seq_ != next_demand_) {
          deferred_seq_ = next_demand_;
          if constexpr (Build::kMetrics) deferred_observed_ns_ = trace_.cur != nullptr ? trace_.cur->observed : 0;
          if (reason == Defer::kRequestSlot) {
            count<kDeferredReuse>();
          } else {
            count<kDeferred>();
          }
        }
        deferred_stamp_ = lease_changes_;
        if constexpr (Build::kMetrics) trace_.cur = nullptr;
        return false;
      }
      if constexpr (Build::kMetrics) {
        if (trace_.cur != nullptr && deferred_seq_ == next_demand_ && deferred_observed_ns_ != 0) {
          trace_.cur->observed = deferred_observed_ns_;
        }
      }
      handle_demand(request);
    }
    deferred_seq_ = 0;
    // demand_done says served, and nothing else: every path that could not serve an armed request aborted inside
    // handle_demand, before this store. The sfence drains the service's non-temporal and write-combining stores (the
    // reader's slab bytes) ahead of the release, which then publishes every RowResult and PieceMask of this request.
    _mm_sfence();
    store_release(page_ + kDemandDone, next_demand_);
    end_stage();
    next_demand_ = skip_zero(next_demand_ + 1u);
    return true;
  }

  // ---- Stage trace: one StageRecord per served request, drained by Python ----

  // Allocates the ring, then turns the trace on. Before the service thread starts, so the flag
  // never flips under a request being served. The instrumented build only (spec M9).
  void enable_trace(size_t capacity) {
    if constexpr (!Build::kMetrics) {
      (void)capacity;
      throw_no_trace();
    } else {
      if (threaded_.load())
        throw std::runtime_error(error_prefix<Layout>() + "enable the stage trace before the service thread starts");
      std::lock_guard<std::mutex> guard(trace_.mutex);
      trace_.ring = std::make_unique<StageRing>(capacity);
      trace_.on.store(true, std::memory_order_release);
    }
  }

  // Up to `max` records into `out` (stage_words() int64 each); returns how many. The count of records
  // dropped for a full ring is `trace_dropped()`.
  int64_t drain_trace(StageRecord* out, int64_t max) {
    if constexpr (!Build::kMetrics) {
      (void)out;
      (void)max;
      throw_no_trace();
    } else {
      std::lock_guard<std::mutex> guard(trace_.mutex);
      return trace_.ring ? trace_.ring->drain(out, max) : 0;
    }
  }

  int64_t trace_dropped() {
    if constexpr (!Build::kMetrics) {
      throw_no_trace();
    } else {
      std::lock_guard<std::mutex> guard(trace_.mutex);
      return trace_.ring ? trace_.ring->dropped() : 0;
    }
  }

  // ---- Python-facing bookkeeping: the single-owner rule (plan 2026-09-29-hotpath-zero-overhead Task 13) ----
  //
  // Everything in the tier but its atomics has exactly one owner at a time: the service thread while it runs; the
  // Python caller that paused it, from the moment the service parks until resume() (RamThread::pause sets parked_);
  // the caller of pump() when there is no thread. An unpaused Python call therefore takes one of three forms:
  //   - a lock-free read of published words: mapping, counters, busy_episode, layer_rows;
  //   - a command through run_as_owner, which the service drains between requests: set_hot, inject_lease, and the
  //     snapshots slot_info, slot_to_expert, lease_entry, lru_order, victim_census;
  //   - a refusal, "needs the service thread paused": has, touch, assign, release, fill_begin.
  // On the owner every one of them runs directly, after draining whatever an earlier caller queued.
  //
  // caller_mutex_ serializes Python-side callers against each other only; the service, copy and fill threads never
  // take it. The tier has no other lock (Task 15): single ownership is what keeps its state consistent. The copy
  // thread touches no tier state (Task 14: it hands its completions back through copy_done_), nor does the prefill
  // fill thread (Task 15: it drives the reader and publishes fill_landed_/fill_state_; its epilogue runs on the owner,
  // finish_fill_owned, after the join).

  bool has(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("contains");
    drain_commands();
    return tiers_[row].expert_slot[expert] >= 0;
  }

  void touch(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("touch");
    drain_commands();
    Tier& tier = tiers_[row];
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0) {
      tier.stamp[slot] = ++tick_;
      // under a share the touch is the prefill's own hit
      if (prefill_share_.load(std::memory_order_relaxed) == 0) disown_locked(tier, slot);
    }
  }

  // A slot for a Python-side read; the map entry is published at once (the device is idle
  // and the thread paused when an eager path calls this). evicted: -1 none, -2 already held.
  int64_t assign(int64_t row, int64_t expert, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("assign");
    drain_commands();
    Tier& tier = tiers_[row];
    if (tier.expert_slot[expert] >= 0) {
      *evicted = -2;
      return tier.expert_slot[expert];
    }
    const int64_t slot = take_admit_slot_locked(row, protect, fallback, evicted);
    if (slot < 0) return -1;
    tier.slot_to_expert[slot] = static_cast<int32_t>(expert);
    tier.state[slot] = kReady;
    tier.stamp[slot] = ++tick_;
    tier.expert_slot[expert] = static_cast<int32_t>(slot);
    publish_map(row, expert, static_cast<int32_t>(slot));
    count<kVersion>();
    return slot;
  }

  void release(int64_t row, int64_t slot) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("release");
    drain_commands();
    if (tiers_[row].state[slot] == kLoading) {
      // The service is filling it and will publish it; freeing it would hand it out twice.
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while it is loading");
    }
    if (leased_locked(tiers_[row], slot)) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while it is leased");
    }
    if (tiers_[row].filling[slot]) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while a fill writes it");
    }
    release_locked(row, slot);
    count<kVersion>();
  }

  // ---- Prefill fills (SGLANG_DSV41_ENABLE_PREFILL_FILLS, plan 2026-09-25-dsv41-prefill-fills) ----
  //
  // An eager caller that holds the pause (or pumps, with no thread) claims slots for `experts` of `row` in order, until
  // one cannot be taken (take_slot_locked: never a hot, leased or filling row, and a `protect`ed one only with
  // `fallback`), and one helper thread reads the claimed rows through the service's reader while the caller gathers.
  // A claimed slot is kReady and mapped at once, as assign() leaves it, and flagged filling until the read ends: no
  // admission can evict it and release() refuses it. The read's progress publishes how many rows have landed, as a
  // prefix of the claim order (fill_wait). The helper holds a busy episode, so the watchdog aborts a hung fill the way
  // it aborts a hung demand. fill_end() joins it; the service thread's resume() joins it first too, so the service
  // thread and a fill never use the reader at once. The helper touches no tier state (ownership rule 3): the claimed
  // slots stay filling, and a failed fill's unlanded rows stay mapped, until the owner joins it and runs the epilogue
  // (fill_join, finish_fill_owned). Returns the count claimed; slots[i] is expert i's slot.
  int64_t fill_begin(
      int64_t row,
      const std::vector<int32_t>& experts,
      const std::vector<int32_t>& protect,
      bool fallback,
      int64_t* slots,
      int64_t* evictions) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("a prefill fill");
    drain_commands();
    if (fill_thread_.joinable()) throw std::runtime_error(error_prefix<Layout>() + "a prefill fill is already running");
    std::vector<int32_t> claimed;
    std::vector<int64_t> taken;
    *evictions = 0;
    {
      Tier& tier = tiers_[row];
      for (const int32_t expert : experts) {
        if (tier.expert_slot[expert] >= 0) {
          throw std::runtime_error(
              error_prefix<Layout>() + "fill of expert " + std::to_string(expert) + " that holds a slot");
        }
        int64_t evicted = -1;
        // A prefetch (no fallback) stops at the prefill share, leaving the rest to gather_rows' chunked admission,
        // which evicts the share's rows the earlier chunks have gathered.
        const int64_t slot = take_admit_slot_locked(row, protect, fallback, &evicted, /*stop_at_share=*/!fallback);
        if (slot < 0) break;
        *evictions += evicted >= 0 ? 1 : 0;
        tier.slot_to_expert[slot] = expert;
        tier.state[slot] = kReady;
        tier.stamp[slot] = ++tick_;
        tier.expert_slot[expert] = static_cast<int32_t>(slot);
        tier.filling[slot] = 1;
        publish_map(row, expert, static_cast<int32_t>(slot));
        claimed.push_back(expert);
        taken.push_back(slot);
      }
      if (!taken.empty()) count<kVersion>();
    }
    for (size_t i = 0; i < taken.size(); ++i)
      slots[i] = taken[i];
    fill_landed_.store(0, std::memory_order_release);
    fill_state_.store(taken.empty() ? kFillOk : kFillRunning, std::memory_order_release);
    if (taken.empty()) return 0;
    fill_row_ = row;
    fill_experts_ = std::move(claimed);
    fill_slots_ = std::move(taken);
    fill_result_ = 0;
    fill_unfinished_ = true;  // the epilogue is owed: fill_join runs it once, on the owner, after the join
    fill_thread_ = std::thread([this] { run_fill(); });
    return static_cast<int64_t>(fill_slots_.size());
  }

  // 1 once the first `rows` claimed rows have landed, 0 when the fill failed first, -1 at the deadline.
  int64_t fill_wait(int64_t rows, int64_t timeout_ns) {
    const int64_t deadline = now_ns() + timeout_ns;
    while (true) {
      const int state = fill_state_.load(std::memory_order_acquire);
      if (fill_landed_.load(std::memory_order_acquire) >= rows) return 1;
      if (state != kFillRunning) return 0;  // ended short of `rows`: failed, or never claimed them
      if (now_ns() > deadline) return -1;
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }

  // How many claimed rows have landed so far, a prefix of the claim order (what fill_wait waits on); never blocks.
  int64_t fill_landed() const {
    return fill_landed_.load(std::memory_order_acquire);
  }

  // Joins the fill: 1 when every claimed row landed (or nothing was claimed), 0 when it failed; a failed fill has
  // released its rows that did not land (here, on the caller: finish_fill_owned). Under caller_mutex_, like every
  // other join of fill_thread_ (resume, stop_thread's final_settle), so no two threads join it at once. The caller
  // owns the tier whenever an epilogue is owed: a fill starts only on the owner (fill_begin), and resume() joins it
  // before it hands the tier back.
  //
  // An owed epilogue writes the tier (finish_fill_owned), so a caller that does not own it is refused, not let race the
  // service. RamThread::start refuses a tier with a fill owed, so this is the backstop, not the gate.
  int64_t fill_end() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (fill_unfinished_) require_owner("fill_end with a prefill fill's epilogue owed");
    fill_join();
    return fill_state_.load(std::memory_order_acquire) == kFillFailed ? 0 : 1;
  }

  // The owner (the caller of fill_end, resume() and stop_thread's final_settle, each under caller_mutex_; or the
  // destructor). The join is the synchronization point: every store the fill thread made (fill_result_, the fill's
  // packed flags, the reader's state, the slab bytes) happens-before the epilogue that follows it here.
  void fill_join() {
    if (fill_thread_.joinable()) fill_thread_.join();
    if (fill_unfinished_) finish_fill_owned();
  }

  // True from fill_begin until the owner's fill_join ran the epilogue. Read under caller_mutex_ by a caller that owns
  // the tier (RamThread::start, before the service exists): the flag is the owner's.
  bool fill_owed() const {
    return fill_unfinished_;
  }

  // Rows a prefill may own per layer; 0 (the default, and always with SGLANG_DSV41_ENABLE_PREFILL_SHARE off) admits as
  // take_slot_locked always has. The service sets it before each forward: the share for a prefill, 0 for decode.
  // Any thread: a relaxed store, read relaxed only by the owner's admissions (a paused caller's, in practice).
  void set_prefill_share(int64_t share) {
    if (share < 0) throw std::runtime_error(error_prefix<Layout>() + "a prefill share cannot be negative");
    prefill_share_.store(share, std::memory_order_relaxed);
  }

  void set_gpu_hot(bool on) {
    if (hot_page_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "GPU hot mode needs a sidecar");
    gpu_hot_mode_.store(on);
  }

  // The hot bitmap of `expected`'s record, copied into the service-owned hot_scratch_ (spec A2): not const, it writes
  // that scratch. request->hot_bitmap points into it until the next call.
  bool read_gpu_hot(uint32_t expected, Request* request) {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    const size_t bytes = hot_scratch_.size();
    std::memcpy(hot_scratch_.data(), record + kHotHeaderBytes, bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 && (hot_scratch_[bytes - 1] & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    request->hot_bitmap = hot_scratch_.data();
    return true;
  }

  void apply_gpu_hot(const Request& request) {
    Tier& tier = tiers_[request.row];
    for (int64_t expert = 0; expert < experts_; ++expert)
      tier.hot[expert] = (request.hot_bitmap[expert / 8] >> (expert % 8)) & 1;
  }

  // Copy engine (LEASE_PROTOCOL.md, "Copy engine"): a thread that copies the reservation hold's hit lanes with the
  // DMA engine. `device` < 0 is the CPU test backend (HostCopyBackend). Before the service thread starts; unarmed until
  // arm(). `wait_timeout_ns`: how long a closed gate may hold the decode stream before the watchdog aborts the process
  // (SGLANG_DSV41_RAM_MISS_TIMEOUT_MS).
  void enable_copy_engine(int64_t device, int64_t spin_ns, int64_t wait_timeout_ns) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable the copy engine before the service thread starts");
    if (copy_engine_ != nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is already enabled");
    const std::string copy_prefix = std::string(Layout::kName) + " RAM miss copy engine: ";
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      backend = std::make_unique<HostCopyBackend>();
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), copy_prefix);
    }
    auto engine = std::make_unique<Engine>(
        std::move(backend),
        layers_,
        spin_ns,
        this,
        copy_prefix,
        std::string(Layout::kName) + "-copy-eng");
    if (wait_timeout_ns <= 0) throw std::runtime_error(error_prefix<Layout>() + "the copy-wait timeout must be positive");
    copy_wait_timeout_ns_ = wait_timeout_ns;
    engine->start();
    copy_engine_ = std::move(engine);
  }

  // ---- The copy wait's gate (LEASE_PROTOCOL.md, "Copy engine") ----
  //
  // The decode stream waits (cuStreamWaitValue32) on area C's gate, which CW closes for G, with G's seq in the word,
  // when G has COPYING or CPU lanes. CW opens it itself when CopyDone already carries G after the close; otherwise the
  // copy thread does, after it stores CopyDone (copy_completed). Both open with the same word, and the host only by a
  // CAS from G's own closed word, so a stale open for G meets closed(G + k) and changes nothing.

  static uint32_t gate_word(uint32_t seq, uint32_t low) {
    return ((seq & kLeaseGateSeqMask) << kLeaseGateSeqShift) | low;
  }

  // The watchdog's view: the gate word, whose closed bit is set while a copy wait holds the decode stream.
  uint32_t copy_gate() const {
    return load_acquire(lease_ + kLeaseCopyGate);
  }

  int64_t copy_wait_timeout_ns() const {
    return copy_wait_timeout_ns_;
  }

  // Teardown (the FFI's stop_thread and close): with no copy thread to follow, a copy wait still closed would hold its
  // stream forever. Opening it lets CC run, which traps unless CopyDone is there; the process is ending either way.
  void open_closed_gate() {
    const uint32_t gate = copy_gate();
    if ((gate & 0x80000000u) != 0) cas_gate(gate, gate & ~0x80000000u);
  }

  // Row `row`'s copy table: `entries` rows of {source slab address, destination tensor address, row bytes}. Bit i
  // of `sm_mask` leaves entry i to the copy wait's SM reads (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES).
  // Precondition (row_layout.h's ExpertRowLayout): `entries` holds one row per Layout::kNames entry, in
  // kNames order, so comparing sm_mask (a copy-table entry index) against Layout::kSmallMask (a layout-name
  // index bitmask) below is comparing the same index space.
  void set_copy_table(int64_t row, const int64_t* entries, int64_t count, int64_t dst_rows, int64_t sm_mask) {
    if ((static_cast<uint64_t>(sm_mask) & ~static_cast<uint64_t>(Layout::kSmallMask)) != 0) {
      throw std::runtime_error(
          error_prefix<Layout>() + "sm_mask names a tensor that is not one of the layout's small ones");
    }
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    if (sm_mask < 0 || (count < 63 && (sm_mask >> count) != 0)) {
      throw std::runtime_error(error_prefix<Layout>() + "the SM mask names a copy-table entry that does not exist");
    }
    std::vector<CopyEntry> table;
    for (int64_t i = 0; i < count; ++i) {
      if (entries[3 * i + 2] <= 0) throw std::runtime_error(error_prefix<Layout>() + "a copy-table entry of no bytes");
      table.push_back(
          CopyEntry{
              static_cast<uint64_t>(entries[3 * i]),
              static_cast<uint64_t>(entries[3 * i + 1]),
              entries[3 * i + 2],
              (sm_mask >> i & 1) != 0});
    }
    copy_engine_->set_table(row, std::move(table), dst_rows);
  }

  void arm_copy_engine(bool on) {
    if (on && copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_armed_.store(on, std::memory_order_release);
  }

  // CPU experts (plan 2026-09-29-dsv41-cpu-experts, "Step B").
  // Of a copy-engine request's resident lanes, the grant publishes the last split[n] with tag kLeaseTagCpu,
  // where n is those lanes, at most kLeaseLanes.
  // The copy thread hands them to the CPU expert thread instead of copying them.
  // Needs the copy engine; call before the service thread starts.
  // Only a captured post (kLeaseLrFlagCaptured) gets CPU lanes: only it stages the layer's input row for them.
  void enable_cpu_experts(CpuExpertConfig config, std::vector<int64_t> split) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable CPU experts before the service thread starts");
    if (copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "CPU experts need the copy engine, which completes their lanes");
    if (cpu_ != nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are already enabled");
    config.rows = layers_;
    if (split.size() != static_cast<size_t>(kLeaseLanes) + 1)
      throw std::runtime_error(error_prefix<Layout>() + "the CPU split table has one entry per n = 0..kLeaseLanes");
    for (size_t n = 0; n < split.size(); ++n) {
      if (split[n] < 0 || split[n] > static_cast<int64_t>(n))
        throw std::runtime_error(error_prefix<Layout>() + "the CPU split table must satisfy 0 <= split[n] <= n");
      cpu_split_[n].store(static_cast<uint8_t>(split[n]), std::memory_order_relaxed);
    }
    const std::string prefix = std::string(Layout::kName) + " CPU experts: ";
    auto engine = std::make_unique<CpuExpertEngine>(std::move(config), prefix, std::string(Layout::kName) + "-cpu-exp");
    engine->start();
    cpu_ = std::move(engine);
    copy_engine_->set_cpu(cpu_.get());
  }

  // CPU experts: `row`'s layer handle, from the trait's register_layer. Any time, once per row; until then no grant
  // sends the row to the CPU.
  void set_cpu_layer(int64_t row, int64_t handle) {
    if (cpu_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    cpu_->set_layer(row, handle);
  }

  // CPU experts: replace the split table (P3's re-tuning between requests). Any time; the grant reads each entry once.
  void set_cpu_split(const int64_t* split, int64_t count) {
    if (cpu_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    if (count != static_cast<int64_t>(kLeaseLanes) + 1)
      throw std::runtime_error(error_prefix<Layout>() + "the CPU split table has one entry per n = 0..kLeaseLanes");
    for (int64_t n = 0; n < count; ++n)
      if (split[n] < 0 || split[n] > n)
        throw std::runtime_error(error_prefix<Layout>() + "the CPU split table must satisfy 0 <= split[n] <= n");
    for (int64_t n = 0; n < count; ++n)
      cpu_split_[n].store(static_cast<uint8_t>(split[n]), std::memory_order_relaxed);
  }

  // CPU experts' metrics, from the CPU expert thread: {jobs, lanes, forward ns}. Zeros when off.
  void cpu_stats(int64_t* out) const {
    out[0] = cpu_ != nullptr ? cpu_->jobs() : 0;
    out[1] = cpu_ != nullptr ? cpu_->lanes() : 0;
    out[2] = cpu_ != nullptr ? cpu_->compute_ns() : 0;
  }

  // The owner (the service thread, the pausing caller, or the caller of pump()): release every lease the copy thread
  // handed back, in completion order. Takes no lock: the copy thread touches no tier state (it only pushes), and
  // every other writer of what this changes is the owner itself. An empty ring costs two loads (its own tail, relaxed;
  // the producer's head, acquire) and no store.
  void drain_copy_completions() {
    CopyJob job;
    while (copy_done_.pop(&job))
      release_copied_owned(job);
  }

  // The owner (RamThread::pause, once the service parked): every job handed to the copy thread has completed (or
  // failed), then everything handed back is released here.
  bool wait_copy_idle_owned(int64_t deadline_ns) {
    const bool idle = copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
    drain_copy_completions();
    return idle;
  }

  // Any thread (the FFI's copy_engine_idle): the same wait, then the drain when this caller owns the tier (pump mode,
  // or a paused service). With the service running unpaused it is the service's own next poll that releases (D7).
  bool wait_copy_idle(int64_t deadline_ns) {
    const bool idle = copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (caller_owns()) drain_copy_completions();
    return idle;
  }

  // stop_thread's final settle, on the caller once the service joined (RamThread::stop released threaded_ after the
  // join, so this caller owns the tier). A stop that arrives mid-pause can find a prefill fill still running: it is
  // joined first and its epilogue runs here (fill_join), so the fill thread's last write happens-before the drain and
  // the settle pass, and they run with no other thread touching the tier. caller_mutex_ orders this against the
  // pausing caller, which may still be in an owned call (or its own fill_end) on another thread: no tier lock, and
  // no deadlock with a snapshot waiter, which saw threaded_ clear and needs nothing from this thread to finish.
  void final_settle() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    fill_join();
    drain_copy_completions();
    retire_leases();
  }

  void copy_engine_ballast(uint64_t dst, uint64_t src, int64_t bytes)
    requires(Build::kFaults)
  {
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_engine_->set_ballast(dst, src, bytes);
  }

  HostCopyBackend& host_copy_backend() {
    HostCopyBackend* backend = copy_engine_ != nullptr ? copy_engine_->host_backend() : nullptr;
    if (backend == nullptr) throw std::runtime_error(error_prefix<Layout>() + "no test copy backend");
    return *backend;
  }

  // Shutdown, after the device was synchronised while the service still served (LEASE_PROTOCOL.md, "Shutdown"): no
  // request is in flight, so the service only stops taking new ones. Retirement goes on.
  void close_admission() {
    admission_closed_.store(true);
  }

  // The seqlock read of the device's lane request for `seq`: false when a later request has already overwritten it.
  bool read_lane_request(uint32_t seq, Request* request) const {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* base = lease_ + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
    const uint64_t word = load_acquire64(base + kLeaseLrGen);
    if ((word & 0xFFFFFFFFull) != seq) return false;
    uint32_t count = 0, flags = 0;
    std::memcpy(&count, base + kLeaseLrCount, 4);
    std::memcpy(&flags, base + kLeaseLrFlags, 4);
    int32_t experts[kLeaseLanes];
    int32_t dst[kLeaseLanes];
    float weights[kLeaseLanes];
    std::memcpy(experts, base + kLeaseLrExpert, sizeof(experts));
    std::memcpy(dst, base + kLeaseLrDst, sizeof(dst));
    std::memcpy(weights, base + kLeaseLrWeight, sizeof(weights));
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire64(base + kLeaseLrGen) != word || count > static_cast<uint32_t>(kLeaseLanes)) return false;
    request->gen = word;
    request->lane_experts.assign(experts, experts + count);
    request->lane_dst.assign(dst, dst + count);
    request->captured = (flags & kLeaseLrFlagCaptured) != 0;
    request->lane_weight.assign(weights, weights + count);
    return true;
  }

  enum class Defer { kNone, kVictims, kRequestSlot };

  // Would this armed demand have to wait for a lease to retire? Counts, changes nothing. The request slot rule: the
  // slot's previous lease row must be fully retired before it is reused, since its RowResults and PieceMask words are
  // rewritten. The victim rule: the tier could serve the request only if leased slots were victims (a dry run, because
  // the take loop evicts a victim per call and keeps that eviction when the request then fails).
  Defer defers(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return Defer::kNone;
    if (!request.lane_experts.empty() && outstanding_[static_cast<int64_t>((request.seq - 1u) % kDemandRecords)].active) {
      return Defer::kRequestSlot;
    }
    FixedVec<int32_t, kWanted> wanted;
    for (std::span<const int32_t> ids : {request.protect.span(), request.lane_experts.span()}) {
      for (int32_t expert : ids) {
        if (expert >= 0 && expert < experts_ && !listed(wanted, expert)) wanted.push_back(expert);
      }
    }
    const Tier& tier = tiers_[request.row];
    int64_t missing = 0;
    for (int32_t expert : wanted) {
      if (tier.expert_slot[expert] < 0) ++missing;
    }
    if (missing == 0) return Defer::kNone;
    const VictimCensus census = census_locked(request.row, wanted);
    if (census.free + census.evictable < missing && census.free + census.evictable + census.leased >= missing) {
      return Defer::kVictims;
    }
    return Defer::kNone;
  }

  // A `_locked` suffix, here and below, means "the owner's": the tier has no mutex (Task 15).
  //
  // Open the ring entry for a request with its full lane count and every lane ungranted. defers() refused an entry
  // still active, so this cannot fail.
  void open_lease_entry_locked(const Request& request) {
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    entry = Outstanding();
    entry.active = true;
    entry.gen = request.gen;
    entry.row = request.row;
    entry.count = static_cast<uint32_t>(request.lane_experts.size());
  }

  // Lease the source slot of every lane and publish its RowResult, in the reservation hold: after the take loop
  // reserved the missing rows and before read(). A hit lane is published READY (or COPYING / CPU, for the copy
  // engine), a miss lane LOADING into the slot `loading` reserved for it, so the device copies the hits while the read
  // runs and each miss piece by piece as the read publishes it. The caller stored and fenced every miss lane's PieceMask
  // word before (init_piece_words_locked), so the word carries G before any ready word of G is visible.
  void grant_lanes_locked(const Request& request, std::span<const int64_t> loading) {
    const size_t count = request.lane_experts.size();
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    Tier& tier = tiers_[request.row];
    int32_t slots[kLeaseLanes];
    uint64_t tags[kLeaseLanes] = {};
    const bool copy_request = copy_engine_ != nullptr && copy_armed_.load(std::memory_order_acquire) && request.captured;
    CopyJob job;
    for (size_t lane = 0; lane < count; ++lane) {
      const int32_t expert = request.lane_experts[lane];
      const int32_t slot = expert >= 0 && expert < experts_ ? tier.expert_slot[expert] : -1;
      const bool still_loading = slot >= 0 && tier.state[slot] == kLoading &&
                                 std::find(loading.begin(), loading.end(), slot) != loading.end();
      if (slot < 0 || tier.slot_to_expert[slot] != expert || (tier.state[slot] != kReady && !still_loading)) {
        fail_stop(
            error_prefix<Layout>() + "request " + std::to_string(request.seq) + ": lane " + std::to_string(lane) +
            " has no slot to lease");
      }
      slots[lane] = slot;
      tags[lane] = still_loading ? kLeaseTagLoading : kLeaseTagReady;
      if (copy_request && !still_loading) {
        const int32_t dst = lane < request.lane_dst.size() ? request.lane_dst[lane] : -1;
        if (copy_engine_->eligible(request.row, dst)) {
          tags[lane] = kLeaseTagCopying;
          const float weight = lane < request.lane_weight.size() ? request.lane_weight[lane] : 0.0f;
          job.lanes[job.count++] = CopyLane{static_cast<int32_t>(lane), slot, dst, weight};
          job.mask |= 1u << lane;
        } else {
          this->template count<kCopyFallbacks>();  // `count` is also this function's lane count
        }
      }
    }
    if (job.count > 0 && cpu_ != nullptr && cpu_->eligible(request.row)) choose_cpu_lanes_locked(&job, tags);
    uint8_t* results = lease_ + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
    for (size_t lane = 0; lane < count; ++lane) {
      const int32_t slot = slots[lane];
      // A lease is counted before its RowResult is published, so no admission can take the slot once a reader may.
      tier.leases[slot] += 1;
      entry.lane[lane] = LaneLease{1, slot, tags[lane] == kLeaseTagCopying || tags[lane] == kLeaseTagCpu};
      // The payload is written without first clearing the ready word: this index was last written for G - 16, whose
      // lease row retired before defers() let G in, so no reader of G - 16 runs, and a reader of G reads the payload
      // only after acquiring a ready word that carries G.
      std::memcpy(results + lane * kLeaseRowResultBytes + kLeaseRrHostSlot, &slot, 4);
    }
    _mm_sfence();  // every payload before any ready word, which the device acquires
    for (size_t lane = 0; lane < count; ++lane)
      store_release64(results + lane * kLeaseRowResultBytes + kLeaseRrReady, tagged_word(tags[lane], request.gen));
    // One writer (the owner): a relaxed load and store, no RMW. graph_leases_outstanding reads it lock-free.
    lanes_outstanding_.store(
        lanes_outstanding_.load(std::memory_order_relaxed) + static_cast<int64_t>(count), std::memory_order_relaxed);
    this->template count<kLeasesGranted>(static_cast<int64_t>(count));
    if (job.count > 0) {
      // After the COPYING words are published: the copy thread may complete and publish CopyDone at once.
      job.gen = request.gen;
      job.idx = idx;
      job.row = request.row;
      if constexpr (Build::kMetrics) job.submit_ns = now_ns();  // copy_latency_ns, a metric
      this->template count<kCopyJobs>();
      this->template count<kCopyLanes>(job.count);
      copy_engine_->submit(job);
    }
  }

  // CPU experts: the last split[n] of the job's n copy-engine lanes go to the CPU. The device plan sorts miss lanes by
  // residency score, highest first, so these are the lowest-scored RAM hits; the rest are copied and inserted.
  // They keep their place in the job (CopyDone covers them), are tagged kLeaseTagCpu, and are named in cpu_mask.
  void choose_cpu_lanes_locked(CopyJob* job, uint64_t* tags) {
    const size_t n = std::min<size_t>(static_cast<size_t>(job->count), kLeaseLanes);
    size_t want = cpu_split_[n].load(std::memory_order_relaxed);
    if (want == 0) return;
    for (int i = job->count - 1; i >= 0 && want > 0; --i) {
      const CopyLane& lane = job->lanes[i];
      job->cpu_mask |= 1u << lane.lane;
      tags[lane.lane] = kLeaseTagCpu;
      --want;
    }
    this->template count<kCpuJobs>();
    this->template count<kCpuLanes>(__builtin_popcount(job->cpu_mask));
  }

  // Release the non-copy leases of every request whose Done word carries its generation: CW stores Done once no kernel
  // of the request reads a leased slot any more. Cheap when nothing is outstanding; never blocks. A COPYING or CPU
  // lease waits for its copy's completion instead (release_copied_owned). The owner only, and no lock (Task 14).
  void retire_leases() {
    if (lanes_outstanding_.load(std::memory_order_relaxed) == 0) return;
    for (int64_t idx = 0; idx < kDemandRecords; ++idx) {
      Outstanding& entry = outstanding_[idx];
      if (!entry.active) continue;
      // The acquire pairs with CW's release of Done: every device read of this request's slots happened before it.
      // Done moves on to G + 16 only after this entry retired, so while it is active Done holds G or an older value.
      if (load_acquire64(lease_ + kLeaseDone + idx * kLeaseDoneBytes) != entry.gen) continue;
      Tier& tier = tiers_[entry.row];
      bool open = false;
      for (uint32_t lane = 0; lane < entry.count; ++lane) {
        LaneLease& held = entry.lane[lane];
        if (held.state == 1 && !held.copy_engine) this->template release_lease_locked<kLeasesAcked>(tier, held);
        open = open || held.state == 1;
      }
      if (!open) entry.active = false;
    }
  }

  // Lanes leased by the device and not yet retired. An eager pause is refused while this is non-zero.
  int64_t graph_leases_outstanding() const {
    return lanes_outstanding_.load(std::memory_order_relaxed);
  }

  // One lease released, exactly once. Called by the owner only (retire_leases, drain_copy_completions). K is a
  // metric (leases_acked, _copied): kept so.
  template <Counter K>
  void release_lease_locked(Tier& tier, LaneLease& held) {
    static_assert(!is_core_counter(K), "the lease release counters are metrics");
    if (tier.leases[held.slot] == 0) fail_stop(error_prefix<Layout>() + "lease underflow on slot " + std::to_string(held.slot));
    tier.leases[held.slot] -= 1;
    held.state = 2;
    lanes_outstanding_.store(lanes_outstanding_.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
    stats_.add(K);
    ++lease_changes_;
  }

  // The extents the introspection methods below write, for the FFI's exact out-buffer checks: none of those
  // methods is given a bound. All are fixed at construction, so no lock.
  int64_t layers() const {
    return layers_;
  }
  int64_t experts() const {
    return experts_;
  }
  int64_t row_capacity(int64_t row) const {
    if (row < 0 || row >= layers_) {
      throw std::runtime_error(error_prefix<Layout>() + "streamed row " + std::to_string(row) + " is out of range");
    }
    return tiers_[row].capacity;
  }

  // Test hooks and introspection. slot_info: [state, expert, leases] per slot. A snapshot.
  void slot_info(int64_t row, int64_t* out) {
    row_capacity(row);  // range-checked on the caller: the owner may be the service thread, which must not throw
    snapshot(row, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Tier& tier = self->tiers_[c.row];
      for (int64_t slot = 0; slot < tier.capacity; ++slot) {
        c.out[3 * slot] = tier.state[slot];
        c.out[3 * slot + 1] = tier.slot_to_expert[slot];
        c.out[3 * slot + 2] = tier.leases[slot];
      }
      return 0;
    });
  }

  // Test only: the service's account of request slot `idx`: [active, count, gen, then per lane (kLeaseLanes) its
  // state, then per lane its slot, then per lane 1 if it is a copy-engine lane]. A snapshot.
  void lease_entry(int64_t idx, int64_t* out) {
    if (idx < 0 || idx >= kDemandRecords)
      throw std::runtime_error(error_prefix<Layout>() + "request slot out of range");
    snapshot(idx, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Outstanding& entry = self->outstanding_[c.row];
      c.out[0] = entry.active;
      c.out[1] = entry.count;
      c.out[2] = static_cast<int64_t>(entry.gen);
      for (int64_t lane = 0; lane < kLeaseLanes; ++lane) {
        c.out[3 + lane] = entry.lane[lane].state;
        c.out[3 + kLeaseLanes + lane] = entry.lane[lane].slot;
        c.out[3 + 2 * kLeaseLanes + lane] = entry.lane[lane].copy_engine ? 1 : 0;
      }
      return 0;
    });
  }

  // Test only: a lease held for a test, standing in for a GPU reader's. A command (run_as_owner) that the caller waits
  // for, so a test's next post sees the lease. InstrBuild only.
  void inject_lease(int64_t row, int64_t slot, int64_t delta) {
    if constexpr (!Build::kFaults) {
      (void)row, (void)slot, (void)delta;
      test_only("inject_lease");
    } else {
      if (slot < 0 || slot >= row_capacity(row))
        throw std::runtime_error(error_prefix<Layout>() + "slot " + std::to_string(slot) + " is out of range");
      Command c;
      c.kind = Command::kInjectLease;
      c.row = row;
      c.arg = slot;
      c.delta = delta;
      int64_t result = 0;
      c.result = &result;
      run_as_owner(c);
      if (result != 0)
        throw std::runtime_error(error_prefix<Layout>() + "lease underflow on slot " + std::to_string(slot));
    }
  }

  // (free, evictable, leased) of `wanted`: a snapshot. `wanted` is the caller's, alive until the answer comes back.
  VictimCensus victim_census(int64_t row, const std::vector<int32_t>& wanted) {
    row_capacity(row);
    int64_t out[3] = {};
    snapshot(row, out, &wanted, [](RamTier* self, const Command& c) -> int64_t {
      const VictimCensus census =
          self->census_locked(c.row, *static_cast<const std::vector<int32_t>*>(c.input));
      c.out[0] = census.free;
      c.out[1] = census.evictable;
      c.out[2] = census.leased;
      return 0;
    });
    return VictimCensus{out[0], out[1], out[2]};
  }

  // Any thread, lock-free: the published slot map, which holds a slot for an expert exactly while that slot is READY
  // (publish_map stores a slot only when it becomes READY, and -1 on every unmap; a kLoading slot is never published).
  void mapping(int64_t row, int64_t* out) const {
    for (int64_t expert = 0; expert < experts_; ++expert)
      out[expert] = __atomic_load_n(map_ + row * experts_ + expert, __ATOMIC_ACQUIRE);
  }

  // A snapshot.
  void slot_to_expert(int64_t row, int64_t* out) {
    row_capacity(row);
    snapshot(row, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Tier& tier = self->tiers_[c.row];
      for (int64_t slot = 0; slot < tier.capacity; ++slot)
        c.out[slot] = tier.slot_to_expert[slot];
      return 0;
    });
  }

  // A snapshot: the READY slots' experts, least recently used first; returns how many. Its vector allocates on the
  // owner: between requests, or -- when it is queued while the service runs unpaused -- mid-read, from read()'s
  // progress hook (answer_snapshots). Production calls it only while paused (NativePinnedSlotTable, inside
  // host_use()), where the caller is the owner and nothing is being served; an unpaused call is test-only.
  int64_t lru_order(int64_t row, int64_t* out) {
    row_capacity(row);
    return snapshot(row, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Tier& tier = self->tiers_[c.row];
      std::vector<int64_t> slots;
      for (int64_t slot = 0; slot < tier.capacity; ++slot) {
        if (tier.state[slot] == kReady) slots.push_back(slot);
      }
      std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t b) { return tier.stamp[a] < tier.stamp[b]; });
      for (size_t i = 0; i < slots.size(); ++i)
        c.out[i] = tier.slot_to_expert[slots[i]];
      return static_cast<int64_t>(slots.size());
    });
  }

  // A command carrying the row's hot set as a bitmap (kMaxHotWords words: at most 1024 experts). Unpaused, it is
  // queued and applied by the service before its next request, in order with every other command; the caller does
  // not wait for it. A full ring makes the caller wait for room, never drops the command.
  void set_hot(int64_t row, const int64_t* experts, int64_t count) {
    row_capacity(row);
    if (experts_ > kMaxHotWords * 64) {
      throw std::runtime_error(
          error_prefix<Layout>() + "set_hot takes at most " + std::to_string(kMaxHotWords * 64) + " experts per row");
    }
    Command c;
    c.kind = Command::kSetHot;
    c.row = row;
    for (int64_t i = 0; i < count; ++i) {
      if (experts[i] >= 0 && experts[i] < experts_) c.hot[experts[i] / 64] |= uint64_t{1} << (experts[i] % 64);
    }
    run_as_owner(c);
  }

  // Any thread, lock-free: relaxed reads of words only the owner's serve() writes (a relaxed store, one writer).
  void layer_rows(int64_t* out) const {
    for (int64_t row = 0; row < layers_; ++row)
      out[row] = __atomic_load_n(&tiers_[row].rows_demand, __ATOMIC_RELAXED);
  }

  // Test-only faults: sleep `delay_ns` before each demand read once `after_demands` demands have read rows; report
  // reads as failed (which fails the process stop). InstrBuild only.
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands) {
    if constexpr (!Build::kFaults) {
      (void)delay_ns, (void)fail_reads, (void)after_demands;
      test_only("inject");
    } else {
      faults_.delay_ns.store(delay_ns);
      faults_.fail_reads.store(fail_reads);
      faults_.delay_after.store(after_demands);
    }
  }

  // Test only: carry a whole ReadFault down to this tier's reader, where inject() reaches it only as a
  // delay or a blanket failure. `words` is the reader tests' fault tensor (kFaultWords
  // int64; see fault_from). Unlike fail_reads the fault does NOT short-circuit ahead of the reader: the read
  // runs, so the fault's part errors, pack delay and the rest act on rows that have already packed. The
  // service thread applies it just before its next read (the reader is that thread's alone), and it then
  // stays until replaced; an all-default tensor clears it. Words 17-18 (abandon_after, step) and 22
  // (piece_stream) are not faults and are ignored, and words 19-20 (formerly pack_workers, pack_split) are reserved:
  // the tier's reader always streams pieces. The reader's counters (submit and
  // completion calls) run over the reader's whole life, so a call-numbered fault (submit_call, cqe_call) is relative to
  // a fresh tier. InstrBuild only: ProdBuild refuses rather than store a fault it would never apply.
  void inject_fault(const int64_t* words) {
    if constexpr (!Build::kFaults) {
      (void)words;
      test_only("inject_fault");
    } else {
      std::lock_guard<std::mutex> guard(faults_.fault_mutex);
      faults_.pending_fault = fault_from(words);
      faults_.fault_pending.store(true, std::memory_order_release);
    }
  }

  // Relaxed reads of every block: a core counter is the sum of its writers' blocks (each word has one writer), a
  // metric is stats_'s (always 0 in ProdBuild; Python reports only the core counters of a production host).
  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = core_.get(i) + copy_core_.get(i) + stats_.get(i);
  }

 private:
  // The owner. Counted (kCommandsApplied, a metric) before `done` is stored, so a caller that saw its answer sees the
  // count too. Nothing escapes: on the service thread an exception would terminate the process.
  void apply_command(const Command& c) {
    int64_t value = 0;
    uint32_t outcome = 1;
    try {
      switch (c.kind) {
        case Command::kSetHot:
          set_hot_owned(c.row, c.hot);
          break;
        case Command::kInjectLease:
          value = inject_lease_owned(c.row, c.arg, c.delta) ? 0 : -1;
          break;
        case Command::kSnapshot:
          value = c.snapshot(this, c);
          break;
      }
    } catch (...) {
      outcome = 2;
    }
    count<kCommandsApplied>();
    if (c.result != nullptr) *c.result = value;
    if (c.done != nullptr) c.done->store(outcome, std::memory_order_release);
  }

  // The service thread's drains inside pump_demand. pump() mode queues nothing (its caller owns the tier), and there
  // the ring's consumer must stay the one Python caller: so only when threaded.
  void drain_commands_on_service() {
    if (threaded_.load(std::memory_order_relaxed)) drain_commands();
  }

  // The service thread, mid-request (the read's progress hook, inject()'s read delay): apply the queue's leading
  // snapshots only. A mutator (set_hot, inject_lease) waits for the end of the request, and every snapshot queued
  // behind it waits too, so the queue's order is kept: a mutator is applied between requests, as a snapshot taken
  // after it sees it.
  void answer_snapshots() {
    if (!threaded_.load(std::memory_order_relaxed)) return;
    Command command;
    for (const Command* next = commands_.front(); next != nullptr && next->kind == Command::kSnapshot;
         next = commands_.front()) {
      commands_.pop(&command);
      apply_command(command);
    }
  }

  // Test only (InstrBuild): inject()'s read delay. It stands in for a slow read, so it answers snapshots as a read's
  // progress hook does, a slice at a time (no clock: the slices are counted).
  void fault_delay(int64_t ns) {
    constexpr int64_t kSliceNs = 1'000'000;
    for (int64_t left = ns; left > 0; left -= kSliceNs) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(std::min(left, kSliceNs)));
      answer_snapshots();
    }
  }

  void command_outcome(uint32_t outcome) const {
    if (outcome == 2) throw std::runtime_error(error_prefix<Layout>() + "a Python command failed on the tier's owner");
  }

  void require_owner(const char* what) const {
    if (!caller_owns()) throw std::runtime_error(error_prefix<Layout>() + what + " needs the service thread paused");
  }

  // A snapshot: `read` runs on the owner (run_as_owner) and its value comes back.
  int64_t snapshot(int64_t row, int64_t* out, const void* input, int64_t (*read)(RamTier*, const Command&)) {
    Command c;
    c.kind = Command::kSnapshot;
    c.row = row;
    c.out = out;
    c.input = input;
    c.snapshot = read;
    int64_t result = 0;
    c.result = &result;
    run_as_owner(c);
    return result;
  }

  void set_hot_owned(int64_t row, const uint64_t* hot) {
    Tier& tier = tiers_[row];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      tier.hot[expert] = (hot[expert / 64] >> (expert % 64)) & 1u;
      // A VRAM-hot row is decode's: kept owned it could never be a victim and would hold the share down.
      if (tier.hot[expert] && tier.expert_slot[expert] >= 0) disown_locked(tier, tier.expert_slot[expert]);
    }
  }

  // False on an underflow, which changes nothing.
  bool inject_lease_owned(int64_t row, int64_t slot, int64_t delta) {
    Tier& tier = tiers_[row];
    if (delta < 0 && tier.leases[slot] < static_cast<uint32_t>(-delta)) return false;
    tier.leases[slot] = static_cast<uint32_t>(static_cast<int64_t>(tier.leases[slot]) + delta);
    if (delta < 0) ++lease_changes_;
    return true;
  }

  // Copy thread. Completion was observed, so no copy of this job reads its slots any more: CopyDone here, so the
  // device's wait ends as soon as the DMA did; the lease itself is released by the owner (drain_copy_completions), one
  // owner poll later (D7). A job whose row has SM entries is not handed back yet (false): CW still reads those slots,
  // and copy_acked hands it back once CW's Done shows it finished. True otherwise.
  bool copy_completed(const CopyJob& job) {
    const uint32_t seq = static_cast<uint32_t>(job.gen);
    store_release64(lease_ + kLeaseCopyDone + job.idx * kLeaseCopyDoneBytes, job.gen);
    // Dekker with CW (row_copy_kernels.cuh: gate close, fence.sc.sys, CopyDone load): CopyDone before the gate load,
    // so if CW missed this store it closed the gate before this load, which then sees closed(G) and opens it.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t closed = gate_word(seq, kLeaseGateClosed);
    if (copy_gate() == closed) cas_gate(closed, gate_word(seq, kLeaseGateOpen));
    if (job.sm) return false;
    hand_back(job);
    return true;
  }

  // Copy thread, a job copy_completed left waiting: true once Done of its ring index reached this request's
  // generation, after which no SM read of the job's slots can be in flight, and the job is handed back to the owner.
  bool copy_acked(const CopyJob& job) {
    if (load_acquire64(lease_ + kLeaseDone + job.idx * kLeaseDoneBytes) < job.gen) return false;
    hand_back(job);
    return true;
  }

  // Copy thread: the job goes back to the owner through copy_done_, before the copy engine counts it finished (its
  // caller finish()es after this returns), so an owner that saw the engine idle (wait_idle's acquire of finished_)
  // pops it. At most kDemandRecords jobs are outstanding, handed-back ones included (a ring index's entry stays active
  // until the owner drains its job), and the ring holds kCopyRing.
  void hand_back(const CopyJob& job) {
    if (!copy_done_.push(job)) fail_stop(error_prefix<Layout>() + "the copy completion ring overflowed");
  }

  // The owner: release a completed job's COPYING and CPU leases, the only place one is released.
  void release_copied_owned(const CopyJob& job) {
    Outstanding& entry = outstanding_[job.idx];
    if (!entry.active || entry.gen != job.gen) {
      fail_stop(error_prefix<Layout>() + "copy of request " + std::to_string(job.gen) + " completed with no lease row");
    }
    Tier& tier = tiers_[entry.row];
    for (int i = 0; i < job.count; ++i) {
      LaneLease& held = entry.lane[job.lanes[i].lane];
      if (held.state == 1 && held.copy_engine) release_lease_locked<kLeasesCopied>(tier, held);
    }
    bool open = false;
    for (uint32_t lane = 0; lane < entry.count; ++lane)
      open = open || entry.lane[lane].state == 1;
    if (!open) entry.active = false;
  }

  // Copy thread, or the submitter when the copy engine's job ring is full. Completion cannot be established, and a
  // lease released without it could hand a slot under an in-flight copy to the next read: fail stop.
  void copy_failed(const CopyJob& job, int error) {
    fail_stop(
        std::string(Layout::kName) + " RAM miss copy engine: copy of request " + std::to_string(job.gen) +
        " failed (" + std::to_string(error) + ")");
  }

  // A locked cmpxchg on the host line is atomic against the device's posted stores to it.
  void cas_gate(uint32_t expected, uint32_t desired) {
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(lease_ + kLeaseCopyGate), &expected, desired, false, __ATOMIC_SEQ_CST,
        __ATOMIC_ACQUIRE);
  }

  // Called the moment a posted record is found. With the trace off this is one relaxed load and
  // no clock read; requests are served one at a time, so one member record serves them all. ProdBuild: nothing.
  void begin_stage(int64_t kind, uint32_t seq, uint32_t backlog) {
    if constexpr (Build::kMetrics) {
      if (!trace_.on.load(std::memory_order_relaxed)) {
        trace_.cur = nullptr;
        return;
      }
      StageRecord& stage = trace_.stage;
      const int64_t observed = stamp(&stage);  // before the reset below: the record is found, not built
      stage = StageRecord{};
      stage.observed = observed;
      stage.kind = kind;
      stage.seq = seq;
      stage.backlog = backlog;
      stage.prev_done = trace_.last_done;
      stage.pack_workers = reader_.pack_workers();
      stage.pack_split = reader_.pack_split();
      stage.piece_stream = reader_.piece_stream() ? 1 : 0;
      trace_.cur = &stage;
    } else {
      (void)kind;
      (void)seq;
      (void)backlog;
    }
  }

  // Service thread, before a read: install the fault inject_fault() left, on the reader only this thread drives.
  // ProdBuild: nothing (its faults are the instrumented build's, plan Task 10).
  void apply_pending_fault() {
    if constexpr (Build::kFaults) {
      if (!faults_.fault_pending.load(std::memory_order_acquire)) return;
      ReadFault fault;
      {
        std::lock_guard<std::mutex> guard(faults_.fault_mutex);
        fault = faults_.pending_fault;
        faults_.fault_pending.store(false, std::memory_order_relaxed);
      }
      reader_.set_fault(fault);
    }
  }

  void end_stage() {
    if constexpr (Build::kMetrics) {
      StageRecord* cur = trace_.cur;
      if (cur == nullptr) return;
      cur->done = stamp(cur);
      trace_.last_done = cur->done;
      trace_.ring->push(*cur);
      trace_.cur = nullptr;
    }
  }

  // The request in service's stage record, or null: always null in ProdBuild.
  StageRecord* stage_record() const {
    if constexpr (Build::kMetrics) {
      return trace_.cur;
    } else {
      return nullptr;
    }
  }

  [[noreturn]] static void throw_no_trace() {
    throw std::runtime_error(
        error_prefix<Layout>() + "the stage trace is in the instrumented host build only "
        "(set SGLANG_DSV41_EXPERT_TRACE_PATH so the service loads it)");
  }

  template <Counter K>
  void count_into(LineCounters<kCounterCount>& core, int64_t n) {
    if constexpr (is_core_counter(K)) {
      core.add(K, n);
    } else {
      stats_.add(K, n);
    }
  }

  void begin_busy() {
    busy_.store(++episodes_, std::memory_order_release);
  }
  void end_busy() {
    busy_.store(0, std::memory_order_release);
  }

  static int64_t round_up_page(int64_t value) {
    return (value + 4095) / 4096 * 4096;
  }

  // Before the service thread or any device exists, so plain stores and one fence suffice.
  void init_lease_block(int64_t lease_bytes) {
    if (lease_ == nullptr || reinterpret_cast<uintptr_t>(lease_) % kLeaseBlockAlign != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "the lease block must be a 4096-byte aligned block");
    }
    if (lease_bytes != kLeaseBlockBytes) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the lease block has " + std::to_string(lease_bytes) + " bytes, not " +
          std::to_string(kLeaseBlockBytes));
    }
    // Open with nothing armed: a copy wait that arms nothing passes its stream wait on this value (0 would block).
    const uint32_t open = gate_word(0, kLeaseGateOpen);
    std::memcpy(lease_ + kLeaseCopyGate, &open, 4);
    _mm_sfence();
  }

  // The eviction predicate's lease half. Task 8 adds host leases here as one more term.
  bool leased_locked(const Tier& tier, int64_t slot) const {
    return tier.leases[slot] > 0;
  }

  VictimCensus census_locked(int64_t row, std::span<const int32_t> wanted) const {
    const Tier& tier = tiers_[row];
    VictimCensus census;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) {
        ++census.free;
      } else if (tier.state[slot] == kReady) {
        const int32_t expert = tier.slot_to_expert[slot];
        if (tier.hot[expert] || tier.filling[slot] || listed(wanted, expert)) continue;
        if (leased_locked(tier, slot)) {
          ++census.leased;
        } else {
          ++census.evictable;
        }
      }
    }
    return census;
  }

  void run_fill() {
    begin_busy();
    apply_pending_fault();  // test only: inject_fault() acts on a fill's read as on a demand's
    std::vector<uint8_t>& packed = fill_packed_;  // reserved to the widest row at construction (spec A10)
    size_t landed = 0;
    auto advance = [&] {
      while (landed < packed.size() && packed[landed] != 0)
        ++landed;
      fill_landed_.store(static_cast<int64_t>(landed), std::memory_order_release);
    };
    int result = 0;
    try {
      // A demand's batch size and no publish target: a fill has no device readiness words.
      result = reader_.read(
          fill_row_,
          fill_experts_,
          fill_slots_,
          kBounceRows,
          [](size_t) { return false; },
          nullptr,
          &packed,
          SIZE_MAX,
          advance,
          nullptr);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "ERROR %sprefill fill: %s\n", error_prefix<Layout>().c_str(), error.what());
      result = 0;
    }
    _mm_sfence();  // the rows' bytes land before the caller is told (fill_landed_)
    // No tier state here (ownership rule 3): the owner runs the epilogue after the join (finish_fill_owned). Until
    // then the claimed slots stay filling, so no admission takes one and release() refuses it.
    fill_result_ = result;
    if (result == 1) {
      fill_landed_.store(static_cast<int64_t>(fill_slots_.size()), std::memory_order_release);
    } else {
      advance();
    }
    end_busy();
    fill_state_.store(result == 1 ? kFillOk : kFillFailed, std::memory_order_release);
  }

  // The fill's epilogue, on the owner after the join (fill_join): a fill thread never writes the tier (ownership rule
  // 3). Clears every claimed slot's filling flag; a failed fill releases (unmaps) its rows that did not land, and
  // counts the read error and the map's change on the owner's counters.
  void finish_fill_owned() {
    fill_unfinished_ = false;
    Tier& tier = tiers_[fill_row_];
    for (size_t i = 0; i < fill_slots_.size(); ++i) {
      const int64_t slot = fill_slots_[i];
      tier.filling[slot] = 0;
      if (fill_result_ != 1 && !(i < fill_packed_.size() && fill_packed_[i] != 0)) release_locked(fill_row_, slot);
    }
    if (fill_result_ != 1) {
      count<kReadErrors>();
      count<kVersion>();
    }
  }

  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
  }

  void disown_locked(Tier& tier, int64_t slot) {
    if (tier.prefill_owned[slot]) {
      tier.prefill_owned[slot] = 0;
      --tier.owned;
    }
  }

  // A slot for a row a Python-side read admits (assign; the prefill fill path). With a prefill share set, no free slot,
  // and the layer already holding that many prefill-owned rows, the victim is the LRU owned row, under every exclusion
  // of take_slot_locked, so a prefill displaces at most `share` of decode's rows. Otherwise, and whenever no owned row
  // qualifies, it is take_slot_locked's choice (a free slot first); with `stop_at_share`, when no owned row qualifies
  // it is none (-1). Under a share the slot becomes prefill-owned.
  int64_t take_admit_slot_locked(
      int64_t row, std::span<const int32_t> protect, bool fallback, int64_t* evicted, bool stop_at_share = false) {
    Tier& tier = tiers_[row];
    int64_t slot = -1;
    const int64_t share = prefill_share_.load(std::memory_order_relaxed);  // once: one admission, one share
    if (share > 0 && tier.owned >= share &&
        std::find(tier.state.begin(), tier.state.end(), kFree) == tier.state.end()) {
      int64_t best = -1;
      for (int64_t s = 0; s < tier.capacity; ++s) {
        if (!tier.prefill_owned[s] || tier.state[s] != kReady || tier.filling[s] || leased_locked(tier, s)) continue;
        const int32_t expert = tier.slot_to_expert[s];
        if (tier.hot[expert] || listed(protect, expert)) continue;
        if (best < 0 || tier.stamp[s] < tier.stamp[best]) best = s;
      }
      if (best >= 0) {
        *evicted = tier.slot_to_expert[best];
        release_locked(row, best);  // unmaps it before its bytes are overwritten (D11) and ends its ownership
        count<kEvictions>();
        slot = best;
      } else if (stop_at_share) {
        *evicted = -1;
        return -1;
      }
    }
    if (slot < 0) slot = take_slot_locked(row, protect, fallback, evicted);
    if (slot >= 0 && share > 0) {
      tier.prefill_owned[slot] = 1;
      ++tier.owned;
    }
    return slot;
  }

  // Takes only a kFree slot or evicts an unleased kReady one: a kLoading slot is never taken.
  int64_t take_slot_locked(int64_t row, std::span<const int32_t> protect, bool fallback, int64_t* evicted) {
    Tier& tier = tiers_[row];
    *evicted = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    int64_t spare = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] != kReady) continue;
      if (tier.filling[slot]) continue;         // a prefill fill is still writing it
      if (leased_locked(tier, slot)) continue;  // a GPU reader may still be reading it
      const int32_t expert = tier.slot_to_expert[slot];
      if (tier.hot[expert]) continue;
      if (listed(protect, expert)) {
        if (spare < 0 || tier.stamp[slot] < tier.stamp[spare]) spare = slot;
        continue;
      }
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0 && fallback) best = spare;
    if (best < 0) {
      count<kNoVictim>();
      return -1;
    }
    const int32_t victim = tier.slot_to_expert[best];
    publish_map(row, victim, -1);  // unmapped before its bytes are overwritten (D11)
    tier.expert_slot[victim] = -1;
    tier.slot_to_expert[best] = -1;
    tier.state[best] = kFree;
    disown_locked(tier, best);
    *evicted = victim;
    count<kEvictions>();
    return best;
  }

  void release_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0) {
      publish_map(row, expert, -1);
      tier.expert_slot[expert] = -1;
    }
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kFree;
    disown_locked(tier, slot);
  }

  // An unarmed demand record: nobody waits on it, so the device may already be gathering any mapped slot (the next
  // token's rows too, once the thread lags). Only refresh the recency of its assigned rows: no eviction, no read.
  void touch_request(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    Tier& tier = tiers_[request.row];
    for (int32_t expert : request.protect) {
      const int32_t slot = expert >= 0 && expert < experts_ ? tier.expert_slot[expert] : -1;
      if (slot >= 0) {
        tier.stamp[slot] = ++tick_;
        disown_locked(tier, slot);
      }
    }
  }

  // An armed demand: touch the request's assigned rows, reserve a slot for every protected or lane expert that is not
  // assigned (D12's recompute: the device is waiting on this record, so no gather is in flight), evicting only
  // unprotected, non-hot, unleased READY rows; lease and publish every lane; read the missing rows; publish the map.
  // Every failure fails stop: nothing of it is recoverable, and demand_done must never say served for it.
  void serve(const Request& request, int64_t* rows) {
    StageRecord* const cur = stage_record();  // null in ProdBuild, so every `if (cur)` below folds away
    if (cur) cur->lanes = static_cast<int64_t>(request.lane_experts.size());
    const auto fail = [&](const std::string& why) {
      fail_stop(error_prefix<Layout>() + "request " + std::to_string(request.seq) + " of row " +
                std::to_string(request.row) + ": " + why);
    };
    if (request.row < 0 || request.row >= layers_) fail("the row is out of range");
    // Fixed-size locals (spec A4, A5): a request names at most kWanted distinct experts, so none of these allocates.
    // A lane's own expert must never be a victim of the request that leases it, whatever the post protected.
    FixedVec<int32_t, kWanted> wanted;
    for (std::span<const int32_t> ids : {request.protect.span(), request.lane_experts.span()}) {
      for (int32_t expert : ids) {
        if (expert < 0 || expert >= experts_) fail("an expert is out of range");
        if (!listed(wanted, expert)) wanted.push_back(expert);  // one slot per expert (device bytes may repeat)
      }
    }
    FixedVec<int32_t, kWanted> missing;
    FixedVec<int64_t, kWanted> slots;
    Tier& tier = tiers_[request.row];
    for (int32_t expert : wanted) {
      const int32_t slot = tier.expert_slot[expert];
      if (slot >= 0) {
        tier.stamp[slot] = ++tick_;
        disown_locked(tier, slot);
      } else {
        missing.push_back(expert);
      }
    }
    // defers() refused this request while leases could still free a victim, so a slot not found now never will be.
    for (int32_t expert : missing) {
      int64_t evicted = -1;
      const int64_t slot = take_slot_locked(request.row, wanted, false, &evicted);
      if (slot < 0) fail("no victim slot for a missing row");
      tier.slot_to_expert[slot] = expert;
      tier.state[slot] = kLoading;
      tier.expert_slot[expert] = static_cast<int32_t>(slot);
      slots.push_back(slot);
    }
    // The reservation hold: from the take loop to read(), nothing else runs on the tier (no other thread writes it,
    // and the owner's own passes run only between requests or from read()'s progress hook). Grant every lane in it:
    // after it, a hit slot could be evicted by a later admission; before the take loop, a hit lane's lease would
    // outlive a request that is then refused.
    bool publishing = false;  // a miss lane has readiness words to publish into
    if (!request.lane_experts.empty()) {
      open_lease_entry_locked(request);
      publishing = init_piece_words_locked(request, missing);
      grant_lanes_locked(request, slots.span());
    }
    if (cur) cur->reserved = stamp(cur);
    int64_t status = kStatusNoRead;
    // Per slot: the row was packed whole (read() sets it). A member, not a local, reserved to kWanted at construction,
    // so read()'s assign() never allocates on the service thread.
    std::vector<uint8_t>& packed = packed_;
    packed.clear();
    if (!missing.empty()) {
      bool fail_reads = false;
      if constexpr (Build::kFaults) {  // the test faults (inject, inject_fault): InstrBuild only
        apply_pending_fault();
        const int64_t delay = faults_.delay_ns.load();
        if (delay > 0 && demands_read_ >= faults_.delay_after.load()) fault_delay(delay);
        fail_reads = faults_.fail_reads.load();
      }
      if (fail_reads) fail("a test fault failed the read");
      const int result = reader_.read(
          request.row,
          missing,
          slots,
          kBounceRows,
          [](size_t) { return false; },
          cur,
          &packed,
          SIZE_MAX,
          // The service stays the tier's owner for the whole read, so it releases the COPYING leases the copy thread
          // handed back (D7), retires Done leases and answers queued snapshots here too (Task 13): none waits for a
          // read's length. read() runs it once per drain-loop turn and once per finished row; each pass early-outs
          // when idle, allocates nothing and takes no lock. A template argument (spec A6): no std::function.
          [this] {
            drain_copy_completions();
            retire_leases();
            answer_snapshots();
          },
          publishing ? &piece_publish_ : nullptr);
      stats_.store(kPiecePublishRefused, reader_.publish_refused());
      if (result != 1) {
        count<kReadErrors>();
        fail("the read failed");
      }
      status = kStatusServed;
      ++demands_read_;
      _mm_sfence();  // the split's memcpy stores land before the map publishes them (D11)
    }
    const int64_t published = static_cast<int64_t>(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
      tier.state[slots[i]] = kReady;
      tier.stamp[slots[i]] = ++tick_;
      publish_map(request.row, missing[i], static_cast<int32_t>(slots[i]));
    }
    if (!slots.empty()) {
      // One writer (the owner), read lock-free by layer_rows: a relaxed store.
      std::atomic_ref<int64_t>(tier.rows_demand).store(tier.rows_demand + published, std::memory_order_relaxed);
      count<kVersion>();
      count<kRowsRead>(published);
    }
    if (cur) {
      cur->mapped = stamp(cur);
      cur->row = request.row;
      cur->ok = 1;
      cur->status = status;
      cur->rows = published;
    }
    *rows = published;
  }

  // Store piece_word(gen) into the readiness word of every lane whose expert is read (in `missing`), then fence,
  // and record those words as the owner's publish targets, by the row's ordinal in the read. Service thread only:
  // it is the only writer of area P. False when no lane names a missing row (nothing to publish).
  bool init_piece_words_locked(const Request& request, std::span<const int32_t> missing) {
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    piece_targets_.assign(missing.size(), PieceTarget{});
    bool any = false;
    for (size_t lane = 0; lane < request.lane_experts.size(); ++lane) {
      const auto found = std::find(missing.begin(), missing.end(), request.lane_experts[lane]);
      if (found == missing.end()) continue;
      uint8_t* word =
          lease_ + kLeasePieceMask + (idx * kLeaseLanes + static_cast<int64_t>(lane)) * kLeasePieceMaskLineBytes;
      store_release64(word, piece_word(request.gen));
      PieceTarget& target = piece_targets_[static_cast<size_t>(found - missing.begin())];
      target.words[target.count++] = reinterpret_cast<uint64_t*>(word);
      any = true;
    }
    _mm_sfence();
    piece_publish_ = PiecePublish{request.gen, piece_targets_.data()};
    return any;
  }

  void handle_demand(const Request& request) {
    begin_busy();
    int64_t rows = 0;
    if (request.armed) {
      serve(request, &rows);
      if (rows == 0) {
        count<kTouchOnly>();
      } else {
        count<kServedRequests>();
      }
    } else {
      touch_request(request);
      count<kTouchOnly>();
      if (StageRecord* const cur = stage_record()) {
        cur->kind = kStageTouch;
        cur->row = request.row;
        cur->ok = 1;
        cur->status = kStatusTouch;
      }
    }
    end_busy();
  }

  uint8_t* page_;
  int32_t* map_;
  uint8_t* lease_;  // the lease block (lease_layout.h)
  uint8_t* hot_page_ = nullptr;
  int64_t hot_stride_ = 0;
  std::atomic<bool> gpu_hot_mode_{false};
  // The copy engine, when enabled (before the service thread starts); armed separately, and only then used.
  std::unique_ptr<Engine> copy_engine_;
  // CPU experts, when enabled (after the copy engine, before the service thread); stopped after the copy thread.
  std::unique_ptr<CpuExpertEngine> cpu_;
  std::array<std::atomic<uint8_t>, kLeaseLanes + 1> cpu_split_{};
  std::atomic<bool> copy_armed_{false};
  int64_t copy_wait_timeout_ns_ = 0;           // set with the copy engine, before any thread reads it
  // Piece streaming: serve()'s readiness words per row it reads, reused every request (like packed_).
  std::vector<PieceTarget> piece_targets_;
  PiecePublish piece_publish_;
  Outstanding outstanding_[kDemandRecords];  // by request slot; the owner's
  // Lanes GRANTED and not yet retired: an early-out for retire_leases. Written by the owner only (a relaxed load and
  // store), read lock-free by graph_leases_outstanding.
  std::atomic<int64_t> lanes_outstanding_{0};
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new; retirement continues
  uint64_t lease_changes_ = 0;  // the owner's: bumped whenever a lease is released, what wakes a deferred demand
  // The demand held back, if any (service thread only): its sequence, the changes seen when it was last refused and
  // when it was first observed (for the stage record).
  uint32_t deferred_seq_ = 0;
  uint64_t deferred_stamp_ = 0;
  int64_t deferred_observed_ns_ = 0;
  int64_t layers_;
  int64_t experts_;
  Source reader_;
  std::vector<uint8_t> packed_;  // serve()'s per-row packed flags, reserved to kWanted, reused every request
  std::vector<uint8_t> hot_scratch_;  // (experts_+7)/8 bytes, sized at construction: the hot bitmap of the record read
  std::vector<uint8_t> fill_packed_;  // run_fill's per-row packed flags, reserved to the largest row capacity
  // Prefill fills: written by fill_begin before the thread starts and read by it; the caller reads only the atomics
  // until it joins (fill_join), and then the rest.
  static constexpr int kFillOk = 0;
  static constexpr int kFillRunning = 1;
  static constexpr int kFillFailed = 2;
  std::thread fill_thread_;
  int64_t fill_row_ = 0;
  std::vector<int32_t> fill_experts_;
  std::vector<int64_t> fill_slots_;
  std::atomic<int64_t> fill_landed_{0};
  std::atomic<int> fill_state_{kFillOk};
  int fill_result_ = 0;           // run_fill's read result: the fill thread's, read by the owner after the join
  bool fill_unfinished_ = false;  // an epilogue is owed (fill_begin started a thread); caller_mutex_ / the owner's
  std::vector<Tier> tiers_;  // the owner's, with every other non-atomic member: the tier has no mutex (Task 15)
  uint64_t tick_ = 0;
  std::atomic<int64_t> prefill_share_{0};  // any thread stores it, relaxed; see set_prefill_share
  uint32_t next_demand_ = 1;
  int64_t demands_read_ = 0;
  std::atomic<bool> threaded_{false};
  // The single owner (Task 13): the pausing caller owns the tier while parked_ (see caller_owns). caller_mutex_
  // serializes Python-side callers only; the service, copy and fill threads never take it. commands_ carries the
  // unpaused callers' commands to the service (producer: the caller_mutex_ holder; consumer: the owner).
  std::atomic<bool> parked_{false};
  std::mutex caller_mutex_;
  SpscRing<Command, 64> commands_;
  // Copy thread -> owner (Task 14): jobs whose copies completed (and whose SM reads CW finished), CopyDone-published on
  // the copy thread. The consumer is always the current owner (the same
  // handoffs as commands_); drain_copy_completions releases their leases.
  SpscRing<CopyJob, kCopyRing> copy_done_;
  // The watchdog's hung-request marker (D6), see busy_episode(): a new value per demand or fill in service, 0 when
  // none. episodes_ is the service thread's, or a fill's (they never run at once: a fill needs the pause).
  std::atomic<uint64_t> busy_{0};
  uint64_t episodes_ = 0;
  // Test-only faults (inject, inject_fault): InstrBuild only (plan Task 10).
  struct TierFaults {
    std::atomic<int64_t> delay_ns{0};
    std::atomic<int64_t> delay_after{0};
    std::atomic<bool> fail_reads{false};
    std::mutex fault_mutex;                 // guards pending_fault between inject_fault() and the service thread
    ReadFault pending_fault{};
    std::atomic<bool> fault_pending{false};
  };
  struct NoTierFaults {};
  [[no_unique_address]] std::conditional_t<Build::kFaults, TierFaults, NoTierFaults> faults_;
  // Counters, see count(). One line-private block per writer thread; the metrics only in InstrBuild.
  LineCounters<kCounterCount> core_;       // the tier's owner: the service thread, or the caller while it owns it
  LineCounters<kCounterCount> copy_core_;  // copy thread only
  [[no_unique_address]] Stats<Build::kMetrics, kCounterCount> stats_;  // InstrBuild: any thread, relaxed RMW
  // Stage trace (spec M9), InstrBuild only. cur points at stage while a traced request is in service, else null.
  struct TraceState {
    std::atomic<bool> on{false};
    std::mutex mutex;  // guards ring against a drain racing enable_trace (Python only)
    std::unique_ptr<StageRing> ring;
    StageRecord stage{};
    StageRecord* cur = nullptr;
    int64_t last_done = 0;
  };
  struct NoTraceState {};
  [[no_unique_address]] std::conditional_t<Build::kMetrics, TraceState, NoTraceState> trace_;
  // The type-system half of the prod proof (nm cannot see state whose names are inlined away): ProdBuild's fault,
  // metric and trace members are empty types, which [[no_unique_address]] gives no storage.
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(faults_)>, "ProdBuild has no faults");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(stats_)>, "ProdBuild has no metrics");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(trace_)>, "ProdBuild has no trace");
};

}  // namespace expert_stream
}  // namespace sglang
