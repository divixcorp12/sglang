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
  std::vector<uint32_t>
      leases;  // GPU-reader leases per slot (LEASE_PROTOCOL.md section 8); 0 frees a slot for eviction
  std::vector<uint32_t> generation;  // bumped before a slot's bytes change; mirrored into the lease block's SlotGen
  std::vector<uint8_t> filling;      // a prefill fill is still writing this kReady slot: never a victim, never released
  // Prefill share (plan 2026-09-25-dsv41-prefill-eviction): 1 while a row a prefill admitted has not been used by
  // decode. `owned` counts them. Only take_admit_slot_locked sets it, so it stays all zero with the share off.
  std::vector<uint8_t> prefill_owned;
  int64_t owned = 0;
  // Written by the owner's serve() only (a relaxed store through std::atomic_ref), read relaxed by layer_rows from
  // any thread.
  int64_t rows_demand = 0;
  int64_t rows_advisory = 0;
};

// What a request could take from a tier, counted without taking anything.
struct VictimCensus {
  int64_t free = 0;       // FREE slots
  int64_t evictable = 0;  // READY, not hot, not requested, not leased
  int64_t leased = 0;     // as evictable, but leased: they would be victims if the leases retired
};

// The pinned-slot bookkeeping of every streamed layer (plan D12) and the service of one
// request at a time. pump_demand/pump_advice are called by one caller at a time: a test's
// pump(), or the Task 12 thread. The Python-facing methods follow the single-owner rule (plan
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
      tier.generation.assign(tier.capacity, 0);
      tier.filling.assign(tier.capacity, 0);
      tier.prefill_owned.assign(tier.capacity, 0);
    }
    if (lease_ != nullptr) init_lease_block(capacity, lease_bytes);
    // The request path's buffers, sized once (spec A2, A10): nothing on it grows after construction.
    hot_scratch_.assign(static_cast<size_t>((experts_ + 7) / 8), 0);
    packed_.reserve(kWanted);
    piece_targets_.reserve(kWanted);
    int64_t widest = 0;
    for (int64_t c : capacity)
      widest = std::max(widest, c);
    fill_packed_.reserve(static_cast<size_t>(widest));
  }

  // The copy thread's callbacks use slot_gen_, the lease block, copy_done_ and the counter blocks, which are destroyed
  // before copy_engine_ would be.
  ~RamTier() {
    fill_join();  // the fill thread reads through reader_ into the slabs; nothing else holds the tier by now
    if (copy_engine_ != nullptr) copy_engine_->stop(5'000'000'000LL);
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = load_acquire(page_ + kDemandDone) + 1u;
    if (next_demand_ == 0) next_demand_ = 1;
    next_advice_ = load_acquire(page_ + kAdviseDone) + 1u;
    if (next_advice_ == 0) next_advice_ = 1;
    return true;
  }

  uint8_t* page() const {
    return page_;
  }
  // The watchdog's hung-request marker (D6): nonzero while a demand, an advisory or a fill is in service, a new value
  // per episode. The watchdog thread times how long one value persists; the service reads no clock for it.
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
  void request_pause(bool paused) {
    pause_requested_.store(paused);
  }
  void request_stop(bool stopping) {
    stop_requested_.store(stopping);
  }
  void skip_advice_posted_so_far() {
    skip_advice_upto_.store(load_acquire(page_ + kAdviseHead));
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
    release_copy_gate();  // a copy wait armed after its CopyDone was published: no copy-thread call opens it
    // The head is read before the retirement pass, so a posted demand makes that pass a settle pass (retire_leases):
    // every signal of each older request is then visible. Once per demand, so a deferred one's polls stay cheap.
    const uint32_t head = load_acquire(page_ + kDemandHead);
    const bool posted = head != 0 && reached(head, next_demand_);
    // Commands a caller queued before it posted this demand are applied before it is served: the demand head's
    // acquire above made the caller's earlier push visible. The loop drains between requests too (RamThread::run).
    // Pump mode queues nothing (its caller owns the tier), so only the service thread looks.
    if (posted) drain_commands_on_service();
    drain_copy_completions();  // D7: the COPYING leases the copy thread handed back, released before the pass below
    retire_leases(posted && settled_seq_ != next_demand_);  // first, so that an idle pump still retires
    if (posted) settled_seq_ = next_demand_;
    if (admission_closed_.load()) return false;
    if (!posted) return false;
    // A deferred demand is not looked at again until a lease retires: no stage record, no clock read per poll.
    if (deferred_seq_ == next_demand_ && !deferral_may_retry()) return false;
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq.
      count<kOverruns>(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    if (read_record(record, next_demand_, &request)) {
      judge_prefetch(request);
      const bool gpu_hot = gpu_hot_mode_.load() && request.armed;
      const bool hot_ok =
          !gpu_hot || (request.row >= 0 && request.row < layers_ && read_gpu_hot(next_demand_, &request) &&
                       load_acquire(record + kRecSeq) == next_demand_);
      if (!hot_ok) {
        count<kOverruns>();
        set_status(record, kFailed);
      } else if (lease_mode_ && request.armed && !read_lane_request(next_demand_, &request)) {
        count<kOverruns>();  // a later request overwrote the lane request: a lapped record
      } else if (lease_mode_ && request.armed && terminal_seen(request)) {
        count<kLateAfterTerminal>();  // the device gave up on it: serve nothing, lease nothing
      } else {
        if (gpu_hot) apply_gpu_hot(request);
        const Defer reason = request.armed ? defers(request) : Defer::kNone;
        if (reason != Defer::kNone) {
          // Held back, not failed and not served: return before handle_demand (so the busy episode and kBusySeq stay
          // untouched, or the watchdog would count the wait as a hung read) and before the tail (no demand_done, no
          // advance). No stage record is pushed; the first observation time is kept for the one written when it is
          // served.
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
          deferred_gen_ = request.gen;
          if constexpr (Build::kMetrics) trace_.cur = nullptr;
          return false;
        } else {
          if constexpr (Build::kMetrics) {
            if (trace_.cur != nullptr && deferred_seq_ == next_demand_ && deferred_observed_ns_ != 0) {
              trace_.cur->observed = deferred_observed_ns_;
            }
          }
          handle_demand(request, record);
        }
      }
    } else {
      count<kOverruns>();  // status stays pending: a waiting layer fails stop
    }
    deferred_seq_ = 0;
    if constexpr (Build::kFaults) {
      if (const int64_t stall = faults_.done_stall_ns.load(); stall > 0) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(stall));  // test only: see inject_done_stall
      }
    }
    _mm_sfence();
    store_release(page_ + kDemandDone, next_demand_);
    end_stage();
    next_demand_ = skip_zero(next_demand_ + 1u);
    return true;
  }

  // Serve (or skip) the next posted advisory record, if any. True when it handled one.
  bool pump_advice() {
    if (admission_closed_.load()) return false;
    const uint32_t head = load_acquire(page_ + kAdviseHead);
    if (head == 0 || !reached(head, next_advice_)) return false;
    begin_stage(kStageAdvisory, next_advice_, head - next_advice_);
    if (head - next_advice_ >= kAdviseRecords) {
      count<kAdvisoriesSkipped>(head - next_advice_ - (kAdviseRecords - 2));
      next_advice_ = skip_zero(head - kAdviseRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kAdviseRing, kAdviseRecords, next_advice_);
    Request request;
    const uint32_t skip_upto = skip_advice_upto_.load();
    const bool stale = !read_record(record, next_advice_, &request) ||
                       (skip_upto != 0 && reached(skip_upto, next_advice_)) ||
                       reached(load_acquire(page_ + kDemandHead), request.after + 1u) ||
                       load_acquire(page_ + kFatal) != 0 || pause_requested_.load();
    if (stale) {
      if constexpr (Build::kMetrics) trace_.cur = nullptr;  // a skipped advisory is no service: no stage record
      count<kAdvisoriesSkipped>();
    } else {
      in_advice_.store(true);
      count<kAdvisories>();
      // An advisory gives up only between rows, not inside a blocking read: the watchdog's
      // stuck rule covers it like a demand, or a hung read would block stop()'s join forever.
      begin_busy();
      int64_t rows = 0;
      serve(request, true, &rows);
      end_busy();
      in_advice_.store(false);
    }
    store_release(page_ + kAdviseDone, next_advice_);
    end_stage();
    next_advice_ = skip_zero(next_advice_ + 1u);
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
  //     snapshots slot_info, slot_to_expert, lease_entry, lru_order, victim_census, prefetch_lease;
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
    bump_generation_locked(row, slot);  // the caller writes the bytes after this returns
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
        bump_generation_locked(row, slot);  // the fill writes the bytes after this returns
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

  // Lease mode: the service reads each armed request's lane request, leases every lane's source slot and publishes
  // a row result per lane before it answers (LEASE_PROTOCOL.md 7). Off leaves every request as it always was.
  void set_lease_mode(bool on) {
    if (lease_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "lease mode needs a lease block");
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "set lease mode before the service thread starts");
    lease_mode_ = on;
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
    if (!lease_mode_) throw std::runtime_error(error_prefix<Layout>() + "GPU hot mode needs leases");
    gpu_hot_mode_.store(on);
  }

  // The hot bitmap of `expected`'s record, copied into the service-owned hot_scratch_ (spec A2): not const, it writes
  // that scratch. request->hot_bitmap points into it until the next call.
  bool read_gpu_hot(uint32_t expected, Request* request) {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    uint32_t count = 0;
    std::memcpy(&count, record + 4, 4);
    if (count != experts_) return false;
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

  // Two-phase mode (Task 6 V1): grant the resident lanes inside serve()'s reservation hold, before read(), so the
  // device can copy them while the missing rows are still being read. Off leaves lease mode exactly as Task 5
  // shipped it, which is the A1 arm every Task 6 measurement is reported against.
  void set_two_phase(bool on) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "set two-phase mode before the service thread starts");
    two_phase_ = on;
  }

  // Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM): the reader reads each part as sub-reads and vets
  // rows piece by piece. Before the thread starts. The service refuses it without two-phase and lease mode.
  void set_piece_stream(bool on) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "set piece streaming before the service thread starts");
    reader_.set_piece_stream(on);
    piece_stream_ = on;
  }

  // Copy engine (LEASE_PROTOCOL.md 7.6): a thread that copies the reservation hold's hit lanes with the DMA engine.
  // `device` < 0 is the CPU test backend (HostCopyBackend). Before the service thread starts; unarmed until arm().
  // `wait_timeout_ns`: how long an armed copy wait may hold the decode stream before the watchdog opens its gate as a
  // timeout (SGLANG_DSV41_RAM_MISS_TIMEOUT_MS).
  void enable_copy_engine(int64_t device, int64_t spin_ns, int64_t wait_timeout_ns) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable the copy engine before the service thread starts");
    if (copy_engine_ != nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is already enabled");
    // Only the piece-streaming chain grants every lane in the reservation hold and runs the copy wait.
    if (!(lease_mode_ && two_phase_ && piece_stream_)) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the copy engine needs lease mode, two-phase and piece streaming");
    }
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

  // ---- The copy wait's gate (LEASE_PROTOCOL.md 7.6, "The stream-ordered copy wait") ----
  //
  // The decode stream waits (cuStreamWaitValue32) on area C's gate, which CW closed for G (the word names G's seq)
  // before it published CopyArm = G. CW opens it itself when CopyDone already carries G at the arm (its Dekker half);
  // otherwise a host releaser does. Any thread may call release_copy_gate: it records G opened at most once (the CAS
  // on gate_released_) and changes the gate only by a CAS from G's own closed word, so a gate CW already opened, or
  // one a later request closed, is left alone however long the releaser stalls. The copy thread calls it after
  // publishing CopyDone, the service on every
  // pump_demand, close_admission and abort_copy_waits after the shutdown word, and the watchdog every poll, with the
  // timeout. No lock, clock or allocation: atomics and pinned words only, so it touches no tier state and needs no
  // owner.

  static uint32_t gate_word(uint32_t seq, uint32_t low) {
    return ((seq & kLeaseGateSeqMask) << kLeaseGateSeqShift) | low;
  }

  // The generation of the request whose copy wait is armed and whose gate is still closed for it, else 0.
  uint64_t armed_copy_wait() const {
    if (lease_ == nullptr || copy_engine_ == nullptr) return 0;
    const uint64_t arm = load_acquire64(lease_ + lease_c_ + kLeaseCopyArm);
    if (tag_of(arm) != kLeaseTagCopyArm) return 0;
    const uint64_t gen = generation_of(arm);
    if (gate_released_.load(std::memory_order_acquire) == gen) return 0;
    const uint32_t gate = load_acquire(lease_ + lease_c_ + kLeaseCopyGate);
    return gate == gate_word(static_cast<uint32_t>(gen), kLeaseGateClosed) ? gen : 0;
  }

  bool copy_waits_aborted() const {
    return copy_waits_aborted_.load();
  }

  int64_t copy_wait_timeout_ns() const {
    return copy_wait_timeout_ns_;
  }

  // `expired`: the watchdog saw generation `expired` armed for longer than the copy-wait timeout (0: none). True when
  // this call opened the gate.
  bool release_copy_gate(uint64_t expired = 0) {
    if (lease_ == nullptr || copy_engine_ == nullptr) return false;
    uint8_t* area_c = lease_ + lease_c_;
    uint64_t released = gate_released_.load(std::memory_order_acquire);
    const uint64_t arm = load_acquire64(area_c + kLeaseCopyArm);
    if (tag_of(arm) != kLeaseTagCopyArm) return false;
    const uint64_t gen = generation_of(arm);
    if (released == gen) return false;
    const uint32_t seq = static_cast<uint32_t>(gen);
    const uint32_t closed = gate_word(seq, kLeaseGateClosed);
    if (load_acquire(area_c + kLeaseCopyGate) != closed) {
      // CW opened G itself (CopyArm is stored after the close, so a gate read after CopyArm == G is G's close or
      // later). Record it, so the watchdog stops timing it; nothing to store.
      gate_released_.compare_exchange_strong(released, gen, std::memory_order_acq_rel);
      return false;
    }
    const uint8_t* done = area_c + static_cast<int64_t>((seq - 1u) % kDemandRecords) * kLeaseCopyDoneBytes;
    uint32_t outcome = kLeaseGateOpen;
    if (load_acquire64(done + kLeaseCdGen) == tagged_word(kLeaseTagCopied, gen)) {
      outcome = kLeaseGateOpen;
    } else if (load_acquire(page_ + kFatal) != 0 || load_acquire(lease_ + kLeaseHeaderShutdown) != 0) {
      outcome = kLeaseGateAborted;
    } else if (expired == gen) {
      outcome = kLeaseGateTimeout;
    } else {
      return false;
    }
    // A releaser that read an older `released` loses here to the one that opened G, or to one that opened a later
    // request (then G's gate was opened before that request could arm).
    if (!gate_released_.compare_exchange_strong(released, gen, std::memory_order_acq_rel)) return false;
    if (outcome == kLeaseGateTimeout) {
      std::fprintf(
          stderr,
          "ERROR %scopy wait of request %llu timed out after %.3f s; failing stop\n",
          error_prefix<Layout>().c_str(),
          static_cast<unsigned long long>(gen),
          static_cast<double>(copy_wait_timeout_ns_) / 1e9);
      std::fflush(stderr);
      // Before the gate, and sticky: a CW open that overwrites this timeout still meets it in the commit kernel.
      raise_fatal(seq);
    }
    // Only G's exact closed word changes: a CAS, not a check and a store, so a releaser stalled here past CW's own
    // open of G and the next request's close finds closed(G + 1) and changes nothing. A locked cmpxchg on the host
    // line is atomic against the device's posted stores to it.
    uint32_t expected = closed;
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(area_c + kLeaseCopyGate),
        &expected,
        gate_word(seq, outcome),
        false,
        __ATOMIC_SEQ_CST,
        __ATOMIC_ACQUIRE);
    return true;
  }

  // Teardown with no service to follow (RamThread's stop, via the FFI's stop_thread and close): raise the header's
  // shutdown word and open an armed gate, so an in-flight replay's copy wait ends aborted instead of never. Only with
  // the copy engine, the one thing that arms a copy wait; the shutdown word is sticky, like close_admission's.
  void abort_copy_waits() {
    if (lease_ == nullptr || copy_engine_ == nullptr) return;
    copy_waits_aborted_.store(true);  // RamThread::start refuses from now on: the shutdown word never clears
    store_release(lease_ + kLeaseHeaderShutdown, 1u);
    std::atomic_thread_fence(std::memory_order_seq_cst);  // the shutdown word before the read of CopyArm
    release_copy_gate();
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

  // Native prefetch (plan 2026-09-25-dsv41-native-prefetch): serve the device's advisory next-layer copy requests
  // from `page` (kPrefetchPageBytes, pinned). Needs the copy engine; before the service thread starts.
  void enable_native_prefetch(uint8_t* page) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable native prefetch before the service thread starts");
    if (copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "native prefetch needs the copy engine");
    if (page == nullptr) throw std::runtime_error(error_prefix<Layout>() + "native prefetch needs its page");
    prefetch_page_ = page;
    last_prefetch_gen_ = generation_of(load_acquire64(page + kPfReqGen));
    judge_.assign(static_cast<size_t>(layers_), PrefetchJudge{});
  }

  // Serve the device's posted prefetch request, if a new one is there. True when it handled one. Called after
  // pump_demand, so a demand posted meanwhile is always served first. A prefetch never reads NVMe: a row that is not
  // READY in the pinned tier is skipped. A served one leases its pinned slot like a COPYING lane and is handed to the
  // copy thread; only that thread's observed completion, handed back to the owner, publishes COPIED and releases the
  // lease (prefetch_completed_owned).
  bool pump_prefetch() {
    if (prefetch_page_ == nullptr) return false;
    const uint64_t word = load_acquire64(prefetch_page_ + kPfReqGen);
    if (tag_of(word) != kPfTagRequest) return false;
    const uint64_t gen = generation_of(word);
    if (gen == last_prefetch_gen_) return false;
    int32_t row = -1, expert = -1, dst = -1;
    std::memcpy(&row, prefetch_page_ + kPfReqRow, 4);
    std::memcpy(&expert, prefetch_page_ + kPfReqExpert, 4);
    std::memcpy(&dst, prefetch_page_ + kPfReqDst, 4);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire64(prefetch_page_ + kPfReqGen) != word) return false;  // rewritten under us: read it again
    last_prefetch_gen_ = gen;
    count<kPrefetchRequests>();
    int64_t read_ns = 0;
    if constexpr (Build::kMetrics) read_ns = now_ns();  // prefetch_latency_ns, a metric
    uint32_t skip = 0;
    CopyJob job;
    if (admission_closed_.load() || copy_engine_ == nullptr || !copy_armed_.load(std::memory_order_acquire) ||
        load_acquire(page_ + kFatal) != 0) {
      skip = kPfSkipUnarmed;
    } else if (row < 0 || row >= layers_ || expert < 0 || expert >= experts_ || !copy_engine_->eligible(row, dst)) {
      skip = kPfSkipInvalid;
    } else {
      Tier& tier = tiers_[row];
      const int32_t slot = tier.expert_slot[expert];
      if (slot < 0 || tier.state[slot] != kReady || tier.slot_to_expert[slot] != expert || prefetch_lease_.active) {
        skip = kPfSkipNotReady;
      } else {
        tier.leases[slot] += 1;  // E1: the slot is neither evicted nor rewritten while the copy may read it
        tier.stamp[slot] = ++tick_;
        disown_locked(tier, slot);
        prefetch_lease_ = PrefetchLease{true, gen, row, slot, tier.generation[slot], expert};
        job.gen = gen;
        job.idx = -1;
        job.row = row;
        job.mask = 1u;
        job.count = 1;
        job.lanes[0] = CopyLane{0, slot, dst, tier.generation[slot]};
        job.submit_ns = read_ns;
        job.prefetch = true;
      }
    }
    if (skip != 0) {
      if (skip == kPfSkipUnarmed) {
        count<kPrefetchSkippedUnarmed>();
      } else if (skip == kPfSkipNotReady) {
        count<kPrefetchSkippedNotReady>();
      } else {
        count<kPrefetchSkippedInvalid>();
      }
      publish_prefetch_done(kPfTagSkipped, gen, skip);
      return true;
    }
    count<kPrefetchIssued>();
    copy_engine_->submit(job);
    return true;
  }

  // Test only: the service's prefetch lease, {active, row, slot}. A snapshot (run_as_owner).
  void prefetch_lease(int64_t* out) {
    snapshot(0, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      c.out[0] = self->prefetch_lease_.active ? 1 : 0;
      c.out[1] = self->prefetch_lease_.row;
      c.out[2] = self->prefetch_lease_.slot;
      return 0;
    });
  }

  // The owner (the service thread, the pausing caller, or the caller of pump()): release every lease the copy thread
  // handed back, in completion order. Takes no lock: the copy thread touches no tier state (it only pushes), and
  // every other writer of what this changes is the owner itself. An empty ring costs two loads (its own tail, relaxed;
  // the producer's head, acquire) and no store.
  void drain_copy_completions() {
    CopyJob job;
    while (copy_done_.pop(&job)) {
      if (job.prefetch) {
        prefetch_completed_owned(job);
      } else {
        release_copied_owned(job);
      }
    }
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
    retire_leases(true);
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

  // Shutdown, first step (LEASE_PROTOCOL.md 14.3 S1): the header word tells the device to stop waiting on the service,
  // and the service serves nothing new. Retirement goes on, so acknowledgements of work already in flight still land.
  void close_admission() {
    admission_closed_.store(true);
    if (lease_ != nullptr) store_release(lease_ + kLeaseHeaderShutdown, 1u);
    // The shutdown word before the read of CopyArm: a wait armed before it is opened here, one armed after it opens
    // itself (CW re-reads the word after arming).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    release_copy_gate();
  }

  // Test only: sleep `ns` between serving a demand and storing demand_done. InstrBuild only.
  void inject_done_stall(int64_t ns) {
    if constexpr (!Build::kFaults) {
      (void)ns;
      test_only("inject_done_stall");
    } else {
      faults_.done_stall_ns.store(ns);
    }
  }

  // The seqlock read of the device's lane request for `seq`: false when a later request has already overwritten it.
  bool read_lane_request(uint32_t seq, Request* request) const {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* base = lease_ + lease_d_ + kLeaseLaneRequest + idx * kLeaseLaneRequestBytes;
    const uint64_t word = load_acquire64(base + kLeaseLrGen);
    if (tag_of(word) == 0 || (generation_of(word) & 0xFFFFFFFFull) != seq) return false;
    uint32_t count = 0, row = 0;
    std::memcpy(&count, base + kLeaseLrCount, 4);
    std::memcpy(&row, base + kLeaseLrRow, 4);
    int32_t experts[kLeaseLanes];
    int32_t dst[kLeaseLanes];
    uint32_t flags = 0;
    std::memcpy(experts, base + kLeaseLrExpert, sizeof(experts));
    std::memcpy(dst, base + kLeaseLrDst, sizeof(dst));
    std::memcpy(&flags, base + kLeaseLrFlags, 4);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire64(base + kLeaseLrGen) != word) return false;
    if (count > static_cast<uint32_t>(kLeaseLanes) || static_cast<int64_t>(row) != request->row) return false;
    request->gen = generation_of(word);
    request->lane_experts.assign(experts, experts + count);
    request->lane_dst.assign(dst, dst + count);
    request->lane_flags = flags;
    return true;
  }

  // The device has published a terminal for this request: it will never read a source for the lanes it names.
  bool terminal_seen(const Request& request) const {
    return terminal_seen_for(request.seq, request.gen);
  }

  bool terminal_seen_for(uint32_t seq, uint64_t gen) const {
    const int64_t idx = static_cast<int64_t>((seq - 1u) % kDemandRecords);
    const uint8_t* base = lease_ + lease_d_ + kLeaseTerminal + idx * kLeaseTerminalBytes;
    const uint64_t word = load_acquire64(base + kLeaseTermGen);
    return generation_of(word) == gen && tag_of(word) != 0;
  }

  enum class Defer { kNone, kVictims, kRequestSlot };

  // Would this armed demand have to wait for a lease to retire? Counts, changes nothing. The request slot rule (lease
  // mode only): the slot's previous lease row must be fully retired before it is reused. The victim rule: the tier
  // could serve the request only if leased slots were victims (LEASE_PROTOCOL.md section 8, and 20.2c: a dry run,
  // because the take loop evicts a victim per call and keeps that eviction when the request then fails).
  Defer defers(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return Defer::kNone;
    if (lease_mode_ && !request.lane_experts.empty() &&
        outstanding_[static_cast<int64_t>((request.seq - 1u) % kDemandRecords)].active) {
      return Defer::kRequestSlot;
    }
    FixedVec<int32_t, kWanted> wanted;
    for (std::span<const int32_t> ids : {request.protect.span(), request.need.span(), request.lane_experts.span()}) {
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

  // A deferred demand is retried only when a lease was released since it was last refused, or the device gave up on it.
  bool deferral_may_retry() const {
    if (lease_changes_ != deferred_stamp_) return true;
    return lease_mode_ && terminal_seen_for(deferred_seq_, deferred_gen_);
  }

  // A `_locked` suffix, here and below, now means "the owner's": the tier mutex it once named is gone (Task 15).
  //
  // S1. Open the ring entry for a request, ONCE, with its full lane count and every lane ungranted.
  //
  // V1 (two-phase) grants a request's lanes in two calls -- the hit lanes inside serve()'s reservation hold, the
  // miss lanes after read() -- so the entry cannot be (re)initialised by the granter: the second call would erase
  // the first call's leases. Opening is therefore its own step. `grants_pending` records that a further grant is
  // still owed, which is what keeps the entry open across the gap (S4); retire_leases must not close a ring slot
  // whose second grant has not run, or grant_lane_group_locked's `entry.active` guard stops protecting it.
  //
  // False on a still-active ring entry, exactly as the single-phase granter was: nothing is opened then.
  //
  // `grants_pending` is false under piece streaming: the loading grant leases every lane, hit and miss, in the one
  // call that follows, so no second grant is ever owed and S4 has nothing to hold open.
  bool open_lease_entry_locked(const Request& request, bool grants_pending = true) {
    if (request.lane_experts.empty()) return true;
    const size_t count = request.lane_experts.size();
    if (count > kLeaseLanes) return false;
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (entry.active) return false;  // the request slot still holds an unretired lease row
    entry = Outstanding();
    entry.active = true;
    entry.gen = request.gen;
    entry.row = request.row;
    entry.count = static_cast<uint32_t>(count);
    entry.grants_pending = grants_pending;
    return true;
  }

  // No further grant is owed for this request: the entry may close once its granted lanes retire. Called when the
  // second group has been granted, and on every path that answers without granting it (S5). Without this a
  // request that failed between the two grants would hold its ring index open for the life of the process.
  //
  // An entry that never granted a lane has nothing for the device to retire, so nothing would ever clear
  // `active`. Such an entry is retired here instead; one that does hold leases stays active and is retired by the
  // device's acknowledgement or its terminal, which is the whole of S5.
  void close_pending_grants_locked(const Request& request) {
    if (request.lane_experts.empty()) return;
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (!entry.active || entry.gen != request.gen) return;
    entry.grants_pending = false;
    bool held = false;
    for (uint32_t lane = 0; lane < entry.count; ++lane)
      held = held || entry.lane[lane].state == 1;
    if (!held) entry.active = false;
  }

  // S1. Lease the source slot of every lane `select` picks, and publish its row result. Callable twice over
  // disjoint subsets of one open entry. A lease is counted before its row result is published (6.1).
  //
  // All-or-nothing: every selected lane is validated before any is committed, so a false return has granted
  // nothing and left the entry as it found it.
  //
  // The fence is PER GROUP and must stay that way. It separates THIS call's payload writes from THIS call's ready
  // stores; it deliberately does not cover the other group. That is the correctness core of two-phase: the hit
  // group's ready words have to become visible to the device while the miss group's payloads do not yet exist.
  // Hoisting one fence to cover both groups would either publish hit lanes late (losing the entire mechanism) or
  // publish miss lanes whose payloads have not been written (handing the device a torn row result).
  //
  // The loading grant (piece streaming, plan 3.2): `loading` names this request's newly reserved slots, and a lane
  // whose slot is kLoading and among them is granted too, under tag LOADING rather than READY. Hit lanes still need
  // kReady. Both groups' payloads exist at reservation, so one call, one fence covers them without breaking the rule
  // above: nothing is published whose payload is not written. The caller has stored and fenced each miss lane's
  // PieceMask word (init_piece_words_locked) before this call, so the word carries the request's generation before
  // any ready word of the request is visible. Null `loading` is the two-phase grant exactly as it was.
  template <typename Select>
  bool grant_lane_group_locked(
      const Request& request, Select select, bool hit_phase = false, const std::span<const int64_t>* loading = nullptr) {
    if (request.lane_experts.empty()) return true;
    const size_t count = request.lane_experts.size();
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    Outstanding& entry = outstanding_[idx];
    if (!entry.active || entry.gen != request.gen || entry.count != count) return false;
    Tier& tier = tiers_[request.row];
    int32_t slots[kLeaseLanes];
    bool take[kLeaseLanes] = {};
    uint64_t tags[kLeaseLanes] = {};
    size_t taken = 0;
    size_t hits = 0;
    // Hit lanes of the reservation-hold grant go to the copy engine when it is armed and the device allowed it.
    const bool copy_request = hit_phase && copy_engine_ != nullptr && copy_armed_.load(std::memory_order_acquire) &&
                              (request.lane_flags & kLeaseLrFlagCopyEngine) != 0;
    CopyJob job;
    for (size_t lane = 0; lane < count; ++lane) {
      if (!select(lane)) continue;
      if (entry.lane[lane].state != 0) return false;  // granted already: the two groups must be disjoint
      const int32_t expert = request.lane_experts[lane];
      if (expert < 0 || expert >= experts_) return false;
      const int32_t slot = tier.expert_slot[expert];
      if (slot < 0 || tier.slot_to_expert[slot] != expert) return false;
      const bool still_loading = loading != nullptr && tier.state[slot] == kLoading &&
                                 std::find(loading->begin(), loading->end(), slot) != loading->end();
      if (tier.state[slot] != kReady && !still_loading) return false;
      slots[lane] = slot;
      take[lane] = true;
      tags[lane] = still_loading ? kLeaseTagLoading : kLeaseTagReady;
      ++taken;
      if (!still_loading) ++hits;
      if (copy_request && !still_loading) {
        const int32_t dst = lane < request.lane_dst.size() ? request.lane_dst[lane] : -1;
        if (copy_engine_->eligible(request.row, dst)) {
          tags[lane] = kLeaseTagCopying;
          job.lanes[job.count++] = CopyLane{static_cast<int32_t>(lane), slot, dst, tier.generation[slot]};
          job.mask |= 1u << lane;
        } else {
          this->template count<kCopyFallbacks>();  // `count` is also this function's lane count
        }
      }
    }
    if (taken == 0) return true;
    uint8_t* results = lease_ + kLeaseRowResult + idx * kLeaseLanes * kLeaseRowResultBytes;
    // The writer half of the 11.4 re-read: a ready word is cleared before its payload is rewritten, so a device
    // reader whose payload loads overlap the rewrite finds the word changed when it re-reads it. Without the clear
    // the word would still read the old generation until the store below, and a torn payload would pass.
    for (size_t lane = 0; lane < count; ++lane) {
      if (take[lane]) store_release64(results + lane * kLeaseRowResultBytes + kLeaseRrReady, 0);
    }
    _mm_sfence();
    for (size_t lane = 0; lane < count; ++lane) {
      if (!take[lane]) continue;
      const int32_t slot = slots[lane];
      tier.leases[slot] += 1;
      entry.lane[lane] = LaneLease{1, slot, tier.generation[slot], false, tags[lane] == kLeaseTagCopying};
      uint8_t* result = results + lane * kLeaseRowResultBytes;
      const uint32_t generation = tier.generation[slot];
      const int32_t expert = request.lane_experts[lane];
      const uint16_t row16 = static_cast<uint16_t>(request.row), lane16 = static_cast<uint16_t>(lane);
      std::memcpy(result + kLeaseRrSlotGeneration, &generation, 4);
      std::memcpy(result + kLeaseRrHostSlot, &slot, 4);
      std::memcpy(result + kLeaseRrExpert, &expert, 4);
      std::memcpy(result + 20, &row16, 2);
      std::memcpy(result + 22, &lane16, 2);
    }
    _mm_sfence();  // THIS group's payloads land before THIS group's ready words -- see the note above
    for (size_t lane = 0; lane < count; ++lane) {
      if (!take[lane]) continue;
      store_release64(results + lane * kLeaseRowResultBytes + kLeaseRrReady, tagged_word(tags[lane], request.gen));
    }
    // One writer (the owner): a relaxed load and store, no RMW. graph_leases_outstanding reads it lock-free.
    lanes_outstanding_.store(
        lanes_outstanding_.load(std::memory_order_relaxed) + static_cast<int64_t>(taken), std::memory_order_relaxed);
    this->template count<kLeasesGranted>(static_cast<int64_t>(taken));
    if (hit_phase) this->template count<kHitLeasesGranted>(static_cast<int64_t>(hits));  // S7: resident lanes
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
    return true;
  }

  // Lease every lane's source slot and publish its row result, in one critical section, after the rows are ready
  // and before the caller answers. A lease is counted before its row result is published (6.1). False on an
  // internal inconsistency: nothing is granted then.
  //
  // The single-phase composition, kept for the non-two-phase path: open the entry and grant every lane at once.
  bool grant_lanes_locked(const Request& request) {
    if (!open_lease_entry_locked(request)) return false;
    const bool granted = grant_lane_group_locked(request, [](size_t) { return true; });
    close_pending_grants_locked(request);  // retires the entry outright when the grant granted nothing
    return granted;
  }

  // Retire the leases the device has acknowledged, or voided with a terminal, without waiting for either. Cheap when
  // nothing is outstanding. Called from the service loop; never blocks (LEASE_PROTOCOL.md 7.5, 16).
  //
  // lease_double_signal: a lane retired by one signal is compared against the other in every later pass, not only
  // in the pass that retired it, so a second word that lands after the entry closed is still counted. A closed
  // entry is watched until a `settle` pass: one that runs after the caller observed a later demand posted (or
  // synchronized the stream). Under A1 the device emitted every signal of a request before posting the next one,
  // so that pass compares each closed entry's words for the last time and ends its watch. The idle early-out
  // skips unsettled watches, which the next posted demand's settle pass then covers (or pause, or stop_thread's
  // final settle). Clearing `watched` at a settle only bounds the cost of later passes: correctness does not depend
  // on it, since every comparison matches on the generation and a ring index's reuse resets the entry anyway.
  //
  // The owner only, and no lock (Task 14): with the copy thread's releases moved to the owner, nothing else writes an
  // entry or a lease while the owner runs. The service thread calls it every loop turn and every read turn, so a
  // mutex here was a lock per turn.
  void retire_leases(bool settle = false) {
    if (lease_ == nullptr || (!settle && lanes_outstanding_.load(std::memory_order_relaxed) == 0)) return;
    for (int64_t idx = 0; idx < kDemandRecords; ++idx) {
      Outstanding& entry = outstanding_[idx];
      if (!entry.active && !entry.watched) continue;
      Tier& tier = tiers_[entry.row];
      const uint8_t* acks = lease_ + lease_d_ + kLeaseLaneAck + idx * kLeaseLanes * kLeaseLaneAckBytes;
      const uint8_t* terminal = lease_ + lease_d_ + kLeaseTerminal + idx * kLeaseTerminalBytes;
      const uint64_t stamp = load_acquire64(terminal + kLeaseTermGen);
      const bool terminated = tag_of(stamp) != 0 && generation_of(stamp) == entry.gen;
      uint32_t mask = 0;
      if (terminated) std::memcpy(&mask, terminal + kLeaseTermSkippedMask, 4);
      for (uint32_t lane = 0; lane < entry.count; ++lane) {
        LaneLease& held = entry.lane[lane];
        // No kernel reads a COPYING lane's slot, so no LaneAck or Terminal bit can release it; its copy's completion
        // does.
        if (held.copy_engine) continue;
        const uint64_t word = load_acquire64(acks + lane * kLeaseLaneAckBytes);
        const bool acknowledged = tag_of(word) != 0 && generation_of(word) == entry.gen;
        const bool voided = terminated && (mask >> lane & 1u) != 0;
        if (held.state == 1) {
          if (acknowledged) {
            this->template release_lease_locked<kLeasesAcked>(tier, held);  // declared below
            entry.watched = true;
          } else if (voided) {
            this->template release_lease_locked<kLeasesVoided>(tier, held);
            entry.watched = true;
          }
          // Plan 3.3: the last lease of a quarantined slot frees it. Its mapping was cleared on entry, so
          // release_locked unmaps nothing -- the expert may already live in another slot (M3). A kLoading slot only
          // loses the lease here; serve()'s post-read step decides it.
          if (held.state != 1 && tier.state[held.slot] == kQuarantine && !leased_locked(tier, held.slot)) {
            release_locked(entry.row, held.slot);
          }
        }
        // A second signal for a lane already released, in this pass or any later one while the entry is watched:
        // counted once, and it releases nothing. Both words carry the generation, so a later request's signal on
        // the same lane or ring index never matches entry.gen.
        if (!held.counted && ((held.state == 2 && voided) || (held.state == 3 && acknowledged))) {
          held.counted = true;
          count<kLeaseDoubleSignal>();
        }
      }
      // S4. A lane whose grant has not run yet is still open. Under V1 the miss lanes sit ungranted between the
      // two grants, so counting only state == 1 would clear `active` while a grant is still pending -- after
      // which grant_lane_group_locked's `entry.active` guard no longer protects this ring slot.
      bool open = entry.grants_pending;
      for (uint32_t lane = 0; lane < entry.count; ++lane)
        open = open || entry.lane[lane].state == 1;
      if (!open) entry.active = false;
      if (settle && !entry.active) entry.watched = false;
    }
  }

  // Lanes leased by the device and not yet retired. An eager pause is refused while this is non-zero; a lease held by
  // anything else (a promotion, in Task 8) is not counted, because it protects its own slot (LEASE_PROTOCOL.md 17.1
  // R2).
  int64_t graph_leases_outstanding() const {
    return lanes_outstanding_.load(std::memory_order_relaxed);
  }

  // One lease released, exactly once: the per-lane state machine is what makes a second signal harmless. Called by the
  // owner only (retire_leases, drain_copy_completions). K is a metric (leases_acked, _voided, _copied): kept so.
  template <Counter K>
  void release_lease_locked(Tier& tier, LaneLease& held) {
    static_assert(!is_core_counter(K), "the lease release counters are metrics");
    if (tier.leases[held.slot] == 0) {
      throw std::runtime_error(error_prefix<Layout>() + "lease underflow on slot " + std::to_string(held.slot));
    }
    tier.leases[held.slot] -= 1;
    held.state = K == kLeasesVoided ? 3 : 2;
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

  // Test hooks and introspection. slot_info: [state, expert, leases, generation] per slot. A snapshot.
  void slot_info(int64_t row, int64_t* out) {
    row_capacity(row);  // range-checked on the caller: the owner may be the service thread, which must not throw
    snapshot(row, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Tier& tier = self->tiers_[c.row];
      for (int64_t slot = 0; slot < tier.capacity; ++slot) {
        c.out[4 * slot] = tier.state[slot];
        c.out[4 * slot + 1] = tier.slot_to_expert[slot];
        c.out[4 * slot + 2] = tier.leases[slot];
        c.out[4 * slot + 3] = tier.generation[slot];
      }
      return 0;
    });
  }

  // Test only: the service's account of request slot `idx`: [active, grants_pending, count, gen, then per lane
  // (kLeaseLanes) its state, then per lane its slot, then per lane 1 if it is a copy-engine lane]. A snapshot.
  void lease_entry(int64_t idx, int64_t* out) {
    if (idx < 0 || idx >= kDemandRecords)
      throw std::runtime_error(error_prefix<Layout>() + "request slot out of range");
    snapshot(idx, out, nullptr, [](RamTier* self, const Command& c) -> int64_t {
      const Outstanding& entry = self->outstanding_[c.row];
      c.out[0] = entry.active;
      c.out[1] = entry.grants_pending;
      c.out[2] = entry.count;
      c.out[3] = static_cast<int64_t>(entry.gen);
      for (int64_t lane = 0; lane < kLeaseLanes; ++lane) {
        c.out[4 + lane] = entry.lane[lane].state;
        c.out[4 + kLeaseLanes + lane] = entry.lane[lane].slot;
        c.out[4 + 2 * kLeaseLanes + lane] = entry.lane[lane].copy_engine ? 1 : 0;
      }
      return 0;
    });
  }

  // Test only: the service grants leases itself from step 3; until then a test stands in for the device's holder.
  // A command (run_as_owner) that the caller waits for, so a test's next post sees the lease. InstrBuild only.
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
  void layer_rows(int64_t* out, bool advisory) const {
    for (int64_t row = 0; row < layers_; ++row)
      out[row] = __atomic_load_n(advisory ? &tiers_[row].rows_advisory : &tiers_[row].rows_demand, __ATOMIC_RELAXED);
  }

  // Test-only faults: sleep `delay_ns` before each advisory read and before each demand
  // read once `after_demands` demands have read rows; report reads as failed; make an advisory
  // give up once `abandon_after_batches` of its batches (rows) were admitted (0: never).
  // InstrBuild only.
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
    if constexpr (!Build::kFaults) {
      (void)delay_ns, (void)fail_reads, (void)after_demands, (void)abandon_after_batches;
      test_only("inject");
    } else {
      faults_.delay_ns.store(delay_ns);
      faults_.fail_reads.store(fail_reads);
      faults_.delay_after.store(after_demands);
      faults_.abandon_after.store(abandon_after_batches);
    }
  }

  // Test only: carry a whole ReadFault down to this tier's reader, where inject() reaches it only as a
  // delay, a blanket failure or an abandon point. `words` is the reader tests' fault tensor (kFaultWords
  // int64; see fault_from). Unlike fail_reads the fault does NOT short-circuit ahead of the reader: the read
  // runs, so the fault's part errors, pack delay and the rest act on rows that have already packed. The
  // service thread applies it just before its next read (the reader is that thread's alone), and it then
  // stays until replaced; an all-default tensor clears it. Words 17-18 (abandon_after, step) and 22
  // (piece_stream) are not faults and are ignored, and words 19-20 (formerly pack_workers, pack_split) are reserved:
  // use inject() for the abandon point and set_piece_stream() for the mode. The reader's counters (submit and
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

  // Copy thread. Completion was observed, so no copy of this job reads its slots any more. E6 here, on the words this
  // thread may read (slot_gen_, written by the owner with release): a leased slot's generation cannot move (E1); if
  // one did, the bytes are not the lease's, so fail stop, publish nothing and hand nothing back (E5: the leases stay
  // held). Then CopyDone, here, so the device's wait ends as soon as the DMA did; the lease itself is released by the
  // owner (drain_copy_completions), one owner poll later (D7). A job whose row has SM entries is not handed back yet
  // (false): the copy wait still reads those slots, and copy_acked hands it back once it acknowledged. True otherwise.
  // A prefetch job publishes nothing here: PrefetchDone is the owner's (prefetch_completed_owned).
  bool copy_completed(const CopyJob& job) {
    const uint32_t* generations = slot_gen_ + slot_gen_base_[job.row];
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      if (load_acquire(reinterpret_cast<const uint8_t*>(generations + lane.host_slot)) != lane.slot_generation) {
        copy_count<kCopyGenerationMismatches>();
        raise_fatal(static_cast<uint32_t>(job.gen));
        return true;  // E5: the lease stays held
      }
    }
    if (!job.prefetch) {
      uint8_t* done = lease_ + lease_c_ + job.idx * kLeaseCopyDoneBytes;
      std::memcpy(done + kLeaseCdMask, &job.mask, 4);
      store_release64(done + kLeaseCdGen, tagged_word(kLeaseTagCopied, job.gen));
      std::atomic_thread_fence(std::memory_order_seq_cst);  // CopyDone before the read of CopyArm
      release_copy_gate();
      if (job.sm) return false;  // handed back once the copy wait acknowledged its SM reads (copy_acked)
    }
    hand_back(job);
    return true;
  }

  // Copy thread, a job copy_completed left waiting: true once the copy wait's SmAck for its ring index carries this
  // request's generation or a later one (the copy wait of a later request in the index ran, so this one's finished),
  // after which no SM read of the job's slots can be in flight, and the job is handed back to the owner.
  bool copy_acked(const CopyJob& job) {
    const uint64_t word = load_acquire64(lease_ + lease_d_ + kLeaseSmAck + job.idx * kLeaseSmAckBytes);
    if (tag_of(word) != kLeaseTagSmAck || generation_of(word) < job.gen) return false;
    hand_back(job);
    return true;
  }

  // Copy thread: the job goes back to the owner through copy_done_, before the copy engine counts it finished (its
  // caller finish()es after this returns), so an owner that saw the engine idle (wait_idle's acquire of finished_)
  // pops it. At most kDemandRecords + 1 jobs are outstanding, handed-back ones included (a ring index's entry stays
  // active, and the prefetch lease stays active, until the owner drains its job), and the ring holds kCopyRing: a
  // full ring is an internal error, and the answer that keeps every lease held (E5) is to fail stop.
  void hand_back(const CopyJob& job) {
    if (!copy_done_.push(job)) {
      copy_count<kCopyErrors>();
      raise_fatal(static_cast<uint32_t>(job.gen));
    }
  }

  // The owner: release a completed job's COPYING leases, the only place one is released.
  void release_copied_owned(const CopyJob& job) {
    Outstanding& entry = outstanding_[job.idx];
    if (!entry.active || entry.gen != job.gen) {
      // Nothing but this completion releases a COPYING lane, so its entry cannot have closed: an internal error.
      count<kCopyErrors>();
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    Tier& tier = tiers_[entry.row];
    for (int i = 0; i < job.count; ++i) {
      LaneLease& held = entry.lane[job.lanes[i].lane];
      if (held.state == 1 && held.copy_engine) release_lease_locked<kLeasesCopied>(tier, held);
    }
    bool open = entry.grants_pending;
    for (uint32_t lane = 0; lane < entry.count; ++lane)
      open = open || entry.lane[lane].state == 1;
    if (!open) entry.active = false;
  }

  // The owner, a prefetch job the copy thread handed back (its E6 check passed there): the judge entry for the
  // target's next request and COPIED, then the lease release.
  void prefetch_completed_owned(const CopyJob& job) {
    if (!prefetch_lease_.active || prefetch_lease_.gen != job.gen) {
      count<kCopyErrors>();
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    Tier& tier = tiers_[prefetch_lease_.row];
    if (tier.leases[prefetch_lease_.slot] == 0) {
      count<kCopyErrors>();
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    judge_[job.row] = PrefetchJudge{true, prefetch_lease_.expert};
    count<kPrefetchCopied>();
    if constexpr (Build::kMetrics) count<kPrefetchLatencyNs>(now_ns() - job.submit_ns);
    publish_prefetch_done(kPfTagCopied, job.gen, 0);
    tier.leases[prefetch_lease_.slot] -= 1;
    ++lease_changes_;  // a demand deferred on this slot may retry
    prefetch_lease_.active = false;
  }

  void publish_prefetch_done(uint64_t tag, uint64_t gen, uint32_t reason) {
    store_release(prefetch_page_ + kPfDoneReason, reason);
    store_release64(prefetch_page_ + kPfDoneGen, tagged_word(tag, gen));
  }

  // Service thread, every record read: the first request of a row after a COPIED prefetch into it says whether the
  // row was routed. Every layer posts a record per forward (touch-only when it misses nothing), so each copied row
  // is judged by its target layer's own forward.
  void judge_prefetch(const Request& request) {
    if (prefetch_page_ == nullptr || request.row < 0 || request.row >= layers_) return;
    PrefetchJudge& judge = judge_[request.row];
    if (!judge.pending) return;
    judge.pending = false;
    const bool used = listed(request.protect, judge.expert) || listed(request.need, judge.expert);
    if (used) {
      count<kPrefetchUsed>();
    } else {
      count<kPrefetchWasted>();
    }
  }

  // Copy thread, or the submitter when the copy engine's job ring is full (an internal error: error -1000).
  // Completion cannot be established: the leases stay held (E5) and the page fails stop. The message is built only
  // here, on the error path.
  void copy_failed(const CopyJob& job, int error) {
    std::fprintf(
        stderr,
        "ERROR %s copy of request %llu failed (%d); leases held\n",
        (std::string(Layout::kName) + " RAM miss copy engine:").c_str(),
        static_cast<unsigned long long>(job.gen),
        error);
    std::fflush(stderr);
    raise_fatal(static_cast<uint32_t>(job.gen));
  }

  void raise_fatal(uint32_t seq) {
    uint32_t zero = 0;
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(page_ + kFatal),
        &zero,
        seq == 0 ? 0xFFFFFFFFu : seq,
        false,
        __ATOMIC_RELEASE,
        __ATOMIC_RELAXED);
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

  // The service is the only writer of the header, the row table and SlotGen, and writes them before the
  // service thread or any device exists, so plain stores and one fence suffice.
  void init_lease_block(const std::vector<int64_t>& capacity, int64_t lease_bytes) {
    if (reinterpret_cast<uintptr_t>(lease_) % 4096 != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "the lease block must be 4096-byte aligned");
    }
    if (layers_ > (kLeaseRowResult - kLeaseRowTable) / 8) {
      throw std::runtime_error(error_prefix<Layout>() + "too many rows for the lease block's row table");
    }
    slot_gen_base_.assign(static_cast<size_t>(layers_), 0);
    int64_t total_slots = 0;
    for (int64_t row = 0; row < layers_; ++row) {
      slot_gen_base_[row] = total_slots;
      total_slots += capacity[row];
    }
    const int64_t d_offset = round_up_page(kLeaseSlotGen + 4 * total_slots);
    lease_d_ = d_offset;
    const int64_t piece_offset = round_up_page(d_offset + kLeaseSmAck + kLeaseRing * kLeaseSmAckBytes);
    lease_p_ = piece_offset;
    const int64_t copy_offset = round_up_page(piece_offset + kLeaseAreaPieceMaskBytes);
    lease_c_ = copy_offset;
    const int64_t needed = round_up_page(copy_offset + kLeaseAreaCBytes);
    if (lease_bytes < needed) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the lease block has " + std::to_string(lease_bytes) + " bytes, its layout needs " +
          std::to_string(needed));
    }
    auto put_u32 = [&](int64_t offset, uint32_t value) { std::memcpy(lease_ + offset, &value, 4); };
    put_u32(0, 0x4C534531u);  // "LSE1"
    put_u32(4, 4u);           // ABI version (exl3_lease_block.ABI_VERSION; 4: area C's copy-wait gate)
    put_u32(kLeaseHeaderRing, static_cast<uint32_t>(kLeaseRing));
    put_u32(kLeaseHeaderLanes, static_cast<uint32_t>(kLeaseLanes));
    put_u32(16, static_cast<uint32_t>(layers_));
    put_u32(kLeaseHeaderShutdown, 0u);
    put_u32(kLeaseHeaderSlotGenOffset, static_cast<uint32_t>(kLeaseSlotGen));
    put_u32(kLeaseHeaderDOffset, static_cast<uint32_t>(d_offset));
    put_u32(kLeaseHeaderPieceOffset, static_cast<uint32_t>(piece_offset));
    put_u32(kLeaseHeaderCopyOffset, static_cast<uint32_t>(copy_offset));
    for (int64_t row = 0; row < layers_; ++row) {
      put_u32(kLeaseRowTable + 8 * row, static_cast<uint32_t>(slot_gen_base_[row]));
      put_u32(kLeaseRowTable + 8 * row + 4, static_cast<uint32_t>(capacity[row]));
    }
    // Open, with nothing armed: a copy wait that arms nothing passes its stream wait on this value.
    put_u32(copy_offset + kLeaseCopyGate, gate_word(0, kLeaseGateOpen));
    std::memset(lease_ + copy_offset + kLeaseCopyArm, 0, 8);
    slot_gen_ = reinterpret_cast<uint32_t*>(lease_ + kLeaseSlotGen);
    _mm_sfence();
  }

  // A slot is about to hold different bytes: bump its generation, and fence it ahead of the first byte store, so
  // that a GPU reader that re-reads the generation after copying (LEASE_PROTOCOL.md 6.5) sees any rewrite.
  void bump_generation_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const uint32_t next = ++tier.generation[slot];
    if (slot_gen_ != nullptr) {
      store_release(reinterpret_cast<uint8_t*>(slot_gen_ + slot_gen_base_[row] + slot), next);
      _mm_sfence();
    }
  }

  // The eviction predicate's lease half. Task 8 adds host leases here as one more term.
  bool leased_locked(const Tier& tier, int64_t slot) const {
    return tier.leases[slot] > 0;
  }

  // A kQuarantine slot is counted as none of free, evictable or leased: it can never help a deferred request, so it
  // must not make one wait (plan 3.2).
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

  // Takes only a kFree slot or evicts an unleased kReady one: kLoading and kQuarantine slots are never taken.
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

  // Plan 3.2: a leased kLoading slot whose read failed. The mapping is cleared at once, both ways, so the expert can
  // be read into another slot by the next request, and the later release_locked (retire_leases, on the last lease)
  // finds no expert to unmap: it cannot unmap the expert from the slot it was re-read into (M3). Nothing is
  // published, because the map was never published for a kLoading slot. The lease is kept: this slot's bytes may be
  // under a device copy until the lane is acknowledged or voided.
  void quarantine_locked(int64_t row, int64_t slot) {
    Tier& tier = tiers_[row];
    const int32_t expert = tier.slot_to_expert[slot];
    if (expert >= 0 && tier.expert_slot[expert] == slot) tier.expert_slot[expert] = -1;
    tier.slot_to_expert[slot] = -1;
    tier.state[slot] = kQuarantine;
    disown_locked(tier, slot);
    count<kSlotsQuarantined>();
  }

  bool demand_pending() const {
    return !reached(next_demand_ - 1u, load_acquire(page_ + kDemandHead));
  }

  // An unarmed demand record: nobody waits on it, so the device may already be gathering
  // any mapped slot (the next token's rows too, once the thread lags). Only refresh the
  // recency of its assigned rows: no eviction, no read. False for an invalid record.
  bool touch_request(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return false;
    Tier& tier = tiers_[request.row];
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (expert < 0 || expert >= experts_) return false;
        const int32_t slot = tier.expert_slot[expert];
        if (slot >= 0) {
          tier.stamp[slot] = ++tick_;
          disown_locked(tier, slot);
        }
      }
    }
    return true;
  }

  // An armed demand or an advisory: touch the request's assigned rows; read every protected
  // or needed expert that is not assigned (D12's recompute: the device is waiting on this
  // record, so no gather is in flight), evicting only unprotected, non-hot READY rows; publish.
  // An advisory protects only its own ids, has at most one row's I/O outstanding, and stops
  // submitting more when a demand is posted, a pause or a stop is requested. It reaps what it
  // already submitted, publishes the rows that completed and packed (each is a whole, valid row and
  // enters the tier as any READY row: evictable under the usual protection rules) and releases the rest.
  // A demand publishes nothing unless every row landed. *rows: the rows it read (0 when it failed).
  bool serve(const Request& request, bool advisory, int64_t* rows, bool* deferred = nullptr) {
    *rows = 0;
    if (deferred != nullptr) *deferred = false;
    StageRecord* const cur = stage_record();  // null in ProdBuild, so every `if (cur)` below folds away
    if (cur) cur->lanes = request.lanes;
    // Fixed-size locals (spec A4, A5): a request names at most kWanted distinct experts, so none of these allocates.
    FixedVec<int32_t, kWanted> wanted;
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (!listed(wanted, expert)) wanted.push_back(expert);  // one slot per expert (device bytes may repeat)
      }
    }
    // A lane's own expert must never be a victim of the request that leases it, whatever the post kernel protected.
    for (int32_t expert : request.lane_experts) {
      if (!listed(wanted, expert)) wanted.push_back(expert);
    }
    FixedVec<int32_t, kWanted> missing;
    FixedVec<int64_t, kWanted> slots;
    bool ok = request.row >= 0 && request.row < layers_;
    // Piece streaming publishes into the lease block's readiness words, initialised in the reservation hold below
    // before the two-phase hit grant. Without two-phase and lease mode there is neither, so refuse before any slot
    // is taken (the service refuses the flag too; this is the tier's own guard).
    const bool piece_stream = reader_.piece_stream();
    if (ok && piece_stream && !(two_phase_ && lease_mode_)) {
      count<kPieceStreamRefused>();
      ok = false;
    }
    bool publishing = false;  // piece streaming: this request's miss lanes have readiness words to publish into
    if (ok) {
      Tier& tier = tiers_[request.row];
      for (int32_t expert : wanted) {
        if (expert < 0 || expert >= experts_) {
          ok = false;
          break;
        }
        const int32_t slot = tier.expert_slot[expert];
        if (slot >= 0) {
          tier.stamp[slot] = ++tick_;
          disown_locked(tier, slot);
        } else {
          missing.push_back(expert);
        }
      }
      if (ok && !missing.empty()) {
        // A request the tier could serve only once leases retire is refused BEFORE any slot is taken: the take
        // loop unmaps a victim per call and keeps that eviction when the request then fails, so a refusal that
        // ran it would evict a row on every retry. A demand is counted as deferred (step 3 retries it); an
        // advisory simply gives up. When leases could not help either, the loop runs as it always has.
        const VictimCensus census = census_locked(request.row, wanted);
        const int64_t want = static_cast<int64_t>(missing.size());
        if (census.free + census.evictable < want && census.free + census.evictable + census.leased >= want) {
          if (!advisory) {
            count<kDeferred>();
            if (deferred != nullptr) *deferred = true;
          }
          ok = false;
        }
      }
      for (size_t i = 0; ok && i < missing.size(); ++i) {
        int64_t evicted = -1;
        const int64_t slot = take_slot_locked(request.row, wanted, false, &evicted);
        if (slot < 0) {
          ok = false;
          break;
        }
        bump_generation_locked(request.row, slot);  // before any byte of the new row is written
        tier.slot_to_expert[slot] = missing[i];
        tier.state[slot] = kLoading;
        tier.expert_slot[missing[i]] = static_cast<int32_t>(slot);
        slots.push_back(slot);
      }
      // S2. Grant and publish the HIT lanes here: in the same reservation hold as the take loop, after it has
      // completed with ok still true, and before the hold ends at read(). The hold is no lock since Task 15: it is
      // the stretch of the owner's code from the census to read(), which nothing else runs inside -- no other thread
      // writes the tier (ownership rules 1-3), and the owner's own interleaved passes (drain_copy_completions,
      // retire_leases, answer_snapshots, applied commands) run only between requests or from read()'s progress
      // hook, after this block. A lane is a hit iff its
      // expert is not in `missing`. This is the whole of V1: these row results become visible to the device while
      // the missing rows are still being read, so their copies overlap the read instead of following it.
      //
      // Neither ordering below it is available. Before the take loop, the !ok bail here returns with no lease
      // unwind, and the deferral branch above sets ok = false for a request that is retried under the same seq --
      // either way leases outlive a request that never ran. After the hold ends, the hit slot can be evicted (by a
      // later admission on the owner) in exactly the window this task exists to close, and the grant buys nothing.
      //
      // Piece streaming grants the MISS lanes here too (plan 3.2): under tag LOADING, into the slots just reserved,
      // in the same all-or-nothing call and behind the same one fence, after init_piece_words_locked has stored and
      // fenced their PieceMask words. No grant is then owed after the read, so the entry opens with none pending.
      if (ok && two_phase_ && lease_mode_ && !advisory && !request.lane_experts.empty()) {
        if (!open_lease_entry_locked(request, !piece_stream)) {
          ok = false;
        } else {
          // Piece streaming: each miss lane's readiness word starts this request's generation with no piece bit,
          // fenced before any ready word of the request. Only after the entry opened: an active entry means an
          // older request may still be reading this ring index's words, and opening refuses it.
          if (piece_stream) publishing = init_piece_words_locked(request, missing);
          const std::span<const int64_t> slots_span = slots.span();  // after the take loop: every reserved slot
          if (!(piece_stream
                    ? grant_lane_group_locked(
                          request, [](size_t) { return true; }, true, &slots_span)
                    : grant_lane_group_locked(
                          request, [&](size_t lane) { return !listed(missing, request.lane_experts[lane]); }, true))) {
            close_pending_grants_locked(request);  // granted nothing: retire the entry rather than leak the ring slot
            ok = false;
          }
        }
      }
      if (!ok) {
        for (int64_t slot : slots) {
          // S6. §2's first fact as an assertion rather than an argument: `slots` holds only newly taken slots for
          // experts in `missing`, and a hit lane's expert is by definition not in `missing`, so no slot released
          // here is ever leased. release_locked checks nothing, and a leased slot through it is silent
          // corruption -- the next request is handed a slot the GPU may still be reading. Piece streaming leases
          // miss slots in this hold, but only through the grant above, which is all-or-nothing and the last step
          // that can set ok = false: a request that reaches this branch granted nothing, so the rule holds as is.
          assert(!leased_locked(tiers_[request.row], slot));
          release_locked(request.row, slot);
        }
        // Each slot taken may have evicted a row, and that eviction stays: the map moved.
        if (!slots.empty()) count<kVersion>();
        slots.clear();
      }
    }
    if (cur) cur->reserved = stamp(cur);
    int64_t status = kStatusNoRead;
    // Per slot: the row was packed whole (read() sets it). A member, not a local, reserved to kWanted at construction,
    // so read()'s assign() never allocates on the service thread.
    std::vector<uint8_t>& packed = packed_;
    // Cleared, not merely reused: the publish gate below reads packed[i] whenever the vector is long
    // enough, and read() only rewrites it when it actually runs. Carrying the PREVIOUS request's flags
    // into a request that never read would publish a row on the strength of an older row's packing.
    packed.clear();
    bool cancelled = false;
    if (ok && !missing.empty()) {
      bool fail_reads = false;
      int64_t abandon_after = 0;
      if constexpr (Build::kFaults) {  // the test faults (inject, inject_fault): InstrBuild only
        apply_pending_fault();
        const int64_t delay = faults_.delay_ns.load();
        if (delay > 0 && (advisory || demands_read_ >= faults_.delay_after.load())) fault_delay(delay);
        fail_reads = faults_.fail_reads.load();
        abandon_after = faults_.abandon_after.load();
      }
      if (fail_reads) {
        count<kReadErrors>();
        ok = false;
        status = kStatusFailed;
      } else {
        // ProdBuild: the advisory rule alone (abandon_after is the constant 0 there). Passed to read() as its own type
        // (spec A6): its closure is 24 bytes, which a std::function would have heap-allocated per read.
        const auto abandon = [&](size_t admitted) {
          bool stop = advisory && (demand_pending() || pause_requested_.load() || stop_requested_.load());
          if constexpr (Build::kFaults) {
            stop = stop || (advisory && abandon_after > 0 && admitted >= static_cast<size_t>(abandon_after));
          } else {
            (void)admitted;
          }
          return stop;
        };
        const int result = reader_.read(
            request.row,
            missing,
            slots,
            advisory ? 1 : kBounceRows,
            abandon,
            cur,
            &packed,
            advisory ? 1 : SIZE_MAX,
            // The hook takes no lock (the tier has none since Task 15): the drain and retire_leases are the owner's
            // passes, and a snapshot's thunk is a read-only pass on the owner.
            // The service stays the tier's owner for the whole read, so it releases the COPYING leases the copy
            // thread handed back (D7) and answers queued snapshots here too (Task 13): neither waits for a read's
            // length, and a test can observe a slot mid-read. read() runs it once per drain-loop turn and once per
            // finished row: all three passes are the owner's, early-out when idle (retire_leases on
            // lanes_outstanding_, the rings on an empty front), allocate nothing and take no lock, so the extra
            // per-row calls cost a few loads each.
            [this] {  // a template argument (spec A6): no std::function, no allocation
              drain_copy_completions();
              retire_leases();
              answer_snapshots();
            },
            publishing ? &piece_publish_ : nullptr);
        if (piece_stream) stats_.store(kPiecePublishRefused, reader_.publish_refused());
        if (result == 0) count<kReadErrors>();
        ok = result == 1;
        cancelled = result == -1;
        status = result == 1 ? kStatusServed : result == 0 ? kStatusFailed : kStatusCancelled;
      }
      if (!advisory) ++demands_read_;
      _mm_sfence();  // the split's memcpy stores land before the map publishes them (D11)
    }
    // A row is published only when it was packed whole: every row of a request that succeeded, and, for
    // a cancelled advisory, the rows that completed before it stopped. A failed request publishes none.
    int64_t published = 0;
    {
      if (!slots.empty()) {
        Tier& tier = tiers_[request.row];
        for (size_t i = 0; i < slots.size(); ++i) {
          if (ok || (cancelled && i < packed.size() && packed[i] != 0)) {
            tier.state[slots[i]] = kReady;
            tier.stamp[slots[i]] = ++tick_;
            publish_map(request.row, missing[i], static_cast<int32_t>(slots[i]));
            ++published;
          } else if (piece_stream && leased_locked(tier, slots[i])) {
            // Plan 3.3: a miss lane still leases this slot under tag LOADING, and the device may have copied
            // (or be copying) its published pieces. It is quarantined, never released; its last lease frees it.
            quarantine_locked(request.row, slots[i]);
          } else {
            // S6 on the post-read path. With the flag off no miss slot is leased before the S3 grant below. Under
            // piece streaming a leased slot took the branch above, so what reaches here is unleased: its lanes
            // were already voided (a timeout while reading) and nothing can read it any more.
            assert(!leased_locked(tier, slots[i]));
            release_locked(request.row, slots[i]);
          }
        }
        // One writer (the owner), read lock-free by layer_rows: a relaxed store.
        int64_t& rows_total = advisory ? tier.rows_advisory : tier.rows_demand;
        std::atomic_ref<int64_t>(rows_total).store(rows_total + published, std::memory_order_relaxed);
        count<kVersion>();
      }
    }
    if (lease_mode_ && !advisory) {
      // Every lane's source is leased, and its row result published, before the caller answers the request.
      //
      // S3. Under two-phase this grants the MISS lanes only, into the entry S2 opened; it must not reopen that
      // entry, which would erase the hit leases S2 already recorded and their row results with them.
      //
      // S5. The failure paths below now hold hit leases where Task 5 held none, and they deliberately do nothing
      // about them: fail_reads, read() returning 0, a cancel (result == -1, reachable only on the advisory path
      // today -- it becomes reachable for demands if the cancel predicate ever widens), and this grant itself
      // failing. In every one the hit leases STAY OUTSTANDING and the request answers with a non-kServed status.
      // They are retired by the device's stage-1 acknowledgement or voided by its terminal. Releasing them on the
      // host is the silent-corruption path: release_locked checks no lease and take_slot_locked's free-slot loop
      // consults none, so the slot goes straight to the next request while the GPU may still be reading it.
      //
      // Piece streaming has no S3: the miss lanes were granted under tag LOADING at reservation, and their readiness
      // is the PieceMask words plus the request's status, not a second grant.
      if (two_phase_) {
        if (ok && !piece_stream) {
          if (!grant_lane_group_locked(
                  request, [&](size_t lane) { return listed(missing, request.lane_experts[lane]); }))
            ok = false;
        }
        close_pending_grants_locked(request);  // the second grant is owed no longer, however this request answered
      } else if (ok) {
        if (!grant_lanes_locked(request)) ok = false;
      }
    }
    if (cur) {
      cur->mapped = stamp(cur);
      cur->row = request.row;
      cur->ok = ok ? 1 : 0;
      cur->status = ok || status != kStatusNoRead ? status : kStatusFailed;
      cur->rows = published;
    }
    if (published > 0) {
      count<kRowsRead>(published);
      if (advisory) count<kAdvisoryRows>(published);
    }
    if (ok) *rows = published;
    return ok;
  }

  // Store piece_word(gen) into the readiness word of every lane whose expert is read (in `missing`), then fence,
  // and record those words as the owner's publish targets, by the row's ordinal in the read. Service thread only:
  // it is the only writer of area P. False when no lane names a missing row (nothing to publish).
  bool init_piece_words_locked(const Request& request, std::span<const int32_t> missing) {
    const int64_t idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    piece_targets_.assign(missing.size(), PieceTarget{});
    bool any = false;
    for (size_t lane = 0; lane < request.lane_experts.size() && lane < static_cast<size_t>(kLeaseLanes); ++lane) {
      const auto found = std::find(missing.begin(), missing.end(), request.lane_experts[lane]);
      if (found == missing.end()) continue;
      uint8_t* word = lease_ + lease_p_ + (idx * kLeaseLanes + static_cast<int64_t>(lane)) * kLeasePieceMaskLineBytes;
      store_release64(word, piece_word(request.gen));
      PieceTarget& target = piece_targets_[static_cast<size_t>(found - missing.begin())];
      target.words[target.count++] = reinterpret_cast<uint64_t*>(word);
      any = true;
    }
    _mm_sfence();
    piece_publish_ = PiecePublish{
        request.gen,
        piece_targets_.data(),
        reinterpret_cast<const uint64_t*>(lease_ + lease_d_ + kLeaseStreamProbe + idx * kLeaseStreamProbeBytes)};
    return any;
  }

  void handle_demand(const Request& request, uint8_t* record) {
    begin_busy();
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) count<kLateAfterFatal>();
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);
    if (StageRecord* const cur = stage_record(); cur && !request.armed) {
      cur->kind = kStageTouch;
      cur->lanes = request.lanes;
      cur->row = request.row;
      cur->ok = ok ? 1 : 0;
      cur->status = ok ? kStatusTouch : kStatusFailed;
    }
    // Classified by what was read: an empty need whose protect ids had to be read is D12's race.
    if (ok) {
      if (rows == 0) {
        count<kTouchOnly>();
      } else {
        count<kServedRequests>();
      }
    }
    _mm_sfence();
    set_status(record, ok ? kServed : kFailed);
    store_release(page_ + kBusySeq, 0);
    end_busy();
  }

  uint8_t* page_;
  int32_t* map_;
  uint8_t* lease_;  // the lease block, or null when the service runs without one
  uint8_t* hot_page_ = nullptr;
  int64_t hot_stride_ = 0;
  std::atomic<bool> gpu_hot_mode_{false};
  uint32_t* slot_gen_ = nullptr;        // SlotGen[] inside it
  std::vector<int64_t> slot_gen_base_;  // first SlotGen word of each row
  int64_t lease_d_ = 0;                 // byte offset of area D (the device-written words)
  int64_t lease_p_ = 0;                 // byte offset of area P (the piece readiness words)
  int64_t lease_c_ = 0;                 // byte offset of area C (CopyDone)
  bool piece_stream_ = false;           // set before the service thread starts, with the reader's flag
  // The copy engine, when enabled (before the service thread starts); armed separately, and only then used.
  std::unique_ptr<Engine> copy_engine_;
  std::atomic<bool> copy_armed_{false};
  int64_t copy_wait_timeout_ns_ = 0;           // set with the copy engine, before any thread reads it
  std::atomic<uint64_t> gate_released_{0};
  std::atomic<bool> copy_waits_aborted_{false};  // abort_copy_waits ran: no service may start again     // the last armed generation whose copy-wait gate a releaser opened
  // Native prefetch: the page (null when off), the last request generation read (service thread), the one lease a
  // prefetch holds (at most one is outstanding: the device waits for its done word before posting the next), and per
  // row the copied expert its next request judges. The last two are the owner's.
  struct PrefetchLease {
    bool active = false;
    uint64_t gen = 0;
    int64_t row = 0;
    int32_t slot = -1;
    uint32_t slot_generation = 0;
    int32_t expert = -1;
  };
  struct PrefetchJudge {
    bool pending = false;
    int32_t expert = -1;
  };
  uint8_t* prefetch_page_ = nullptr;
  uint64_t last_prefetch_gen_ = 0;
  PrefetchLease prefetch_lease_;
  std::vector<PrefetchJudge> judge_;
  // Piece streaming: serve()'s readiness words per row it reads, reused every request (like packed_).
  std::vector<PieceTarget> piece_targets_;
  PiecePublish piece_publish_;
  bool lease_mode_ = false;  // set before the service thread starts; off is today's protocol
  bool two_phase_ = false;   // Task 6 V1: hit lanes granted before read(); off is the Task 5 batched grant
  Outstanding outstanding_[kDemandRecords];    // by request slot; the owner's
  // Lanes GRANTED and not yet retired: an early-out for retire_leases. Written by the owner only (a relaxed load and
  // store), read lock-free by graph_leases_outstanding.
  std::atomic<int64_t> lanes_outstanding_{0};
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new; retirement continues
  uint64_t lease_changes_ = 0;  // the owner's: bumped whenever a lease is released, what wakes a deferred demand
  // The demand held back, if any (service thread only): its sequence, the changes seen when it was last refused,
  // its generation (a terminal for it also wakes it) and when it was first observed (for the stage record).
  uint32_t deferred_seq_ = 0;
  uint64_t deferred_stamp_ = 0;
  uint64_t deferred_gen_ = 0;
  int64_t deferred_observed_ns_ = 0;
  uint32_t settled_seq_ = 0;  // the demand whose posting last settled the double-signal watch (service thread only)
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
  uint32_t next_advice_ = 1;
  int64_t demands_read_ = 0;
  std::atomic<bool> in_advice_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> threaded_{false};
  // The single owner (Task 13): the pausing caller owns the tier while parked_ (see caller_owns). caller_mutex_
  // serializes Python-side callers only; the service, copy and fill threads never take it. commands_ carries the
  // unpaused callers' commands to the service (producer: the caller_mutex_ holder; consumer: the owner).
  std::atomic<bool> parked_{false};
  std::mutex caller_mutex_;
  SpscRing<Command, 64> commands_;
  // Copy thread -> owner (Task 14): jobs whose copies completed (and whose SM reads were acknowledged), E6-checked and,
  // for a demand job, CopyDone-published on the copy thread. The consumer is always the current owner (the same
  // handoffs as commands_); drain_copy_completions releases their leases.
  SpscRing<CopyJob, kCopyRing> copy_done_;
  std::atomic<uint32_t> skip_advice_upto_{0};
  // The watchdog's hung-request marker (D6), see busy_episode(): a new value per demand, advisory or fill in service,
  // 0 when none. episodes_ is the service thread's, or a fill's (they never run at once: a fill needs the pause).
  std::atomic<uint64_t> busy_{0};
  uint64_t episodes_ = 0;
  // Test-only faults (inject, inject_fault, inject_done_stall): InstrBuild only (plan Task 10).
  struct TierFaults {
    std::atomic<int64_t> delay_ns{0};
    std::atomic<int64_t> delay_after{0};
    std::atomic<int64_t> abandon_after{0};
    std::atomic<bool> fail_reads{false};
    std::atomic<int64_t> done_stall_ns{0};  // sleep between serving a demand and storing demand_done
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
