// The RAM tier: the host side of the RAM-miss service, which owns the pinned-slot bookkeeping and serves demand
// records.
//
// The device posts one record per MoE layer call into a demand ring, typing each routed expert as a hit (served from
// VRAM, the copy engine, the CPU, or SM reads) or a miss (read into a staging slot). The tier validates every lane
// against its own bookkeeping, reads the misses from storage, publishes the resulting map change as a delta, and hands
// copy-engine and CPU lanes to their threads. Any disagreement with the device's map fail-stops the process.
//
//   Tier             per-row slot state: which expert is in which slot, recency, staging, hot and prefill ownership
//   VictimCensus     what a request could take from a row, counted without taking anything
//   RamTier          the tier of every streamed row, the record path, the eager Python paths and the copy/CPU engines
//
// The service thread itself (spin, park, watchdog) is RamThread in ram_thread.h.
//
// The class body is large, so parts of it live in fragments that this file includes inside RamTier, in their original
// order: ram_tier_fill_copy_cpu.inc.h, ram_tier_record_support.inc.h and ram_tier_spec_serve.inc.h. Tier and
// VictimCensus are in ram_tier_slots.h.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The host per record".
#pragma once

#include <bit>
#include <cmath>
#include <future>
#include <limits>

#include "../cpu_token_table.h"
#include "../row_layout.h"
#include "../spec_candidates.h"
#include "copy_engine.h"
#include "drive_load.h"
#include "gpu_drip.h"
#include "host_copy_backend.h"
#include "numa_distributor.h"
#include "ram_prefetch.h"
#include "gate_scorer.h"
#include "split_calibration.h"

#include "ram_tier_slots.h"

namespace sglang {
namespace expert_stream {

// The pinned-slot bookkeeping of every streamed layer, and the service of one request at a time.
//
// `Source` is the row reader (the reader stack in reader_core.h) and supplies the layout and build types.
//
// Threads. The service thread (RamThread) calls pump_demand(); in tests a caller does so through pump(). The copy
// thread (CopyEngine) touches no tier state: it publishes CopyDone and the copy wait's gate, and reads the CPU engine's
// done(). The prefill fill thread drives the reader and publishes fill_landed_/fill_state_, and its epilogue runs on
// the owner after the join. The CPU expert thread (CpuExpertEngine) computes CPU lanes and touches no tier state.
// With several NUMA groups (numa_group.h) each group's service thread serves the lanes homed on its node, touching only
// its own slots of a row's slot-indexed arrays and its own experts of the expert-indexed ones; the single-owner rule
// holds per group.
//
// The single-owner rule. Everything in the tier but its atomics has exactly one owner at a time: the service thread
// while it runs; the Python caller that paused it, from the moment the service parks until resume() (RamThread::pause
// sets parked_); the caller of pump() when there is no thread. An unpaused Python call therefore takes one of two
// forms:
//   - a lock-free read of published words: mapping, counters, busy_episode, layer_rows;
//   - a refusal, "needs the service thread paused": every other call (has, touch, assign, release, fill_begin,
//     set_hot, slot_info, slot_to_expert, lru_order, victim_census, ...).
// caller_mutex_ serializes Python-side callers against each other only. The service, copy and fill threads never take
// it, and the tier has no other lock, so the service never waits on a caller.
//
// Builds. ProdBuild compiles the faults, metrics and stage trace out to empty types (see the static_asserts at the end
// of the class); InstrBuild keeps them for tests and tracing.
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The host per record".
template <class Source>
class RamTier {
 public:
  using Layout = typename Source::LayoutType;
  using CpuExpertEngine = BasicCpuExpertEngine<typename Source::BuildType>;
  using Build = typename Source::BuildType;  // ProdBuild or InstrBuild (build_policy.h)
  using Engine = CopyEngine<Build, RamTier>;
  using Drip = GpuDrip<Build, RamTier>;
  using Group = NumaGroup<Source>;
  // The copy thread's callbacks are copy_completed, copy_failed and the drip's (drip_step); it also reads stats_ for
  // its latency max.
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
      int64_t hot_bytes,
      std::vector<std::vector<std::pair<int64_t, int64_t>>> ranges,
      std::vector<int> sq_thread_cpus)
      : page_(page),
        map_(slot_map),
        lease_(lease),
        hot_page_(hot_page),
        layers_(tables.layers),
        experts_(tables.experts),
        dist_(make_groups(tables, capacity, direct, ranges, sq_thread_cpus), tables.layers),
        tiers_(static_cast<size_t>(layers_)) {
    hot_stride_ =
        ((Wire::kHotHeaderBytes + (experts_ + 7) / 8 + Wire::kHotAlignment - 1) / Wire::kHotAlignment) *
        Wire::kHotAlignment;
    if (hot_page_ != nullptr && hot_bytes != Wire::kHotRecords * hot_stride_)
      throw std::runtime_error(error_prefix<Layout>() + "hot bitmap sidecar size disagrees with expert count");
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      tier.capacity = capacity[row];
      tier.slot_to_expert.assign(tier.capacity, -1);
      tier.state.assign(tier.capacity, kFree);
      tier.stamp.assign(tier.capacity, 0);
      tier.expert_slot.assign(experts_, -1);
      tier.hot.assign(experts_, 0);
      tier.filling.assign(tier.capacity, 0);
      tier.prefill_owned.assign(tier.capacity, 0);
    }
    init_lease_block(lease_bytes);
    for (int g = 0; g < dist_.size(); ++g) {
      Group& group = dist_.group(g);
      // The service streams every miss lane piece by piece: the reader must publish pieces.
      group.reader.set_piece_stream(true);
      // Every group's reader counts into the tier's one drive load: the groups' rings read the same drives.
      group.reader.set_drive_load(&drive_load_);
      // The request path's buffers, sized once: nothing on it grows after construction.
      group.hot_scratch.assign(static_cast<size_t>((experts_ + 7) / 8), 0);
      group.packed.reserve(kWanted);
      group.two_span.reserve(kWanted);
      group.prefix.reserve(kWanted);
      group.piece_targets.reserve(kWanted);
    }
    int64_t widest = 0;
    for (int64_t c : capacity)
      widest = std::max(widest, c);
    fill_packed_.reserve(static_cast<size_t>(widest));
    cpu_layers_ = std::make_unique<CpuExpertLayers>(layers_);
  }

  // Joins the fill thread, then the copy thread, then the CPU thread. The copy thread's callbacks use the lease block
  // and the counter blocks, so it must stop before they are destroyed; the CPU thread stops last because the copy
  // thread reads its done().
  ~RamTier() {
    release_row_hold();
    stop_spec();  // the speculative threads read through the groups' readers into the pool's slabs
    fill_join();  // the fill thread reads through group 0's reader into the slabs; nothing else holds the tier by now
    if (copy_engine_ != nullptr) copy_engine_->stop(5'000'000'000LL);
    for (int g = 0; g < dist_.size(); ++g)
      if (dist_.group(g).cpu != nullptr) dist_.group(g).cpu->stop();
  }

  // Opens every group's reader and positions its demand cursor after the ring's current head. False if a reader
  // cannot open.
  bool open() {
    for (int g = 0; g < dist_.size(); ++g) {
      Group& group = dist_.group(g);
      group.reader.set_sq_thread_cpu(group.sq_thread_cpu);
      if (!group.reader.open()) return false;
      group.next_demand = skip_zero(load_acquire(page_ + Wire::kDemandHead) + 1u);
    }
    return true;
  }

  uint8_t* page() const {
    return page_;
  }
  // The watchdog's hung-request marker: nonzero while a demand or a fill is in service, a new value per episode. The
  // watchdog thread times how long one value persists; the service reads no clock for it.
  uint64_t busy_episode(int g) const {
    return dist_.group(g).busy.load(std::memory_order_acquire);
  }
  // Group g's service thread's set-once words (kRunning, kSpinCpu), written by that thread only.
  void set_counter(int g, int index, int64_t value) {
    dist_.group(g).core.set(index, value);
  }
  int groups() const {
    return dist_.size();
  }

  // Counters. A core counter (is_core_counter) lives in the writing thread's own line-private block. count(group) is
  // that group's service thread's, and count() the eager paths' (the caller owning the tier while every group is paused
  // or pumped: one writer at a time, handed over by the same edges as the tier; it writes group 0's block), and
  // copy_count() is the copy thread's. The fill thread counts nothing: its epilogue runs on the owner
  // (finish_fill_owned). Every other counter is a metric: InstrBuild keeps it as a shared relaxed atomic, ProdBuild has
  // none.
  template <Counter K>
  void count(Group& group, int64_t n = 1) {
    count_into<K>(group.core, n);
  }
  template <Counter K>
  void count(int64_t n = 1) {
    count_into<K>(dist_.group(0).core, n);
  }
  template <Counter K>
  void copy_count(int64_t n = 1) {
    count_into<K>(copy_core_, n);
  }
  bool threaded() const {
    return threaded_.load();
  }
  // Called by RamThread::start before the thread exists and by RamThread::stop after its join. The join makes every
  // tier write of the service thread visible, and this release hands that on to a caller's caller_owns().
  void set_threaded(bool threaded) {
    threaded_.store(threaded, std::memory_order_release);
  }

  // ---- Ownership (see the class comment) ----

  // True when the caller owns the tier: the service is paused (parked_, set by the pausing caller once the service has
  // parked) or there is no thread at all (pump mode, or after stop()'s join).
  bool caller_owns() const {
    return !threaded_.load(std::memory_order_acquire) || parked_.load(std::memory_order_acquire);
  }
  // Set true by RamThread::pause after the service parked, and false by resume before the service may run, both under
  // caller_mutex_: a caller that takes caller_mutex_ afterwards sees who owns the tier.
  void set_parked(bool parked) {
    parked_.store(parked, std::memory_order_release);
  }
  std::mutex& caller_mutex() {
    return caller_mutex_;
  }

  // Serves group g's next posted demand record, if any, and returns whether it handled one. One caller per group at a
  // time: the group's service thread, or a test's pump().
  bool pump_demand(int g) {
    Group& group = dist_.group(g);
    const uint32_t head = load_acquire(page_ + Wire::kDemandHead);
    const bool posted = head != 0 && reached(head, group.next_demand);
    if (admission_closed_.load()) return false;
    if (!posted) return false;
    if (!streamed_[g].empty()) release_streamed(g);
    if constexpr (Build::kFaults) {
      if (const int64_t ns = faults_.group_stall_ns[g].load()) {  // once: the group's own thread is the only reader
        faults_.group_stall_ns[g].store(0);
        fault_delay(ns);
      }
    }
    if (g == 0) begin_stage(kStageDemand, group.next_demand, head - group.next_demand);
    if (head - group.next_demand >= Wire::kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq. Only records nothing
      // waits for lap, since the device posts a record with host work only after the previous one's chain ended.
      count<kOverruns>(group, head - group.next_demand - (Wire::kDemandRecords - 2));
      group.next_demand = skip_zero(head - Wire::kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(Wire::kDemandRing, Wire::kDemandRecords, group.next_demand);
    prefetch_request(record, group.next_demand);
    Request request;
    // A torn record was overwritten by a later post, so nothing waits on it: skip and count it.
    const RecordRead read = read_record(record, group.next_demand, &request);
    if (read == RecordRead::kMalformed) {
      fail_stop(
          error_prefix<Layout>() + "request " + std::to_string(group.next_demand) +
          ": malformed record (a lane kind or "
          "count the device never writes)");
    }
    if (read == RecordRead::kTorn) {
      count<kOverruns>(group);
    } else {
      bool skip = false;
      if (hot_page_ != nullptr && !request.lanes.empty()) {
        if (request.row >= 0 && request.row < layers_ && read_gpu_hot(group, group.next_demand, &request) &&
            load_acquire(record + Wire::kRecSeq) == group.next_demand) {
          apply_gpu_hot(group, request);
        } else if (has_host_lane(group, request)) {
          fail_stop(
              error_prefix<Layout>() + "request " + std::to_string(group.next_demand) + ": no hot set for it");
        } else {
          // No host lane of this group's node: the device does not wait on the group for this record (its misses and
          // host copies are waited for, an SM hit or another node's lane is not), so a later post may have lapped the
          // hot record, or the record itself. The group then reads nothing it needs.
          count<kOverruns>(group);
          skip = true;
        }
      }
      if (spec_ != nullptr) note_serving(group, request);
      if (promote_) publish_due(group, request);
      if (!skip) handle_record(group, request);
      // Every group sees the record, served or skipped, so both groups score the same jobs.
      if (spec_ != nullptr) offer_spec(group, request);
    }
    if (g == 0) end_stage();
    group.handled.store(group.next_demand, std::memory_order_release);
    group.next_demand = skip_zero(group.next_demand + 1u);
    return true;
  }

  // True when the record has a lane of the group's node that the device waits on: any lane it does not copy itself.
  static bool has_host_lane(const Group& group, const Request& request) {
    for (const Lane& lane : request.lanes)
      if (!device_copied(lane.kind) && Wire::home(lane.expert) == group.index) return true;
    return false;
  }

  // Pump mode: serves the next record of every group, in order, and returns whether group 0 handled one.
  bool pump_demand() {
    bool first = false;
    for (int g = 0; g < dist_.size(); ++g) {
      const bool handled = pump_demand(g);
      if (g == 0) first = handled;
    }
    return first;
  }

  // The least seq every group finished, across the wrap. Any thread, lock-free (ChainSim.wait_handled, a test's view).
  uint32_t handled_through() const {
    uint32_t least = dist_.group(0).handled.load(std::memory_order_acquire);
    for (int g = 1; g < dist_.size(); ++g) {
      const uint32_t h = dist_.group(g).handled.load(std::memory_order_acquire);
      if (!reached(h, least)) least = h;
    }
    return least;
  }

  // ---- Stage trace: one StageRecord per served request, drained by Python ----

  // Allocates the ring, then turns the trace on. Must run before the service thread starts, so the flag never flips
  // under a request being served. InstrBuild only; ProdBuild throws.
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

  // Moves up to `max` records into `out` (stage_words() int64 each) and returns how many. InstrBuild only.
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

  // The number of records dropped because the ring was full. InstrBuild only.
  int64_t trace_dropped() {
    if constexpr (!Build::kMetrics) {
      throw_no_trace();
    } else {
      std::lock_guard<std::mutex> guard(trace_.mutex);
      return trace_.ring ? trace_.ring->dropped() : 0;
    }
  }

  // ---- Python-facing bookkeeping (the single-owner rule, see the class comment) ----

  bool has(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("contains");
    return tiers_[row].expert_slot[expert] >= 0;
  }

  void touch(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("touch");
    Tier& tier = tiers_[row];
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0) {
      tier.stamp[slot] = ++dist_.home_group(expert).tick;
      // Under a share the touch is the prefill's own hit, so the slot stays owned.
      if (prefill_share_.load(std::memory_order_relaxed) == 0) disown_slot_locked(row, tier, slot);
    }
  }

  // Takes a slot for `expert` for a Python-side read and maps it at once (the device is idle and the service paused
  // when an eager path calls this). Returns the slot, or -1 if none can be taken. `*evicted` is the expert displaced,
  // -1 for none, or -2 when the expert already held a slot.
  int64_t assign(int64_t row, int64_t expert, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("assign");
    Tier& tier = tiers_[row];
    if (tier.expert_slot[expert] >= 0) {
      *evicted = -2;
      return tier.expert_slot[expert];
    }
    const int64_t slot = take_admit_slot_locked(row, expert, protect, fallback, evicted);
    if (slot < 0) return -1;
    tier.slot_to_expert[slot] = static_cast<int32_t>(expert);
    tier.state[slot] = kReady;
    tier.stamp[slot] = ++dist_.home_group(expert).tick;
    tier.expert_slot[expert] = static_cast<int32_t>(slot);
    publish_map(row, expert, static_cast<int32_t>(slot));
    count<kVersion>();
    return slot;
  }

  // Frees `slot` and unmaps its expert. Throws for a staging or pool slot, or one a fill is still writing.
  void release(int64_t row, int64_t slot) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("release");
    if (tiers_[row].state[slot] == kStaging) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + ": it is a staging slot");
    }
    if (tiers_[row].state[slot] == kSpec) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) +
          ": it is a speculative pool slot");
    }
    if (tiers_[row].filling[slot]) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while a fill writes it");
    }
    release_locked(row, slot);
    count<kVersion>();
  }

#include "ram_tier_fill_copy_cpu.inc.h"

  // ---- Test hooks and introspection, for the tier's owner (refused with "needs the service thread paused" otherwise)
  // ----

  // Writes [state, expert, stamp] per slot of `row`.
  void slot_info(int64_t row, int64_t* out) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("slot_info");
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      out[3 * slot] = tier.state[slot];
      out[3 * slot + 1] = tier.slot_to_expert[slot];
      out[3 * slot + 2] = static_cast<int64_t>(tier.stamp[slot]);
    }
  }

  // Counts the free and evictable slots of `row` for a request that routes `wanted`.
  VictimCensus victim_census(int64_t row, const std::vector<int32_t>& wanted) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("victim_census");
    return census_locked(row, wanted);
  }

  // The published slot map of `row`, from any thread, lock-free. It holds a slot for an expert exactly while that slot
  // is READY: a slot is stored only once its bytes landed, and -1 on every unmap.
  void mapping(int64_t row, int64_t* out) const {
    for (int64_t expert = 0; expert < experts_; ++expert)
      out[expert] = __atomic_load_n(map_ + row * experts_ + expert, __ATOMIC_ACQUIRE);
  }

  // Writes every slot's expert.
  void slot_to_expert(int64_t row, int64_t* out) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("slot_to_expert");
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot)
      out[slot] = tier.slot_to_expert[slot];
  }

  // Writes the READY slots' experts, least recently used first, and returns how many.
  int64_t lru_order(int64_t row, int64_t* out) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("lru_order");
    const Tier& tier = tiers_[row];
    std::vector<int64_t> slots;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kReady) slots.push_back(slot);
    }
    std::sort(slots.begin(), slots.end(), [&](int64_t a, int64_t b) { return tier.stamp[a] < tier.stamp[b]; });
    for (size_t i = 0; i < slots.size(); ++i)
      out[i] = tier.slot_to_expert[slots[i]];
    return static_cast<int64_t>(slots.size());
  }

  // Replaces the row's hot set. A VRAM-hot row is decode's: kept prefill-owned it could never be a victim and would
  // hold the share down, so it is disowned. Plain stores: the owner's pause quiesced the scorers (quiesce_spec).
  void set_hot(int64_t row, const int64_t* experts, int64_t count) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("set_hot");
    Tier& tier = tiers_[row];
    tier.hot.assign(experts_, 0);
    for (int64_t i = 0; i < count; ++i) {
      const int64_t expert = experts[i];
      if (expert < 0 || expert >= experts_) continue;
      tier.hot[expert] = 1;
      if (tier.expert_slot[expert] >= 0) disown_slot_locked(row, tier, tier.expert_slot[expert]);
    }
  }

  // Writes each row's demand-row count. Any thread, lock-free: relaxed reads of words only the owner's serve() writes.
  void layer_rows(int64_t* out) const {
    for (int64_t row = 0; row < layers_; ++row)
      out[row] = __atomic_load_n(&tiers_[row].rows_demand, __ATOMIC_RELAXED);
  }

  // Test only (InstrBuild; ProdBuild throws): sleeps `delay_ns` before each demand read once `after_demands` demands
  // have read rows, and with `fail_reads` reports the reads as failed, which fail-stops the process.
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

  // Test only (InstrBuild; ProdBuild throws): group g's service sleeps `ns` once, before it reads its next record.
  void inject_group_stall(int g, int64_t ns) {
    if constexpr (!Build::kFaults) {
      (void)g, (void)ns;
      test_only("inject_group_stall");
    } else {
      if (g < 0 || g >= groups())
        throw std::runtime_error(error_prefix<Layout>() + "no NUMA group " + std::to_string(g));
      faults_.group_stall_ns[g].store(ns);
    }
  }

  // Test only (InstrBuild; ProdBuild refuses rather than store a fault it would never apply): carries a whole
  // ReadFault down to this tier's reader, where inject() reaches it only as a delay or a blanket failure.
  //
  // `words` is the reader tests' fault tensor (kFaultWords int64; see fault_from). Unlike fail_reads, the fault does
  // not short-circuit ahead of the reader: the read runs, so the fault's part errors, pack delay and the rest act on
  // rows that have already packed. The service thread applies it just before its next read (the reader is that thread's
  // alone), and it stays until replaced; an all-default tensor clears it. Words 17-18 (abandon_after, step), 22
  // (piece_stream), 20 (mirror caps) and 32 (two_span_rows: the tier picks its two-span rows itself) are not faults and
  // are ignored. The reader's call counters run over its whole life, so a call-numbered fault (submit_call, cqe_call) is
  // relative to a fresh tier.
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

  // Test only (InstrBuild): lands `expert`'s row of `row` in an empty pool entry of its home group, as a speculative
  // read would, and returns the slot. On the owner; nothing is mapped or published.
  int64_t spec_place(int64_t row, int64_t expert)
    requires(Build::kFaults)
  {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("spec_place");
    row_capacity(row);
    const std::string prefix = error_prefix<Layout>() + "spec_place: ";
    if (pool_ == nullptr) throw std::runtime_error(prefix + "no speculative pool (reserve_spec_pool)");
    if (expert < 0 || expert >= experts_) throw std::runtime_error(prefix + "expert out of range");
    if (tiers_[row].expert_slot[expert] >= 0) throw std::runtime_error(prefix + "the tier maps the expert");
    const int g = Wire::home(expert);
    const int32_t id = static_cast<int32_t>(expert);
    int i = -1;
    {
      std::lock_guard<std::mutex> lock(pool_->mutex(g));
      for (int k = 0; k < pool_->share() && i < 0; ++k)
        if (pool_->entry(row, g, k).word.load(std::memory_order_relaxed) == kPoolEmpty) i = k;
      if (i < 0) throw std::runtime_error(prefix + "no empty pool entry");
      PoolEntry& entry = pool_->entry(row, g, i);
      entry.for_seq.store(0, std::memory_order_relaxed);  // no source record: every job sees it pooled before
      entry.word.store(pool_word(kPoolReading, id), std::memory_order_release);
    }
    std::vector<uint8_t> packed;
    if (land_pool_row(g, row, i, id, &packed) != 1) throw std::runtime_error(prefix + "the read failed");
    return pool_->entry(row, g, i).slot;
  }

  // Test only (InstrBuild): with `hold`, a helper thread takes group g's row_mutex[row] and keeps it until the next call;
  // the call without `hold` only releases it. Serving `row` meanwhile would block on the lock.
  void hold_row(int64_t row, int g, bool hold)
    requires(Build::kFaults)
  {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    release_row_hold();
    if (!hold) return;
    row_capacity(row);
    if (g < 0 || g >= groups()) throw std::runtime_error(error_prefix<Layout>() + "no NUMA group " + std::to_string(g));
    row_held_.store(false, std::memory_order_relaxed);
    row_hold_release_.store(false, std::memory_order_relaxed);
    row_holder_ = std::thread([this, row, g] {
      std::lock_guard<std::mutex> lock(dist_.group(g).row_mutex[row]);
      row_held_.store(true, std::memory_order_release);
      while (!row_hold_release_.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    while (!row_held_.load(std::memory_order_acquire)) std::this_thread::yield();
  }

  // Test only (InstrBuild): serves group g's next speculative job on the caller, as its speculative thread would;
  // false when the ring is empty. Refused while the service thread runs.
  bool spec_pump(int g)
    requires(Build::kFaults)
  {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    if (spec_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the RAM prefetch is not enabled");
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "spec_pump while the service thread runs");
    if (g < 0 || g >= groups()) throw std::runtime_error(error_prefix<Layout>() + "no NUMA group " + std::to_string(g));
    SpecJob job;
    if (!spec_->groups[g]->ring.pop(&job)) return false;
    serve_spec_job(g, job, /*threaded=*/false);
    return true;
  }

  // Test only (InstrBuild; ProdBuild throws): each speculative read sleeps `delay_ns`, and with `fail` is reported
  // failed without reading. On the shared reader it sleeps first, holding the reader's turn; on an idle-drive reader it
  // sleeps once its first piece is in flight (counted in the drive load), so the read holds that drive meanwhile.
  void inject_spec(int64_t delay_ns, bool fail) {
    if constexpr (!Build::kFaults) {
      (void)delay_ns, (void)fail;
      test_only("inject_spec");
    } else {
      faults_.spec_delay_ns.store(delay_ns);
      faults_.spec_fail.store(fail);
    }
  }

  // Test only (InstrBuild; ProdBuild throws): adds reads[q] demand sub-reads to root q's drive load (negative takes
  // them back), as another reader's demand in flight that an idle-drive read must avoid. kMaxDrives words.
  void inject_demand_load(const int64_t* reads) {
    if constexpr (!Build::kFaults) {
      (void)reads;
      test_only("inject_demand_load");
    } else {
      int64_t now = 0;
      for (int q = 0; q < kMaxDrives; ++q)
        if (reads[q] != 0) drive_load_.change(q, kDemandRead, static_cast<int>(reads[q]), 0, now);
    }
  }

  // ---- The speculative threads (RamThread drives them with the service threads) ----

  // Starts each group's speculative thread, pinned to its cores. Throws, after joining them, when one cannot be
  // pinned. A no-op with the RAM prefetch off.
  void start_spec() {
    if (spec_ == nullptr) return;
    spec_->stop.store(false);
    spec_->hold.store(false);
    std::vector<std::future<int>> pins;
    for (size_t g = 0; g < spec_->groups.size(); ++g) {
      SpecGroup& spec = *spec_->groups[g];
      spec.pinned = std::promise<int>();  // a previous start's thread was joined by stop_spec
      pins.push_back(spec.pinned.get_future());
      spec.thread = std::thread([this, g, &spec] {
        const int error = pin_spec(static_cast<int>(g));
        spec.pinned.set_value(error);
        if (error == 0) run_spec(static_cast<int>(g));
      });
    }
    int failed = -1;
    int error = 0;
    for (size_t g = 0; g < pins.size(); ++g) {
      const int e = pins[g].get();
      if (failed < 0 && e != 0) {
        failed = static_cast<int>(g);
        error = e;
      }
    }
    if (failed >= 0) {
      stop_spec();
      throw std::runtime_error(
          error_prefix<Layout>() + "could not pin the speculative thread of group " + std::to_string(failed) + ": " +
          std::strerror(error));
    }
  }

  // Stops and joins the speculative threads; each finishes the read it is in first. Idempotent.
  void stop_spec() {
    if (spec_ == nullptr) return;
    spec_->stop.store(true, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
    for (auto& spec : spec_->groups)
      if (spec->thread.joinable()) spec->thread.join();
  }

  // Holds the speculative threads between reads and returns once none reads or scores: a pausing caller and a prefill
  // fill use the readers, and set_hot writes the hot flags the scorer reads. Unbounded; the watchdog times a hung read.
  void quiesce_spec() {
    if (spec_ == nullptr) return;
    spec_->hold.store(true, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
    // `idle` rather than the thread object, which a racing stop_spec joins: a thread not running is idle.
    for (auto& spec : spec_->groups)
      while (!spec->idle.load(std::memory_order_seq_cst))
        std::this_thread::sleep_for(std::chrono::microseconds(20));
  }

  void resume_spec() {
    if (spec_ == nullptr) return;
    spec_->hold.store(false, std::memory_order_seq_cst);
    for (auto& spec : spec_->groups)
      spec->bell.ring();
  }

  // The watchdog's marker for group g's speculative read: nonzero while one is in flight, a new value per read.
  uint64_t spec_busy_episode(int g) const {
    return spec_ == nullptr ? 0 : spec_->groups[g]->busy.load(std::memory_order_acquire);
  }

  // Writes every counter with relaxed reads: a core counter is the sum of its writers' blocks (each word has one
  // writer; kSpinCpu is group 0's alone), a metric is stats_'s (always 0 in ProdBuild; Python reports only the core
  // counters of a production host).
  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i) {
      int64_t core = 0;
      for (int g = 0; g < dist_.size(); ++g) {
        if (i == kSpinCpu && g > 0) break;
        core += dist_.group(g).core.get(i);
      }
      if (spec_ != nullptr)
        for (const auto& spec : spec_->groups)
          core += spec->core.get(i);
      out[i] = core + copy_core_.get(i) + stats_.get(i);
    }
  }

  // SGLANG_MOE_EXPERT_MIRROR_DYNAMIC: every group's reader picks each sub-read's root under `caps`, one in-flight cap
  // per mirror root (ReaderCore::set_mirror_caps); empty turns it off. Demand and speculative reads both go through
  // the groups' readers, so both choose. On the owner (before the thread starts, or paused).
  void set_mirror_caps(std::span<const int64_t> caps) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("set_mirror_caps");
    for (int g = 0; g < dist_.size(); ++g)
      dist_.group(g).reader.set_mirror_caps(caps);
  }

  // The drive load (drive_load.h), kDriveLoadWords words, relaxed reads from any thread: exact once every reader is
  // idle, as at the shutdown line.
  void drive_load(int64_t* out) const {
    drive_load_.snapshot(out, dist_.group(0).reader.tables().parts);
  }

  // Group g's own core counters: its service thread's block and its speculative thread's.
  void group_counters(int g, int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = dist_.group(g).core.get(i) + (spec_ != nullptr ? spec_->groups[g]->core.get(i) : 0);
  }

 private:
#include "ram_tier_record_support.inc.h"

#include "ram_tier_spec_serve.inc.h"

  // Per group, owned by its service thread: the current record's time split (trace_svc), InstrBuild only.
  struct alignas(64) SvcClock {
    int64_t last_end = 0;
    int64_t cpu_ns = 0;
    int64_t read_ns = 0;
  };
  std::array<SvcClock, Wire::kNodes> svc_{};
  // Adds its scope's duration to `total` in InstrBuild; ProdBuild reads no clock.
  struct SvcTimer {
    int64_t* total;
    int64_t start;
    explicit SvcTimer(int64_t& t) : total(&t), start(Build::kMetrics ? now_ns() : 0) {}
    ~SvcTimer() {
      if constexpr (Build::kMetrics) *total += now_ns() - start;
    }
  };

  // Per group, owned by its service thread: the forward counts up when a record's row is not past the last one's.
  struct alignas(64) ForwardState {
    int64_t last_row = -1;
    uint64_t id = 0;
  };
  std::array<ForwardState, Wire::kNodes> forward_{};
  bool promote_ = false;  // set in enable_ram_prefetch, before the service thread starts
  bool pool_serve_ = false;  // likewise
  // Per group, under its pool mutex: the entries S streams uncached rows from (point_pooled_locked), emptied at the
  // group's next record (release_streamed). One record's misses at most.
  struct StreamedEntry {
    int64_t row = 0;
    int entry = 0;
  };
  std::array<FixedVec<StreamedEntry, Wire::kLanes>, Wire::kNodes> streamed_;
  // hold_row's helper (InstrBuild tests only); joined by release_row_hold.
  std::thread row_holder_;
  std::atomic<bool> row_held_{false};
  std::atomic<bool> row_hold_release_{false};
  void release_row_hold() {
    if (!row_holder_.joinable()) return;
    row_hold_release_.store(true, std::memory_order_release);
    row_holder_.join();
  }

  uint8_t* page_;
  int32_t* map_;
  uint8_t* lease_;               // the lease block (lease_layout.h)
  uint8_t* hot_page_ = nullptr;  // the hot bitmap sidecar: when given, each record's hot set is applied
  int64_t hot_stride_ = 0;
  // The copy engine, when enabled (before the service thread starts); armed separately, and only then used.
  std::unique_ptr<Engine> copy_engine_;
  std::atomic<bool> copy_armed_{false};
  // The GPU drip, when enabled (before the service thread starts): per [row, expert] the mirror's change count, and
  // the drip the copy thread steps (drip_, published once drip_owner_ is complete).
  std::unique_ptr<std::atomic<uint32_t>[]> drip_gen_;
  std::unique_ptr<Drip> drip_owner_;
  std::atomic<Drip*> drip_{nullptr};
  // The copy thread's side of the drip (CopyEngine::run).
  bool drip_on() const {
    return drip_.load(std::memory_order_acquire) != nullptr;
  }
  bool drip_step(CopyBackend& backend, bool demand_idle) {
    return drip_.load(std::memory_order_acquire)->step(backend, demand_idle);
  }
  bool drip_busy() const {
    return drip_.load(std::memory_order_acquire)->busy();
  }
  int64_t copy_wait_timeout_ns_ = 0;  // set with the copy engine, before any thread reads it
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new
  // The eager paths' map changes, {row, expert, slot}: the owner's, taken by take_bulk_delta. Grows only on those
  // paths (assign, prefill fills, release), never on the record path.
  std::vector<std::array<int32_t, 3>> bulk_;
  int64_t layers_;
  int64_t experts_;
  // Every row's CPU expert layer, shared by every group's engine; declared before dist_, so destroyed after the engines.
  std::unique_ptr<CpuExpertLayers> cpu_layers_;
  CpuExpertEngine& draft_cpu(int g) {
    if (g < 0 || g >= groups() || dist_.group(g).cpu == nullptr)
      throw std::runtime_error(std::string(CpuExpertEngine::kDraftPrefix) + "group " + std::to_string(g) +
                               " has no CPU expert engine (SGLANG_DSV41_CPU_EXPERTS)");
    return *dist_.group(g).cpu;
  }
  std::array<std::unique_ptr<DraftSource>, Wire::kNodes> pending_draft_;
  std::array<int64_t, Wire::kNodes> draft_fatal_ns_{};
  const cpu_experts::CpuExpertKernel* cpu_kernel_ = nullptr;  // every group's engine runs this one kernel
  // The NUMA groups: each one's reader, serve state and CPU experts (when enabled, group 0's, after the copy engine and
  // before the service threads; stopped after the copy thread). Declared before tiers_: the groups' ranges are checked
  // against the capacities first.
  NumaNodeDistributor<Source> dist_;
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
  std::vector<Tier> tiers_;       // the owner's, with every other non-atomic member: the tier has no mutex
  // The speculative pool, once reserved (reserve_spec_pool); null with the RAM prefetch off.
  std::unique_ptr<SpecPool> pool_;
  std::unique_ptr<SpecState> spec_;  // null with the RAM prefetch off
  std::atomic<int64_t> prefill_share_{0};  // any thread stores it, relaxed; see set_prefill_share
  std::atomic<bool> threaded_{false};
  // The pausing caller owns the tier while parked_ (see caller_owns). caller_mutex_ serializes Python-side callers
  // only.
  std::atomic<bool> parked_{false};
  std::mutex caller_mutex_;
  // Test-only faults (inject, inject_fault): InstrBuild only.
  struct TierFaults {
    std::atomic<int64_t> delay_ns{0};
    std::atomic<int64_t> delay_after{0};
    std::atomic<bool> fail_reads{false};
    std::mutex fault_mutex;  // guards pending_fault between inject_fault() and the service thread
    ReadFault pending_fault{};
    std::atomic<bool> fault_pending{false};
    std::array<std::atomic<int64_t>, Wire::kNodes> group_stall_ns{};  // inject_group_stall, once per group
    std::atomic<int64_t> spec_delay_ns{0};  // inject_spec: each speculative read sleeps (see inject_spec)
    std::atomic<bool> spec_fail{false};
  };
  struct NoTierFaults {};
  [[no_unique_address]] std::conditional_t<Build::kFaults, TierFaults, NoTierFaults> faults_;
  // Counters, see count(). One line-private block per writer thread; the metrics only in InstrBuild.
  LineCounters<kCounterCount> copy_core_;  // copy thread only (each group's service thread has its own, in Group::core)
  [[no_unique_address]] Stats<Build::kMetrics, kCounterCount> stats_;  // InstrBuild: any thread, relaxed RMW
  DriveLoad drive_load_;  // every group's reader counts its reads in flight here (drive_load.h); relaxed atomics
  // Stage trace, InstrBuild only. cur points at stage while a traced request is in service, else null.
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
  // The type-system half of the proof that ProdBuild carries no test machinery (nm cannot see state whose names are
  // inlined away): its fault, metric and trace members are empty types, which [[no_unique_address]] gives no storage.
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(faults_)>, "ProdBuild has no faults");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(stats_)>, "ProdBuild has no metrics");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(trace_)>, "ProdBuild has no trace");
};

}  // namespace expert_stream
}  // namespace sglang
