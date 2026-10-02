// Tier/RamTier: slot state and admission.
#pragma once

#include "../row_layout.h"
#include "copy_engine.h"
#include "split_calibration.h"

namespace sglang {
namespace expert_stream {

struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (READY) or -1
  std::vector<uint8_t> hot;
  std::vector<uint8_t> filling;      // a prefill fill is still writing this kReady slot: never a victim, never released
  // Prefill share: 1 while a row a prefill admitted has not been used by
  // decode. `owned` counts them. Only take_admit_slot_locked sets it, so it stays all zero with the share off.
  std::vector<uint8_t> prefill_owned;
  int64_t owned = 0;
  // Written by the owner's serve() only (a relaxed store through std::atomic_ref), read relaxed by layer_rows from
  // any thread.
  int64_t rows_demand = 0;
  // The row's staging slots (kStaging, never mapped), in the order the device assigns them to miss lanes, and the
  // map-chain number of the last delta published. 0: reserve_staging has not run, so the row serves no miss.
  FixedVec<int32_t, kLeaseLanes> staging;
  uint64_t chain = 0;
};

// What a request could take from a tier, counted without taking anything.
struct VictimCensus {
  int64_t free = 0;       // FREE slots
  int64_t evictable = 0;  // READY, not hot, not requested, not filling
};

// The pinned-slot bookkeeping of every streamed layer and the service of one request at a time. pump_demand is called
// by one caller at a time: a test's pump(), or the service thread. The Python-facing methods follow the single-owner
// rule, stated at "Python-facing bookkeeping" below.
template <class Source>
class RamTier {
 public:
  using Layout = typename Source::LayoutType;
  using Build = typename Source::BuildType;  // ProdBuild or InstrBuild (build_policy.h)
  using Engine = CopyEngine<Build, RamTier>;
  // The copy engine calls copy_completed and copy_failed, and reads stats_ for its latency max.
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
      tier.filling.assign(tier.capacity, 0);
      tier.prefill_owned.assign(tier.capacity, 0);
    }
    init_lease_block(lease_bytes);
    // The service streams every miss lane piece by piece: the reader must publish pieces.
    reader_.set_piece_stream(true);
    // The request path's buffers, sized once: nothing on it grows after construction.
    hot_scratch_.assign(static_cast<size_t>((experts_ + 7) / 8), 0);
    packed_.reserve(kWanted);
    piece_targets_.reserve(kWanted);
    int64_t widest = 0;
    for (int64_t c : capacity)
      widest = std::max(widest, c);
    fill_packed_.reserve(static_cast<size_t>(widest));
  }

  // The copy thread's callbacks use the lease block and the counter blocks, which are destroyed before copy_engine_
  // would be.
  ~RamTier() {
    fill_join();  // the fill thread reads through reader_ into the slabs; nothing else holds the tier by now
    if (copy_engine_ != nullptr) copy_engine_->stop(5'000'000'000LL);
    if (cpu_ != nullptr) cpu_->stop();  // after the copy thread, which reads its done()
  }

  bool open() {
    if (!reader_.open()) return false;
    next_demand_ = skip_zero(load_acquire(page_ + kDemandHead) + 1u);
    return true;
  }

  uint8_t* page() const {
    return page_;
  }
  // The watchdog's hung-request marker: nonzero while a demand or a fill is in service, a new value per episode. The
  // watchdog thread times how long one value persists; the service reads no clock for it.
  uint64_t busy_episode() const {
    return busy_.load(std::memory_order_acquire);
  }
  // The service thread's set-once words (kRunning, kSpinCpu), written by that thread only.
  void set_counter(int index, int64_t value) {
    core_.set(index, value);
  }

  // Counters. A core counter (is_core_counter) lives in the writing thread's own line-private block: count() is the
  // tier's owner's (the service thread, or the caller owning the tier while it is paused or pumped: one writer at a
  // time, handed over by the same edges as the tier), copy_count() the copy thread's. The prefill fill thread counts
  // nothing: its epilogue runs on the owner (finish_fill_owned). Every other counter is a metric: InstrBuild keeps it
  // as a shared relaxed atomic, ProdBuild has none.
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
  // RamThread::start before the thread exists, and RamThread::stop after its join: the join is what makes
  // every tier write of the service thread visible, and this release hands that on to a caller's caller_owns().
  void set_threaded(bool threaded) {
    threaded_.store(threaded, std::memory_order_release);
  }

  // ---- The single owner (see "Python-facing bookkeeping") ----

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

  // Serve the next posted demand record, if any. True when it handled one.
  bool pump_demand() {
    const uint32_t head = load_acquire(page_ + kDemandHead);
    const bool posted = head != 0 && reached(head, next_demand_);
    if (admission_closed_.load()) return false;
    if (!posted) return false;
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq. Only records nothing
      // waits for lap: the device posts one with host work only after the previous one's chain ended.
      count<kOverruns>(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    prefetch_request(record, next_demand_);
    Request request;
    // A torn record was overwritten by a later post, so nothing waits on it: skipped, and counted.
    const RecordRead read = read_record(record, next_demand_, &request);
    if (read == RecordRead::kMalformed) {
      fail_stop(error_prefix<Layout>() + "request " + std::to_string(next_demand_) + ": malformed record (a lane kind or "
                "count the device never writes)");
    }
    if (read == RecordRead::kTorn) {
      count<kOverruns>();
    } else {
      bool skip = false;
      if (hot_page_ != nullptr && !request.lanes.empty()) {
        if (request.row >= 0 && request.row < layers_ && read_gpu_hot(next_demand_, &request) &&
            load_acquire(record + kRecSeq) == next_demand_) {
          apply_gpu_hot(request);
        } else if (request.host_work()) {
          fail_stop(error_prefix<Layout>() + "request " + std::to_string(next_demand_) + ": no hot set for it");
        } else {
          // SM hits only: the device did not wait for this record, so a later post may have lapped its hot record.
          count<kOverruns>();
          skip = true;
        }
      }
      if (!skip) handle_record(request);
    }
    end_stage();
    handled_.store(next_demand_, std::memory_order_release);
    next_demand_ = skip_zero(next_demand_ + 1u);
    return true;
  }

  // Any thread, lock-free: the last seq the service finished (ChainSim.wait_handled; a test's view).
  uint32_t handled_through() const {
    return handled_.load(std::memory_order_acquire);
  }

  // ---- Stage trace: one StageRecord per served request, drained by Python ----

  // Allocates the ring, then turns the trace on. Before the service thread starts, so the flag
  // never flips under a request being served. The instrumented build only.
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

  // ---- Python-facing bookkeeping: the single-owner rule ----
  //
  // Everything in the tier but its atomics has exactly one owner at a time: the service thread while it runs; the
  // Python caller that paused it, from the moment the service parks until resume() (RamThread::pause sets parked_);
  // the caller of pump() when there is no thread. An unpaused Python call therefore takes one of two forms:
  //   - a lock-free read of published words: mapping, counters, busy_episode, layer_rows;
  //   - a refusal, "needs the service thread paused": every other call (has, touch, assign, release, fill_begin,
  //     set_hot, slot_info, slot_to_expert, lru_order, victim_census, ...).
  //
  // caller_mutex_ serializes Python-side callers against each other only; the service, copy and fill threads never
  // take it, and the tier has no other lock. The copy thread touches no tier state (it publishes CopyDone and the gate
  // only), nor does the prefill fill thread (it drives the reader and publishes fill_landed_/fill_state_; its epilogue
  // runs on the owner, finish_fill_owned, after the join).

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
    if (tiers_[row].state[slot] == kStaging) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + ": it is a staging slot");
    }
    if (tiers_[row].filling[slot]) {
      throw std::runtime_error(
          error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while a fill writes it");
    }
    release_locked(row, slot);
    count<kVersion>();
  }

  // ---- Prefill fills (SGLANG_DSV41_ENABLE_PREFILL_FILLS) ----
  //
  // fill_begin: an eager caller that holds the pause (or pumps, with no thread) claims slots for `experts` of `row` in
  // order, until one cannot be taken (take_slot_locked), and one helper thread reads the claimed rows through the
  // service's reader while the caller gathers. A claimed slot is kReady and mapped at once, as assign() leaves it, and
  // flagged filling until the read ends: no admission can evict it and release() refuses it. The read's progress
  // publishes how many rows have landed, as a prefix of the claim order (fill_wait). The helper holds a busy episode,
  // so the watchdog aborts a hung fill the way it aborts a hung demand. fill_end() joins it, and so does resume(), so
  // the service thread and a fill never use the reader at once. The helper touches no tier state: the claimed slots
  // stay filling, and a failed fill's unlanded rows stay mapped, until the owner joins it and runs the epilogue
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
  // released its rows that did not land (here, on the caller: finish_fill_owned). Under caller_mutex_, like every other
  // join of fill_thread_ (resume, stop_thread's final_settle), so no two threads join it at once. An owed epilogue
  // writes the tier, so a caller that does not own it is refused, not let race the service (a fill starts only on the
  // owner, and resume() joins it before handing the tier back). RamThread::start refuses a tier with a fill owed, so
  // this is the backstop, not the gate.
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

  // The hot bitmap of `expected`'s record, copied into the service-owned hot_scratch_: not const, it writes
  // that scratch. request->hot_bitmap points into it until the next call.
  bool read_gpu_hot(uint32_t expected, Request* request) {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    const size_t bytes = hot_scratch_.size();
    std::memcpy(hot_scratch_.data(), record + kHotHeaderBytes, bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    asm volatile("" ::: "memory");  // the copy's plain loads must stay before the seq re-check
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 && (hot_scratch_[bytes - 1] & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    request->hot_bitmap = hot_scratch_.data();
    return true;
  }

  // Starts the loads of every line the request's read will touch: the record's two lines and its hot record's. The
  // device has just written each, so each is an L3 miss; issued together they overlap instead of queueing behind
  // read_record's branches. Called once the head shows the post: a line prefetched earlier would be refetched.
  void prefetch_request(const uint8_t* record, uint32_t seq) const {
    static_assert(kRecordBytes == 128, "a record is two lines");
    _mm_prefetch(reinterpret_cast<const char*>(record), _MM_HINT_T0);
    _mm_prefetch(reinterpret_cast<const char*>(record + 64), _MM_HINT_T0);
    if (hot_page_ == nullptr) return;
    const uint8_t* hot = hot_page_ + static_cast<int64_t>((seq - 1u) % kHotRecords) * hot_stride_;
    for (int64_t line = 0; line < hot_stride_; line += 64)
      _mm_prefetch(reinterpret_cast<const char*>(hot + line), _MM_HINT_T0);
  }

  void apply_gpu_hot(const Request& request) {
    Tier& tier = tiers_[request.row];
    for (int64_t expert = 0; expert < experts_; ++expert)
      tier.hot[expert] = (request.hot_bitmap[expert / 8] >> (expert % 8)) & 1;
  }

  // Copy engine (LEASE_PROTOCOL.md, "Copy engine"): a thread that copies a record's HIT_COPY lanes with the
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

  // The device reads kCopyArmed at every post and types copy-engine and CPU lanes only while it is 1.
  void arm_copy_engine(bool on) {
    if (on && copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_armed_.store(on, std::memory_order_release);
    store_release(lease_ + kCopyArmed, on ? 1u : 0u);
  }

  // CPU experts. The device types a captured post's CPU lanes: the
  // last split[n] of its n eligible lanes (kSplit, which this and set_cpu_split write). The service hands them to the
  // CPU expert thread and the copy thread waits for them, so CopyDone covers them. Needs the copy engine; call before
  // the service thread starts.
  void enable_cpu_experts(CpuExpertConfig config, std::vector<int64_t> split) {
    if (threaded_.load())
      throw std::runtime_error(error_prefix<Layout>() + "enable CPU experts before the service thread starts");
    if (copy_engine_ == nullptr)
      throw std::runtime_error(error_prefix<Layout>() + "CPU experts need the copy engine, which completes their lanes");
    if (cpu_ != nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are already enabled");
    config.rows = layers_;
    store_split(split.data(), static_cast<int64_t>(split.size()));
    const std::string prefix = std::string(Layout::kName) + " CPU experts: ";
    auto engine = std::make_unique<CpuExpertEngine>(std::move(config), prefix, std::string(Layout::kName) + "-cpu-exp");
    engine->start();
    cpu_ = std::move(engine);
    copy_engine_->set_cpu(cpu_.get());
  }

  // CPU experts: `row`'s layer handle, from the trait's register_layer. Any time, once per row; until then no post
  // types a CPU lane for the row.
  void set_cpu_layer(int64_t row, int64_t handle) {
    if (cpu_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    cpu_->set_layer(row, handle);
  }

  // CPU experts: replace the split table (P3's re-tuning). Any time; the device reads each entry once per post.
  void set_cpu_split(const int64_t* split, int64_t count) {
    if (cpu_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "CPU experts are not enabled");
    store_split(split, count);
  }

  // CPU experts' metrics, from the CPU expert thread: {jobs, lanes, forward ns}. Zeros when off.
  void cpu_stats(int64_t* out) const {
    out[0] = cpu_ != nullptr ? cpu_->jobs() : 0;
    out[1] = cpu_ != nullptr ? cpu_->lanes() : 0;
    out[2] = cpu_ != nullptr ? cpu_->compute_ns() : 0;
  }

  // The CPU experts' cores, empty without CPU experts. The caller's, before the service thread starts.
  std::vector<int> cpu_cores() const {
    return cpu_ != nullptr ? cpu_->cores() : std::vector<int>{};
  }

  // The bytes the DMA moves per expert of `row` (its copy table less SM entries).
  int64_t copy_expert_bytes(int64_t row) const {
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    return calibration_expert_bytes(copy_engine_->dma_entries(row));
  }

  // Startup calibration (split_calibration.h): fills out, float64 [kCalibRows][kCalibCols] ms. The caller owns the
  // tier: it claims CPU job sequences, the owner's. device -1 copies with the test backend; scratch holds
  // kCalibLanes experts on that device.
  void calibrate_cpu_split(int64_t row, int64_t device, int64_t reps, uint64_t scratch, int64_t scratch_bytes,
                           int64_t timeout_ns, double* out) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("calibrate_cpu_split");
    const std::string prefix = error_prefix<Layout>() + "calibration: ";
    if (cpu_ == nullptr) throw std::runtime_error(prefix + "CPU experts are not enabled");
    if (copy_engine_ == nullptr) throw std::runtime_error(prefix + "the copy engine is not enabled");
    if (row < 0 || row >= layers_) throw std::runtime_error(prefix + "row out of range");
    if (!cpu_->eligible(row))
      throw std::runtime_error(prefix + "row " + std::to_string(row) + " has no registered CPU layer");
    if (tiers_[row].capacity < kCalibLanes)
      throw std::runtime_error(prefix + "it needs 8 RAM slots in row " + std::to_string(row) + ", the row has " +
                               std::to_string(tiers_[row].capacity));
    if (reps < 1 || timeout_ns <= 0) throw std::runtime_error(prefix + "reps and the timeout must be positive");
    CalibrationSetup s;
    s.cpu = cpu_.get();
    s.row = row;
    s.entries = copy_engine_->dma_entries(row);
    const int64_t need = kCalibLanes * calibration_expert_bytes(s.entries);
    if (need == 0) throw std::runtime_error(prefix + "row " + std::to_string(row) + " copies no bytes");
    if (scratch == 0 || scratch_bytes < need)
      throw std::runtime_error(prefix + "the scratch holds " + std::to_string(scratch_bytes) + " bytes, it needs " +
                               std::to_string(need));
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      auto host = std::make_unique<HostCopyBackend>();
      s.host_backend = host.get();
      backend = std::move(host);
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device), prefix + "copy: ");
    }
    if (const std::string error = backend->init(); !error.empty()) throw std::runtime_error(prefix + error);
    // Freed only when idle: after a failure a copy may still be in flight into the scratch.
    struct Shutdown {
      CopyBackend* backend;
      bool idle = false;
      ~Shutdown() {
        backend->shutdown(idle);
      }
    } shutdown{backend.get()};
    s.backend = backend.get();
    s.scratch = scratch;
    s.reps = static_cast<int>(reps);
    s.timeout_ns = timeout_ns;
    calibrate_split(s, out);
    shutdown.idle = true;
  }

  // The owner (RamThread::pause, once the service parked): every job handed to the copy thread has completed (or
  // failed). False at the deadline.
  bool wait_copy_idle_owned(int64_t deadline_ns) {
    return copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
  }

  // Any thread (the FFI's copy_engine_idle): the same wait.
  bool wait_copy_idle(int64_t deadline_ns) {
    return copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
  }

  // stop_thread's final settle, on the caller once the service joined (RamThread::stop released threaded_ after the
  // join, so this caller owns the tier). A stop that arrives mid-pause can find a prefill fill still running: it is
  // joined first and its epilogue runs here (fill_join). caller_mutex_ orders this against the pausing caller.
  void final_settle() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    fill_join();
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
  // request is in flight, so the service only stops taking new ones.
  void close_admission() {
    admission_closed_.store(true);
  }

  // Every row's staging slots (LEASE_PROTOCOL.md): the first min(k, capacity - 1) FREE slots become kStaging, and the
  // tag-1 delta names them, which the device's first post of the row applies. Once, on the tier's owner, before any
  // slot is filled: no row is evicted, so no unmap needs a bulk delta. A row with fewer than 2 slots could serve no
  // miss and no hit at once: refused.
  void reserve_staging(int64_t k) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("reserve_staging");
    if (k < 1 || k > kLeaseLanes)
      throw std::runtime_error(error_prefix<Layout>() + "a row has 1.." + std::to_string(kLeaseLanes) + " staging slots");
    for (int64_t row = 0; row < layers_; ++row) {
      const Tier& tier = tiers_[row];
      if (tier.chain != 0) throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is once");
      if (tier.capacity < 2)
        throw std::runtime_error(error_prefix<Layout>() + "row " + std::to_string(row) + " has too few slots to stage");
      for (int64_t slot = 0; slot < tier.capacity; ++slot)
        if (tier.state[slot] != kFree)
          throw std::runtime_error(error_prefix<Layout>() + "reserve_staging is before any slot is filled");
    }
    for (int64_t row = 0; row < layers_; ++row) {
      Tier& tier = tiers_[row];
      const int64_t want = std::min(k, tier.capacity - 1);
      for (int64_t slot = 0; slot < want; ++slot) {
        tier.state[slot] = kStaging;
        tier.staging.push_back(static_cast<int32_t>(slot));
      }
      tier.chain = 1;
      publish_delta_locked(row, 1, tier.staging, nullptr, 0);
    }
  }

  // The eager paths' map changes since the last take, {row, expert, slot} each, slot -1 an unmap: the bulk delta
  // map_bulk_apply writes on the device. The owner only (a paused caller). The count joins a running fill first, so a
  // failed fill's unmaps are in it.
  int64_t bulk_delta_count() {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("bulk_delta_count");
    fill_join();
    return static_cast<int64_t>(bulk_.size());
  }

  void take_bulk_delta(int32_t* out, int64_t n) {
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("take_bulk_delta");
    if (n != static_cast<int64_t>(bulk_.size()))
      throw std::runtime_error(error_prefix<Layout>() + "take_bulk_delta: the count changed since bulk_delta_count");
    for (int64_t i = 0; i < n; ++i)
      for (int c = 0; c < 3; ++c)
        out[3 * i + c] = bulk_[i][c];
    bulk_.clear();
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

  // Test hooks and introspection, for the tier's owner (a paused or pumping caller; "needs the service thread paused"
  // otherwise). slot_info: [state, expert, stamp] per slot.
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

  // (free, evictable) of `wanted`.
  VictimCensus victim_census(int64_t row, const std::vector<int32_t>& wanted) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("victim_census");
    return census_locked(row, wanted);
  }

  // Any thread, lock-free: the published slot map, which holds a slot for an expert exactly while that slot is READY
  // (a slot is stored only once its bytes landed, and -1 on every unmap).
  void mapping(int64_t row, int64_t* out) const {
    for (int64_t expert = 0; expert < experts_; ++expert)
      out[expert] = __atomic_load_n(map_ + row * experts_ + expert, __ATOMIC_ACQUIRE);
  }

  // The owner's: every slot's expert.
  void slot_to_expert(int64_t row, int64_t* out) {
    row_capacity(row);
    std::lock_guard<std::mutex> caller(caller_mutex_);
    require_owner("slot_to_expert");
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot)
      out[slot] = tier.slot_to_expert[slot];
  }

  // The owner's: the READY slots' experts, least recently used first; returns how many.
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

  // The owner's: replace the row's hot set. A VRAM-hot row is decode's: kept owned it could never be a victim and
  // would hold the share down.
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
      if (tier.expert_slot[expert] >= 0) disown_locked(tier, tier.expert_slot[expert]);
    }
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

  // Test only: carry a whole ReadFault down to this tier's reader, where inject() reaches it only as a delay or a
  // blanket failure. `words` is the reader tests' fault tensor (kFaultWords int64; see fault_from). Unlike fail_reads
  // the fault does NOT short-circuit ahead of the reader: the read runs, so the fault's part errors, pack delay and the
  // rest act on rows that have already packed. The service thread applies it just before its next read (the reader is
  // that thread's alone), and it then stays until replaced; an all-default tensor clears it. Words 17-18
  // (abandon_after, step) and 22 (piece_stream) are not faults and are ignored, and words 19-20 are reserved: the
  // tier's reader always streams pieces. The reader's counters (submit and completion calls) run over its whole life,
  // so a call-numbered fault (submit_call, cqe_call) is relative to a fresh tier. InstrBuild only: ProdBuild refuses
  // rather than store a fault it would never apply.
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
  // Test only (InstrBuild): inject()'s read delay, standing in for a slow read.
  void fault_delay(int64_t ns) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
  }

  void require_owner(const char* what) const {
    if (!caller_owns()) throw std::runtime_error(error_prefix<Layout>() + what + " needs the service thread paused");
  }

  // Copy thread. Completion was observed: every DMA and CPU job of the record is done. CopyDone here, so the device's
  // wait ends as soon as the work did.
  void copy_completed(const CopyJob& job) {
    const uint32_t seq = static_cast<uint32_t>(job.gen);
    store_release64(lease_ + kLeaseCopyDone + job.idx * kLeaseCopyDoneBytes, job.gen);
    // Dekker with CW (row_copy_kernels.cuh: gate close, fence.sc.sys, CopyDone load): CopyDone before the gate load,
    // so if CW missed this store it closed the gate before this load, which then sees closed(G) and opens it.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const uint32_t closed = gate_word(seq, kLeaseGateClosed);
    if (copy_gate() == closed) cas_gate(closed, gate_word(seq, kLeaseGateOpen));
  }

  // Copy thread, or the submitter when the copy engine's job ring is full. Completion cannot be established, and the
  // device would wait on CopyDone until the watchdog: fail stop.
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
  // ProdBuild: nothing (its faults are the instrumented build's).
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

  // The completion block, then one delta record per row (expert_lease_block.lease_block_bytes).
  static int64_t lease_block_bytes(int64_t rows) {
    return kLeaseBlockBytes + round_up_page(rows * kDeltaStride);
  }

  // Before the service thread or any device exists, so plain stores and one fence suffice.
  void init_lease_block(int64_t lease_bytes) {
    if (lease_ == nullptr || reinterpret_cast<uintptr_t>(lease_) % kLeaseBlockAlign != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "the lease block must be a 4096-byte aligned block");
    }
    if (lease_bytes != lease_block_bytes(layers_)) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the lease block has " + std::to_string(lease_bytes) + " bytes, not " +
          std::to_string(lease_block_bytes(layers_)) + " for " + std::to_string(layers_) + " rows");
    }
    // Open with nothing armed: a copy wait that arms nothing passes its stream wait on this value (0 would block).
    const uint32_t open = gate_word(0, kLeaseGateOpen);
    std::memcpy(lease_ + kLeaseCopyGate, &open, 4);
    _mm_sfence();
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
        ++census.evictable;
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
    // No tier state here: the owner runs the epilogue after the join (finish_fill_owned). Until
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

  // The fill's epilogue, on the owner after the join (fill_join): a fill thread never writes the tier.
  // Clears every claimed slot's filling flag; a failed fill releases (unmaps) its rows that did not land, and
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

  // The host mirror (mapping()) only: a change the device learns from a record's delta.
  void publish_mirror(int64_t row, int64_t expert, int32_t slot) {
    __atomic_store_n(map_ + row * experts_ + expert, slot, __ATOMIC_RELEASE);
  }

  // An eager path's change: the mirror, and the bulk delta the device applies before the next forward.
  void publish_map(int64_t row, int64_t expert, int32_t slot) {
    publish_mirror(row, expert, slot);
    bulk_.push_back({static_cast<int32_t>(row), static_cast<int32_t>(expert), slot});
  }

  // Row `row`'s delta record: the payload with plain stores, an sfence, then the tag with a release (the device
  // acquires the tag before it reads the rest). The device read the previous delta in this row's last post, which
  // came before the record now being served, so nothing reads the record while it is rewritten.
  void publish_delta_locked(
      int64_t row, uint64_t tag, const FixedVec<int32_t, kLeaseLanes>& staging, const int32_t (*entries)[2],
      int count) {
    uint8_t* d = lease_ + kDeltaBase + row * kDeltaStride;
    const uint32_t n = static_cast<uint32_t>(count);
    std::memcpy(d + kDeltaCount, &n, 4);
    for (int k = 0; k < kLeaseLanes; ++k) {
      const int16_t slot = static_cast<int16_t>(k < static_cast<int>(staging.size()) ? staging[k] : -1);
      std::memcpy(d + kDeltaStaging + 2 * k, &slot, 2);
    }
    for (int i = 0; i < count; ++i) {
      const int16_t entry[2] = {static_cast<int16_t>(entries[i][0]), static_cast<int16_t>(entries[i][1])};
      std::memcpy(d + kDeltaEntries + 4 * i, entry, 4);
    }
    _mm_sfence();
    store_release64(d + kDeltaTag, tag);
  }

  void store_split(const int64_t* split, int64_t count) {
    if (count != static_cast<int64_t>(kLeaseLanes) + 1)
      throw std::runtime_error(error_prefix<Layout>() + "the CPU split table has one entry per n = 0..kLeaseLanes");
    for (int64_t n = 0; n < count; ++n)
      if (split[n] < 0 || split[n] > n)
        throw std::runtime_error(error_prefix<Layout>() + "the CPU split table must satisfy 0 <= split[n] <= n");
    for (int64_t n = 0; n < count; ++n)
      __atomic_store_n(reinterpret_cast<int32_t*>(lease_ + kSplit) + n, static_cast<int32_t>(split[n]), __ATOMIC_RELAXED);
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
        if (!tier.prefill_owned[s] || tier.state[s] != kReady || tier.filling[s]) continue;
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

  // An eager admission's slot: a kFree one, or an evicted kReady one; never kStaging.
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
      if (tier.filling[slot]) continue;  // a prefill fill is still writing it
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

  void stamp_routed_locked(Tier& tier, int32_t expert) {
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0 && tier.state[slot] == kReady) {
      tier.stamp[slot] = ++tick_;
      disown_locked(tier, slot);
    }
  }

  // A record with no lanes: refresh the recency of its routed experts' RAM rows. No eviction, no read.
  void touch_request(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return;
    Tier& tier = tiers_[request.row];
    for (int32_t expert : request.protect) {
      if (expert >= 0 && expert < experts_) stamp_routed_locked(tier, expert);
    }
  }

  // A decode miss's RAM victim: a free slot first, else the least recently used READY slot that is not VRAM-hot, not
  // routed by this record (`wanted`), and not filling. Its expert is unmapped (on the host mirror; the device by the
  // delta) and returned in `old` (-1 for a free slot). -1 when none qualifies: the miss is not cached.
  int64_t take_victim_locked(int64_t row, std::span<const int32_t> wanted, int32_t* old) {
    Tier& tier = tiers_[row];
    *old = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] == kFree) return slot;
    }
    int64_t best = -1;
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      if (tier.state[slot] != kReady || tier.filling[slot]) continue;
      const int32_t expert = tier.slot_to_expert[slot];
      if (tier.hot[expert] || listed(wanted, expert)) continue;
      if (best < 0 || tier.stamp[slot] < tier.stamp[best]) best = slot;
    }
    if (best < 0) return -1;
    *old = tier.slot_to_expert[best];
    publish_mirror(row, *old, -1);
    tier.expert_slot[*old] = -1;
    tier.slot_to_expert[best] = -1;
    disown_locked(tier, best);
    count<kEvictions>();
    return best;
  }

  // What serve_record's phases hand on: the copy job, the misses in lane order, and every expert the record routes.
  struct RecordPlan {
    int64_t idx = 0;  // the record's index in the demand ring
    CopyJob job;
    FixedVec<int32_t, kWanted> wanted;
    FixedVec<int32_t, kLeaseLanes> missing;
    FixedVec<int64_t, kLeaseLanes> slots;
    FixedVec<int32_t, kLeaseLanes> miss_lane;  // the lane index of each read row
  };

  // The record's CPU-miss jobs still to submit.
  struct CpuMissBatch {
    uint32_t sent = 0;  // bit i: miss i went to the CPU
    int left = 0;
    uint32_t next = 0;  // the next CPU-miss job's sequence, short of the last (job.late_seq)
  };

  [[noreturn]] void fail_record(const Request& request, const std::string& why) {
    fail_stop(error_prefix<Layout>() + "request " + std::to_string(request.seq) + " of row " +
              std::to_string(request.row) + ": " + why);
  }

  // The record's routed experts, each once: protect's, stamped as routed, then the lanes'.
  void collect_wanted_locked(Tier& tier, const Request& request, RecordPlan* plan) {
    for (int32_t expert : request.protect) {
      if (expert < 0 || expert >= experts_) fail_record(request, "an expert is out of range");
      if (!listed(plan->wanted, expert)) plan->wanted.push_back(expert);
      stamp_routed_locked(tier, expert);
    }
    FixedVec<int32_t, kLeaseLanes> lane_experts;
    for (const Lane& lane : request.lanes) {
      if (lane.expert < 0 || lane.expert >= experts_) fail_record(request, "an expert is out of range");
      if (listed(lane_experts, lane.expert)) fail_record(request, "a lane names an expert twice");
      lane_experts.push_back(lane.expert);
      if (!listed(plan->wanted, lane.expert)) plan->wanted.push_back(lane.expert);
    }
  }

  // Checks every lane against the tier: a miss goes to the read lists, a copy-engine or CPU hit (stamped) to the copy
  // job, an SM hit nowhere. Then the map chain: the row's next one when the record misses, else 0.
  void classify_lanes_locked(Tier& tier, const Request& request, RecordPlan* plan) {
    const bool host_lanes = copy_engine_ != nullptr && copy_armed_.load(std::memory_order_acquire) && request.captured;
    const bool cpu_row = host_lanes && cpu_ != nullptr && cpu_->eligible(request.row);
    CopyJob& job = plan->job;
    for (size_t j = 0; j < request.lanes.size(); ++j) {
      const Lane& lane = request.lanes[j];
      const auto fail = [&](const char* why) {  // the message is built only to fail: the hot path allocates nothing
        fail_record(request, "lane " + std::to_string(j) + " (expert " + std::to_string(lane.expert) + ", slot " +
                                 std::to_string(lane.slot) + ")" + why);
      };
      if (is_miss(lane.kind)) {
        if (!listed(tier.staging, lane.slot) || tier.state[lane.slot] != kStaging) fail(" is not a staging slot");
        if (listed(plan->slots, static_cast<int64_t>(lane.slot))) fail(" shares its staging slot with another miss");
        if (tier.expert_slot[lane.expert] >= 0) fail(" misses an expert the tier holds");
        if (lane.kind == kKindMissCpu) {
          if (!cpu_row || cpu_->parts() < 2) fail(": a CPU miss on a row without CPU experts' miss part");
          ++job.late_cpu;
        }
        plan->missing.push_back(lane.expert);
        plan->slots.push_back(lane.slot);
        plan->miss_lane.push_back(static_cast<int32_t>(j));
        continue;
      }
      if (lane.slot < 0 || lane.slot >= tier.capacity || tier.expert_slot[lane.expert] != lane.slot ||
          tier.state[lane.slot] != kReady) {
        fail(": the device maps it there, the tier does not");
      }
      tier.stamp[lane.slot] = ++tick_;
      disown_locked(tier, lane.slot);
      if (lane.kind == kKindHitCopy) {
        if (!host_lanes || !copy_engine_->eligible(request.row, lane.dst)) fail(": a copy-engine lane on an ineligible row");
      } else if (lane.kind == kKindHitCpu) {
        if (!cpu_row) fail(": a CPU lane on a row without CPU experts");
        job.cpu_mask |= 1u << j;
      } else {
        continue;  // kKindHitSm: C1 copies it, the host has nothing to do
      }
      job.lanes[job.count++] = CopyLane{static_cast<int32_t>(j), lane.slot, lane.dst, lane.weight};
      job.mask |= 1u << j;
    }
    if (!plan->missing.empty()) {
      if (request.chain != tier.chain + 1) {
        fail_record(request, "map chain " + std::to_string(request.chain) + ", the row expects " +
                                 std::to_string(tier.chain + 1));
      }
    } else if (request.chain != 0) {
      fail_record(request, "map chain " + std::to_string(request.chain) + " on a record without a miss");
    }
  }

  void submit_cpu_job(const Request& request, const CpuJob& cpu_job) {
    if (!cpu_->submit(cpu_job)) fail_record(request, "the CPU expert ring is full");
    this->template count<kCpuJobs>();
  }

  // The record's CPU-hit job, then its copy job, so the CPU starts first. Returns the CPU misses' batch, whose jobs
  // read_misses submits as their rows land.
  CpuMissBatch submit_host_lanes(const Request& request, RecordPlan* plan) {
    CopyJob& job = plan->job;
    CpuMissBatch misses;
    misses.left = job.late_cpu;
    if (job.count == 0 && job.late_cpu == 0) return misses;
    job.gen = request.gen;
    job.idx = plan->idx;
    job.row = request.row;
    if constexpr (Build::kMetrics) job.submit_ns = now_ns();  // copy_latency_ns, a metric
    this->template count<kCopyJobs>();
    this->template count<kCopyLanes>(job.count + job.late_cpu);
    const int cpu_hits = __builtin_popcount(job.cpu_mask);
    this->template count<kCpuLanes>(cpu_hits + job.late_cpu);
    if (cpu_hits > 0 || job.late_cpu > 0) {
      // One sequence per job the record can need: the hits' and one per CPU miss. The last miss job takes the last,
      // so the copy thread's done(late_seq) covers however the misses were batched.
      const uint32_t first = cpu_->claim((cpu_hits > 0 ? 1 : 0) + job.late_cpu);
      job.cpu_seq = first;
      misses.next = first + (cpu_hits > 0 ? 1u : 0u);
      job.late_seq = misses.next + static_cast<uint32_t>(job.late_cpu) - 1u;
      if (cpu_hits > 0) {
        CpuJob cpu_job;
        cpu_job.row = request.row;
        cpu_job.part = 0;
        cpu_job.seq = first;
        for (int i = 0; i < job.count; ++i) {
          const CopyLane& lane = job.lanes[i];
          if ((job.cpu_mask >> lane.lane & 1u) == 0) continue;
          cpu_job.slots[cpu_job.k] = lane.host_slot;
          cpu_job.weights[cpu_job.k] = lane.weight;
          ++cpu_job.k;
        }
        submit_cpu_job(request, cpu_job);
      }
    }
    copy_engine_->submit(job);
    return misses;
  }

  // Victims and the delta, before any read. A miss lands in its staging slot whatever happens here; whether it is
  // cached is the victim's question. inserted[i]: miss i took a victim and is cached once read.
  void reserve_victims_locked(Tier& tier, const Request& request, const RecordPlan& plan, bool* inserted) {
    int32_t entries[kDeltaMaxEntries][2];
    int count = 0;
    FixedVec<int32_t, kLeaseLanes> staging = tier.staging;
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      int32_t old = -1;
      const int64_t victim = take_victim_locked(request.row, plan.wanted, &old);
      if (victim < 0) {
        this->template count<kRamInsertSkipped>();
        continue;
      }
      if (old >= 0) {
        entries[count][0] = old;
        entries[count][1] = -1;
        ++count;
      }
      entries[count][0] = plan.missing[i];
      entries[count][1] = static_cast<int32_t>(plan.slots[i]);
      ++count;
      const int32_t slot = static_cast<int32_t>(plan.slots[i]);
      tier.state[victim] = kStaging;
      for (int32_t& s : staging)
        if (s == slot) s = static_cast<int32_t>(victim);
      inserted[i] = true;
    }
    tier.staging = staging;
    tier.chain = request.chain;
    publish_delta_locked(request.row, request.chain, tier.staging, entries, count);
  }

  // The CPU misses whose rows landed since the last call, as one part-1 job: from read()'s progress hook, and after
  // the read. Every job after the record's first adds into the part.
  void submit_landed_cpu_misses(const Request& request, const RecordPlan& plan, CpuMissBatch* misses) {
    if (misses->left == 0) return;
    const CopyJob& job = plan.job;
    CpuJob cpu_job;
    cpu_job.row = request.row;
    cpu_job.part = 1;
    cpu_job.accumulate = misses->left < job.late_cpu;
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      const Lane& lane = request.lanes[plan.miss_lane[i]];
      if (lane.kind != kKindMissCpu || (misses->sent >> i & 1u) != 0) continue;
      if (!(i < packed_.size() && packed_[i] != 0)) continue;
      cpu_job.slots[cpu_job.k] = static_cast<int32_t>(plan.slots[i]);
      cpu_job.weights[cpu_job.k] = lane.weight;
      ++cpu_job.k;
      misses->sent |= 1u << i;
    }
    if (cpu_job.k == 0) return;
    misses->left -= cpu_job.k;
    cpu_job.seq = misses->left == 0 ? job.late_seq : misses->next++;
    _mm_sfence();  // the rows' bytes before the CPU thread reads them
    submit_cpu_job(request, cpu_job);
  }

  // Reads the misses into their staging slots, each CPU miss going to the CPU as its row lands. Returns the stage
  // status.
  int64_t read_misses(const Request& request, const RecordPlan& plan, CpuMissBatch* misses, StageRecord* cur) {
    packed_.clear();
    const bool publishing = init_piece_words_locked(request, plan.miss_lane, plan.idx);
    bool fail_reads = false;
    if constexpr (Build::kFaults) {  // the test faults (inject, inject_fault): InstrBuild only
      apply_pending_fault();
      const int64_t delay = faults_.delay_ns.load();
      if (delay > 0 && demands_read_ >= faults_.delay_after.load()) fault_delay(delay);
      fail_reads = faults_.fail_reads.load();
    }
    if (fail_reads) fail_record(request, "a test fault failed the read");
    const int result = reader_.read(
        request.row,
        plan.missing,
        plan.slots,
        kBounceRows,
        [](size_t) { return false; },
        cur,
        &packed_,
        SIZE_MAX,
        // read() runs this once per drain-loop turn and once per finished row.
        [&] { submit_landed_cpu_misses(request, plan, misses); },
        publishing ? &piece_publish_ : nullptr);
    stats_.store(kPiecePublishRefused, reader_.publish_refused());
    if (result != 1) {
      count<kReadErrors>();
      fail_record(request, "the read failed");
    }
    ++demands_read_;
    _mm_sfence();  // the split's memcpy stores land before the mirror publishes them (D11)
    submit_landed_cpu_misses(request, plan, misses);
    if (misses->left != 0) fail_record(request, "a CPU miss's row never landed");
    return kStatusServed;
  }

  // Maps each read miss that took a victim into the tier and the host mirror.
  void commit_inserted_locked(Tier& tier, const Request& request, const RecordPlan& plan, const bool* inserted) {
    for (size_t i = 0; i < plan.missing.size(); ++i) {
      if (!inserted[i]) continue;
      tier.slot_to_expert[plan.slots[i]] = plan.missing[i];
      tier.expert_slot[plan.missing[i]] = static_cast<int32_t>(plan.slots[i]);
      tier.state[plan.slots[i]] = kReady;
      tier.stamp[plan.slots[i]] = ++tick_;
      publish_mirror(request.row, plan.missing[i], static_cast<int32_t>(plan.slots[i]));
    }
    // One writer (the owner), read lock-free by layer_rows: a relaxed store.
    std::atomic_ref<int64_t>(tier.rows_demand)
        .store(tier.rows_demand + static_cast<int64_t>(plan.missing.size()), std::memory_order_relaxed);
    count<kVersion>();
    count<kRowsRead>(static_cast<int64_t>(plan.missing.size()));
  }

  // A record with lanes (LEASE_PROTOCOL.md, "The host per record"). The device typed every lane from its copy of the
  // map, so the host checks each against the tier and fail-stops on any disagreement. Order: stamps, checks, the copy
  // job (hits start at once), victims and the map delta (before any read, so a served chain always has its delta
  // published), then the misses' reads.
  void serve_record(const Request& request, int64_t* rows) {
    StageRecord* const cur = stage_record();  // null in ProdBuild, so every `if (cur)` below folds away
    if (cur) cur->lanes = static_cast<int64_t>(request.lanes.size());
    if (request.row < 0 || request.row >= layers_) fail_record(request, "the row is out of range");
    Tier& tier = tiers_[request.row];
    RecordPlan plan;
    plan.idx = static_cast<int64_t>((request.seq - 1u) % kDemandRecords);
    collect_wanted_locked(tier, request, &plan);
    classify_lanes_locked(tier, request, &plan);
    CpuMissBatch misses = submit_host_lanes(request, &plan);
    const bool reads = !plan.missing.empty();
    bool inserted[kLeaseLanes] = {};
    if (reads) reserve_victims_locked(tier, request, plan, inserted);
    if (cur) cur->reserved = stamp(cur);
    const int64_t status = reads ? read_misses(request, plan, &misses, cur) : kStatusNoRead;
    if (reads) commit_inserted_locked(tier, request, plan, inserted);
    if (cur) {
      cur->mapped = stamp(cur);
      cur->row = request.row;
      cur->ok = 1;
      cur->status = status;
      cur->rows = static_cast<int64_t>(plan.missing.size());
    }
    *rows = static_cast<int64_t>(plan.missing.size());
  }

  // Store piece_word(gen) into the readiness word of every kMissGpu lane, then fence, and record those words as the
  // reader's publish targets, by the row's ordinal in the read. A kMissCpu lane has no device reader: no word. False
  // when no lane has one.
  bool init_piece_words_locked(const Request& request, std::span<const int32_t> miss_lane, int64_t idx) {
    piece_targets_.assign(miss_lane.size(), PieceTarget{});
    bool any = false;
    for (size_t i = 0; i < miss_lane.size(); ++i) {
      const int32_t lane = miss_lane[i];
      if (request.lanes[lane].kind != kKindMissGpu) continue;
      uint8_t* word =
          lease_ + kLeasePieceMask + (idx * kLeaseLanes + static_cast<int64_t>(lane)) * kLeasePieceMaskLineBytes;
      store_release64(word, piece_word(request.gen));
      PieceTarget& target = piece_targets_[i];
      target.words[target.count++] = reinterpret_cast<uint64_t*>(word);
      any = true;
    }
    _mm_sfence();
    piece_publish_ = PiecePublish{request.gen, piece_targets_.data()};
    return any;
  }

  void handle_record(const Request& request) {
    begin_busy();
    int64_t rows = 0;
    if (!request.lanes.empty()) {
      serve_record(request, &rows);
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
  uint8_t* hot_page_ = nullptr;  // the hot bitmap sidecar: when given, each record's hot set is applied
  int64_t hot_stride_ = 0;
  // The copy engine, when enabled (before the service thread starts); armed separately, and only then used.
  std::unique_ptr<Engine> copy_engine_;
  // CPU experts, when enabled (after the copy engine, before the service thread); stopped after the copy thread.
  std::unique_ptr<CpuExpertEngine> cpu_;
  std::atomic<bool> copy_armed_{false};
  int64_t copy_wait_timeout_ns_ = 0;           // set with the copy engine, before any thread reads it
  // Piece streaming: serve_record's readiness words per row it reads, reused every request (like packed_).
  std::vector<PieceTarget> piece_targets_;
  PiecePublish piece_publish_;
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new
  std::atomic<uint32_t> handled_{0};           // the last seq the owner finished (handled_through)
  // The eager paths' map changes, {row, expert, slot}: the owner's, taken by take_bulk_delta. Grows only on those
  // paths (assign, prefill fills, release), never on the record path.
  std::vector<std::array<int32_t, 3>> bulk_;
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
  // The pausing caller owns the tier while parked_ (see caller_owns). caller_mutex_ serializes Python-side callers only.
  std::atomic<bool> parked_{false};
  std::mutex caller_mutex_;
  // The watchdog's hung-request marker, see busy_episode(): a new value per demand or fill in service, 0 when none.
  // episodes_ is the service thread's, or a fill's (they never run at once: a fill needs the pause).
  std::atomic<uint64_t> busy_{0};
  uint64_t episodes_ = 0;
  // Test-only faults (inject, inject_fault): InstrBuild only.
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
  // The type-system half of the prod proof (nm cannot see state whose names are inlined away): ProdBuild's fault,
  // metric and trace members are empty types, which [[no_unique_address]] gives no storage.
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(faults_)>, "ProdBuild has no faults");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(stats_)>, "ProdBuild has no metrics");
  static_assert(!std::is_same_v<Build, ProdBuild> || std::is_empty_v<decltype(trace_)>, "ProdBuild has no trace");
};

}  // namespace expert_stream
}  // namespace sglang
