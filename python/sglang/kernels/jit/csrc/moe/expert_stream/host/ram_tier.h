// Tier/RamTier: slot state and admission.
#pragma once

#include "copy_engine.h"
#include "../row_layout.h"

namespace sglang {
namespace expert_stream {

struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (LOADING or READY) or -1
  std::vector<uint8_t> hot;
  std::vector<uint32_t> leases;      // GPU-reader leases per slot (LEASE_PROTOCOL.md section 8); 0 frees a slot for eviction
  std::vector<uint32_t> generation;  // bumped before a slot's bytes change; mirrored into the lease block's SlotGen
  std::vector<uint8_t> filling;      // a prefill fill is still writing this kReady slot: never a victim, never released
  // Prefill share (plan 2026-09-25-dsv41-prefill-eviction): 1 while a row a prefill admitted has not been used by
  // decode. `owned` counts them. Only take_admit_slot_locked sets it, so it stays all zero with the share off.
  std::vector<uint8_t> prefill_owned;
  int64_t owned = 0;
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
// pump(), or the Task 12 thread. The Python-facing methods take the same mutex.
template <class Source>
class RamTier {
 public:
  using Layout = typename Source::LayoutType;

  RamTier(
      uint8_t* page, int32_t* slot_map, uint8_t* lease, int64_t lease_bytes, Tables tables, std::vector<int64_t> capacity,
      bool direct, int64_t pack_workers, uint8_t* hot_page, int64_t hot_bytes)
      : page_(page),
        map_(slot_map),
        lease_(lease),
        hot_page_(hot_page),
        layers_(tables.layers),
        experts_(tables.experts),
        reader_(std::move(tables), direct, pack_workers),
        tiers_(static_cast<size_t>(layers_)) {
    hot_stride_ = ((kHotHeaderBytes + (experts_ + 7) / 8 + kHotAlignment - 1) / kHotAlignment) * kHotAlignment;
    if (hot_page_ != nullptr && hot_bytes != kHotRecords * hot_stride_)
      throw std::runtime_error(error_prefix<Layout>() + "hot bitmap sidecar size disagrees with expert count");
    for (auto& counter : counters_)
      counter.store(0);
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
  }

  // The copy thread's callbacks use tiers_, mutex_ and counters_, which are destroyed before copy_engine_ would be.
  ~RamTier() {
    fill_join();  // the fill thread reads through reader_ into the slabs
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

  std::vector<int> packing_cpus() const { return reader_.packing_cpus(); }

  uint8_t* page() const {
    return page_;
  }
  int64_t busy_since() const {
    return busy_since_.load();
  }
  void set_counter(int index, int64_t value) {
    counters_[index].store(value);
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
  void set_threaded(bool threaded) {
    threaded_.store(threaded);
  }

  // Serve the next posted demand record, if any. True when it handled one.
  bool pump_demand() {
    retire_leases();  // first, so that an idle pump still retires what the device has acknowledged
    if (admission_closed_.load()) return false;
    const uint32_t head = load_acquire(page_ + kDemandHead);
    if (head == 0 || !reached(head, next_demand_)) return false;
    // A deferred demand is not looked at again until a lease retires: no stage record, no clock read per poll.
    if (deferred_seq_ == next_demand_ && !deferral_may_retry()) return false;
    begin_stage(kStageDemand, next_demand_, head - next_demand_);
    if (head - next_demand_ >= kDemandRecords) {
      // Lapped: resume at head - 14 (head - 15 may be mid-rewrite) and count every skipped seq.
      counters_[kOverruns].fetch_add(head - next_demand_ - (kDemandRecords - 2));
      next_demand_ = skip_zero(head - kDemandRecords + 2u);
    }
    uint8_t* record = page_ + record_offset(kDemandRing, kDemandRecords, next_demand_);
    Request request;
    if (read_record(record, next_demand_, &request)) {
      judge_prefetch(request);
      const bool gpu_hot = gpu_hot_mode_.load() && request.armed;
      const bool hot_ok = !gpu_hot ||
          (request.row >= 0 && request.row < layers_ && read_gpu_hot(next_demand_, &request) &&
           load_acquire(record + kRecSeq) == next_demand_);
      if (!hot_ok) {
        counters_[kOverruns].fetch_add(1);
        set_status(record, kFailed);
      } else if (lease_mode_ && request.armed && !read_lane_request(next_demand_, &request)) {
        counters_[kOverruns].fetch_add(1);  // a later request overwrote the lane request: a lapped record
      } else if (lease_mode_ && request.armed && terminal_seen(request)) {
        counters_[kLateAfterTerminal].fetch_add(1);  // the device gave up on it: serve nothing, lease nothing
      } else {
        if (gpu_hot) apply_gpu_hot(request);
        const Defer reason = request.armed ? defers(request) : Defer::kNone;
        if (reason != Defer::kNone) {
          // Held back, not failed and not served: return before handle_demand (so busy_since_ and kBusySeq stay
          // untouched, or the watchdog would count the wait as a hung read) and before the tail (no demand_done, no
          // advance). No stage record is pushed; the first observation time is kept for the one written when it is served.
          if (deferred_seq_ != next_demand_) {
            deferred_seq_ = next_demand_;
            deferred_observed_ns_ = cur_ != nullptr ? cur_->observed : 0;
            counters_[reason == Defer::kRequestSlot ? kDeferredReuse : kDeferred].fetch_add(1);
          }
          deferred_stamp_ = lease_changes_.load();
          deferred_gen_ = request.gen;
          cur_ = nullptr;
          return false;
        } else {
          if (cur_ != nullptr && deferred_seq_ == next_demand_ && deferred_observed_ns_ != 0) {
            cur_->observed = deferred_observed_ns_;
          }
          handle_demand(request, record);
        }
      }
    } else {
      counters_[kOverruns].fetch_add(1);  // status stays pending: a waiting layer fails stop
    }
    deferred_seq_ = 0;
    if (const int64_t stall = done_stall_ns_.load(); stall > 0) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(stall));  // test only: see inject_done_stall
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
      counters_[kAdvisoriesSkipped].fetch_add(head - next_advice_ - (kAdviseRecords - 2));
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
      cur_ = nullptr;  // a skipped advisory is no service: no stage record
      counters_[kAdvisoriesSkipped].fetch_add(1);
    } else {
      in_advice_.store(true);
      counters_[kAdvisories].fetch_add(1);
      // An advisory gives up only between rows, not inside a blocking read: the watchdog's
      // stuck rule covers it like a demand, or a hung read would block stop()'s join forever.
      busy_since_.store(now_ns());
      int64_t rows = 0;
      serve(request, true, &rows);
      busy_since_.store(0);
      in_advice_.store(false);
    }
    store_release(page_ + kAdviseDone, next_advice_);
    end_stage();
    next_advice_ = skip_zero(next_advice_ + 1u);
    return true;
  }

  // ---- Stage trace: one StageRecord per served request, drained by Python ----

  // Allocates the ring, then turns the trace on. Before the service thread starts, so the flag
  // never flips under a request being served.
  void enable_trace(size_t capacity) {
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "enable the stage trace before the service thread starts");
    std::lock_guard<std::mutex> guard(trace_mutex_);
    ring_ = std::make_unique<StageRing>(capacity);
    trace_on_.store(true, std::memory_order_release);
  }

  // Up to `max` records into `out` (stage_words() int64 each); returns how many. The count of records
  // dropped for a full ring is `trace_dropped()`.
  int64_t drain_trace(StageRecord* out, int64_t max) {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    return ring_ ? ring_->drain(out, max) : 0;
  }

  int64_t trace_dropped() {
    std::lock_guard<std::mutex> guard(trace_mutex_);
    return ring_ ? ring_->dropped() : 0;
  }

  // ---- Python-facing bookkeeping; eager callers pause the thread first (Task 12) ----

  bool has(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    return tiers_[row].expert_slot[expert] >= 0;
  }

  void touch(int64_t row, int64_t expert) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    const int32_t slot = tier.expert_slot[expert];
    if (slot >= 0) {
      tier.stamp[slot] = ++tick_;
      if (prefill_share_ == 0) disown_locked(tier, slot);  // under a share the touch is the prefill's own hit
    }
  }

  // A slot for a Python-side read; the map entry is published at once (the device is idle
  // and the thread paused when an eager path calls this). evicted: -1 none, -2 already held.
  int64_t assign(int64_t row, int64_t expert, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
    std::lock_guard<std::mutex> guard(mutex_);
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
    counters_[kVersion].fetch_add(1);
    return slot;
  }

  void release(int64_t row, int64_t slot) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (tiers_[row].state[slot] == kLoading) {
      // The service is filling it and will publish it; freeing it would hand it out twice.
      throw std::runtime_error(error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while it is loading");
    }
    if (leased_locked(tiers_[row], slot)) {
      throw std::runtime_error(error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while it is leased");
    }
    if (tiers_[row].filling[slot]) {
      throw std::runtime_error(error_prefix<Layout>() + "release of pinned slot " + std::to_string(slot) + " while a fill writes it");
    }
    release_locked(row, slot);
    counters_[kVersion].fetch_add(1);
  }

  // ---- Prefill fills (SGLANG_DSV41_ENABLE_PREFILL_FILLS, plan 2026-09-25-dsv41-prefill-fills) ----
  //
  // An eager caller that holds the pause (or pumps, with no thread) claims slots for `experts` of `row` in order, until
  // one cannot be taken (take_slot_locked: never a hot, leased or filling row, and a `protect`ed one only with
  // `fallback`), and one helper thread reads the claimed rows through the service's reader while the caller gathers.
  // A claimed slot is kReady and mapped at once, as assign() leaves it, and flagged filling until the read ends: no
  // admission can evict it and release() refuses it. The read's progress publishes how many rows have landed, as a
  // prefix of the claim order (fill_wait). The helper holds busy_since_, so the watchdog aborts a hung fill the way it
  // aborts a hung demand. fill_end() joins it; the service thread's resume() joins it first too, so the service
  // thread and a fill never use the reader at once. Returns the count claimed; slots[i] is expert i's slot.
  int64_t fill_begin(
      int64_t row, const std::vector<int32_t>& experts, const std::vector<int32_t>& protect, bool fallback,
      int64_t* slots, int64_t* evictions) {
    if (threaded_.load() && !pause_requested_.load()) {
      throw std::runtime_error(error_prefix<Layout>() + "a prefill fill needs the service thread paused");
    }
    if (fill_thread_.joinable()) throw std::runtime_error(error_prefix<Layout>() + "a prefill fill is already running");
    std::vector<int32_t> claimed;
    std::vector<int64_t> taken;
    *evictions = 0;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      Tier& tier = tiers_[row];
      for (const int32_t expert : experts) {
        if (tier.expert_slot[expert] >= 0) {
          throw std::runtime_error(error_prefix<Layout>() + "fill of expert " + std::to_string(expert) + " that holds a slot");
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
      if (!taken.empty()) counters_[kVersion].fetch_add(1);
    }
    for (size_t i = 0; i < taken.size(); ++i) slots[i] = taken[i];
    fill_landed_.store(0, std::memory_order_release);
    fill_state_.store(taken.empty() ? kFillOk : kFillRunning, std::memory_order_release);
    if (taken.empty()) return 0;
    fill_row_ = row;
    fill_experts_ = std::move(claimed);
    fill_slots_ = std::move(taken);
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
  // released its rows that did not land.
  int64_t fill_end() {
    fill_join();
    return fill_state_.load(std::memory_order_acquire) == kFillFailed ? 0 : 1;
  }

  void fill_join() {
    if (fill_thread_.joinable()) fill_thread_.join();
  }

  // Lease mode: the service reads each armed request's lane request, leases every lane's source slot and publishes
  // a row result per lane before it answers (LEASE_PROTOCOL.md 7). Off leaves every request as it always was.
  void set_lease_mode(bool on) {
    if (lease_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "lease mode needs a lease block");
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "set lease mode before the service thread starts");
    lease_mode_ = on;
  }

  // Rows a prefill may own per layer; 0 (the default, and always with SGLANG_DSV41_ENABLE_PREFILL_SHARE off) admits as
  // take_slot_locked always has. The service sets it before each forward: the share for a prefill, 0 for decode.
  void set_prefill_share(int64_t share) {
    if (share < 0) throw std::runtime_error(error_prefix<Layout>() + "a prefill share cannot be negative");
    std::lock_guard<std::mutex> guard(mutex_);
    prefill_share_ = share;
  }

  void set_gpu_hot(bool on) {
    if (hot_page_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "GPU hot mode needs a sidecar");
    if (!lease_mode_) throw std::runtime_error(error_prefix<Layout>() + "GPU hot mode needs leases");
    gpu_hot_mode_.store(on);
  }

  bool read_gpu_hot(uint32_t expected, Request* request) const {
    if (hot_page_ == nullptr) return false;
    const uint8_t* record = hot_page_ + static_cast<int64_t>((expected - 1u) % kHotRecords) * hot_stride_;
    if (load_acquire(record) != expected) return false;
    uint32_t count = 0;
    std::memcpy(&count, record + 4, 4);
    if (count != experts_) return false;
    const int64_t bytes = (experts_ + 7) / 8;
    request->hot_bitmap.assign(record + kHotHeaderBytes, record + kHotHeaderBytes + bytes);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (load_acquire(record) != expected) return false;
    if (experts_ % 8 != 0 &&
        (request->hot_bitmap.back() & static_cast<uint8_t>(~((1u << (experts_ % 8)) - 1u))) != 0)
      return false;
    return true;
  }

  void apply_gpu_hot(const Request& request) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[request.row];
    for (int64_t expert = 0; expert < experts_; ++expert)
      tier.hot[expert] = (request.hot_bitmap[expert / 8] >> (expert % 8)) & 1;
  }

  // Two-phase mode (Task 6 V1): grant the resident lanes inside serve()'s reservation hold, before read(), so the
  // device can copy them while the missing rows are still being read. Off leaves lease mode exactly as Task 5
  // shipped it, which is the A1 arm every Task 6 measurement is reported against.
  void set_two_phase(bool on) {
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "set two-phase mode before the service thread starts");
    two_phase_ = on;
  }

  // Piece streaming (SGLANG_DSV41_ENABLE_RAM_MISS_PIECE_STREAM): the reader reads each part as sub-reads and vets
  // rows piece by piece. Before the thread starts. The reader refuses it without packing workers; the service
  // refuses it without two-phase and lease mode.
  void set_piece_stream(bool on) {
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "set piece streaming before the service thread starts");
    reader_.set_piece_stream(on);
    piece_stream_ = on;
  }

  // Copy engine (LEASE_PROTOCOL.md 7.6): a thread that copies the reservation hold's hit lanes with the DMA engine.
  // `device` < 0 is the CPU test backend (HostCopyBackend). Before the service thread starts; unarmed until arm().
  void enable_copy_engine(int64_t device, int64_t spin_ns) {
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "enable the copy engine before the service thread starts");
    if (copy_engine_ != nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is already enabled");
    // Only the piece-streaming chain grants every lane in the reservation hold and runs the copy wait.
    if (!(lease_mode_ && two_phase_ && piece_stream_)) {
      throw std::runtime_error(error_prefix<Layout>() + "the copy engine needs lease mode, two-phase and piece streaming");
    }
    std::unique_ptr<CopyBackend> backend;
    if (device < 0) {
      backend = std::make_unique<HostCopyBackend>();
    } else {
      backend = std::make_unique<CudaCopyBackend>(static_cast<int>(device));
    }
    auto engine = std::make_unique<CopyEngine>(
        std::move(backend), layers_, spin_ns, counters_,
        [this](const CopyJob& job) { return copy_completed(job); },
        [this](const CopyJob& job) { return copy_acked(job); },
        [this](const CopyJob& job, int error) { copy_failed(job, error); },
        std::string(Layout::kName) + " RAM miss copy engine: ", std::string(Layout::kName) + "-copy-eng");
    engine->start();
    copy_engine_ = std::move(engine);
  }

  // Row `row`'s copy table: `entries` rows of {source slab address, destination tensor address, row bytes}. Bit i
  // of `sm_mask` leaves entry i to the copy wait's SM reads (SGLANG_DSV41_ENABLE_RAM_MISS_SM_SMALL_COPIES).
  // Precondition (row_layout.h's ExpertRowLayout): `entries` holds one row per Layout::kNames entry, in
  // kNames order, so comparing sm_mask (a copy-table entry index) against Layout::kSmallMask (a layout-name
  // index bitmask) below is comparing the same index space.
  void set_copy_table(int64_t row, const int64_t* entries, int64_t count, int64_t dst_rows, int64_t sm_mask) {
    if ((static_cast<uint64_t>(sm_mask) & ~static_cast<uint64_t>(Layout::kSmallMask)) != 0) {
      throw std::runtime_error(error_prefix<Layout>() + "sm_mask names a tensor that is not one of the layout's small ones");
    }
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    if (sm_mask < 0 || (count < 63 && (sm_mask >> count) != 0)) {
      throw std::runtime_error(error_prefix<Layout>() + "the SM mask names a copy-table entry that does not exist");
    }
    std::vector<CopyEntry> table;
    for (int64_t i = 0; i < count; ++i) {
      if (entries[3 * i + 2] <= 0) throw std::runtime_error(error_prefix<Layout>() + "a copy-table entry of no bytes");
      table.push_back(CopyEntry{
          static_cast<uint64_t>(entries[3 * i]), static_cast<uint64_t>(entries[3 * i + 1]), entries[3 * i + 2],
          (sm_mask >> i & 1) != 0});
    }
    copy_engine_->set_table(row, std::move(table), dst_rows);
  }

  void arm_copy_engine(bool on) {
    if (on && copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "the copy engine is not enabled");
    copy_armed_.store(on, std::memory_order_release);
  }

  // Native prefetch (plan 2026-09-25-dsv41-native-prefetch): serve the device's advisory next-layer copy requests
  // from `page` (kPrefetchPageBytes, pinned). Needs the copy engine; before the service thread starts.
  void enable_native_prefetch(uint8_t* page) {
    if (threaded_.load()) throw std::runtime_error(error_prefix<Layout>() + "enable native prefetch before the service thread starts");
    if (copy_engine_ == nullptr) throw std::runtime_error(error_prefix<Layout>() + "native prefetch needs the copy engine");
    if (page == nullptr) throw std::runtime_error(error_prefix<Layout>() + "native prefetch needs its page");
    prefetch_page_ = page;
    last_prefetch_gen_ = generation_of(load_acquire64(page + kPfReqGen));
    judge_.assign(static_cast<size_t>(layers_), PrefetchJudge{});
  }

  // Serve the device's posted prefetch request, if a new one is there. True when it handled one. Called after
  // pump_demand, so a demand posted meanwhile is always served first. A prefetch never reads NVMe: a row that is not
  // READY in the pinned tier is skipped. A served one leases its pinned slot like a COPYING lane and is handed to the
  // copy thread; only that thread's observed completion publishes COPIED and releases the lease.
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
    counters_[kPrefetchRequests].fetch_add(1);
    const int64_t read_ns = now_ns();
    uint32_t skip = 0;
    CopyJob job;
    if (admission_closed_.load() || copy_engine_ == nullptr || !copy_armed_.load(std::memory_order_acquire) ||
        load_acquire(page_ + kFatal) != 0) {
      skip = kPfSkipUnarmed;
    } else if (row < 0 || row >= layers_ || expert < 0 || expert >= experts_ || !copy_engine_->eligible(row, dst)) {
      skip = kPfSkipInvalid;
    } else {
      std::lock_guard<std::mutex> guard(mutex_);
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
      counters_[skip == kPfSkipUnarmed    ? kPrefetchSkippedUnarmed
                : skip == kPfSkipNotReady ? kPrefetchSkippedNotReady
                                          : kPrefetchSkippedInvalid]
          .fetch_add(1);
      publish_prefetch_done(kPfTagSkipped, gen, skip);
      return true;
    }
    counters_[kPrefetchIssued].fetch_add(1);
    copy_engine_->submit(job);
    return true;
  }

  // Test only: the service's prefetch lease, {active, row, slot}.
  void prefetch_lease(int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    out[0] = prefetch_lease_.active ? 1 : 0;
    out[1] = prefetch_lease_.row;
    out[2] = prefetch_lease_.slot;
  }

  // Every job handed to the copy thread has completed (or failed) and been retired.
  bool wait_copy_idle(int64_t deadline_ns) {
    return copy_engine_ == nullptr || copy_engine_->wait_idle(deadline_ns);
  }

  void copy_engine_ballast(uint64_t dst, uint64_t src, int64_t bytes) {
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
  }

  void inject_done_stall(int64_t ns) {
    done_stall_ns_.store(ns);
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
    std::lock_guard<std::mutex> guard(mutex_);
    if (lease_mode_ && !request.lane_experts.empty() &&
        outstanding_[static_cast<int64_t>((request.seq - 1u) % kDemandRecords)].active) {
      return Defer::kRequestSlot;
    }
    std::vector<int32_t> wanted;
    for (const auto* ids : {&request.protect, &request.need, &request.lane_experts}) {
      for (int32_t expert : *ids) {
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
    if (lease_changes_.load() != deferred_stamp_) return true;
    return lease_mode_ && terminal_seen_for(deferred_seq_, deferred_gen_);
  }

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
    for (uint32_t lane = 0; lane < entry.count; ++lane) held = held || entry.lane[lane].state == 1;
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
      const Request& request, Select select, bool hit_phase = false, const std::vector<int64_t>* loading = nullptr) {
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
          counters_[kCopyFallbacks].fetch_add(1);
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
    lanes_outstanding_.fetch_add(static_cast<int64_t>(taken));
    counters_[kLeasesGranted].fetch_add(static_cast<int64_t>(taken));
    if (hit_phase) counters_[kHitLeasesGranted].fetch_add(static_cast<int64_t>(hits));  // S7: resident lanes
    if (job.count > 0) {
      // After the COPYING words are published: the copy thread may complete and publish CopyDone at once.
      job.gen = request.gen;
      job.idx = idx;
      job.row = request.row;
      job.submit_ns = now_ns();
      counters_[kCopyJobs].fetch_add(1);
      counters_[kCopyLanes].fetch_add(job.count);
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
  void retire_leases() {
    if (lease_ == nullptr || lanes_outstanding_.load(std::memory_order_relaxed) == 0) return;
    std::lock_guard<std::mutex> guard(mutex_);
    for (int64_t idx = 0; idx < kDemandRecords; ++idx) {
      Outstanding& entry = outstanding_[idx];
      if (!entry.active) continue;
      Tier& tier = tiers_[entry.row];
      const uint8_t* acks = lease_ + lease_d_ + kLeaseLaneAck + idx * kLeaseLanes * kLeaseLaneAckBytes;
      const uint8_t* terminal = lease_ + lease_d_ + kLeaseTerminal + idx * kLeaseTerminalBytes;
      const uint64_t stamp = load_acquire64(terminal + kLeaseTermGen);
      const bool terminated = tag_of(stamp) != 0 && generation_of(stamp) == entry.gen;
      uint32_t mask = 0;
      if (terminated) std::memcpy(&mask, terminal + kLeaseTermSkippedMask, 4);
      for (uint32_t lane = 0; lane < entry.count; ++lane) {
        LaneLease& held = entry.lane[lane];
        // No kernel reads a COPYING lane's slot, so no LaneAck or Terminal bit can release it; its copy's completion does.
        if (held.copy_engine) continue;
        const uint64_t word = load_acquire64(acks + lane * kLeaseLaneAckBytes);
        const bool acknowledged = tag_of(word) != 0 && generation_of(word) == entry.gen;
        const bool voided = terminated && (mask >> lane & 1u) != 0;
        if (held.state == 1) {
          if (acknowledged) {
            release_lease_locked(tier, held, kLeasesAcked);
          } else if (voided) {
            release_lease_locked(tier, held, kLeasesVoided);
          }
          // Plan 3.3: the last lease of a quarantined slot frees it. Its mapping was cleared on entry, so
          // release_locked unmaps nothing -- the expert may already live in another slot (M3). A kLoading slot only
          // loses the lease here; serve()'s post-read step decides it.
          if (held.state != 1 && tier.state[held.slot] == kQuarantine && !leased_locked(tier, held.slot)) {
            release_locked(entry.row, held.slot);
          }
        }
        // A second signal for a lane already released: counted once, and it releases nothing.
        if (!held.counted && ((held.state == 2 && voided) || (held.state == 3 && acknowledged))) {
          held.counted = true;
          counters_[kLeaseDoubleSignal].fetch_add(1);
        }
      }
      // S4. A lane whose grant has not run yet is still open. Under V1 the miss lanes sit ungranted between the
      // two grants, so counting only state == 1 would clear `active` while a grant is still pending -- after
      // which grant_lane_group_locked's `entry.active` guard no longer protects this ring slot.
      bool open = entry.grants_pending;
      for (uint32_t lane = 0; lane < entry.count; ++lane) open = open || entry.lane[lane].state == 1;
      if (!open) entry.active = false;
    }
  }

  // Lanes leased by the device and not yet retired. An eager pause is refused while this is non-zero; a lease held by
  // anything else (a promotion, in Task 8) is not counted, because it protects its own slot (LEASE_PROTOCOL.md 17.1 R2).
  int64_t graph_leases_outstanding() const {
    return lanes_outstanding_.load();
  }

  // One lease released, exactly once: the per-lane state machine is what makes a second signal harmless.
  void release_lease_locked(Tier& tier, LaneLease& held, int counter) {
    if (tier.leases[held.slot] == 0) {
      throw std::runtime_error(error_prefix<Layout>() + "lease underflow on slot " + std::to_string(held.slot));
    }
    tier.leases[held.slot] -= 1;
    held.state = counter == kLeasesVoided ? 3 : 2;
    lanes_outstanding_.fetch_sub(1);
    counters_[counter].fetch_add(1);
    lease_changes_.fetch_add(1);
  }

  // Test hooks and introspection. slot_info: [state, expert, leases, generation] per slot.
  void slot_info(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot) {
      out[4 * slot] = tier.state[slot];
      out[4 * slot + 1] = tier.slot_to_expert[slot];
      out[4 * slot + 2] = tier.leases[slot];
      out[4 * slot + 3] = tier.generation[slot];
    }
  }

  // Test only: the service's account of request slot `idx`: [active, grants_pending, count, gen, then per lane
  // (kLeaseLanes) its state, then per lane its slot, then per lane 1 if it is a copy-engine lane].
  void lease_entry(int64_t idx, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Outstanding& entry = outstanding_[idx];
    out[0] = entry.active;
    out[1] = entry.grants_pending;
    out[2] = entry.count;
    out[3] = static_cast<int64_t>(entry.gen);
    for (int64_t lane = 0; lane < kLeaseLanes; ++lane) {
      out[4 + lane] = entry.lane[lane].state;
      out[4 + kLeaseLanes + lane] = entry.lane[lane].slot;
      out[4 + 2 * kLeaseLanes + lane] = entry.lane[lane].copy_engine ? 1 : 0;
    }
  }

  // Test only: the service grants leases itself from step 3; until then a test stands in for the device's holder.
  void inject_lease(int64_t row, int64_t slot, int64_t delta) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    if (delta < 0 && tier.leases[slot] < static_cast<uint32_t>(-delta)) {
      throw std::runtime_error(error_prefix<Layout>() + "lease underflow on slot " + std::to_string(slot));
    }
    tier.leases[slot] = static_cast<uint32_t>(static_cast<int64_t>(tier.leases[slot]) + delta);
    if (delta < 0) lease_changes_.fetch_add(1);
  }

  VictimCensus victim_census(int64_t row, const std::vector<int32_t>& wanted) {
    std::lock_guard<std::mutex> guard(mutex_);
    return census_locked(row, wanted);
  }

  void mapping(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t expert = 0; expert < experts_; ++expert) {
      const int32_t slot = tier.expert_slot[expert];
      out[expert] = slot >= 0 && tier.state[slot] == kReady ? slot : -1;
    }
  }

  void slot_to_expert(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
    const Tier& tier = tiers_[row];
    for (int64_t slot = 0; slot < tier.capacity; ++slot)
      out[slot] = tier.slot_to_expert[slot];
  }

  int64_t lru_order(int64_t row, int64_t* out) {
    std::lock_guard<std::mutex> guard(mutex_);
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

  void set_hot(int64_t row, const int64_t* experts, int64_t count) {
    std::lock_guard<std::mutex> guard(mutex_);
    Tier& tier = tiers_[row];
    std::fill(tier.hot.begin(), tier.hot.end(), 0);
    for (int64_t i = 0; i < count; ++i) {
      if (experts[i] >= 0 && experts[i] < experts_) {
        tier.hot[experts[i]] = 1;
        // A VRAM-hot row is decode's: kept owned it could never be a victim and would hold the share down.
        if (tier.expert_slot[experts[i]] >= 0) disown_locked(tier, tier.expert_slot[experts[i]]);
      }
    }
  }

  void layer_rows(int64_t* out, bool advisory) {
    std::lock_guard<std::mutex> guard(mutex_);
    for (int64_t row = 0; row < layers_; ++row)
      out[row] = advisory ? tiers_[row].rows_advisory : tiers_[row].rows_demand;
  }

  // Test-only faults: sleep `delay_ns` before each advisory read and before each demand
  // read once `after_demands` demands have read rows; report reads as failed; make an advisory
  // give up once `abandon_after_batches` of its batches (rows) were admitted (0: never).
  void inject(int64_t delay_ns, bool fail_reads, int64_t after_demands, int64_t abandon_after_batches) {
    delay_ns_.store(delay_ns);
    fail_reads_.store(fail_reads);
    delay_after_.store(after_demands);
    abandon_after_.store(abandon_after_batches);
  }

  // Test only: carry a whole ReadFault down to this tier's reader, where inject() reaches it only as a
  // delay, a blanket failure or an abandon point. `words` is the reader tests' fault tensor (kFaultWords
  // int64; see fault_from). Unlike fail_reads the fault does NOT short-circuit ahead of the reader: the read
  // runs, so the fault's part errors, pack delay and the rest act on rows that have already packed. The
  // service thread applies it just before its next read (the reader is that thread's alone), and it then
  // stays until replaced; an all-default tensor clears it. Words 17-20 (abandon_after, step, pack_workers,
  // pack_split) and 22 (piece_stream) are not faults and are ignored: use inject() for the abandon point, the
  // tier's own constructor for the packing pool and set_piece_stream() for the mode. The reader's counters (submit and completion calls) run over the
  // reader's whole life, so a call-numbered fault (submit_call, cqe_call) is relative to a fresh tier.
  void inject_fault(const int64_t* words) {
    std::lock_guard<std::mutex> guard(fault_mutex_);
    pending_fault_ = fault_from(words);
    fault_pending_.store(true, std::memory_order_release);
  }

  void counters(int64_t* out) const {
    for (int i = 0; i < kCounterCount; ++i)
      out[i] = counters_[i].load();
  }

 private:
  // Copy thread. Completion was observed, so no copy of this job reads its slots any more: publish CopyDone, then
  // release. A leased slot's generation cannot move (E1); if one did, the bytes are not the lease's, so fail stop.
  // A job whose row has SM entries is not released here (false): the copy wait still reads those slots, and
  // copy_acked releases them once it acknowledged. True otherwise, fail-stops included (the leases stay held, E5).
  bool copy_completed(const CopyJob& job) {
    if (job.prefetch) {
      prefetch_completed(job);
      return true;
    }
    const uint32_t* generations = slot_gen_ + slot_gen_base_[job.row];
    for (int i = 0; i < job.count; ++i) {
      const CopyLane& lane = job.lanes[i];
      if (load_acquire(reinterpret_cast<const uint8_t*>(generations + lane.host_slot)) != lane.slot_generation) {
        counters_[kCopyGenerationMismatches].fetch_add(1);
        raise_fatal(static_cast<uint32_t>(job.gen));
        return true;
      }
    }
    uint8_t* done = lease_ + lease_c_ + job.idx * kLeaseCopyDoneBytes;
    std::memcpy(done + kLeaseCdMask, &job.mask, 4);
    store_release64(done + kLeaseCdGen, tagged_word(kLeaseTagCopied, job.gen));
    if (job.sm) return false;
    release_copied(job);
    return true;
  }

  // Copy thread, a job copy_completed left waiting: true once the copy wait's SmAck for its ring index carries this
  // request's generation or a later one (the copy wait of a later request in the index ran, so this one's finished),
  // after which no SM read of the job's slots can be in flight, and the leases are released.
  bool copy_acked(const CopyJob& job) {
    const uint64_t word = load_acquire64(lease_ + lease_d_ + kLeaseSmAck + job.idx * kLeaseSmAckBytes);
    if (tag_of(word) != kLeaseTagSmAck || generation_of(word) < job.gen) return false;
    release_copied(job);
    return true;
  }

  // Copy thread: release a completed job's COPYING leases, the only place one is released.
  void release_copied(const CopyJob& job) {
    std::lock_guard<std::mutex> guard(mutex_);
    Outstanding& entry = outstanding_[job.idx];
    if (!entry.active || entry.gen != job.gen) {
      // Nothing but this completion releases a COPYING lane, so its entry cannot have closed: an internal error.
      counters_[kCopyErrors].fetch_add(1);
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    Tier& tier = tiers_[entry.row];
    for (int i = 0; i < job.count; ++i) {
      LaneLease& held = entry.lane[job.lanes[i].lane];
      if (held.state == 1 && held.copy_engine) release_lease_locked(tier, held, kLeasesCopied);
    }
    bool open = entry.grants_pending;
    for (uint32_t lane = 0; lane < entry.count; ++lane) open = open || entry.lane[lane].state == 1;
    if (!open) entry.active = false;
  }

  // Copy thread, a prefetch job: the same E6 check as a COPYING lane, then the judge entry for the target's next
  // request and COPIED, then the lease release (all after the observed completion). A mismatch fails stop and
  // publishes nothing, so the device's wait for this request ends only on the fatal word.
  void prefetch_completed(const CopyJob& job) {
    const CopyLane& lane = job.lanes[0];
    const uint32_t* generations = slot_gen_ + slot_gen_base_[job.row];
    if (load_acquire(reinterpret_cast<const uint8_t*>(generations + lane.host_slot)) != lane.slot_generation) {
      counters_[kCopyGenerationMismatches].fetch_add(1);
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    if (!prefetch_lease_.active || prefetch_lease_.gen != job.gen) {
      counters_[kCopyErrors].fetch_add(1);
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    Tier& tier = tiers_[prefetch_lease_.row];
    if (tier.leases[prefetch_lease_.slot] == 0) {
      counters_[kCopyErrors].fetch_add(1);
      raise_fatal(static_cast<uint32_t>(job.gen));
      return;
    }
    judge_[job.row] = PrefetchJudge{true, prefetch_lease_.expert};
    counters_[kPrefetchCopied].fetch_add(1);
    counters_[kPrefetchLatencyNs].fetch_add(now_ns() - job.submit_ns);
    publish_prefetch_done(kPfTagCopied, job.gen, 0);
    tier.leases[prefetch_lease_.slot] -= 1;
    lease_changes_.fetch_add(1);  // a demand deferred on this slot may retry
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
    std::lock_guard<std::mutex> guard(mutex_);
    PrefetchJudge& judge = judge_[request.row];
    if (!judge.pending) return;
    judge.pending = false;
    const bool used = listed(request.protect, judge.expert) || listed(request.need, judge.expert);
    counters_[used ? kPrefetchUsed : kPrefetchWasted].fetch_add(1);
  }

  // Copy thread. Completion cannot be established: the leases stay held (E5) and the page fails stop.
  void copy_failed(const CopyJob& job, int error) {
    std::fprintf(stderr, "ERROR %s copy of request %llu failed (%d); leases held\n",
                 (std::string(Layout::kName) + " RAM miss copy engine:").c_str(),
                 static_cast<unsigned long long>(job.gen), error);
    std::fflush(stderr);
    raise_fatal(static_cast<uint32_t>(job.gen));
  }

  void raise_fatal(uint32_t seq) {
    uint32_t zero = 0;
    __atomic_compare_exchange_n(
        reinterpret_cast<uint32_t*>(page_ + kFatal), &zero, seq == 0 ? 0xFFFFFFFFu : seq, false, __ATOMIC_RELEASE,
        __ATOMIC_RELAXED);
  }

  // Called the moment a posted record is found. With the trace off this is one relaxed load and
  // no clock read; requests are served one at a time, so one member record serves them all.
  void begin_stage(int64_t kind, uint32_t seq, uint32_t backlog) {
    if (!trace_on_.load(std::memory_order_relaxed)) {
      cur_ = nullptr;
      return;
    }
    const int64_t observed = stamp(&stage_);  // before the reset below: the record is found, not built
    stage_ = StageRecord{};
    stage_.observed = observed;
    stage_.kind = kind;
    stage_.seq = seq;
    stage_.backlog = backlog;
    stage_.prev_done = last_done_;
    stage_.pack_workers = reader_.pack_workers();
    stage_.pack_split = reader_.pack_split();
    stage_.piece_stream = reader_.piece_stream() ? 1 : 0;
    cur_ = &stage_;
  }

  // Service thread, before a read: install the fault inject_fault() left, on the reader only this thread drives.
  void apply_pending_fault() {
    if (!fault_pending_.load(std::memory_order_acquire)) return;
    ReadFault fault;
    {
      std::lock_guard<std::mutex> guard(fault_mutex_);
      fault = pending_fault_;
      fault_pending_.store(false, std::memory_order_relaxed);
    }
    reader_.set_fault(fault);
  }

  void end_stage() {
    if (cur_ == nullptr) return;
    cur_->done = stamp(cur_);
    last_done_ = cur_->done;
    ring_->push(*cur_);
    cur_ = nullptr;
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
    const int64_t needed = round_up_page(copy_offset + kLeaseAreaCopyDoneBytes);
    if (lease_bytes < needed) {
      throw std::runtime_error(
          error_prefix<Layout>() + "the lease block has " + std::to_string(lease_bytes) + " bytes, its layout needs " +
          std::to_string(needed));
    }
    auto put_u32 = [&](int64_t offset, uint32_t value) { std::memcpy(lease_ + offset, &value, 4); };
    put_u32(0, 0x4C534531u);  // "LSE1"
    put_u32(4, 3u);           // ABI version (exl3_lease_block.ABI_VERSION; 3: 128-byte LaneRequest, area C)
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
  VictimCensus census_locked(int64_t row, const std::vector<int32_t>& wanted) const {
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
    busy_since_.store(now_ns());
    apply_pending_fault();  // test only: inject_fault() acts on a fill's read as on a demand's
    std::vector<uint8_t> packed;
    size_t landed = 0;
    auto advance = [&] {
      while (landed < packed.size() && packed[landed] != 0) ++landed;
      fill_landed_.store(static_cast<int64_t>(landed), std::memory_order_release);
    };
    int result = 0;
    try {
      // A demand's batch size and no publish target: a fill has no device readiness words.
      result = reader_.read(
          fill_row_, fill_experts_, fill_slots_, kBounceRows, [](size_t) { return false; }, nullptr, &packed, SIZE_MAX,
          advance, nullptr);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "ERROR %sprefill fill: %s\n", error_prefix<Layout>().c_str(), error.what());
      result = 0;
    }
    _mm_sfence();  // the rows' bytes land before the caller is told (fill_landed_)
    {
      std::lock_guard<std::mutex> guard(mutex_);
      Tier& tier = tiers_[fill_row_];
      for (size_t i = 0; i < fill_slots_.size(); ++i) {
        const int64_t slot = fill_slots_[i];
        tier.filling[slot] = 0;
        if (result != 1 && !(i < packed.size() && packed[i] != 0)) release_locked(fill_row_, slot);
      }
      if (result != 1) {
        counters_[kReadErrors].fetch_add(1);
        counters_[kVersion].fetch_add(1);
      }
    }
    if (result == 1) {
      fill_landed_.store(static_cast<int64_t>(fill_slots_.size()), std::memory_order_release);
    } else {
      advance();
    }
    busy_since_.store(0);
    fill_state_.store(result == 1 ? kFillOk : kFillFailed, std::memory_order_release);
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
      int64_t row, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted, bool stop_at_share = false) {
    Tier& tier = tiers_[row];
    int64_t slot = -1;
    if (prefill_share_ > 0 && tier.owned >= prefill_share_ &&
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
        counters_[kEvictions].fetch_add(1);
        slot = best;
      } else if (stop_at_share) {
        *evicted = -1;
        return -1;
      }
    }
    if (slot < 0) slot = take_slot_locked(row, protect, fallback, evicted);
    if (slot >= 0 && prefill_share_ > 0) {
      tier.prefill_owned[slot] = 1;
      ++tier.owned;
    }
    return slot;
  }

  // Takes only a kFree slot or evicts an unleased kReady one: kLoading and kQuarantine slots are never taken.
  int64_t take_slot_locked(int64_t row, const std::vector<int32_t>& protect, bool fallback, int64_t* evicted) {
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
      counters_[kNoVictim].fetch_add(1);
      return -1;
    }
    const int32_t victim = tier.slot_to_expert[best];
    publish_map(row, victim, -1);  // unmapped before its bytes are overwritten (D11)
    tier.expert_slot[victim] = -1;
    tier.slot_to_expert[best] = -1;
    tier.state[best] = kFree;
    disown_locked(tier, best);
    *evicted = victim;
    counters_[kEvictions].fetch_add(1);
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
    counters_[kSlotsQuarantined].fetch_add(1);
  }

  bool demand_pending() const {
    return !reached(next_demand_ - 1u, load_acquire(page_ + kDemandHead));
  }

  // An unarmed demand record: nobody waits on it, so the device may already be gathering
  // any mapped slot (the next token's rows too, once the thread lags). Only refresh the
  // recency of its assigned rows: no eviction, no read. False for an invalid record.
  bool touch_request(const Request& request) {
    if (request.row < 0 || request.row >= layers_) return false;
    std::lock_guard<std::mutex> guard(mutex_);
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
    if (cur_) cur_->lanes = request.lanes;
    std::vector<int32_t> wanted;
    for (const auto* ids : {&request.protect, &request.need}) {
      for (int32_t expert : *ids) {
        if (!listed(wanted, expert)) wanted.push_back(expert);  // one slot per expert (device bytes may repeat)
      }
    }
    // A lane's own expert must never be a victim of the request that leases it, whatever the post kernel protected.
    for (int32_t expert : request.lane_experts) {
      if (!listed(wanted, expert)) wanted.push_back(expert);
    }
    std::vector<int32_t> missing;
    std::vector<int64_t> slots;
    bool ok = request.row >= 0 && request.row < layers_;
    // Piece streaming publishes into the lease block's readiness words, initialised in the reservation hold below
    // before the two-phase hit grant. Without two-phase and lease mode there is neither, so refuse before any slot
    // is taken (the service refuses the flag too; this is the tier's own guard).
    const bool piece_stream = reader_.piece_stream();
    if (ok && piece_stream && !(two_phase_ && lease_mode_)) {
      counters_[kPieceStreamRefused].fetch_add(1);
      ok = false;
    }
    bool publishing = false;  // piece streaming: this request's miss lanes have readiness words to publish into
    if (ok) {
      std::lock_guard<std::mutex> guard(mutex_);
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
            counters_[kDeferred].fetch_add(1);
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
      // S2. Grant and publish the HIT lanes here: in the same mutex_ hold as the reservation, after the take loop
      // has completed with ok still true, and before the hold is dropped for read(). A lane is a hit iff its
      // expert is not in `missing`. This is the whole of V1: these row results become visible to the device while
      // the missing rows are still being read, so their copies overlap the read instead of following it.
      //
      // Neither ordering below it is available. Before the take loop, the !ok bail here returns with no lease
      // unwind, and the deferral branch above sets ok = false for a request that is retried under the same seq --
      // either way leases outlive a request that never ran. After the hold is dropped, the hit slot can be
      // evicted in exactly the window this task exists to close, and the grant buys nothing.
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
          if (!(piece_stream ? grant_lane_group_locked(request, [](size_t) { return true; }, true, &slots)
                             : grant_lane_group_locked(
                                   request,
                                   [&](size_t lane) { return !listed(missing, request.lane_experts[lane]); },
                                   true))) {
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
        if (!slots.empty()) counters_[kVersion].fetch_add(1);
        slots.clear();
      }
    }
    if (cur_) cur_->reserved = stamp(cur_);
    int64_t status = kStatusNoRead;
    // Per slot: the row was packed whole (read() sets it). A member, not a local, so that the buffer
    // the pipeline writes into is allocated once rather than per served demand on the service thread
    // - read()'s assign() below only grows it, and it never shrinks.
    std::vector<uint8_t>& packed = packed_;
    // Cleared, not merely reused: the publish gate below reads packed[i] whenever the vector is long
    // enough, and read() only rewrites it when it actually runs. Carrying the PREVIOUS request's flags
    // into a request that never read would publish a row on the strength of an older row's packing.
    packed.clear();
    bool cancelled = false;
    if (ok && !missing.empty()) {
      apply_pending_fault();
      const int64_t delay = delay_ns_.load();
      if (delay > 0 && (advisory || demands_read_ >= delay_after_.load())) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
      }
      if (fail_reads_.load()) {
        counters_[kReadErrors].fetch_add(1);
        ok = false;
        status = kStatusFailed;
      } else {
        const int64_t abandon_after = abandon_after_.load();
        const int result = reader_.read(
            request.row,
            missing,
            slots,
            advisory ? 1 : kBounceRows,
            [&](size_t admitted) {
              return advisory && (demand_pending() || pause_requested_.load() || stop_requested_.load() ||
                                  (abandon_after > 0 && admitted >= static_cast<size_t>(abandon_after)));
            },
            cur_,
            &packed,
            advisory ? 1 : SIZE_MAX,
            // Safety, not just style: read() runs here with mutex_ NOT held -- the only lock_guard in
            // serve() near it is the scoped S2 reservation hold above, which closes before this call.
            // retire_leases() takes mutex_ itself, so this is deadlock-free today. That is a property
            // of the code as it stands, not a guarantee: if a future yield point is ever added to
            // read()'s drain loop under a lock, this callback would deadlock against it.
            [this] { retire_leases(); },
            publishing ? &piece_publish_ : nullptr);
        if (piece_stream) counters_[kPiecePublishRefused].store(reader_.publish_refused());
        if (result == 0) counters_[kReadErrors].fetch_add(1);
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
      std::lock_guard<std::mutex> guard(mutex_);
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
        (advisory ? tier.rows_advisory : tier.rows_demand) += published;
        counters_[kVersion].fetch_add(1);
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
      std::lock_guard<std::mutex> guard(mutex_);
      if (two_phase_) {
        if (ok && !piece_stream) {
          if (!grant_lane_group_locked(request, [&](size_t lane) { return listed(missing, request.lane_experts[lane]); }))
            ok = false;
        }
        close_pending_grants_locked(request);  // the second grant is owed no longer, however this request answered
      } else if (ok) {
        if (!grant_lanes_locked(request)) ok = false;
      }
    }
    if (cur_) {
      cur_->mapped = stamp(cur_);
      cur_->row = request.row;
      cur_->ok = ok ? 1 : 0;
      cur_->status = ok || status != kStatusNoRead ? status : kStatusFailed;
      cur_->rows = published;
    }
    if (published > 0) {
      counters_[kRowsRead].fetch_add(published);
      if (advisory) counters_[kAdvisoryRows].fetch_add(published);
    }
    if (ok) *rows = published;
    return ok;
  }

  // Store piece_word(gen) into the readiness word of every lane whose expert is read (in `missing`), then fence,
  // and record those words as the owner's publish targets, by the row's ordinal in the read. Service thread only:
  // it is the only writer of area P. False when no lane names a missing row (nothing to publish).
  bool init_piece_words_locked(const Request& request, const std::vector<int32_t>& missing) {
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
        request.gen, piece_targets_.data(),
        reinterpret_cast<const uint64_t*>(lease_ + lease_d_ + kLeaseStreamProbe + idx * kLeaseStreamProbeBytes)};
    return any;
  }

  void handle_demand(const Request& request, uint8_t* record) {
    busy_since_.store(now_ns());
    store_release(page_ + kBusySeq, request.seq);
    if (load_acquire(page_ + kFatal) != 0) counters_[kLateAfterFatal].fetch_add(1);
    int64_t rows = 0;
    const bool ok = request.armed ? serve(request, false, &rows) : touch_request(request);
    if (cur_ && !request.armed) {
      cur_->kind = kStageTouch;
      cur_->lanes = request.lanes;
      cur_->row = request.row;
      cur_->ok = ok ? 1 : 0;
      cur_->status = ok ? kStatusTouch : kStatusFailed;
    }
    // Classified by what was read: an empty need whose protect ids had to be read is D12's race.
    if (ok) counters_[rows == 0 ? kTouchOnly : kServedRequests].fetch_add(1);
    _mm_sfence();
    set_status(record, ok ? kServed : kFailed);
    store_release(page_ + kBusySeq, 0);
    busy_since_.store(0);
  }

  uint8_t* page_;
  int32_t* map_;
  uint8_t* lease_;             // the lease block, or null when the service runs without one
  uint8_t* hot_page_ = nullptr;
  int64_t hot_stride_ = 0;
  std::atomic<bool> gpu_hot_mode_{false};
  uint32_t* slot_gen_ = nullptr;  // SlotGen[] inside it
  std::vector<int64_t> slot_gen_base_;  // first SlotGen word of each row
  int64_t lease_d_ = 0;                 // byte offset of area D (the device-written words)
  int64_t lease_p_ = 0;                 // byte offset of area P (the piece readiness words)
  int64_t lease_c_ = 0;                 // byte offset of area C (CopyDone)
  bool piece_stream_ = false;           // set before the service thread starts, with the reader's flag
  // The copy engine, when enabled (before the service thread starts); armed separately, and only then used.
  std::unique_ptr<CopyEngine> copy_engine_;
  std::atomic<bool> copy_armed_{false};
  // Native prefetch: the page (null when off), the last request generation read (service thread), the one lease a
  // prefetch holds (at most one is outstanding: the device waits for its done word before posting the next), and per
  // row the copied expert its next request judges. The last two are guarded by mutex_.
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
  bool lease_mode_ = false;             // set before the service thread starts; off is today's protocol
  bool two_phase_ = false;              // Task 6 V1: hit lanes granted before read(); off is the Task 5 batched grant
  Outstanding outstanding_[kDemandRecords];  // by request slot; guarded by mutex_
  std::atomic<int64_t> lanes_outstanding_{0};  // lanes GRANTED and not yet retired: an early-out for retire_leases
  std::atomic<bool> admission_closed_{false};  // shutdown: serve nothing new; retirement continues
  std::atomic<int64_t> done_stall_ns_{0};      // test only: sleep between serving a demand and storing demand_done
  std::atomic<uint64_t> lease_changes_{0};     // bumped whenever a lease is released: what wakes a deferred demand
  // The demand held back, if any (service thread only): its sequence, the changes seen when it was last refused,
  // its generation (a terminal for it also wakes it) and when it was first observed (for the stage record).
  uint32_t deferred_seq_ = 0;
  uint64_t deferred_stamp_ = 0;
  uint64_t deferred_gen_ = 0;
  int64_t deferred_observed_ns_ = 0;
  int64_t layers_;
  int64_t experts_;
  Source reader_;
  std::vector<uint8_t> packed_;  // serve()'s per-row packed flags, sized by read(), reused every request
  // Prefill fills: written by fill_begin before the thread starts and read by it; the caller reads only the atomics.
  static constexpr int kFillOk = 0;
  static constexpr int kFillRunning = 1;
  static constexpr int kFillFailed = 2;
  std::thread fill_thread_;
  int64_t fill_row_ = 0;
  std::vector<int32_t> fill_experts_;
  std::vector<int64_t> fill_slots_;
  std::atomic<int64_t> fill_landed_{0};
  std::atomic<int> fill_state_{kFillOk};
  std::vector<Tier> tiers_;
  std::mutex mutex_;
  uint64_t tick_ = 0;
  int64_t prefill_share_ = 0;  // under mutex_; see set_prefill_share
  uint32_t next_demand_ = 1;
  uint32_t next_advice_ = 1;
  int64_t demands_read_ = 0;
  std::atomic<bool> in_advice_{false};
  std::atomic<bool> pause_requested_{false};
  std::atomic<bool> stop_requested_{false};
  std::atomic<bool> threaded_{false};
  std::atomic<uint32_t> skip_advice_upto_{0};
  std::atomic<int64_t> busy_since_{0};
  std::atomic<int64_t> delay_ns_{0};
  std::atomic<int64_t> delay_after_{0};
  std::atomic<int64_t> abandon_after_{0};
  std::atomic<bool> fail_reads_{false};
  std::mutex fault_mutex_;  // guards pending_fault_ between inject_fault() and the service thread
  ReadFault pending_fault_{};
  std::atomic<bool> fault_pending_{false};
  std::atomic<int64_t> counters_[kCounterCount];
  // Stage trace. cur_ points at stage_ while a traced request is in service, else null.
  std::atomic<bool> trace_on_{false};
  std::mutex trace_mutex_;  // guards ring_ against a drain racing enable_trace
  std::unique_ptr<StageRing> ring_;
  StageRecord stage_{};
  StageRecord* cur_ = nullptr;
  int64_t last_done_ = 0;
};


}  // namespace expert_stream
}  // namespace sglang
