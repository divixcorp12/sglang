// The DSpark draft channel as a CPU expert engine's second job source (cpu_experts.h): the source the host attaches to
// a running engine, the engine thread's poll and preparation of the channel's records, their completion, and the
// draft's counters.
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The second client: the DSpark draft".
#pragma once

#include "../draft_channel.h"
#include "build_policy.h"
#include "cpu_experts/kernel.hpp"
#include "cpu_experts/team.hpp"
#include "job_trace.h"
#include "lease_channel.h"
#include "reader_base.h"
#include "tier_protocol.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
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

/// The DSpark draft channel (draft_channel.h) as a CpuExpertEngine's second job source (plan 2026-10-06 Task 11): each
/// posted record is a stage's M-row forward over the stage's pinned areas, completed through the lease channel (done,
/// then the Dekker open of the gate). The stages' layers read the draft's own slabs (pageable host memory).
template <class Layer>
struct DraftSourceOf {
  uint8_t* channel = nullptr;      // draft::kChannelBytes, pinned: the page and the completion block
  const uint8_t* x = nullptr;      // fp16 [stages, kMaxRows, hidden]
  const int32_t* slots = nullptr;  // [stages, kMaxRows, kMaxK]
  const float* weights = nullptr;  // [stages, kMaxRows, kMaxK]
  float* out = nullptr;            // [stages, kMaxRows, hidden]
  int64_t hidden = 0;
  std::vector<Layer> layers;  // one per stage, each made by the format's kernel
  std::vector<char> made;     // per stage: its layer is set
  bool test_hooks = false;                       // the instr build's: honour draft::g_test_poll_pause_us
};

/// A CPU expert engine's draft source and its read of the channel. Two threads use it: the host's (attach, detach,
/// open_gate, stats, unserved_head) and the engine's (source, pending, prepare, leave; finish runs on whichever worker
/// finishes the record's job), which meet on one word, `source_`: null (none, or detached), the attached source, or the
/// detach sentinel while a detach waits for the engine thread to let go of the source. The engine thread reads the word
/// once per idle quantum and keeps a source across one only while a record is in the window, so a detach returns within
/// a quantum plus the job in progress.
template <BuildPolicy Build>
class DraftExperts {
 public:
  using Kernel = typename Build::CpuKernel;
  using Layer = typename Kernel::Layer;
  using DraftSource = DraftSourceOf<Layer>;
  static constexpr const char* kPrefix = "DSpark draft CPU experts: ";

  struct Stats {
    int64_t jobs, rows, forward_ns, collided_jobs, shared_routes, collided_forward_ns;
  };

  /// Every draft job runs in the engine's window. `trace` and `trigger` are the engine's, written by the engine thread
  /// and, in finish(), by the one thread that finishes the record in flight.
  DraftExperts(JobTrace<Build::kMetrics>& trace, DraftDelayTrigger<kDraftDelayTriggerOnly && !Build::kMetrics>& trigger)
      : trace_(trace), trigger_(trigger) {}
  DraftExperts(const DraftExperts&) = delete;
  DraftExperts& operator=(const DraftExperts&) = delete;

  /// Host: checks the source against the team and publishes it; the engine thread serves its records from its next
  /// quantum on. Once.
  void attach(std::unique_ptr<DraftSource> source) {
    if (owned_ != nullptr) throw std::runtime_error(std::string(kPrefix) + "a draft source is attached already");
    const DraftSource& d = *source;
    if (!d.channel || !d.x || !d.slots || !d.weights || !d.out || d.hidden <= 0 || d.layers.empty())
      throw std::runtime_error(std::string(kPrefix) + "the channel, the stage areas and a layer per stage are required");
    for (size_t s = 0; s < d.layers.size(); ++s) {
      if (d.layers[s].hidden != d.hidden)
        throw std::runtime_error(std::string(kPrefix) + "stage " + std::to_string(s) + "'s layer has hidden " +
                                 std::to_string(d.layers[s].hidden) + ", the areas " + std::to_string(d.hidden));
    }
    if (Kernel::max_routes() < draft::kMaxK || Kernel::max_rows() < draft::kMaxRows)
      throw std::runtime_error(std::string(kPrefix) + "kernel " + Kernel::name() +
                               " takes fewer rows or routes than a draft call");
    const uint32_t head = channel::head<DraftChannel>(d.channel);
    completed_.store(head, std::memory_order_relaxed);
    next_ = channel::skip_zero(head + 1u);  // the engine thread reads it after acquiring source_
    owned_ = std::move(source);
    source_.store(owned_.get(), std::memory_order_release);
  }

  /// Host: stops the engine thread serving the channel (it finishes a job in progress and lets go of the source, or
  /// has already, on its exit), then opens the gate a wait still holds closed (no completer is left). Idempotent.
  void detach() {
    if (owned_ == nullptr) return;
    DraftSource* d = source_.load(std::memory_order_acquire);
    while (d != nullptr && d != detaching() && !source_.compare_exchange_weak(d, detaching(), std::memory_order_seq_cst)) {
    }
    while (source_.load(std::memory_order_acquire) != nullptr) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    channel::open_closed_gate<DraftChannel>(owned_->channel);
  }

  /// Host, once the engine thread has exited: opens the gate a wait still holds closed.
  void open_gate() {
    if (owned_ != nullptr) channel::open_closed_gate<DraftChannel>(owned_->channel);
  }

  Stats stats() const {
    return {jobs_.load(std::memory_order_relaxed),          rows_.load(std::memory_order_relaxed),
            forward_ns_.load(std::memory_order_relaxed),    collided_jobs_.load(std::memory_order_relaxed),
            shared_routes_.load(std::memory_order_relaxed), collided_forward_ns_.load(std::memory_order_relaxed)};
  }

  /// Any thread: the posted record the engine thread has not completed, 0 when there is none (the watchdog's).
  uint32_t unserved_head() const {
    const DraftSource* d = source_.load(std::memory_order_acquire);
    if (d == nullptr || d == detaching()) return 0;
    const uint32_t head = channel::head<DraftChannel>(d->channel);
    return head == completed_.load(std::memory_order_acquire) ? 0 : head;
  }

  /// Engine thread, each quantum: the attached source, null when there is none or a detach is waiting (it lets go of
  /// the source here, which is what the detach waits for). The instr build's poll pause (test_hooks) follows.
  DraftSource* source() {
    DraftSource* d = source_.load(std::memory_order_acquire);
    if (d == detaching()) {
      // A record in the window finishes first: its completion still writes the channel the detach then closes.
      if (in_flight_.load(std::memory_order_acquire)) return nullptr;
      source_.store(nullptr, std::memory_order_release);
      return nullptr;
    }
    if (d != nullptr && d->test_hooks)
      if (const int64_t pause = draft::g_test_poll_pause_us.load(std::memory_order_relaxed); pause > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(pause));
    return d;
  }

  /// Engine thread: the record posted and not yet served, 0 when there is none or one is in the window (its completion
  /// moves next_, and the device posts one record per wait). A record past the one expected fail-stops.
  uint32_t pending(const DraftSource& d) {
    if (in_flight_.load(std::memory_order_acquire)) return 0;
    const uint32_t head = channel::head<DraftChannel>(d.channel);
    if (head == 0 || !channel::reached(head, next_)) return 0;
    if constexpr (Build::kMetrics) trace_.draft_observe(&d, head);
    if (head != next_)
      fail_stop(std::string(kPrefix) + "record " + std::to_string(next_) + " lapped (head " + std::to_string(head) +
                "); the device posts one record per wait");
    return next_;
  }

  /// One record's call as prepare() builds it, in storage the window's job reads while it runs: the record's slots and
  /// weights compacted to [rows, k], and what finish() needs.
  struct Prepared {
    cpu_experts::ForwardCall call;
    const Layer* layer = nullptr;
    const DraftSource* source = nullptr;
    uint32_t seq = 0, epoch = 0;
    int stage = 0, rows = 0, k = 0, shared = 0;
    int32_t slots[draft::kMaxRows * draft::kMaxK];
    float weights[draft::kMaxRows * draft::kMaxK];
  };

  /// Engine thread: reads record `seq` (pending()'s) into `p`, the call a window admits; the record is in flight until
  /// finish(). A torn or malformed record fail-stops.
  void prepare(const DraftSource& d, uint32_t seq, Prepared& p) {
    int64_t selected = 0;
    if constexpr (Build::kMetrics) selected = now_ns();
    alignas(64) uint8_t raw[DraftChannel::kRecordBytes];
    const uint8_t* rec = channel::record_at<DraftChannel>(d.channel, seq);
    if (!channel::read_seqlocked<DraftChannel>(rec, seq, raw))
      fail_stop(std::string(kPrefix) + "record " + std::to_string(seq) + " torn");
    uint32_t word, epoch;
    std::memcpy(&word, raw + draft::kRecStage, 4);
    std::memcpy(&epoch, raw + draft::kRecEpoch, 4);
    const int stage = static_cast<int>(word & 0xFFFFu), rows = static_cast<int>((word >> 16) & 0xFFu),
              k = static_cast<int>(word >> 24);
    if (stage >= static_cast<int>(d.layers.size()) || rows < 1 || rows > draft::kMaxRows || k < 1 || k > draft::kMaxK)
      fail_stop(std::string(kPrefix) + "record " + std::to_string(seq) + " malformed (stage " + std::to_string(stage) +
                ", rows " + std::to_string(rows) + ", k " + std::to_string(k) + ")");
    if constexpr (Build::kMetrics) {
      uint64_t gpu_ns = 0;
      std::memcpy(&gpu_ns, raw + draft::kRecPublishNs, sizeof(gpu_ns));
      trace_.draft_selected(selected, stage, epoch, seq, gpu_ns);
      int64_t offset_high = 0;
      std::memcpy(&offset_high, d.channel + draft::kClockOffsetHigh, sizeof(offset_high));
      trace_.draft_arrival(stage, epoch, seq, gpu_ns, offset_high);
    }
    // The slot and weight areas are kMaxK wide per token; the kernel reads [rows, k] contiguous, so compact them.
    int32_t* const slots = p.slots;
    float* const weights = p.weights;
    const int32_t* s = d.slots + static_cast<int64_t>(stage) * draft::kMaxRows * draft::kMaxK;
    const float* w = d.weights + static_cast<int64_t>(stage) * draft::kMaxRows * draft::kMaxK;
    for (int t = 0; t < rows; ++t)
      for (int i = 0; i < k; ++i) {
        slots[t * k + i] = s[t * draft::kMaxK + i];
        weights[t * k + i] = w[t * draft::kMaxK + i];
      }
    const int shared = draft::count_shared_routes(slots, rows * k);
    p.layer = &d.layers[stage];
    p.source = &d;
    p.seq = seq;
    p.epoch = epoch;
    p.stage = stage;
    p.rows = rows;
    p.k = k;
    p.shared = shared;
    cpu_experts::ForwardCall& call = p.call;
    call = {};
    call.rows = rows;
    call.k = k;
    call.x = d.x + static_cast<int64_t>(stage) * draft::kMaxRows * d.hidden * 2;
    call.slots = slots;
    call.weights = weights;
    call.out = d.out + static_cast<int64_t>(stage) * draft::kMaxRows * d.hidden;
    if constexpr (Build::kMetrics) trace_.draft_prepared(now_ns(), stage, epoch, seq);
    if constexpr (Build::kMetrics) {
      // draft_shape, before the clock starts: a = routes the CPU computes (slot >= 0; the rest are the draft's
      // GPU-resident experts), b = distinct experts among them, c = rows; for verify jobs cpu_shape gives the same.
      int live = 0, distinct = 0;
      for (int i = 0; i < rows * k; ++i) {
        if (slots[i] < 0) continue;
        ++live;
        bool seen = false;
        for (int j = 0; j < i && !seen; ++j)
          seen = slots[j] == slots[i];
        distinct += seen ? 0 : 1;
      }
      trace_.emit("draft_shape", stage, epoch, seq, -1, live, distinct, rows);
    }
    in_flight_.store(true, std::memory_order_release);
  }

  /// Engine thread, just before the record's job is admitted: its start events.
  void admitting(const Prepared& p) {
    if constexpr (Build::kMetrics) trace_.emit("draft_start", p.stage, p.epoch, p.seq, -1, p.rows, p.k, p.shared);
    if constexpr (Build::kMetrics)
      trace_.resources("draft_faults_start", "draft_switches_start", p.stage, p.epoch, p.seq);
  }

  /// The thread that finishes the record's job (any worker; one record is in flight at a time): its counters, its
  /// completion through the channel, its trace, and the engine's next record. `elapsed` is the job's admission to its
  /// end, in ns.
  void finish(const Prepared& p, int64_t elapsed) {
    add(forward_ns_, elapsed);
    add(jobs_, 1);
    add(rows_, p.rows);
    if (p.shared > 0) {
      add(collided_jobs_, 1);
      add(shared_routes_, p.shared);
      add(collided_forward_ns_, elapsed);
    }
    if constexpr (Build::kMetrics) trace_.emit("draft_end", p.stage, p.epoch, p.seq, -1, p.rows, p.k, p.shared);
    channel::complete<DraftChannel>(p.source->channel, p.seq, static_cast<uint64_t>(p.epoch) << 32 | p.seq);
    completed_.store(p.seq, std::memory_order_release);
    if constexpr (Build::kMetrics) trace_.draft_finished(elapsed, p.stage, p.epoch, p.seq);
    trigger_.finished(p.seq, elapsed);
    next_ = channel::skip_zero(p.seq + 1u);
    in_flight_.store(false, std::memory_order_release);
  }

  /// Engine thread, after finish(): the record's end snapshot of the thread's faults and switches.
  void resources_end(const Prepared& p) {
    if constexpr (Build::kMetrics)
      trace_.resources("draft_faults_end", "draft_switches_end", p.stage, p.epoch, p.seq);
  }

  /// Engine thread, on exit: lets go of the source, so a detach in flight returns and a later one has nothing to wait
  /// for.
  void leave() {
    source_.store(nullptr, std::memory_order_release);
  }

 private:
  /// The word's value while a detach waits: never dereferenced.
  static DraftSource* detaching() {
    static DraftSource tag;
    return &tag;
  }

  /// A counter only the engine thread writes: no locked add needed.
  static void add(std::atomic<int64_t>& counter, int64_t n) {
    counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }

  JobTrace<Build::kMetrics>& trace_;
  DraftDelayTrigger<kDraftDelayTriggerOnly && !Build::kMetrics>& trigger_;
  std::unique_ptr<DraftSource> owned_;         // the host's, for the engine's lifetime
  std::atomic<DraftSource*> source_{nullptr};  // the word the two threads meet on (the class comment)
  uint32_t next_ = 0;                          // the record served next: the engine thread's, moved by finish()
  std::atomic<bool> in_flight_{false};         // a record is in the window, from prepare() to finish()
  std::atomic<uint32_t> completed_{0};
  std::atomic<int64_t> jobs_{0}, rows_{0}, forward_ns_{0}, collided_jobs_{0}, shared_routes_{0},
      collided_forward_ns_{0};
};

}  // namespace sglang::expert_stream
