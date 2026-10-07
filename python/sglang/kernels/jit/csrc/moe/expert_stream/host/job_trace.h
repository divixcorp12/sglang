// Bounded host-clock events for CPU/copy attribution. Production has no storage, branches or clock reads.
#pragma once

#include "build_policy.h"
#include "trace_gate.h"
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace sglang::expert_stream {

// A stable uprobe target in optimized builds. The asm prevents an empty-call elimination;
// noinline keeps the call boundary. Arguments also identify the matching buffered event.
extern "C" __attribute__((noinline, visibility("default"))) inline
void sglang_draft_delay_trigger(uint32_t seq, int64_t delay_ns, int reason) {
  asm volatile("" : : "r"(seq), "r"(delay_ns), "r"(reason) : "memory");
}


template <bool On>
class JobTrace;

template <>
class JobTrace<false> {
 public:
  explicit JobTrace(const std::string&) {}
  bool enabled() const { return false; }
  void emit(const char*, int64_t, uint64_t, uint32_t, int, int64_t = 0, int64_t = 0, int64_t = 0) {}
  void resources(const char*, const char*, int64_t, uint64_t, uint32_t) {}
};

// Writers reserve distinct slots; no reader touches them until every producer and consumer has joined.
// No ring overwrite, allocation, formatting or IO while serving. Overflow is explicit in the footer.
template <>
class JobTrace<true> {
 public:
  explicit JobTrace(const std::string& name) {
    const char* prefix = std::getenv("SGLANG_DSV41_EXPERT_JOB_TRACE_PREFIX");
    if (!prefix || !*prefix) return;
    path_ = std::string(prefix) + "." + std::to_string(getpid()) + "." + name + "." +
            std::to_string(instances_.fetch_add(1, std::memory_order_relaxed)) + ".jsonl";
    if (const char* value = std::getenv("SGLANG_DSV41_EXPERT_JOB_TRACE_CAPACITY")) {
      const std::string_view text(value);
      if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos)
        throw std::invalid_argument("expert job trace capacity must be an integer in [1, 1048576]");
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(value, &end, 10);
      if (*end || parsed < 1 || parsed > 1048576)
        throw std::invalid_argument("expert job trace capacity must be an integer in [1, 1048576]");
      capacity_ = static_cast<size_t>(parsed);
    }
    pending_threshold_ = threshold("SGLANG_DRAFT_PENDING_TRIGGER_US");
    arrival_threshold_ = threshold("SGLANG_DRAFT_ARRIVAL_TRIGGER_US");
    forward_threshold_ = threshold("SGLANG_DRAFT_FORWARD_TRIGGER_US");
    events_ = std::make_unique<Event[]>(capacity_);
    const char* resources = std::getenv("SGLANG_DSV41_EXPERT_JOB_RESOURCE_TRACE");
    resources_ = resources && std::string_view(resources) == "1";
    monotonic_ns_ = clock_ns();
    timespec wall{};
    clock_gettime(CLOCK_REALTIME, &wall);
    epoch_ns_ = static_cast<int64_t>(wall.tv_sec) * 1'000'000'000 + wall.tv_nsec;
  }
  ~JobTrace() {
    if (!events_) return;
    FILE* out = std::fopen(path_.c_str(), "w");
    if (!out) { std::perror(path_.c_str()); return; }
    std::fprintf(out, "{\"schema\":1,\"clock\":\"CLOCK_MONOTONIC\",\"monotonic_ns\":%lld,\"epoch_ns\":%lld}\n",
                 static_cast<long long>(monotonic_ns_), static_cast<long long>(epoch_ns_));
    const size_t count = next_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < std::min(count, capacity_); ++i) {
      const Event& e = events_[i];
      std::fprintf(out, "{\"event\":\"%s\",\"ns\":%lld,\"row\":%lld,\"gen\":%llu,\"seq\":%u,\"group\":%d,\"a\":%lld,\"b\":%lld,\"c\":%lld}\n",
                   e.kind, static_cast<long long>(e.ns), static_cast<long long>(e.row),
                   static_cast<unsigned long long>(e.gen), e.seq, e.group,
                   static_cast<long long>(e.a), static_cast<long long>(e.b), static_cast<long long>(e.c));
    }
    std::fprintf(out, "{\"dropped\":%llu}\n", static_cast<unsigned long long>(count > capacity_ ? count - capacity_ : 0));
    std::fclose(out);
  }
  bool enabled() const { return events_ != nullptr; }
  bool active() const { return events_ && gate_.enabled(); }
  // Called only on a pending head, before target-queue selection. No clock read per idle poll.
  void draft_observe(const void* source, uint32_t seq) {
    if (source == observed_source_ && seq == observed_seq_) return;
    observed_source_ = source;
    observed_seq_ = seq;
    observed_ns_ = active() ? clock_ns() : 0;
  }
  void draft_selected(int64_t selected, int stage, uint32_t epoch, uint32_t seq, uint64_t gpu_ns) {
    if (!observed_ns_ || !active()) return;
    emit_at("draft_observed", observed_ns_, stage, epoch, seq, -1, gpu_ns, syscall(SYS_gettid));
    emit_at("draft_selected", selected, stage, epoch, seq, -1);
    emit("draft_record_ready", stage, epoch, seq, -1);
  }
  // Startup-correlated arrival estimate. The lower endpoint is conservative for the
  // measured offset interval, but later clock drift must be checked by the analysis.
  void draft_arrival(int stage, uint32_t epoch, uint32_t seq, uint64_t gpu_ns, int64_t offset_high) {
    if (observed_ns_ && active() && gpu_ns && offset_high)
      trigger(2, observed_ns_ - static_cast<int64_t>(gpu_ns) - offset_high,
              arrival_threshold_, stage, epoch, seq);
  }
  void draft_prepared(int64_t ready, int stage, uint32_t epoch, uint32_t seq) {
    if (!observed_ns_ || !active()) return;
    emit_at("draft_payload_ready", ready, stage, epoch, seq, -1);
    trigger(0, ready - observed_ns_, pending_threshold_, stage, epoch, seq);
  }
  void draft_finished(int64_t elapsed, int stage, uint32_t epoch, uint32_t seq) {
    if (observed_ns_ && active()) trigger(1, elapsed, forward_threshold_, stage, epoch, seq);
    observed_ns_ = 0;
    observed_seq_ = 0;
  }
  void emit(const char* kind, int64_t row, uint64_t gen, uint32_t seq, int group,
            int64_t a = 0, int64_t b = 0, int64_t c = 0) {
    if (!events_ || !gate_.enabled()) return;
    emit_at(kind, clock_ns(), row, gen, seq, group, a, b, c);
  }
  void emit_at(const char* kind, int64_t ns, int64_t row, uint64_t gen, uint32_t seq, int group,
               int64_t a = 0, int64_t b = 0, int64_t c = 0) {
    if (!active()) return;
    const size_t slot = next_.fetch_add(1, std::memory_order_relaxed);
    if (slot < capacity_) events_[slot] = Event{kind, ns, row, gen, seq, group, a, b, c};
  }
  // Only the engine's owning CPU thread calls this. These are its counters, not the worker team's sum.
  void resources(const char* faults, const char* switches, int64_t row, uint64_t gen, uint32_t seq) {
    if (!resources_ || !gate_.enabled()) return;
    if (!resource_tid_) resource_tid_ = static_cast<int>(syscall(SYS_gettid));
    rusage usage{};
    if (getrusage(RUSAGE_THREAD, &usage) != 0) {
      emit(faults, row, gen, seq, -1, -1, -1, resource_tid_);
      emit(switches, row, gen, seq, -1, -1, -1, -1);
      return;
    }
    const int64_t cpu_ns = (static_cast<int64_t>(usage.ru_utime.tv_sec) + usage.ru_stime.tv_sec) * 1'000'000'000 +
                           (static_cast<int64_t>(usage.ru_utime.tv_usec) + usage.ru_stime.tv_usec) * 1000;
    emit(faults, row, gen, seq, -1, usage.ru_minflt, usage.ru_majflt, resource_tid_);
    emit(switches, row, gen, seq, -1, usage.ru_nvcsw, usage.ru_nivcsw, cpu_ns);
  }
 private:
  static int64_t threshold(const char* name) {
    const char* value = std::getenv(name);
    if (!value) return 0;
    const std::string_view text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string_view::npos)
      throw std::invalid_argument(std::string(name) + " must be an integer in [0, 1000000000]");
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (*end || parsed > 1000000000)
      throw std::invalid_argument(std::string(name) + " must be an integer in [0, 1000000000]");
    return static_cast<int64_t>(parsed) * 1000;
  }
  void trigger(int reason, int64_t elapsed, int64_t threshold_ns, int stage, uint32_t epoch, uint32_t seq) {
    if (!threshold_ns || elapsed < threshold_ns || triggered_) return;
    triggered_ = true;  // one snapshot per engine; admission gate keeps warm-up from consuming it
    emit("draft_trigger", stage, epoch, seq, -1, reason, elapsed, threshold_ns);
    sglang_draft_delay_trigger(seq, elapsed, reason);
  }
  static int64_t clock_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
  }
  struct Event {
    const char* kind;
    int64_t ns, row;
    uint64_t gen;
    uint32_t seq;
    int group;
    int64_t a, b, c;
  };
  inline static std::atomic<size_t> instances_{0};
  TraceGate gate_;
  size_t capacity_ = 131072;
  std::string path_;
  std::unique_ptr<Event[]> events_;
  std::atomic<size_t> next_{0};
  int64_t monotonic_ns_ = 0, epoch_ns_ = 0;
  bool resources_ = false;
  int resource_tid_ = 0;
  const void* observed_source_ = nullptr;
  uint32_t observed_seq_ = 0;
  int64_t observed_ns_ = 0, pending_threshold_ = 0, forward_threshold_ = 0, arrival_threshold_ = 0;
  bool triggered_ = false;
};

static_assert(std::is_empty_v<JobTrace<false>>, "production job tracing is empty");
}  // namespace sglang::expert_stream
