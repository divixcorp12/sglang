// Bounded host-clock events for CPU/copy attribution. Production has no storage, branches or clock reads.
#pragma once

#include "build_policy.h"
#include <time.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <unistd.h>

namespace sglang::expert_stream {

template <bool On>
class JobTrace;

template <>
class JobTrace<false> {
 public:
  explicit JobTrace(const std::string&) {}
  bool enabled() const { return false; }
  void emit(const char*, int64_t, uint64_t, uint32_t, int, int64_t = 0, int64_t = 0, int64_t = 0) {}
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
    events_ = std::make_unique<Event[]>(kCapacity);
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
    for (size_t i = 0; i < std::min(count, kCapacity); ++i) {
      const Event& e = events_[i];
      std::fprintf(out, "{\"event\":\"%s\",\"ns\":%lld,\"row\":%lld,\"gen\":%llu,\"seq\":%u,\"group\":%d,\"a\":%lld,\"b\":%lld,\"c\":%lld}\n",
                   e.kind, static_cast<long long>(e.ns), static_cast<long long>(e.row),
                   static_cast<unsigned long long>(e.gen), e.seq, e.group,
                   static_cast<long long>(e.a), static_cast<long long>(e.b), static_cast<long long>(e.c));
    }
    std::fprintf(out, "{\"dropped\":%llu}\n", static_cast<unsigned long long>(count > kCapacity ? count - kCapacity : 0));
    std::fclose(out);
  }
  bool enabled() const { return events_ != nullptr; }
  void emit(const char* kind, int64_t row, uint64_t gen, uint32_t seq, int group,
            int64_t a = 0, int64_t b = 0, int64_t c = 0) {
    if (!events_) return;
    const int64_t ns = clock_ns();
    const size_t slot = next_.fetch_add(1, std::memory_order_relaxed);
    if (slot < kCapacity) events_[slot] = Event{kind, ns, row, gen, seq, group, a, b, c};
  }
 private:
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
  static constexpr size_t kCapacity = 131072;
  std::string path_;
  std::unique_ptr<Event[]> events_;
  std::atomic<size_t> next_{0};
  int64_t monotonic_ns_ = 0, epoch_ns_ = 0;
};

static_assert(std::is_empty_v<JobTrace<false>>, "production job tracing is empty");
}  // namespace sglang::expert_stream
