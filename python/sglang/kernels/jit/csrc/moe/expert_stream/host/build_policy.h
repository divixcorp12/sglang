// The compile-time build policy of the expert-stream host transport (plan 2026-09-29-hotpath-zero-overhead, spec
// section 4.1). ProdBuild carries no metrics, trace or fault state on the request path; InstrBuild carries all of it.
// Each is instantiated in its own module: exl3_ram_miss_host.cpp (prod), exl3_ram_miss_host_instr.cpp (instr).
#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace sglang::expert_stream {

struct ProdBuild {
  static constexpr bool kMetrics = false;
  static constexpr bool kFaults = false;
  static constexpr std::string_view kName = "prod";
};

struct InstrBuild {
  static constexpr bool kMetrics = true;
  static constexpr bool kFaults = true;
  static constexpr std::string_view kName = "instr";
};

template <class B>
concept BuildPolicy = std::is_same_v<B, ProdBuild> || std::is_same_v<B, InstrBuild>;

// The refusal of a test-only entry point (fault injection, SQE logs, stress harnesses) on ProdBuild, which compiles
// none of their machinery (plan Task 10). One message, so Python and C++ refusals read alike.
[[noreturn]] inline void test_only(const char* name) {
  throw std::runtime_error(std::string(name) + " is test-only: it exists in the instrumented host build");
}

// Counters with one writer each (the thread that owns the instance): add() is a relaxed load and a relaxed store of
// the sum, which on x86 is a plain add with no lock prefix and no fence. Readers on other threads load relaxed; a
// count may lag its writer, never tear. Own cache line(s), so a writer never shares a line with another thread.
template <int N>
struct alignas(64) LineCounters {
  int64_t v[N] = {};
  void add(int k, int64_t n = 1) {
    std::atomic_ref<int64_t> word(v[k]);
    word.store(word.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
  }
  void set(int k, int64_t value) {
    std::atomic_ref<int64_t>(v[k]).store(value, std::memory_order_relaxed);
  }
  int64_t get(int k) const {
    return std::atomic_ref<int64_t>(const_cast<int64_t&>(v[k])).load(std::memory_order_relaxed);
  }
};

// Metrics: InstrBuild keeps today's shared atomics (several writers per slot); ProdBuild has none.
template <bool On, int N>
struct Stats;

template <int N>
struct Stats<false, N> {
  void add(int, int64_t = 1) {}
  void store(int, int64_t) {}
  int64_t get(int) const {
    return 0;
  }
};

template <int N>
struct Stats<true, N> {
  std::atomic<int64_t> v[N]{};
  void add(int k, int64_t n = 1) {
    v[k].fetch_add(n, std::memory_order_relaxed);
  }
  void store(int k, int64_t value) {
    v[k].store(value, std::memory_order_relaxed);
  }
  int64_t get(int k) const {
    return v[k].load(std::memory_order_relaxed);
  }
};

static_assert(std::is_empty_v<Stats<false, 1>>, "the production build carries no metric storage");

}  // namespace sglang::expert_stream
