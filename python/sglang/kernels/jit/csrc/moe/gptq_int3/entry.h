#pragma once
#include "../expert_stream/host/cpu_experts/kernel.hpp"
#include <memory>
#include <span>

namespace sglang::gptq_int3 {
struct Params {
  int32_t group_size = 128;
};
// Six slabs per slot: gate/up low2, high1, FP16 scales, then down low2, high1, FP16 scales.
// Projections are row-major [output][input]; gate rows precede up rows. No requantization.
struct __attribute__((visibility("hidden"))) GptqInt3CpuKernel {
  using Params = ::sglang::gptq_int3::Params;
  using Layer = ::sglang::cpu_experts::ExpertLayer<Params>;
  static constexpr bool usable() noexcept {
    return true;
  }
  static const char* name() noexcept;
  static Params params(std::span<const int64_t>);
  static Layer make_layer(const Layer&);
  static int32_t max_routes() noexcept;
  static int32_t max_rows() noexcept;
  static void check(const Layer&, const ::sglang::cpu_experts::ForwardCall&);
  static int32_t warm(const uint32_t*, uint32_t, int64_t);
  struct Job {
    struct Impl;
    std::unique_ptr<Impl> impl;
    Job();
    ~Job();
    Job(const Job&) = delete;
    Job& operator=(const Job&) = delete;
  };
  static constexpr int kPhases = 4, kGatePhase = 1, kOutPhase = 3;
  static void prepare(Job&, const Layer&, const ::sglang::cpu_experts::ForwardCall&);
  static ::sglang::cpu_experts::Tasks tasks(const Job&, int);
  static void run(Job&, int, int, int, int);
  static int status(const Job&);
};
}  // namespace sglang::gptq_int3
