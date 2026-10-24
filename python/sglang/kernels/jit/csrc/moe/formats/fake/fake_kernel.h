// The fake CPU expert kernel of the test format `fake` (format.h), declared here and defined in fake_kernel.cpp, its own
// translation unit as a real format's kernel is (cpu_experts/binding.hpp lists what a kernel type offers).
//
// Its layers read only their hidden (and capacity, which a call records). A forward spins k * ns_per_expert, waits while
// its worker-0 core is held (hold), throws when made failing, else for each row t < max(rows, 1) writes out[t * hidden +
// j] for j < max(hidden, 1) -- (accumulate ? out[..] : j) + sum_i weights[t * k + i] * (slots[t * k + i] + 1), or a zero
// partial (accumulate ? out[..] : 0) when made zeroing -- and records one call per row. Its warm loop PAUSEs until its
// word moves or its deadline passes. Over a layer of the EXL3 row's six slabs, a call also records the byte sums of its
// live slots' first three slabs (w13) at its start and of the last three (w2) after ForwardCall::stage_two, which a
// forward calls between the two when set: what a two-stage forward's GateUp and Down would read.
#pragma once
#include "../../expert_stream/host/cpu_experts/kernel.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sglang::formats::fake {

/// \brief The fake's params: it takes no list.
struct FakeParams {};

/// \brief The most lanes a recorded call keeps (the wire's widest build).
inline constexpr int kFakeMaxLanes = 32;

struct __attribute__((visibility("hidden"))) FakeCpuKernel {
  using Params = FakeParams;
  using Layer = ::sglang::cpu_experts::ExpertLayer<Params>;

  /// One forward row as the fake records it.
  struct Call {
    int32_t core, affinity, threads, accumulate, k;
    std::array<int32_t, kFakeMaxLanes> slots;
    std::array<float, kFakeMaxLanes> weights;
    int32_t capacity;  // the layer's: which layer the call ran
    int32_t staged;    // ForwardCall::stage_two was set (and called)
    double w13, w2;    // the slab byte sums (six-slab layers only, else 0)
  };

  static constexpr bool usable() noexcept { return true; }
  static const char* name() noexcept;
  static Params params(std::span<const int64_t>);
  static Layer make_layer(const Layer& shape);
  static int32_t max_routes() noexcept;
  static int32_t max_rows() noexcept;
  static void check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
  static void forward(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call, ::sglang::cpu_experts::Team& team);
  static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

  /// Resets the recorded calls and the controls; forwards then spin k * ns_per_expert, fail when `fail` is nonzero, and
  /// write a zero partial when `zero`.
  static void reset(int64_t ns_per_expert, int64_t fail, bool zero);
  static std::vector<Call> calls();
  /// While on, a forward whose worker-0 core is `core` waits (one held core at a time).
  static void hold(int64_t core, bool on);
  /// The largest row count a forward takes (a real kernel's is 65536); reset restores it.
  static void set_max_rows(int32_t rows);
};

}  // namespace sglang::formats::fake
