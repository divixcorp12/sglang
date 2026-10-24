// The fake CPU expert kernel of the test format `fake` (format.h), declared here and defined in fake_kernel.cpp, its own
// translation unit as a real format's kernel is (cpu_experts/binding.hpp lists what a kernel type offers).
//
// Its layers read only their hidden (and capacity, which a call records). A job's prepare throws when the kernel is made
// failing; its out phase spins k * ns_per_expert, waits while its worker-0 core is held (hold), and for each row t <
// max(rows, 1) writes out[t * hidden + j] for j < max(hidden, 1) -- (accumulate ? out[..] : j) + sum_i weights[t * k + i]
// * (slots[t * k + i] + 1), or a zero partial (accumulate ? out[..] : 0) when made zeroing. A job records one call per
// row when it finishes. Its warm loop PAUSEs until its word moves or its deadline passes. Over a layer of the EXL3 row's
// six slabs, a call also records the byte sums of its live slots' first three slabs (w13), read in the first phase, and
// of the last three (w2), read in the phase a staged job's gate holds: what a two-stage forward's GateUp and Down would
// read.
#pragma once
#include "../../expert_stream/host/cpu_experts/kernel.hpp"
#include <array>
#include <atomic>
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

  /// One job row as the fake records it.
  struct Call {
    int32_t core, affinity, threads, accumulate, k;
    std::array<int32_t, kFakeMaxLanes> slots;
    std::array<float, kFakeMaxLanes> weights;
    int32_t capacity;  // the layer's: which layer the call ran
    int32_t staged;    // ForwardCall::stage_two was set (and waited on)
    double w13, w2;    // the slab byte sums (six-slab layers only, else 0)
  };

  static constexpr bool usable() noexcept { return true; }
  static const char* name() noexcept;
  static Params params(std::span<const int64_t>);
  static Layer make_layer(const Layer& shape);
  static int32_t max_routes() noexcept;
  static int32_t max_rows() noexcept;
  static void check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
  static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns);

  // The job interface (cpu_experts/binding.hpp's CpuExpertJobKernel), for the scheduler tests: three phases over a
  // call's rows. Phase 0 sums each row's w13 slabs, phase 1 (kGatePhase, the one a staged job's gate holds) its w2
  // slabs, phase 2 (kOutPhase) spins, waits on hold and writes the rows: task t of a phase with C tasks takes rows t, t +
  // C, ... (task 0 of phase 2 spins and waits first). Tasks of a phase default to one per row (max(rows, 1)). Each task
  // can spin first, and every run is recorded.
  // A job whose first slot is kRefuseSlot reports status 2; one whose first slot is kRejectSlot, or any prepare after
  // reset(..., fail != 0), throws std::invalid_argument.
  static constexpr int kPhases = 3;
  static constexpr int kGatePhase = 1;
  static constexpr int kOutPhase = 2;
  static constexpr int32_t kRefuseSlot = -7;
  static constexpr int32_t kRejectSlot = -9;

  struct Job {
    int32_t id = 0;  // prepare's ordinal since job_reset, which events and the test hooks name the job by
    int32_t rows = 0;
    int32_t status = 0;
    const Layer* layer = nullptr;
    const ::sglang::cpu_experts::ForwardCall* call = nullptr;
    std::array<int32_t, kPhases> count{};
    std::array<std::atomic<int32_t>, kPhases> executed{};  // tasks done per phase
    std::vector<double> w13, w2;  // phase 0 and 1's per-row results, stored as sum + 1; phase 1 and 2 read the earlier
    std::vector<Call> calls;      // the job's rows as calls() reports them, published when its last phase has run
    int32_t team_core = -1;       // worker 0's core, which hold() names
  };

  /// One run of a phase's tasks as the fake records it.
  struct JobEvent {
    int32_t job, phase, t0, t1, worker;
    uint64_t tsc0, tsc1;  // rdtsc before and after the run
  };
  /// The test's controls: tasks per (job, phase, rows) (null or a negative result: one per row), ns each task spins
  /// first (null: none), and the phases' claim alignment.
  struct JobTest {
    int32_t (*count)(int32_t job, int32_t phase, int32_t rows) = nullptr;
    int64_t (*delay_ns)(int32_t job, int32_t phase, int32_t task) = nullptr;
    std::array<int32_t, kPhases> align{1, 1, 1};
  };

  static void prepare(Job& job, const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call);
  static ::sglang::cpu_experts::Tasks tasks(const Job& job, int phase);
  static void run(Job& job, int phase, int t0, int t1, int worker);
  static int status(const Job& job);

  /// Clears the job events, the ordinal and the violation count, and sets the controls. Not while a job is in flight.
  static void job_reset(const JobTest& test);
  /// Every run recorded since job_reset, in no order (read once the jobs are done).
  static std::vector<JobEvent> job_events();
  /// How many times a phase started before the previous one had finished, or read a result it never got.
  static int64_t job_violations();

  /// Resets the recorded calls and the controls; jobs then spin k * ns_per_expert, are refused at prepare when `fail` is
  /// nonzero, and write a zero partial when `zero`.
  static void reset(int64_t ns_per_expert, int64_t fail, bool zero);
  static std::vector<Call> calls();
  /// While on, a job whose worker-0 core is `core` waits in its out phase (one held core at a time).
  static void hold(int64_t core, bool on);
  /// The largest row count a job takes (a real kernel's is 65536); reset restores it.
  static void set_max_rows(int32_t rows);
};

}  // namespace sglang::formats::fake
