// What a format's CPU expert kernel type (format.h's `using CpuKernel`) offers the host, as the concept
// CpuExpertKernel, and the stub of a format with none.
//
// A kernel type is concrete: static functions declared where the host can see them (the kernel's entry.h; no kernel
// internals) and defined in the kernel's own translation unit, which the host module compiles with the kernel flags
// (-Ofast, -march; expert_stream_transport.py's kernel sources and load_jit's source_cflags). The host calls them
// directly. A kernel type has:
//   Params, Layer                 the format's parameter struct and ExpertLayer<Params>
//   usable()                      constexpr, true
//   name()                        the format's name
//   params(span<const int64_t>)   the Params from the int64 list Python passes for a layer (std::invalid_argument for a
//                                 count or value the struct cannot hold; make_layer checks the values themselves)
//   make_layer(shape)             `shape` (slabs and params set), accepted: the layer jobs run over; throws
//                                 std::invalid_argument for a shape, slab or parameter the quant cannot run
//   max_routes(), max_rows()      the largest k and rows a job takes; the host checks its jobs fit when it enables
//                                 the kernel
//   check(layer, call)            throws std::invalid_argument for a call a job may not take
//   warm(word, seen, deadline_ns) register-only work until *word != seen or the clock reaches deadline_ns (a Team's
//                                 idle loop, so a core keeps the license the jobs run at)
//   the job interface             CpuExpertJobKernel below: a call is a job of phases of tasks, which a JobWindow
//                                 (job_window.hpp) or a JobRunner (single_job.hpp) schedules over a Team
// A kernel type may also offer the format's options (plans_json, chunk_m, isa, configure; the host's
// cpu_kernel_* exports). BuildWithKernel and ExpertFormatTraits require CpuExpertKernel, so a kernel type missing one
// of these, or declaring one with another signature, fails there with the concept's name.
//
// UnsupportedCpuKernel<Name> is the kernel of a format with none; Name::value is the format's name. usable() is false,
// so the engine offers no CPU lanes and refuses a request for them at startup; its functions abort naming the format,
// so a bug that reaches one is loud.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <type_traits>

#include "kernel.hpp"

namespace sglang::cpu_experts {

/// \brief A kernel type's job interface (job_window.hpp): one call as phases of tasks that any worker may run.
///
/// Job is one slot's state, default-constructible and reused across jobs. kGatePhase is the first phase that reads
/// stage-two rows (-1: none). kOutPhase, the phase that writes `out`, is optional and defaults to kPhases - 1; only it
/// waits for the previous job on the same output. prepare runs serially at admission and may throw
/// std::invalid_argument; tasks is read when a phase becomes ready, so a phase may report 0 tasks; run takes tasks
/// [t0, t1) of a phase, with `worker` indexing per-worker scratch only, and does not throw; status is 0, or 2 when the
/// input was refused.
template <class K>
concept CpuExpertJobKernel = requires(
    typename K::Job& job, const typename K::Job& cjob, const typename K::Layer& layer, const ForwardCall& call,
    int phase, int t0, int t1, int worker) {
  typename K::Job;
  requires std::default_initializable<typename K::Job>;
  requires(K::kPhases >= 1);
  requires(K::kGatePhase >= -1 && K::kGatePhase < K::kPhases);
  { K::prepare(job, layer, call) } -> std::same_as<void>;
  { K::tasks(cjob, phase) } -> std::same_as<Tasks>;
  { K::run(job, phase, t0, t1, worker) } -> std::same_as<void>;
  { K::status(cjob) } -> std::same_as<int>;
};

/// \brief A format's CPU expert kernel type: the list above, by signature, and its job interface.
template <class K>
concept CpuExpertKernel = requires(
    const typename K::Layer& layer, const ForwardCall& call, std::span<const int64_t> values, const uint32_t* word,
    uint32_t seen, int64_t deadline_ns) {
  typename K::Params;
  requires std::same_as<typename K::Layer, ExpertLayer<typename K::Params>>;
  typename std::bool_constant<K::usable()>;
  { K::name() } -> std::same_as<const char*>;
  { K::params(values) } -> std::same_as<typename K::Params>;
  { K::make_layer(layer) } -> std::same_as<typename K::Layer>;
  { K::max_routes() } -> std::same_as<int32_t>;
  { K::max_rows() } -> std::same_as<int32_t>;
  { K::check(layer, call) } -> std::same_as<void>;
  { K::warm(word, seen, deadline_ns) } -> std::same_as<int32_t>;
} && CpuExpertJobKernel<K>;

/// \brief The parameters of a format with no kernel.
struct NoParams {};

/// \brief The tag of an UnsupportedCpuKernel whose format is not named (HostExports' default binding).
struct UnnamedFormat {
  static constexpr char value[] = "unnamed";
};

/// \brief The kernel type of a format with no CPU expert kernel.
template <class Name>
struct UnsupportedCpuKernel {
  using Params = NoParams;
  using Layer = ExpertLayer<Params>;

  static constexpr bool usable() noexcept { return false; }
  static constexpr std::string_view kFormat = Name::value;

  static const char* name() noexcept { return Name::value; }
  static Params params(std::span<const int64_t>) { unsupported("params"); }
  static Layer make_layer(const Layer&) { unsupported("make_layer"); }
  static int32_t max_routes() noexcept { return 0; }
  static int32_t max_rows() noexcept { return 0; }
  static void check(const Layer&, const ForwardCall&) { unsupported("check"); }
  static int32_t warm(const uint32_t*, uint32_t, int64_t) { unsupported("warm"); }

  struct Job {};
  static constexpr int kPhases = 1;
  static constexpr int kGatePhase = -1;
  static constexpr int kOutPhase = 0;
  static void prepare(Job&, const Layer&, const ForwardCall&) { unsupported("prepare"); }
  static Tasks tasks(const Job&, int) { unsupported("tasks"); }
  static void run(Job&, int, int, int, int) { unsupported("run"); }
  static int status(const Job&) { unsupported("status"); }

 private:
  [[noreturn]] static void unsupported(const char* call) noexcept {
    std::fprintf(
        stderr, "%s CPU experts: %s called, but this format has no CPU expert kernel (UnsupportedCpuKernel)\n",
        Name::value, call);
    std::fflush(stderr);
    std::abort();
  }
};
static_assert(CpuExpertKernel<UnsupportedCpuKernel<UnnamedFormat>>);
static_assert(CpuExpertJobKernel<UnsupportedCpuKernel<UnnamedFormat>>);

}  // namespace sglang::cpu_experts
