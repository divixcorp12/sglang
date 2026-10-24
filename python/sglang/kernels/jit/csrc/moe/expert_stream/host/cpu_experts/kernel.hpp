// The data a CPU expert kernel and the expert-stream host exchange. A format's kernel is a concrete type (F::CpuKernel,
// binding.hpp's CpuExpertKernel concept lists what it offers) declared where the host can see it and defined in the
// kernel's own translation unit, compiled into the host module with its own flags; the host calls it directly. Only
// data crosses between the two: no host callback runs inside a forward.
//
// The layer types are quant-agnostic views of the pinned tier's bytes plus the quant's typed parameters. A quant reads
// the bytes through its own typed view (EXL3's Exl3Projection, NVFP4's Projection), made from an ExpertRow's pointers
// and its Params.
#pragma once
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>

namespace sglang::cpu_experts {
// Whether v is neither NaN nor infinite, from its exponent bits: a kernel file is built -Ofast (finite-math-only), under
// which std::isfinite may fold to true.
inline bool is_finite_bits(float v) { return (std::bit_cast<uint32_t>(v) & 0x7f800000u) != 0x7f800000u; }

inline constexpr int kMaxSlabs = 8;

class Team;  // team.hpp: the workers a forward runs on

// One expert slot's bytes: its row's first byte in each slab, null for an absent optional slab.
struct ExpertRow {
  std::array<const uint8_t*, kMaxSlabs> slab{};
};

// One layer: the pinned tier's slabs as a strided array of ExpertRows (row s of slab i starts at slabs[i] + s *
// slot_bytes[i]), the expert shape, and the quant's parameters (Params, the format's own struct). A value of views: the
// caller keeps every slab alive for the layer's lifetime. The kernel's make_layer is the only way to a layer forward
// takes.
template <class Params>
struct ExpertLayer {
  int32_t capacity = 0, hidden = 0, intermediate = 0, activation = 0;
  float act_limit = 0;
  int32_t slab_count = 0;
  std::array<const void*, kMaxSlabs> slabs{};
  std::array<uint64_t, kMaxSlabs> slot_bytes{};
  Params params{};

  ExpertRow operator[](int s) const {
    ExpertRow row;
    for (int i = 0; i < kMaxSlabs; ++i)
      row.slab[i] = slabs[i] ? static_cast<const uint8_t*>(slabs[i]) + std::size_t(s) * slot_bytes[i] : nullptr;
    return row;
  }
};

// How a forward writes out (fp32 [rows][hidden]).
enum class OutMode : uint8_t {
  kOverwrite,         // out = each row's routes' sum; a row no route writes reads 0
  kOverwriteByStore,  // kOverwrite's bits, by storing each row's first sum and zeroing only the rows no route writes
                      // (SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT); a kernel without that path runs kOverwrite
  kAccumulate,        // out += each row's routes' sum
};

// A two-stage CPU miss's gate (SGLANG_EXPERT_STREAM_CPU_TWO_STAGE): its w2 slabs may still be landing when the job is
// admitted. The window holds the job's kGatePhase, the first phase that reads w2, until *word == open or *stop is set,
// and records what it saw (job_window.hpp); a kernel never waits on it.
struct StageGate {
  const std::atomic<uint32_t>* word = nullptr;
  uint32_t open = 0;
  const std::atomic<bool>* stop = nullptr;  // null: no early exit
  // Read the clock (called_ns, waited_ns): the engine sets it under Build::kMetrics, so a prod job reads none.
  bool timed = false;

  int32_t calls = 0;      // the gate phase's arrivals: one per staged job (a JobRunner copies the window's here)
  int64_t called_ns = 0;  // CLOCK_MONOTONIC at the arrival (timed only)
  bool spun = false;      // the gate was shut on arrival
  int64_t waited_ns = 0;  // how long it was shut (timed only; 0 when it was open on arrival)
};

// A job phase's tasks as the scheduler reads them (job_window.hpp): `count` tasks, claimed in runs of a multiple of
// `align` (the last run of a phase may be shorter).
struct Tasks {
  int count = 0;
  int align = 1;
};

// One job's call: `rows` token rows; row t's experts are slots[t*k+i] weighted by weights[t*k+i], -1 skipped, written to
// out as out_mode says. The slots, weights, x and out stay valid until the job has completed.
struct ForwardCall {
  int32_t rows = 0, k = 0;
  const void* x = nullptr;
  const int32_t* slots = nullptr;
  const float* weights = nullptr;
  float* out = nullptr;
  OutMode out_mode = OutMode::kOverwrite;
  StageGate* stage_two = nullptr;  // a two-stage CPU miss's gate; null: one stage
  // Set by the window that admits the call (the caller's values are ignored): the team's workers and worker 0's core (-1
  // unpinned), and whether the call is staged (stage_two is the window's, so a kernel sees it null). For the fake
  // kernel's records and a kernel's per-team sizing.
  int32_t team_workers = 0;
  int32_t team_core = -1;
  bool staged = false;

  bool accumulate() const { return out_mode == OutMode::kAccumulate; }
  bool store_first() const { return out_mode == OutMode::kOverwriteByStore; }
};
}  // namespace sglang::cpu_experts
