// The data a CPU expert kernel and the expert-stream host exchange. A format's kernel is a concrete type (F::CpuKernel,
// binding.hpp lists what it offers) declared where the host can see it and defined in the kernel's own translation unit,
// compiled into the host module with its own flags; the host calls it directly. Nothing here is virtual.
//
// The layer types are quant-agnostic views of the pinned tier's bytes plus the quant's typed parameters. A quant reads
// the bytes through its own typed view (EXL3's Exl3Projection, NVFP4's Projection), made from an ExpertRow's pointers
// and its Params.
#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

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

// One forward: `rows` token rows; row t's experts are slots[t*k+i] weighted by weights[t*k+i], -1 skipped. out (fp32
// [rows][hidden]) is overwritten, or added to when accumulate. `threads` and `cores` are the team the call is checked
// against, and the team a standalone forward (forward_alone, team.hpp) makes: worker i on cores[i], empty: unpinned.
struct ForwardCall {
  int32_t rows = 0, k = 0, threads = 1;
  const void* x = nullptr;
  const int32_t* slots = nullptr;
  const float* weights = nullptr;
  float* out = nullptr;
  bool accumulate = false;
  std::span<const int> cores;
  // A two-stage CPU miss (SGLANG_EXPERT_STREAM_CPU_TWO_STAGE): worker 0 calls stage_two(stage_two_ctx) once the gate/up GEMVs
  // are done and before anything reads the second-stage slabs (w2), the rest of the team waiting until it returns. A
  // kernel that cannot split its forward calls it first. Null: one stage.
  void (*stage_two)(void*) = nullptr;
  void* stage_two_ctx = nullptr;
  // An overwrite (accumulate false) may store each row's first sum and zero only the rows no route writes, instead of
  // zeroing every row and then adding (SGLANG_EXPERT_STREAM_ENABLE_CPU_EXPERTS_STORE_OUT). The bits are the same either way; a kernel
  // that ignores it overwrites as before.
  bool store_first = false;
};
}  // namespace sglang::cpu_experts
