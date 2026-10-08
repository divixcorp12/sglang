// The CPU expert kernel interface: the one C++ boundary between the expert-stream host (CpuExpertEngine) and each
// quant's CPU kernel library (EXL3's torch extension, NVFP4's tvm-ffi library). Each library keeps its own build flags;
// the host calls through the vtable. Signatures use plain structs, std::span and std::array only (no std::string or
// std::vector), so the two sides may differ in _GLIBCXX_USE_CXX11_ABI. Not in an anonymous namespace: every library
// must name the same classes.
//
// The layer types are quant-agnostic views of the pinned tier's bytes. A quant reads them through its own typed view
// (EXL3's Exl3Projection, NVFP4's Projection), made from an ExpertRow's pointers and its Params.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace sglang::cpu_experts {
inline constexpr int kMaxSlabs = 8;
inline constexpr size_t kMaxParamBytes = 16;

class CpuExpertKernel;
class Team;  // team.hpp: the workers a forward runs on

// One expert slot's bytes: its row's first byte in each slab, null for an absent optional slab.
struct ExpertRow {
  std::array<const uint8_t*, kMaxSlabs> slab{};
};

// One layer: the pinned tier's slabs as a strided array of ExpertRows (row s of slab i starts at slabs[i] + s *
// slot_bytes[i]), the expert shape, and the quant's parameters. A value of views: the caller keeps every slab alive for
// the layer's lifetime. `kernel` and `params` are set by the kernel's make_layer, which is the only way to a layer
// forward takes.
struct ExpertLayer {
  int32_t capacity = 0, hidden = 0, intermediate = 0, activation = 0;
  float act_limit = 0;
  int32_t slab_count = 0;
  std::array<const void*, kMaxSlabs> slabs{};
  std::array<uint64_t, kMaxSlabs> slot_bytes{};
  const CpuExpertKernel* kernel = nullptr;
  alignas(8) std::array<std::byte, kMaxParamBytes> params{};

  ExpertRow operator[](int s) const {
    ExpertRow row;
    for (int i = 0; i < kMaxSlabs; ++i)
      row.slab[i] = slabs[i] ? static_cast<const uint8_t*>(slabs[i]) + std::size_t(s) * slot_bytes[i] : nullptr;
    return row;
  }

  // The quant's parameters, as make_layer stored them.
  template <class P>
  P params_as() const {
    static_assert(std::is_trivially_copyable_v<P> && sizeof(P) <= kMaxParamBytes);
    P p;
    std::memcpy(&p, params.data(), sizeof(P));
    return p;
  }
};

// One forward: `rows` token rows; row t's experts are slots[t*k+i] weighted by weights[t*k+i], -1 skipped. out (fp32
// [rows][hidden]) is overwritten, or added to when accumulate. `threads` and `cores` are the team the call is checked
// against, and the team a standalone forward (CpuExpertKernel::forward without a Team) makes: worker i on cores[i],
// empty: unpinned.
struct ForwardCall {
  int32_t rows = 0, k = 0, threads = 1;
  const void* x = nullptr;
  const int32_t* slots = nullptr;
  const float* weights = nullptr;
  float* out = nullptr;
  bool accumulate = false;
  std::span<const int> cores;
};

class CpuExpertKernel {
 public:
  virtual ~CpuExpertKernel() = default;
  virtual const char* name() const noexcept = 0;
  // `shape` with its slabs, accepted: kernel set to this one and `params` (the quant's parameter struct, exactly its
  // size) stored. Throws std::invalid_argument for a shape, slab or parameter the quant cannot run.
  virtual ExpertLayer make_layer(const ExpertLayer& shape, std::span<const std::byte> params) const = 0;
  // The largest k and rows a forward takes. The host checks its jobs fit once, when it enables the kernel.
  virtual int32_t max_routes() const noexcept = 0;
  virtual int32_t max_rows() const noexcept = 0;
  // Throws std::invalid_argument for a call forward may not take: a layer of another kernel, rows, k, threads or a
  // buffer out of range, cores out of range or repeated, or fewer cores than threads, a slot outside the layer's
  // capacity or unusable, a non-finite weight. forward checks none of it; a caller whose calls are not built to fit
  // (a test, a harness) calls check first.
  virtual void check(const ExpertLayer&, const ForwardCall&) const = 0;
  // Runs a call check would pass on `team`: every worker of the team, the caller (the team's owner) as worker 0.
  // Throws std::invalid_argument when the input itself refuses (the quant cannot represent it), std::runtime_error
  // for a failure; out is untouched when it throws.
  virtual void forward(const ExpertLayer&, const ForwardCall&, Team& team) const = 0;
  // Register-only work at the forward's vector width (keep_warm.hpp) until *word != seen or CLOCK_MONOTONIC reaches
  // deadline_ns: a Team's idle loop, so a core keeps the license the forward runs at. Returns a value for the caller
  // to sink.
  virtual int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns) const = 0;
  // forward on a team made for this call alone (team.hpp), for a caller that holds no Team.
  void forward(const ExpertLayer&, const ForwardCall&) const;
};
}  // namespace sglang::cpu_experts
