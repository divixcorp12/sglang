// The CPU expert kernel interface: the one C++ boundary between the expert-stream host (CpuExpertEngine) and each
// quant's CPU kernel library (EXL3's torch extension, NVFP4's tvm-ffi library). Each library keeps its own build flags;
// the host calls through the vtable. Signatures use plain structs, std::span, std::array and std::unique_ptr only
// (no std::string or std::vector), so the two sides may differ in _GLIBCXX_USE_CXX11_ABI. Not in an anonymous
// namespace: every library must name the same classes. No RTTI crosses a library (a kernel static_casts its own
// layers after the kernel() check).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace sglang::cpu_experts {
inline constexpr int kMaxSlabs = 8;

// One layer's pinned host tier: slot s of slab i starts at slabs[i] + s * slot_bytes[i]. Views only: the caller keeps
// every slab alive for the layer's lifetime. A quant's layer keeps it as made and addresses its slots through it.
struct LayerSlabs {
  int32_t capacity = 0, hidden = 0, intermediate = 0, activation = 0;
  float act_limit = 0;
  int32_t slab_count = 0;
  std::array<const void*, kMaxSlabs> slabs{};
  std::array<uint64_t, kMaxSlabs> slot_bytes{};

  // Slot s's first byte in slab i; null when the slab is absent (an optional one).
  const uint8_t* slot(int i, int s) const {
    return slabs[i] ? static_cast<const uint8_t*>(slabs[i]) + std::size_t(s) * slot_bytes[i] : nullptr;
  }
  // Slot s's first byte in each of the first N slabs, as a quant's decode reads them.
  template <int N>
  std::array<const uint8_t*, N> slot_bases(int s) const {
    std::array<const uint8_t*, N> base;
    for (int i = 0; i < N; ++i) base[i] = slot(i, s);
    return base;
  }
};

// One forward: `rows` token rows; row t's experts are slots[t*k+i] weighted by weights[t*k+i], -1 skipped. out (fp32
// [rows][hidden]) is overwritten, or added to when accumulate. Worker i runs on cores[i]; empty: unpinned workers.
struct ForwardCall {
  int32_t rows = 0, k = 0, threads = 1;
  const void* x = nullptr;
  const int32_t* slots = nullptr;
  const float* weights = nullptr;
  float* out = nullptr;
  bool accumulate = false;
  std::span<const int> cores;
};

class CpuExpertKernel;

// A registered layer, opaque to the host. It names the kernel that made it, so a layer passed to another kernel is
// refused rather than misread.
class CpuExpertLayer {
 public:
  virtual ~CpuExpertLayer() = default;
  const CpuExpertKernel& kernel() const { return *kernel_; }

 protected:
  explicit CpuExpertLayer(const CpuExpertKernel& k) : kernel_(&k) {}

 private:
  const CpuExpertKernel* kernel_;
};

class CpuExpertKernel {
 public:
  virtual ~CpuExpertKernel() = default;
  virtual const char* name() const noexcept = 0;
  // Validates (the old C ABI's registration checks) and stores views; throws std::invalid_argument. `params` is the quant's
  // own parameter struct, size-checked against it (empty when the quant has none).
  virtual std::unique_ptr<CpuExpertLayer> make_layer(const LayerSlabs&, std::span<const std::byte> params) const = 0;
  // The largest k and rows a forward takes. The host checks its jobs fit once, when it enables the kernel.
  virtual int32_t max_routes() const noexcept = 0;
  virtual int32_t max_rows() const noexcept = 0;
  // Throws std::invalid_argument for a call forward may not take: a layer of another kernel, rows, k, threads or a
  // buffer out of range, cores out of range or repeated, or fewer cores than threads, a slot outside the layer's
  // capacity or unusable, a non-finite weight. forward checks none of it; a caller whose calls are not built to fit
  // (a test, a harness, an upstream API) calls check first.
  virtual void check(const CpuExpertLayer&, const ForwardCall&) const = 0;
  // Runs a call check would pass. Throws std::invalid_argument when the input itself refuses (the quant cannot
  // represent it), std::runtime_error for a failure; out is untouched when it throws. Pins the calling thread and its
  // workers to cores.
  virtual void forward(const CpuExpertLayer&, const ForwardCall&) const = 0;
  // Register-only work at the forward's vector width on `threads` workers pinned to cores, until *word != seen or
  // CLOCK_MONOTONIC reaches deadline_ns (keep_warm.hpp). The cores are not checked (in range, distinct): the caller
  // passes cores it has checked, as forward's. Throws for no worker, no word or more workers than cores.
  virtual void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
                         int64_t deadline_ns) const = 0;
};
}  // namespace sglang::cpu_experts
