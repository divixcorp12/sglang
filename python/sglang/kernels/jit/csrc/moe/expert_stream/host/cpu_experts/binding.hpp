// How an expert format binds to a CPU expert kernel (format.h's `using CpuKernel`).
//
// A format's CPU kernel is a CpuExpertKernel (kernel.hpp) compiled into the host module as its own translation unit,
// with the kernel flags (-Ofast, -march) the host files do not take (expert_stream_transport.py's kernel sources and
// load_jit's source_cflags). ExpertForward<Quant> has internal linkage there, so the host cannot name that type; the
// translation unit exports a named accessor instead, and the host calls the kernel through the vtable. The two
// bindings below say, at compile time, which world a format is in:
//   ExternalCpuKernel<Entry>    the format has a compiled-in kernel. Entry names it: `static const CpuExpertKernel&
//                               kernel()` and `static std::vector<std::byte> params(std::span<const int64_t>)`, the
//                               int64 list Python passes for a layer made into the params bytes make_layer takes
//                               (validated there). An Entry may also give the format's kernel options (plans_json,
//                               plan_calls, chunk_m, configure, isa; ffi_exports.h's cpu_kernel_* exports). usable()
//                               is true.
//   UnsupportedCpuKernel<Name>  the format has none; Name::value is the format's name. usable() is false, so the engine
//                               offers no CPU lanes and refuses a request for them at startup; kernel() is a stand-in
//                               CpuExpertKernel whose make_layer, check, forward and warm abort naming the format, so a
//                               bug that reaches it is loud.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string_view>
#include <vector>

#include "kernel.hpp"

namespace sglang::cpu_experts {

/// \brief A format whose CPU kernel is compiled into the host module (see the file comment for `Entry`).
template <class Entry>
struct ExternalCpuKernel {
  using Kernel = Entry;

  static constexpr bool usable() noexcept { return true; }
  static const CpuExpertKernel* kernel() noexcept { return &Entry::kernel(); }

  /// \brief A layer's params bytes from the int64 list Python passes; throws std::invalid_argument for a list the
  /// kernel's params cannot take.
  static std::vector<std::byte> params(std::span<const int64_t> values) { return Entry::params(values); }
};

/// \brief The tag of an UnsupportedCpuKernel whose format is not named (HostExports' default binding).
struct UnnamedFormat {
  static constexpr char value[] = "unnamed";
};

/// \brief The stub of a format with no CPU expert kernel.
template <class Name>
class UnsupportedCpuKernel final : public CpuExpertKernel {
 public:
  static constexpr bool usable() noexcept { return false; }
  static constexpr std::string_view kFormat = Name::value;

  /// The stub's address, in place of a library's: what a harness that must hold some CpuExpertKernel passes.
  static const CpuExpertKernel* kernel() noexcept {
    static const UnsupportedCpuKernel stub;
    return &stub;
  }

  using CpuExpertKernel::forward;

  const char* name() const noexcept override { return Name::value; }
  ExpertLayer make_layer(const ExpertLayer&, std::span<const std::byte>) const override { unsupported("make_layer"); }
  int32_t max_routes() const noexcept override { return 0; }
  int32_t max_rows() const noexcept override { return 0; }
  void check(const ExpertLayer&, const ForwardCall&) const override { unsupported("check"); }
  void forward(const ExpertLayer&, const ForwardCall&, Team&) const override { unsupported("forward"); }
  int32_t warm(const uint32_t*, uint32_t, int64_t) const override { unsupported("warm"); }

 private:
  [[noreturn]] static void unsupported(const char* call) noexcept {
    std::fprintf(
        stderr, "%s CPU experts: %s called, but this format has no CPU expert kernel (UnsupportedCpuKernel)\n",
        Name::value, call);
    std::fflush(stderr);
    std::abort();
  }
};

}  // namespace sglang::cpu_experts
