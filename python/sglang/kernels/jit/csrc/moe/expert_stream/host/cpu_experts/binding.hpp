// How an expert format binds to a CPU expert kernel (format.h's `using CpuKernel`).
//
// A format's CPU kernel is a CpuExpertKernel (kernel.hpp) that lives in its own library with its own build flags and
// ABI (EXL3's torch extension, NVFP4's tvm-ffi library); ExpertForward<Quant> has internal linkage there, so the host
// cannot name that type. Python hands the host the kernel's address (trait.py's kernel_address()), and the host calls
// through the vtable. The two bindings below say, at compile time, which of the two worlds a format is in, without
// touching that address-passing path. Each takes a tag type with a `static constexpr char value[]` (not a string
// template argument, so nvcc and older host compilers see only plain types):
//   ExternalCpuKernel<Symbol>   the format has a kernel library; Symbol::value names its address accessor. usable() is
//                               true and kernel() is null: the address arrives from Python at enable_cpu_experts.
//   UnsupportedCpuKernel<Name>  the format has none; Name::value is the format's name. usable() is false, so the engine
//                               offers no CPU lanes and refuses a request for them at startup; kernel() is a stand-in
//                               CpuExpertKernel whose make_layer, check, forward and warm abort naming the format, so a
//                               bug that reaches it is loud.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "kernel.hpp"

namespace sglang::cpu_experts {

/// \brief The tag of an ExternalCpuKernel whose accessor is not named.
struct UnnamedKernelSymbol {
  static constexpr char value[] = "caller-supplied";
};

/// \brief A format whose CPU kernel is a separate library; the host receives its address from Python.
template <class Symbol = UnnamedKernelSymbol>
struct ExternalCpuKernel {
  static constexpr bool usable() noexcept { return true; }
  static constexpr const CpuExpertKernel* kernel() noexcept { return nullptr; }
  static constexpr std::string_view kAddressSymbol = Symbol::value;
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
