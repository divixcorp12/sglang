// ExpertFormatTraits: every fact about an expert quant format the engine's C++ needs, in one type.
//
// The transport (reader, RAM tier, copy engine, leases, drip) sees only bytes and a row layout; a format supplies:
//   Layout      the streamed row's tensor names and masks (row_layout.h's ExpertRowLayout)
//   CpuKernel   how the format reaches a CPU expert kernel (host/cpu_experts/binding.hpp): ExternalCpuKernel<Symbol>,
//               a kernel library whose address Python passes to the host (the existing set_cpu_layer path), or
//               UnsupportedCpuKernel<Name>, no kernel. usable() says which, so policy never routes work to a stub.
//   Staging     the dtype of the activations the host stages for CPU experts and the draft channel
//   kName       the format's name; it is also Layout::kName, the prefix of every service error
//
// Concrete traits live in ../formats/<name>/format.h; the engine's translation units pick one with
// -DSGLANG_EXPERT_FORMAT=<name> (engine_host.cpp, engine_device.cuh).
#pragma once

#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "host/cpu_experts/binding.hpp"
#include "row_layout.h"

namespace sglang::expert_stream {

/// \brief fp16 activation staging (today's every format).
struct HalfStaging {
  static constexpr std::string_view kName = "half";
  static constexpr int kBytes = 2;
};

template <class F>
concept ExpertFormatTraits = requires {
  typename F::Layout;
  typename F::CpuKernel;
  typename F::Staging;
  { F::kName } -> std::convertible_to<std::string_view>;
  { F::CpuKernel::usable() } -> std::same_as<bool>;
  { F::CpuKernel::kernel() } -> std::convertible_to<const cpu_experts::CpuExpertKernel*>;
  { F::Staging::kBytes } -> std::convertible_to<int>;
} && ExpertRowLayout<typename F::Layout> && (std::string_view(F::kName) == std::string_view(F::Layout::kName));

}  // namespace sglang::expert_stream
