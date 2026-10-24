// ExpertFormatTraits: every fact about an expert quant format the engine's C++ needs, in one type.
//
// The transport (reader, RAM tier, copy engine, leases, drip) sees only bytes and a row layout; a format supplies:
//   Layout      the streamed row's tensor names and masks (row_layout.h's ExpertRowLayout)
//   CpuKernel   the format's CPU expert kernel, a concrete type compiled into the host module (the interface is in
//               host/cpu_experts/binding.hpp), or UnsupportedCpuKernel<Name>, no kernel. usable() says which, so
//               policy never routes work to a stub.
//   Staging     the dtype of the activations the host stages for CPU experts and the draft channel
//   kName       the format's name, the directory name -DSGLANG_EXPERT_FORMAT selects it by. A real format's Layout
//               has the same kName, the prefix of every service error; a test format (kTestFormat) borrows one.
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
  typename F::CpuKernel::Layer;
  { F::Staging::kBytes } -> std::convertible_to<int>;
} && ExpertRowLayout<typename F::Layout> &&
    (std::string_view(F::kName) == std::string_view(F::Layout::kName) || requires { requires F::kTestFormat; });

}  // namespace sglang::expert_stream
