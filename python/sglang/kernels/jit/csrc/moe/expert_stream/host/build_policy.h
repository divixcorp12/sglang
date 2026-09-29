// The compile-time build policy of the expert-stream host transport (plan 2026-09-29-hotpath-zero-overhead, spec
// section 4.1). ProdBuild carries no metrics, trace or fault state on the request path; InstrBuild carries all of it.
// Each is instantiated in its own module: exl3_ram_miss_host.cpp (prod), exl3_ram_miss_host_instr.cpp (instr).
#pragma once

#include <string_view>
#include <type_traits>

namespace sglang::expert_stream {

struct ProdBuild {
  static constexpr bool kMetrics = false;
  static constexpr bool kFaults = false;
  static constexpr std::string_view kName = "prod";
};

struct InstrBuild {
  static constexpr bool kMetrics = true;
  static constexpr bool kFaults = true;
  static constexpr std::string_view kName = "instr";
};

template <class B>
concept BuildPolicy = std::is_same_v<B, ProdBuild> || std::is_same_v<B, InstrBuild>;

}  // namespace sglang::expert_stream
