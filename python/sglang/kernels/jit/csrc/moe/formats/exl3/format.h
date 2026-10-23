// The EXL3 expert format as the engine's C++ sees it (expert_stream/format.h): its row layout, its CPU kernel and its
// staging dtype.
#pragma once

#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::exl3 {

struct KernelSymbol {
  static constexpr char value[] = "sglang_exl3_cpu::kernel_address";
};

struct Format {
  static constexpr std::string_view kName = "exl3";
  using Layout = ::sglang::exl3::Exl3RowLayout;
  // The torch extension csrc/exl3/optimized hands Python the kernel's address (sglang_exl3_cpu::kernel_address); the
  // host receives it at enable_cpu_experts, as before.
  using CpuKernel = ::sglang::cpu_experts::ExternalCpuKernel<KernelSymbol>;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::exl3
