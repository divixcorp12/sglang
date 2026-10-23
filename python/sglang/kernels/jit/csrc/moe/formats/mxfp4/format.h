// The MXFP4 expert format as the engine's C++ sees it (expert_stream/format.h). It has no CPU expert kernel yet: every
// routed expert is computed on the GPU, and UnsupportedCpuKernel makes a request for CPU lanes fail at startup.
#pragma once

#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::mxfp4 {

struct NoCpuKernelName {
  static constexpr char value[] = "mxfp4";
};

struct Format {
  static constexpr std::string_view kName = "mxfp4";
  using Layout = ::sglang::mxfp4::Mxfp4RowLayout;
  using CpuKernel = ::sglang::cpu_experts::UnsupportedCpuKernel<NoCpuKernelName>;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);
static_assert(!Format::CpuKernel::usable());

}  // namespace sglang::formats::mxfp4
