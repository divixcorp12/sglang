// The EXL3 expert format as the engine's C++ sees it (expert_stream/format.h): its row layout, its CPU kernel and its
// staging dtype.
#pragma once

#include "../../../exl3/optimized/entry.h"
#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::exl3 {

struct Format {
  static constexpr std::string_view kName = "exl3";
  using Layout = ::sglang::exl3::Exl3RowLayout;
  // csrc/exl3/optimized/kernel.cpp, compiled into this module with the kernel flags.
  using CpuKernel = ::sglang::exl3_cpu::Exl3CpuKernel;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::exl3
