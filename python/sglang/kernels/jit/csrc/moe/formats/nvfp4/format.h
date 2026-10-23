// The NVFP4 expert format as the engine's C++ sees it (expert_stream/format.h).
#pragma once

#include "../../../nvfp4/optimized/entry.h"
#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::nvfp4 {

struct Format {
  static constexpr std::string_view kName = "nvfp4";
  using Layout = ::sglang::nvfp4::Nvfp4RowLayout;
  // csrc/nvfp4/optimized/kernel.cpp (with the vendored upstream/nvfp4.c, compiled as C), compiled into this module with
  // the kernel flags. It has no options of its own.
  using CpuKernel = ::sglang::nvfp4_cpu::Nvfp4CpuKernel;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::nvfp4
