// The NVFP4 expert format as the engine's C++ sees it (expert_stream/format.h).
#pragma once

#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::nvfp4 {

struct KernelSymbol {
  static constexpr char value[] = "nvfp4_cpu_kernel_address";
};

struct Format {
  static constexpr std::string_view kName = "nvfp4";
  using Layout = ::sglang::nvfp4::Nvfp4RowLayout;
  // The tvm-ffi library csrc/nvfp4/optimized exports the kernel's address as nvfp4_cpu_kernel_address; Python passes it
  // to the host at enable_cpu_experts, the same path as EXL3's.
  using CpuKernel = ::sglang::cpu_experts::ExternalCpuKernel<KernelSymbol>;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::nvfp4
