#pragma once
#include "../../expert_stream/format.h"
#include "../../gptq_int3/entry.h"
#include "row_layout.h"
namespace sglang::formats::gptq_int3 {
struct Format {
  static constexpr std::string_view kName = "gptq_int3";
  using Layout = ::sglang::gptq_int3::RowLayout;
  using CpuKernel = ::sglang::gptq_int3::GptqInt3CpuKernel;
  using Staging = ::sglang::expert_stream::HalfStaging;
};
static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);
}  // namespace sglang::formats::gptq_int3
