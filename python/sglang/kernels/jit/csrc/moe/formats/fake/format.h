// The test format `fake` as the engine's C++ sees it (expert_stream/format.h): EXL3's row layout and a fake CPU expert
// kernel (fake_kernel.h) that records its calls, for the tests that drive the host's CPU expert engine without a real
// quant. Built only for tests (expert_stream_transport.FAKE_BUILD); its layout keeps the name "exl3", the prefix of every
// service error the EXL3 tests read.
#pragma once

#include "../../expert_stream/format.h"
#include "../exl3/row_layout.h"
#include "fake_kernel.h"

namespace sglang::formats::fake {

struct Format {
  static constexpr std::string_view kName = "fake";
  static constexpr bool kTestFormat = true;
  using Layout = ::sglang::exl3::Exl3RowLayout;
  using CpuKernel = FakeCpuKernel;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::fake
