// The EXL3 expert format as the engine's C++ sees it (expert_stream/format.h): its row layout, its CPU kernel and its
// staging dtype.
#pragma once

#include "../../../exl3/optimized/entry.h"
#include "../../expert_stream/format.h"
#include "row_layout.h"

namespace sglang::formats::exl3 {

// The kernel csrc/exl3/optimized/kernel.cpp, compiled into this module with the kernel flags.
struct KernelEntry {
  static const ::sglang::cpu_experts::CpuExpertKernel& kernel() { return ::sglang::exl3_cpu::exl3_cpu_kernel(); }
  static std::vector<std::byte> params(std::span<const int64_t> values) {
    return ::sglang::exl3_cpu::exl3_cpu_param_bytes(values);
  }
  static std::string plans_json() { return ::sglang::exl3_cpu::exl3_cpu_plans_json(); }
  static std::vector<int64_t> plan_calls() { return ::sglang::exl3_cpu::exl3_cpu_plan_call_counts(); }
  static int64_t chunk_m() { return ::sglang::exl3_cpu::exl3_cpu_chunk_m(); }
  static std::string isa() { return ::sglang::exl3_cpu::exl3_cpu_isa(); }
  static void configure(
      std::string worker_trace_prefix, int64_t capacity, int64_t min_us, std::string scratch_prefix,
      std::string gate_path) {
    ::sglang::exl3_cpu::exl3_cpu_configure(
        {std::move(worker_trace_prefix), capacity, min_us, std::move(scratch_prefix), std::move(gate_path)});
  }
};

struct Format {
  static constexpr std::string_view kName = "exl3";
  using Layout = ::sglang::exl3::Exl3RowLayout;
  using CpuKernel = ::sglang::cpu_experts::ExternalCpuKernel<KernelEntry>;
  using Staging = ::sglang::expert_stream::HalfStaging;
};

static_assert(::sglang::expert_stream::ExpertFormatTraits<Format>);

}  // namespace sglang::formats::exl3
