// What the expert-stream host module calls in the EXL3 kernel's translation unit (kernel.cpp), in plain std types so
// the host and the device translation units can include it through formats/exl3/format.h without the kernel's headers.
// kernel.cpp defines every function here; kernel.h adds the kernel's own typed interface.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sglang::exl3_cpu {
// The kernel's options from Python's config, before its first forward: the worker trace (SGLANG_EXPERT_STREAM_TRACE's
// cpu_worker and gate_path; worker_trace.hpp), which only a -DEXL3_MOE_CPU_WORKER_TRACE=1 build records.
struct Exl3CpuOptions {
    std::string worker_trace_prefix;
    int64_t worker_trace_capacity = 65536;
    int64_t worker_trace_min_us = 6000;
    std::string scratch_trace_prefix;
    std::string trace_gate_path;
};

// Hidden: the host module's other symbols stay private to its .so, and a standalone build never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
// The tier this build runs at (isa.hpp's isa_name): scalar, avx2, bw, vnni or vbmi.
__attribute__((visibility("hidden"))) const char* exl3_cpu_isa();
// A layer's params bytes (kernel.h's SglangExl3CpuParams) from [bits_w13, bits_w2, swizzled, row_weighted, plan];
// std::invalid_argument for a wrong count or a value its field cannot hold.
__attribute__((visibility("hidden"))) std::vector<std::byte> exl3_cpu_param_bytes(std::span<const int64_t> values);
// The compiled plans (kernel.h's SglangExl3CpuPlanInfo) as a JSON array of objects, the generic plan first; `available`
// follows the build's -march.
__attribute__((visibility("hidden"))) std::string exl3_cpu_plans_json();
// The forwards since load by the plan they took: [dsv41, generic].
__attribute__((visibility("hidden"))) std::vector<int64_t> exl3_cpu_plan_call_counts();
// The most tokens a chunk holds in this build (math.hpp's CHUNK_M = MAX_M / ACT_ROWS).
__attribute__((visibility("hidden"))) int32_t exl3_cpu_chunk_m();
__attribute__((visibility("hidden"))) void exl3_cpu_configure(const Exl3CpuOptions& options);
}  // namespace sglang::exl3_cpu
