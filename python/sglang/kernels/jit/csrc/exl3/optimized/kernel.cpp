// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
// The arithmetic is math.hpp and each tier's math_scalar.hpp, math_avx2.hpp and math_avx512.hpp; the forward is
// forward_plan.hpp (with shapes.hpp). This file holds the kernel's accessor and its params, plans and options.
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "kernel.h"
#include "quant.hpp"

#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Last: the definition order the kernels were validated in (bit-exact per tier); another order changes what GCC
// inlines and clones.
#include "forward_plan.hpp"

namespace sglang::exl3_cpu {
const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel()
{
    static const ::sglang::cpu_experts::ExpertForward<Exl3Quant> kernel{};
    return kernel;
}

const char* exl3_cpu_isa() { return ::sglang::cpu_experts::isa_name(::sglang::cpu_experts::ExpertForward<Exl3Quant>::isa()); }

std::vector<std::byte> exl3_cpu_param_bytes(std::span<const int64_t> values)
{
    if (values.size() != 5)
        throw std::invalid_argument("EXL3 CPU params are [bits_w13, bits_w2, swizzled, row_weighted, plan], got "
                                    + std::to_string(values.size()) + " values");
    // Refused here rather than narrowed into a value make_layer would accept.
    for (size_t i = 0; i < 4; ++i)
        if (values[i] < INT16_MIN || values[i] > INT16_MAX)
            throw std::invalid_argument("EXL3 CPU params: " + std::to_string(values[i]) + " does not fit int16");
    if (values[4] < INT32_MIN || values[4] > INT32_MAX)
        throw std::invalid_argument("EXL3 CPU params: plan " + std::to_string(values[4]) + " does not fit int32");
    const SglangExl3CpuParams params{static_cast<int16_t>(values[0]), static_cast<int16_t>(values[1]),
                                     static_cast<int16_t>(values[2]), static_cast<int16_t>(values[3]),
                                     static_cast<int32_t>(values[4])};
    const auto* bytes = reinterpret_cast<const std::byte*>(&params);
    return {bytes, bytes + sizeof(params)};
}

SglangExl3CpuPlanCalls exl3_cpu_plan_calls()
{
    return {g_plan_calls[0].load(std::memory_order_relaxed), g_plan_calls[1].load(std::memory_order_relaxed)};
}

std::vector<int64_t> exl3_cpu_plan_call_counts()
{
    const SglangExl3CpuPlanCalls calls = exl3_cpu_plan_calls();
    return {calls.dsv41, calls.generic};
}

std::vector<SglangExl3CpuPlanInfo> exl3_cpu_plans() { return plan_infos(); }

std::string exl3_cpu_plans_json()
{
    std::string out = "[";
    char buf[512];
    for (const SglangExl3CpuPlanInfo& p : plan_infos()) {
        std::snprintf(buf, sizeof(buf),
                      "%s{\"id\": %d, \"name\": \"%s\", \"fixed\": %s, \"hidden\": %d, \"intermediate\": %d, "
                      "\"bits_w13\": %d, \"bits_w2\": %d, \"swizzled\": %d, \"act_limit\": %.9g, \"min_isa\": \"%s\", "
                      "\"available\": %s}",
                      out.size() > 1 ? ", " : "", p.id, p.name, p.fixed ? "true" : "false", p.hidden, p.intermediate,
                      p.bits_w13, p.bits_w2, p.swizzled, static_cast<double>(p.act_limit), p.min_isa,
                      p.available ? "true" : "false");
        out += buf;
    }
    return out + "]";
}

int32_t exl3_cpu_match_plan(
    int32_t hidden, int32_t intermediate, int32_t bits_w13, int32_t bits_w2, int32_t swizzled, float act_limit)
{
    for (const SglangExl3CpuPlanInfo& plan : plan_infos())
        if (plan.fixed && plan.available && plan.hidden == hidden && plan.intermediate == intermediate
            && plan.bits_w13 == bits_w13 && plan.bits_w2 == bits_w2 && plan.swizzled == swizzled
            && plan.act_limit == act_limit)
            return plan.id;
    return kSglangExl3CpuPlanGeneric;
}

int32_t exl3_cpu_chunk_m() { return CHUNK_M; }

void exl3_cpu_configure(const Exl3CpuOptions& options)
{
    worker_trace::set_options({options.worker_trace_prefix, options.worker_trace_capacity, options.worker_trace_min_us,
                               options.scratch_trace_prefix, options.trace_gate_path});
}
}  // namespace sglang::exl3_cpu
