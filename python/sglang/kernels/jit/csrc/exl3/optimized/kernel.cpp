// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
// The arithmetic is math.hpp and each tier's math_scalar.hpp, math_avx2.hpp and math_avx512.hpp; the forward is
// forward_plan.hpp (with shapes.hpp). This file defines the kernel type Exl3CpuKernel (entry.h): the quant's
// operations (ExpertForward<Exl3Quant>), its params, plans and options.
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
namespace {
using Forward = ::sglang::cpu_experts::ExpertForward<Exl3Quant>;
}  // namespace

const char* Exl3CpuKernel::name() noexcept { return Forward::name(); }

Exl3CpuKernel::Params Exl3CpuKernel::params(std::span<const int64_t> values)
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
    return {static_cast<int16_t>(values[0]), static_cast<int16_t>(values[1]), static_cast<int16_t>(values[2]),
            static_cast<int16_t>(values[3]), static_cast<int32_t>(values[4])};
}

Exl3CpuKernel::Layer Exl3CpuKernel::make_layer(const Layer& shape) { return Forward::make_layer(shape); }
int32_t Exl3CpuKernel::max_routes() noexcept { return Forward::max_routes(); }
int32_t Exl3CpuKernel::max_rows() noexcept { return Forward::max_rows(); }
void Exl3CpuKernel::check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call) { Forward::check(layer, call); }

void Exl3CpuKernel::forward(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call, ::sglang::cpu_experts::Team& team)
{
    Forward::forward(layer, call, team);
}

int32_t Exl3CpuKernel::warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    return Forward::warm(word, seen, deadline_ns);
}

std::string Exl3CpuKernel::isa() { return ::sglang::cpu_experts::isa_name(Forward::isa()); }

SglangExl3CpuPlanCalls exl3_cpu_plan_calls()
{
    return {g_plan_calls[0].load(std::memory_order_relaxed), g_plan_calls[1].load(std::memory_order_relaxed)};
}

std::vector<int64_t> Exl3CpuKernel::plan_calls()
{
    const SglangExl3CpuPlanCalls calls = exl3_cpu_plan_calls();
    return {calls.dsv41, calls.generic};
}

std::vector<SglangExl3CpuPlanInfo> exl3_cpu_plans() { return plan_infos(); }

std::string Exl3CpuKernel::plans_json()
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

int64_t Exl3CpuKernel::chunk_m() { return CHUNK_M; }

void Exl3CpuKernel::configure(
    std::string worker_trace_prefix, int64_t worker_trace_capacity, int64_t worker_trace_min_us,
    std::string scratch_trace_prefix, std::string trace_gate_path)
{
    worker_trace::set_options({std::move(worker_trace_prefix), worker_trace_capacity, worker_trace_min_us,
                               std::move(scratch_trace_prefix), std::move(trace_gate_path)});
}
}  // namespace sglang::exl3_cpu
