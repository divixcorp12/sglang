// The EXL3 CPU expert kernel's torch ops: sglang_exl3_cpu::kernel_address, how Python hands the kernel to the
// expert-stream host (enable_cpu_experts) and to the host's test exports, and sglang_exl3_cpu::params, a layer's
// params as make_layer reads them (uint8 bytes of SglangExl3CpuParams), sglang_exl3_cpu::plans, the compiled plans a
// layer's params may name (JSON; call it after configure), sglang_exl3_cpu::plan_calls, the forwards each
// plan took ([dsv41, generic]), sglang_exl3_cpu::chunk_m, the build's CHUNK_M, and sglang_exl3_cpu::configure, the
// kernel's options (kernel.h's Exl3CpuOptions), which Python calls before kernel_address. Built into the optimized
// extension only (quantization/exl3/ext.py), not into benches or the standalone build.
#include "kernel.h"
#include <ATen/ATen.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <torch/library.h>
#include <vector>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::exl3_cpu::exl3_cpu_kernel()); }

void configure(std::string worker_trace_prefix, int64_t worker_trace_capacity, int64_t worker_trace_min_us,
               std::string scratch_trace_prefix, std::string trace_gate_path)
{
    ::sglang::exl3_cpu::exl3_cpu_configure({std::move(worker_trace_prefix), worker_trace_capacity,
                                            worker_trace_min_us, std::move(scratch_trace_prefix),
                                            std::move(trace_gate_path)});
}

at::Tensor params(int64_t bits_w13, int64_t bits_w2, int64_t swizzled, int64_t row_weighted, int64_t plan)
{
    // Refused here rather than narrowed into a value make_layer would accept.
    for (const int64_t v : {bits_w13, bits_w2, swizzled, row_weighted})
        TORCH_CHECK(v >= INT16_MIN && v <= INT16_MAX, "sglang_exl3_cpu::params: ", v, " does not fit int16");
    TORCH_CHECK(plan >= INT32_MIN && plan <= INT32_MAX, "sglang_exl3_cpu::params: plan ", plan, " does not fit int32");
    const SglangExl3CpuParams p{static_cast<int16_t>(bits_w13), static_cast<int16_t>(bits_w2),
                                static_cast<int16_t>(swizzled), static_cast<int16_t>(row_weighted),
                                static_cast<int32_t>(plan)};
    at::Tensor out = at::empty({static_cast<int64_t>(sizeof(p))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(out.data_ptr(), &p, sizeof(p));
    return out;
}

// The compiled plans (kernel.h's SglangExl3CpuPlanInfo) as a JSON array of objects, the generic plan first.
std::string plans()
{
    std::string out = "[";
    char buf[512];
    for (const SglangExl3CpuPlanInfo& p : ::sglang::exl3_cpu::exl3_cpu_plans()) {
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

std::vector<int64_t> plan_calls()
{
    const SglangExl3CpuPlanCalls c = ::sglang::exl3_cpu::exl3_cpu_plan_calls();
    return {c.dsv41, c.generic};
}

int64_t chunk_m() { return ::sglang::exl3_cpu::exl3_cpu_chunk_m(); }
}  // namespace

TORCH_LIBRARY(sglang_exl3_cpu, m)
{
    m.def("kernel_address() -> int", &kernel_address);
    m.def("params(int bits_w13, int bits_w2, int swizzled, int row_weighted, int plan) -> Tensor", &params);
    m.def("plans() -> str", &plans);
    m.def("plan_calls() -> int[]", &plan_calls);
    m.def("chunk_m() -> int", &chunk_m);
    m.def("configure(str worker_trace_prefix, int worker_trace_capacity, int worker_trace_min_us, "
          "str scratch_trace_prefix, str trace_gate_path) -> ()",
          &configure);
}
