// The EXL3 CPU expert kernel (kernel.cpp): the kernel type Exl3CpuKernel (entry.h) the expert-stream host module
// compiles in and calls (formats/exl3/format.h), and the one the benches link, with the kernel's own typed plan
// interface below. Its layers are slab layers only (make_layer). No torch.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/single_job.hpp"
#include "entry.h"
#include <cstdint>
#include <vector>

// The forward plans this library compiles (forward_plan.hpp's run_plan), by the id a layer's params name. The generic
// plan runs any layer make_layer accepts; a fixed plan runs only the layer facts its Shape (shapes.hpp) fixes.
enum SglangExl3CpuPlan : int32_t {
    kSglangExl3CpuPlanGeneric = 0,  // ForwardPlan<GenericShape>
    kSglangExl3CpuPlanDsv41 = 1,    // ForwardPlan<Dsv41Shape>
};

// One compiled plan as exl3_cpu_plans lists it (and the host export expert_stream_cpu_kernel_plans as JSON). A fixed
// plan accepts exactly the layer facts below; the generic plan (fixed false, facts 0) accepts any layer. available:
// min_isa is within the build's tier; make_layer refuses an unavailable plan.
struct SglangExl3CpuPlanInfo {
    int32_t id;
    const char* name;
    bool fixed;
    int32_t hidden, intermediate, bits_w13, bits_w2, swizzled;
    float act_limit;
    const char* min_isa;
    bool available;
};

namespace sglang::exl3_cpu {
// The compiled plans, the generic plan first; `available` follows the build's -march.
__attribute__((visibility("hidden"))) std::vector<SglangExl3CpuPlanInfo> exl3_cpu_plans();
// The first available fixed plan that accepts these layer facts, else the generic plan: for native harnesses (the
// benches); Python chooses from the plans' list itself.
__attribute__((visibility("hidden"))) int32_t exl3_cpu_match_plan(
    int32_t hidden, int32_t intermediate, int32_t bits_w13, int32_t bits_w2, int32_t swizzled, float act_limit);
}  // namespace sglang::exl3_cpu
