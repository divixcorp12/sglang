// The EXL3 CPU expert kernel (kernel.cpp): the CpuExpertKernel the expert-stream host module compiles in and calls
// (formats/exl3/format.h), and the one the benches link. Its layers are slab layers only (make_layer). No torch.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/team.hpp"
#include "entry.h"
#include <cstdint>
#include <vector>

// Forwards since load by the plan they took (forward_plan.hpp's run_plan): dsv41, ForwardPlan<Dsv41Shape, Bw>;
// generic, ForwardPlan<GenericShape, *>. Read by tests and benches to see which plan a call took.
struct SglangExl3CpuPlanCalls {
    int64_t dsv41;
    int64_t generic;
};

// The forward plans this library compiles (forward_plan.hpp's run_plan), by the id a layer's params name. The generic
// plan runs any layer make_layer accepts; a fixed plan runs only the layer facts its Shape (shapes.hpp) fixes.
enum SglangExl3CpuPlan : int32_t {
    kSglangExl3CpuPlanGeneric = 0,  // ForwardPlan<GenericShape, *>
    kSglangExl3CpuPlanDsv41 = 1,    // ForwardPlan<Dsv41Shape, Bw>
};

// A layer's params: make_layer's `params` bytes are this struct, as exl3_cpu_params builds it from the int64 list Python
// passes, so Python never restates its layout. slabs: w13_trellis, w13_suh, w13_svh,
// w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts s * slot_bytes[i]
// in, w13 rows holding gate then up. hidden and intermediate are multiples of 128 in [128, 8192]; activation 0 (gated SiLU, clamped
// at act_limit when it is nonzero). A layer stores views. At most kMaxParamBytes (16, cpu_experts/kernel.hpp).
struct SglangExl3CpuParams {
    int16_t bits_w13;      // gate and up trellis bits per weight, 1..8
    int16_t bits_w2;       // down trellis bits per weight, 1..8
    int16_t swizzled;      // 0 or 1: band-contiguous trellis layout; an 8-bit projection is never swizzled
    int16_t row_weighted;  // 0 or 1: the layer's gate/up and down tiles are split by chunk row count
                           // (tile_assignment.hpp), the same arithmetic in another worker's hands
    int32_t plan;          // a SglangExl3CpuPlan; make_layer refuses one that does not accept the layer
};

// One compiled plan as exl3_cpu_plans lists it (and the host export expert_stream_cpu_kernel_plans as JSON). A fixed
// plan accepts exactly the layer facts below; the generic plan (fixed false, facts 0) accepts any layer. available: the build conditions hold (the activation layout
// the plan's kernels assume) and min_isa is within the kernel's tier; make_layer refuses an unavailable plan.
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
// The forwards since load by the plan they took (entry.h lists them as [dsv41, generic]).
__attribute__((visibility("hidden"))) SglangExl3CpuPlanCalls exl3_cpu_plan_calls();
// The compiled plans, the generic plan first; `available` follows the build's -march.
__attribute__((visibility("hidden"))) std::vector<SglangExl3CpuPlanInfo> exl3_cpu_plans();
// The first available fixed plan that accepts these layer facts, else the generic plan: for native harnesses (the
// benches); Python chooses from the plans' list itself.
__attribute__((visibility("hidden"))) int32_t exl3_cpu_match_plan(
    int32_t hidden, int32_t intermediate, int32_t bits_w13, int32_t bits_w2, int32_t swizzled, float act_limit);
}  // namespace sglang::exl3_cpu
