// The EXL3 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the torch extension's
// sglang_exl3_cpu::kernel_address op hands to Python, and the one the benches link. Its layers are slab layers only
// (make_layer); upstream's per-expert tensor API (exl3_moe_cpu_make_layer and its forwards) refuses.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/team.hpp"
#include <cstdint>

// Forwards since load by the plan they took (forward_plan.hpp's run_plan): dsv41, ForwardPlan<Dsv41Shape, Bw>;
// generic, ForwardPlan<GenericShape, *>. Read by tests and benches to see which plan a call took.
struct SglangExl3CpuPlanCalls {
    int64_t dsv41;
    int64_t generic;
};

// A layer's params: make_layer's `params` bytes are this struct, as the extension's torch op
// sglang_exl3_cpu::params packs them, so Python never restates its layout. slabs: w13_trellis, w13_suh, w13_svh,
// w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts s * slot_bytes[i]
// in, w13 rows holding gate then up. hidden and intermediate are multiples of 128 in [128, 8192]; activation 0 (gated SiLU, clamped
// at act_limit when it is nonzero). A layer stores views.
struct SglangExl3CpuParams {
    int32_t bits;          // trellis bits per weight, 1..8
    int32_t swizzled;      // 0 or 1: band-contiguous trellis layout; make_layer stores 0 at 8 bits, never swizzled
    int32_t row_weighted;  // 0 or 1: the layer's gate/up and down tiles are split by chunk row count
                           // (tile_assignment.hpp), the same arithmetic in another worker's hands
};

namespace sglang::exl3_cpu {
// Hidden: another library defining it in the same process (the extension and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
__attribute__((visibility("hidden"))) SglangExl3CpuPlanCalls exl3_cpu_plan_calls();
// The most tokens a chunk holds in this build (math.hpp's CHUNK_M = MAX_M / ACT_ROWS).
__attribute__((visibility("hidden"))) int32_t exl3_cpu_chunk_m();
}  // namespace sglang::exl3_cpu
