// The EXL3 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the torch extension's
// sglang_exl3_cpu::kernel_address op hands to Python, and the one the benches link. Its layers are slab layers only
// (make_layer); upstream's per-expert tensor API (exl3_moe_cpu_make_layer and its forwards) refuses.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstdint>

// A layer's params: make_layer's `params` bytes are this struct, as the extension's torch op
// sglang_exl3_cpu::params packs them, so Python never restates its layout. slabs: w13_trellis, w13_suh, w13_svh,
// w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts s * slot_bytes[i]
// in, w13 rows holding gate then up. hidden and intermediate are multiples of 128 in [128, 8192]; activation 0 (gated SiLU, clamped
// at act_limit when it is nonzero). A layer stores views.
struct SglangExl3CpuParams {
    int32_t bits;      // trellis bits per weight, 1..8
    int32_t swizzled;  // 0 or 1: band-contiguous trellis layout; make_layer stores 0 at 8 bits, never swizzled
};

namespace sglang::exl3_cpu {
// Hidden: another library defining it in the same process (the extension and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
}  // namespace sglang::exl3_cpu
