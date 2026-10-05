// The NVFP4 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the tvm-ffi export
// nvfp4_cpu_kernel_address hands to Python, and the one the benches link.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstdint>

// A layer's params: make_layer's `params` bytes are this struct, as the library's tvm-ffi export nvfp4_cpu_params
// packs them, so Python never restates its layout. Packed row-major E2M1 weights (low nibble first), GPU-ready
// 128x4-swizzled E4M3 scales. Slab strides are BYTES per host slot. slabs: w13, w2, sf13, sf2, gate_alpha,
// down_alpha, optional up_alpha (null shares gate_alpha). Alphas are FP32 scalars per slot. activation 0 is ordinary
// SiLU(gate)*up; act_limit L > 0 clamps gate=min(gate,L), up=clamp(up,-L,L) before SiLU. A layer stores views, never
// repacks.
struct SglangNvfp4CpuParams {
    // 0: [gate,up], 1: [up,gate], 2: alternating 64-row [up,gate] chunks.
    int32_t w13_layout;
    // Cancel activation scales folded into GPU GEMM alphas; use 1 for weight-only alphas.
    float inv_input_scale13, inv_input_scale2;
};

namespace sglang::nvfp4_cpu {
// Hidden: another library defining it in the same process (the library and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& nvfp4_cpu_kernel();
}  // namespace sglang::nvfp4_cpu
