// The NVFP4 CPU expert kernel (kernel.cpp): the CpuExpertKernel the expert-stream host module compiles in and calls
// (formats/nvfp4/format.h), and the one the benches link. entry.h declares what the host calls.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/team.hpp"
#include "entry.h"
#include <cstdint>
#include <string>

// A layer's params: make_layer's `params` bytes are this struct, as nvfp4_cpu_param_bytes builds it from the int64 list
// Python passes, so Python never restates its layout. Packed row-major E2M1 weights (low nibble first), GPU-ready
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
