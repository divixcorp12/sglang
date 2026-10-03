#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define NVFP4_NOEXCEPT noexcept
#else
#define NVFP4_NOEXCEPT
#endif

// ABI v1. Packed row-major E2M1 weights (low nibble first), GPU-ready
// 128x4-swizzled E4M3 scales. Slab strides are BYTES per host slot.
// slabs: w13, w2, sf13, sf2, gate_alpha, down_alpha, optional up_alpha.
// Alphas are FP32 scalars per slot. A null up_alpha shares gate_alpha.
// inv_input_scale cancels activation scales folded into GPU GEMM alphas;
// use 1 for weight-only alphas. Registration stores views, never repacks.
typedef struct SglangNvfp4CpuLayer {
    uint32_t abi_version;
    int32_t capacity, hidden, intermediate;
    // 0: [gate,up], 1: [up,gate], 2: alternating 64-row [up,gate] chunks.
    int32_t w13_layout;
    // 0: ordinary SiLU(gate)*up, with optional pre-SiLU gate/up clamp.
    // Other activation conventions must get a separate ABI value.
    int32_t activation;
    float act_limit; // 0 disables clamp; gate=min(gate,L), up=clamp(up,-L,L)
    float inv_input_scale13, inv_input_scale2;
    const void* slabs[7];
    uint64_t slot_bytes[7];
} SglangNvfp4CpuLayer;

// Status: 0 success, 1 internal error, 2 invalid arguments, 3 concurrent forward.
int sglang_nvfp4_cpu_experts_register_slabs(const SglangNvfp4CpuLayer*, int64_t* handle) NVFP4_NOEXCEPT;
int sglang_nvfp4_cpu_experts_free_layer(int64_t handle) NVFP4_NOEXCEPT;
// Same CpuExpertForward signature as expert_stream/host/cpu_experts.h.
// x is FP16[hidden]; GGML Q8_0 is used internally for input and SwiGLU.
// Routing weights and out are FP32. -1 slots are skipped.
int sglang_nvfp4_cpu_experts_forward(int64_t layer, const void* x,
    const int32_t* slots, const float* weights, int32_t k, float* out,
    int32_t threads, int32_t accumulate) NVFP4_NOEXCEPT;
// Configure once before first forward. Worker 0 is the calling engine thread.
int sglang_nvfp4_cpu_experts_set_cores(const int32_t* cores, int32_t n) NVFP4_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef NVFP4_NOEXCEPT
