#pragma once
#include <stdint.h>
#include "../../../../../kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"

#ifdef __cplusplus
extern "C" {
#define NVFP4_NOEXCEPT noexcept
#else
#define NVFP4_NOEXCEPT
#endif

// A layer registers as an SglangCpuExpertsLayer (cpu_experts_abi.h) with slab_count 7 and params pointing at this.
// Packed row-major E2M1 weights (low nibble first), GPU-ready 128x4-swizzled E4M3 scales. Slab strides are BYTES per
// host slot. slabs: w13, w2, sf13, sf2, gate_alpha, down_alpha, optional up_alpha (null shares gate_alpha). Alphas are
// FP32 scalars per slot. activation 0 is ordinary SiLU(gate)*up; act_limit L > 0 clamps gate=min(gate,L),
// up=clamp(up,-L,L) before SiLU. Registration stores views, never repacks.
typedef struct SglangNvfp4CpuParams {
    // 0: [gate,up], 1: [up,gate], 2: alternating 64-row [up,gate] chunks.
    int32_t w13_layout;
    // Cancel activation scales folded into GPU GEMM alphas; use 1 for weight-only alphas.
    float inv_input_scale13, inv_input_scale2;
} SglangNvfp4CpuParams;

// Status: 0 success, 1 internal error, 2 invalid arguments, 3 concurrent use.
int sglang_nvfp4_cpu_experts_register_layer(const SglangCpuExpertsLayer*, int64_t* handle) NVFP4_NOEXCEPT;
int sglang_nvfp4_cpu_experts_free_layer(int64_t handle) NVFP4_NOEXCEPT;
// CpuExpertForward (expert_stream/host/cpu_experts.h): SglangCpuExpertsForward's rows, x FP16 [rows][hidden], out FP32
// [rows][hidden]; rows at most 65536, k at most 8. GGML Q8_0 is used internally for input and SwiGLU. Tokens sharing a
// slot share each weight row's decode; every row's output is bitwise its own one-row call's. A call is all or nothing:
// when Q8_0 cannot represent any row's input or intermediate it returns 2 and leaves every row of out untouched.
int sglang_nvfp4_cpu_experts_forward(const SglangCpuExpertsForward* call) NVFP4_NOEXCEPT;
// Holds `threads` pinned workers in register-only work at the forward's vector width until *word != seen or
// CLOCK_MONOTONIC reaches deadline_ns (cpu_experts_common/keep_warm.hpp).
int sglang_nvfp4_cpu_experts_keep_warm(int32_t threads, const uint32_t* word, uint32_t seen,
                                       int64_t deadline_ns) NVFP4_NOEXCEPT;
// Configure before the first forward or keep-warm (refused with 2 after it). Worker i runs on cores[i]; worker 0 is
// the calling engine thread.
int sglang_nvfp4_cpu_experts_set_cores(const int32_t* cores, int32_t n) NVFP4_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef NVFP4_NOEXCEPT
