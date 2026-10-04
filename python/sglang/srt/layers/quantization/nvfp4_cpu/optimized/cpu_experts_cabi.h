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

// Status: 0 success, 1 internal error, 2 invalid arguments, 3 a free_layer while a forward runs.
int sglang_nvfp4_cpu_experts_register_layer(const SglangCpuExpertsLayer*, int64_t* handle) NVFP4_NOEXCEPT;
int sglang_nvfp4_cpu_experts_free_layer(int64_t handle) NVFP4_NOEXCEPT;
// CpuExpertForward (expert_stream/host/cpu_experts.h): SglangCpuExpertsForward's rows, x FP16 [rows][hidden], out FP32
// [rows][hidden]; rows at most 65536, k at most 8. GGML Q8_0 is used internally for input and SwiGLU. Tokens sharing a
// slot share each weight row's decode; every row's output is bitwise its own one-row call's. A call is all or nothing:
// when Q8_0 cannot represent any row's input or intermediate it returns 2 and leaves every row of out untouched.
// call->engine names the engine whose cores the workers run on (0: unpinned); an unknown or freed engine, or more
// threads than its cores, is refused (2). Forwards on any engines run at once from different threads.
// A call naming an engine leaves its calling thread (worker 0) and its OpenMP workers pinned to that engine's cores
// after it returns; engine 0 does not unpin them (it only leaves an unpinned thread unpinned).
int sglang_nvfp4_cpu_experts_forward(const SglangCpuExpertsForward* call) NVFP4_NOEXCEPT;
// Holds `threads` workers of `engine` (0: unpinned), pinned as the forward pins them, in register-only work at the
// forward's vector width until *word != seen or CLOCK_MONOTONIC reaches deadline_ns (cpu_experts_common/keep_warm.hpp).
// 2 as the forward refuses an engine.
int sglang_nvfp4_cpu_experts_keep_warm(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen,
                                       int64_t deadline_ns) NVFP4_NOEXCEPT;
// An engine: worker i of each forward and keep-warm naming it runs on cores[i], the calling thread as worker 0. Cores
// must be distinct and in [0, CPU_SETSIZE); they are not checked against the caller's affinity, and a core that cannot
// be pinned fails the call's pin (1). Engines are immutable. Returns 0 and the handle (never 0), 1, or 2.
int sglang_nvfp4_cpu_experts_engine_create(const int32_t* cores, int32_t n, int64_t* engine) NVFP4_NOEXCEPT;
// Frees `engine`; a call already running on it keeps its cores. Returns 0, or 2 for a handle never created or freed.
int sglang_nvfp4_cpu_experts_engine_free(int64_t engine) NVFP4_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef NVFP4_NOEXCEPT
