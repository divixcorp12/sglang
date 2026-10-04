#pragma once
#include <stdint.h>
#include "../../moe/expert_stream/host/cpu_experts_abi.h"

#ifdef __cplusplus
extern "C" {
#define EXL3_CPU_NOEXCEPT noexcept
#else
#define EXL3_CPU_NOEXCEPT
#endif

// A layer registers as an SglangCpuExpertsLayer (cpu_experts_abi.h) with slab_count 6 and params pointing at this.
// slabs: w13_trellis, w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order);
// slot s of each starts s * slot_bytes[i] in, w13 rows holding gate then up. hidden and intermediate are multiples of
// 128 in [128, 8192]; activation 0 (gated SiLU, clamped at act_limit when it is nonzero). Registration stores views.
typedef struct SglangExl3CpuParams {
    int32_t bits;      // trellis bits per weight, 1..8
    int32_t swizzled;  // 0 or 1: band-contiguous trellis layout; ignored at 8 bits, which is never swizzled
} SglangExl3CpuParams;

// Status: 0 success, 1 internal error, 2 invalid arguments, 3 a free_layer while a forward runs.
int sglang_exl3_cpu_experts_register_layer(const SglangCpuExpertsLayer*, int64_t* handle) EXL3_CPU_NOEXCEPT;
int sglang_exl3_cpu_experts_free_layer(int64_t handle) EXL3_CPU_NOEXCEPT;
// CpuExpertForward (expert_stream/host/cpu_experts.h): SglangCpuExpertsForward's rows, x FP16 [rows][hidden], out FP32
// [rows][hidden]; rows at most 65536, k at most 32. Routing weights are converted to FP16, preserving the registered
// EXL3 kernel's convention. A failed call (1 or 2) leaves out untouched.
// call->engine names the engine whose cores the workers run on (0: unpinned); an unknown or freed engine, or more
// threads than its cores, is refused (2). Forwards on any engines run at once from different threads.
// A call naming an engine leaves its calling thread (worker 0) and its OpenMP workers pinned to that engine's cores
// after it returns; engine 0 does not unpin them (it only leaves an unpinned thread unpinned).
int sglang_exl3_cpu_experts_forward(const SglangCpuExpertsForward* call) EXL3_CPU_NOEXCEPT;
// Holds `threads` workers of `engine` (0: unpinned), pinned as the forward pins them, in register-only work at the
// forward's vector width until *word != seen or CLOCK_MONOTONIC reaches deadline_ns
// (expert_stream/host/cpu_experts/keep_warm.hpp). 2 as the forward refuses an engine.
int sglang_exl3_cpu_experts_keep_warm(int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen,
                                      int64_t deadline_ns) EXL3_CPU_NOEXCEPT;
// An engine: worker i of each forward and keep-warm naming it runs on cores[i], the calling thread as worker 0. Cores
// must be distinct and in [0, CPU_SETSIZE); they are not checked against the caller's affinity, and a core that cannot
// be pinned fails the call's pin (1). Engines are immutable. Returns 0 and the handle (never 0), 1, or 2.
int sglang_exl3_cpu_experts_engine_create(const int32_t* cores, int32_t n, int64_t* engine) EXL3_CPU_NOEXCEPT;
// Frees `engine`; a call already running on it keeps its cores. Returns 0, or 2 for a handle never created or freed.
int sglang_exl3_cpu_experts_engine_free(int64_t engine) EXL3_CPU_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef EXL3_CPU_NOEXCEPT
