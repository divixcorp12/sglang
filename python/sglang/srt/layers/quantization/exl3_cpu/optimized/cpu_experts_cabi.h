#pragma once
#include <stdint.h>
#include "../../../../../kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"

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

// Status: 0 success, 1 internal error, 2 invalid arguments, 3 concurrent use.
int sglang_exl3_cpu_experts_register_layer(const SglangCpuExpertsLayer*, int64_t* handle) EXL3_CPU_NOEXCEPT;
int sglang_exl3_cpu_experts_free_layer(int64_t handle) EXL3_CPU_NOEXCEPT;
// CpuExpertForward (expert_stream/host/cpu_experts.h): SglangCpuExpertsForward's rows, x FP16 [rows][hidden], out FP32
// [rows][hidden]; rows at most 65536, k at most 32. Routing weights are converted to FP16, preserving the registered
// EXL3 kernel's convention. A refused call (2) leaves out untouched.
int sglang_exl3_cpu_experts_forward(const SglangCpuExpertsForward* call) EXL3_CPU_NOEXCEPT;
// Holds `threads` pinned workers in register-only work at the forward's vector width until *word != seen or
// CLOCK_MONOTONIC reaches deadline_ns (cpu_experts_common/keep_warm.hpp).
int sglang_exl3_cpu_experts_keep_warm(int32_t threads, const uint32_t* word, uint32_t seen,
                                      int64_t deadline_ns) EXL3_CPU_NOEXCEPT;
// Configure before the first forward or keep-warm (refused with 2 after it). Worker i runs on cores[i]; worker 0 is
// the calling engine thread.
int sglang_exl3_cpu_experts_set_cores(const int32_t* cores, int32_t n) EXL3_CPU_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef EXL3_CPU_NOEXCEPT
