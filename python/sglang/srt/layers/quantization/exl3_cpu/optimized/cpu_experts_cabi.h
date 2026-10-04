#pragma once
#include <stdint.h>
#include "../../../../../kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"

#ifdef __cplusplus
extern "C" {
#define EXL3_CPU_NOEXCEPT noexcept
#else
#define EXL3_CPU_NOEXCEPT
#endif

// SglangCpuExpertsForward's rows through layer `call->layer`: x FP16, out FP32; k at most 32. Routing weights are
// converted to FP16, preserving the registered EXL3 kernel's convention. Caller counts as worker 0.
// Returns 0 on success, 1 on a kernel error, 2 on invalid arguments.
int sglang_exl3_cpu_experts_forward(const SglangCpuExpertsForward* call) EXL3_CPU_NOEXCEPT;

// Holds `threads` workers in register-only work of the forward's vector width until *word != seen or
// CLOCK_MONOTONIC reaches deadline_ns, so they keep the forward's frequency license through an idle gap. Caller
// counts as worker 0. Returns 0 on success, 1 on a kernel error, 2 on invalid arguments.
int sglang_exl3_cpu_experts_keep_warm(int32_t threads, const uint32_t* word, uint32_t seen,
    int64_t deadline_ns) EXL3_CPU_NOEXCEPT;

// Before the first forward: worker i uses cores[i].
// Core IDs must be distinct and valid; configure at least `threads` cores.
int sglang_exl3_cpu_experts_set_cores(const int32_t* cores, int32_t n) EXL3_CPU_NOEXCEPT;

// Register `capacity` expert slots laid out as the pinned tier's slab rows. slabs[6] are the bases of w13_trellis,
// w13_suh, w13_svh, w2_trellis, w2_suh, w2_svh (exl3_expert_format.EXL3_STREAMED_NAMES order); slot s of each starts
// s rows in, w13 rows holding gate then up. Gated SiLU clamped at act_limit. The kernel stores only the pointers:
// keep the slabs alive until exl3_moe_cpu_free_layer(*handle). Returns 0 and the layer handle, 1 on a kernel error,
// 2 on invalid arguments.
int sglang_exl3_cpu_experts_register_slabs(const void* const* slabs, int32_t capacity, int32_t hidden,
    int32_t intermediate, int32_t bits, int32_t swizzled, float act_limit, int64_t* handle) EXL3_CPU_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef EXL3_CPU_NOEXCEPT
