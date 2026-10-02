#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define EXL3_CPU_NOEXCEPT noexcept
#else
#define EXL3_CPU_NOEXCEPT
#endif

// FP16 x[hidden], FP32 output[hidden], overwritten, or added to when accumulate
// is nonzero; routing weights are converted to FP16, preserving the registered
// EXL3 kernel's convention. Caller counts as worker 0.
// Returns 0 on success, 1 on a kernel error, 2 on invalid arguments.
int sglang_exl3_cpu_experts_forward(int64_t layer, const void* x,
    const int32_t* slots, const float* weights, int32_t k, float* out,
    int32_t threads, int32_t accumulate) EXL3_CPU_NOEXCEPT;

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
