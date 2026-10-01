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

// Before the first forward/staging-pool use: worker i uses cores[i].
// Core IDs must be distinct and valid; configure at least `threads` cores.
int sglang_exl3_cpu_experts_set_cores(const int32_t* cores, int32_t n) EXL3_CPU_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef EXL3_CPU_NOEXCEPT
