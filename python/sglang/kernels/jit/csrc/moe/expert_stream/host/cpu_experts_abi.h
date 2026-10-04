// The one argument of every CPU expert format's forward (CpuExpertForward in cpu_experts.h): the EXL3 and NVFP4
// kernels' sglang_*_cpu_experts_forward take a pointer to this. Plain C, so a format's standalone build and a ctypes
// caller can use it without the engine's headers.
#pragma once
#include <stdint.h>

#define SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION 1u

// `rows` token rows through layer `layer`. Token t's input is row t of x (the format's x type, FP16 for EXL3 and
// NVFP4, [rows][hidden] contiguous); its experts are slots[t * k + i] weighted by weights[t * k + i], i < k, a -1 slot
// skipped. Its output, row t of out (FP32 [rows][hidden] contiguous), is overwritten with the weighted sum, or the sum is
// added to it when `accumulate` is nonzero. `threads` workers, the calling thread counted as one. A kernel refuses
// (returns 2) an abi_version other than SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION.
typedef struct SglangCpuExpertsForward {
    uint32_t abi_version;
    int32_t rows;
    int64_t layer;
    const void* x;
    const int32_t* slots;
    const float* weights;
    float* out;
    int32_t k;
    int32_t threads;
    int32_t accumulate;
} SglangCpuExpertsForward;

#define SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION 1u
#define SGLANG_CPU_EXPERTS_MAX_SLABS 8

// One layer's pinned host tier, for every quant's register_layer: slot s of slab i starts at slabs[i] + s *
// slot_bytes[i]. slab_count is the quant's own count (a slab the quant marks optional may be null); params points at
// the quant's parameter struct (cpu_experts_cabi.h), or is null when the quant has none. activation 0 is gated SiLU
// with the pre-SiLU clamp act_limit (0: none); a quant refuses (2) an activation it does not implement. The kernel
// stores the pointers only: the registrant keeps every slab and params alive until free_layer.
typedef struct SglangCpuExpertsLayer {
    uint32_t abi_version;
    int32_t capacity;
    int32_t hidden;
    int32_t intermediate;
    int32_t activation;
    float act_limit;
    int32_t slab_count;
    const void* slabs[SGLANG_CPU_EXPERTS_MAX_SLABS];
    uint64_t slot_bytes[SGLANG_CPU_EXPERTS_MAX_SLABS];
    const void* params;
} SglangCpuExpertsLayer;
