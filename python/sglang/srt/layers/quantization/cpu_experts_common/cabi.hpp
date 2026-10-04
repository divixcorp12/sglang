// The five C functions every CPU expert quant exports, as one-line wrappers over ExpertForward<Quant>.
#pragma once
#include "expert_forward.hpp"
#include "keep_warm.hpp"
#include "team.hpp"

#define SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant)                                                             \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_register_layer(          \
        const SglangCpuExpertsLayer* d, int64_t* handle) noexcept                                               \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::register_layer(d, handle); }                          \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_free_layer(              \
        int64_t handle) noexcept                                                                                \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::free_layer(handle); }                                 \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_forward(                 \
        const SglangCpuExpertsForward* call) noexcept                                                           \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::forward(call); }                                      \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_keep_warm(               \
        int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept                     \
    { return ::sglang::cpu_experts::ExpertForward<Quant>::keep_warm(threads, word, seen, deadline_ns); }         \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_set_cores(               \
        const int32_t* cores, int32_t n) noexcept                                                               \
    { return ::sglang::cpu_experts::Cores::configure(cores, n); }
