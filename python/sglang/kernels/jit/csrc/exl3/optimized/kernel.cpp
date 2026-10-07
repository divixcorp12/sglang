// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
// The arithmetic is math.hpp and each tier's math_scalar.hpp, math_avx2.hpp and math_avx512.hpp; the forward is
// forward_plan.hpp (with shapes.hpp). This file holds the kernel's accessor and the torch wrappers.
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include "kernel.h"
#include "quant.hpp"
#include <c10/util/Half.h>
#include <ATen/ATen.h>

#include <cstdint>
#include <vector>

// Last: the definition order the kernels were validated in (bit-exact per tier); another order changes what GCC
// inlines and clones.
#include "forward_plan.hpp"

namespace sglang::exl3_cpu {
const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel()
{
    static const ::sglang::cpu_experts::ExpertForward<Exl3Quant> kernel{};
    return kernel;
}

SglangExl3CpuPlanCalls exl3_cpu_plan_calls()
{
    return {g_plan_calls[0].load(std::memory_order_relaxed), g_plan_calls[1].load(std::memory_order_relaxed)};
}

int32_t exl3_cpu_chunk_m() { return CHUNK_M; }
}  // namespace sglang::exl3_cpu

// Kept for upstream's bindings. Phase timing is compile-time here (ForwardPlan's Profile, forward_plan.hpp).
void exl3_moe_cpu_set_prof(bool) {}

namespace {
using Exl3Forward = ::sglang::cpu_experts::ExpertForward<::sglang::exl3_cpu::Exl3Quant>;
}  // namespace

// -------------------------------------------------------------------------------------------
//   Upstream link compatibility
// -------------------------------------------------------------------------------------------

// This file replaces upstream's cpu/moe_mul1.cpp inside the exllamav3 extension, whose
// bindings.cpp and cpu/moe_handoff.cu still reference these symbols. sglang never calls the
// ones below that fail: the staging copy belongs to upstream's handoff worker, the pool they
// exercised was replaced by the OpenMP forward, and the per-expert layers by slab layers. They
// exist only so the extension links.

void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_stage_experts is not supported by the optimized CPU expert kernel");
}

int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_pool_stress is not supported by the optimized CPU expert kernel");
    return 0;
}

bool exl3_moe_cpu_has_avx2() { return Exl3Forward::isa() != ::sglang::cpu_experts::Isa::Scalar; }
bool exl3_moe_cpu_has_avx512_bw() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Bw; }
bool exl3_moe_cpu_has_avx512_vnni() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Vnni; }
bool exl3_moe_cpu_has_avx512_vbmi() { return Exl3Forward::isa() == ::sglang::cpu_experts::Isa::Vbmi; }

// Upstream's per-expert tensor API: its layers are tables of tensors anywhere in memory, which the kernel does not take
// (its layers are slab layers, made by its make_layer). bindings.cpp still binds these, so they exist and refuse.

int64_t exl3_moe_cpu_make_layer(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,
                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, int64_t, double, int64_t)
{
    TORCH_CHECK(false, "exl3_moe_cpu_make_layer is not supported by the optimized CPU expert kernel: make a slab layer "
                       "with its make_layer (Exl3CpuQuantTrait.layer_spec)");
    return 0;
}

void exl3_moe_cpu_free_layer(int64_t)
{
    TORCH_CHECK(false, "exl3_moe_cpu_free_layer is not supported by the optimized CPU expert kernel");
}

void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_forward_raw is not supported by the optimized CPU expert kernel");
}

void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t)
{
    TORCH_CHECK(false, "exl3_moe_cpu_forward is not supported by the optimized CPU expert kernel");
}
