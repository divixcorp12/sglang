// The EXL3 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the torch extension's
// sglang_exl3_cpu::kernel_address op hands to Python, and the one the benches link.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstdint>

namespace sglang::exl3_cpu {
// Hidden: another library defining it in the same process (the extension and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
// The layer behind a handle exl3_moe_cpu_make_layer returned (upstream's per-expert tensor API): the handle is the
// layer's address, so no table and no lock stand between it and a forward. Valid until exl3_moe_cpu_free_layer(handle);
// an unknown or freed handle is undefined. Upstream's child worker (moe_handoff.cu) names layers by registration index,
// so it does not run against this kernel.
inline const ::sglang::cpu_experts::CpuExpertLayer& exl3_cpu_table_layer(int64_t handle)
{
    return *reinterpret_cast<const ::sglang::cpu_experts::CpuExpertLayer*>(handle);
}
}  // namespace sglang::exl3_cpu
