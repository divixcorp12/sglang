// The EXL3 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the torch extension's
// sglang_exl3_cpu::kernel_address op hands to Python, and the one the benches link.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstdint>
#include <memory>

namespace sglang::exl3_cpu {
// Hidden: another library defining it in the same process (the extension and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& exl3_cpu_kernel();
// The layer exl3_moe_cpu_make_layer registered as `handle` (upstream's per-expert tensor API); valid until
// exl3_moe_cpu_free_layer(handle) and after it for the holder of the returned reference. Throws std::invalid_argument
// for an unknown or freed handle.
__attribute__((visibility("hidden"))) std::shared_ptr<const ::sglang::cpu_experts::CpuExpertLayer> exl3_cpu_table_layer(
    int64_t handle);
}  // namespace sglang::exl3_cpu
