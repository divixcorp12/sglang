// The NVFP4 CPU expert kernel of this library (kernel.cpp): the CpuExpertKernel the tvm-ffi export
// nvfp4_cpu_kernel_address hands to Python, and the one the benches link.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"

namespace sglang::nvfp4_cpu {
// Hidden: another library defining it in the same process (the library and a standalone build) never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& nvfp4_cpu_kernel();
}  // namespace sglang::nvfp4_cpu
