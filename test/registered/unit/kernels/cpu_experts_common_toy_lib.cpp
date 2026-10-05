// The toy quant as a shared library, for test_cpu_experts_common.py's per-library and portable-build checks.
#include "cpu_experts_common_toy.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/cabi.hpp"

namespace toy {
__attribute__((visibility("default"))) const sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL()
{
    static const sglang::cpu_experts::ExpertForward<ToyQuant> kernel{};
    return kernel;
}
}  // namespace toy

SGLANG_CPU_EXPERTS_DEFINE_CABI(toy, toy::ToyQuant, toy::TOY_KERNEL)
