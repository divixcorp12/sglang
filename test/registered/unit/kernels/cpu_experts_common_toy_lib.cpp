// The toy quant as a shared library, for test_cpu_experts_common.py: its kernel behind the accessor TOY_KERNEL, and the
// portable-build check.
#include "cpu_experts_common_toy.hpp"

namespace toy {
__attribute__((visibility("default"))) const sglang::cpu_experts::CpuExpertKernel& TOY_KERNEL()
{
    static const sglang::cpu_experts::ExpertForward<ToyQuant> kernel{};
    return kernel;
}
}  // namespace toy
