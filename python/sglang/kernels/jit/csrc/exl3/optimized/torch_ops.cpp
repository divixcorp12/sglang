// The EXL3 CPU expert kernel's address as the torch op sglang_exl3_cpu::kernel_address: how Python hands the kernel to
// the expert-stream host (enable_cpu_experts) and to the host's test exports. Built into the optimized extension only
// (quantization/exl3/ext.py), not into benches or the standalone build.
#include "kernel.h"
#include <torch/library.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::exl3_cpu::exl3_cpu_kernel()); }
}  // namespace

TORCH_LIBRARY(sglang_exl3_cpu, m) { m.def("kernel_address() -> int", &kernel_address); }
