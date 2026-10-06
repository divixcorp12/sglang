// The EXL3 CPU expert kernel's torch ops: sglang_exl3_cpu::kernel_address, how Python hands the kernel to the
// expert-stream host (enable_cpu_experts) and to the host's test exports, and sglang_exl3_cpu::params, a layer's
// params as make_layer reads them (uint8 bytes of SglangExl3CpuParams). Built into the optimized extension only
// (quantization/exl3/ext.py), not into benches or the standalone build.
#include "kernel.h"
#include <ATen/ATen.h>
#include <cstring>
#include <torch/library.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::exl3_cpu::exl3_cpu_kernel()); }

at::Tensor params(int64_t bits, int64_t swizzled)
{
    const SglangExl3CpuParams p{static_cast<int32_t>(bits), static_cast<int32_t>(swizzled)};
    at::Tensor out = at::empty({static_cast<int64_t>(sizeof(p))}, at::TensorOptions().dtype(at::kByte));
    std::memcpy(out.data_ptr(), &p, sizeof(p));
    return out;
}
}  // namespace

TORCH_LIBRARY(sglang_exl3_cpu, m)
{
    m.def("kernel_address() -> int", &kernel_address);
    m.def("params(int bits, int swizzled) -> Tensor", &params);
}
