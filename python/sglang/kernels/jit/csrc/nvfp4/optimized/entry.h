// What the expert-stream host module calls in the NVFP4 kernel's translation unit (kernel.cpp), in plain std types so
// formats/nvfp4/format.h can include it from the host and device translation units. kernel.h adds the kernel's own
// interface.
#pragma once
#include "../../moe/expert_stream/host/cpu_experts/kernel.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace sglang::nvfp4_cpu {
// Hidden: the host module's other symbols stay private to its .so, and a standalone build never interposes.
__attribute__((visibility("hidden"))) const ::sglang::cpu_experts::CpuExpertKernel& nvfp4_cpu_kernel();
// A layer's params bytes (kernel.h's SglangNvfp4CpuParams) from [w13_layout, inv_input_scale13, inv_input_scale2]:
// the scales are IEEE binary32 bit patterns in the low 32 bits. std::invalid_argument for a wrong count or a value
// its field cannot hold; make_layer checks the values themselves.
__attribute__((visibility("hidden"))) std::vector<std::byte> nvfp4_cpu_param_bytes(std::span<const int64_t> values);
}  // namespace sglang::nvfp4_cpu
