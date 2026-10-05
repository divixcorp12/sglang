// The NVFP4 CPU expert library's tvm-ffi exports: its kernel's address (nvfp4_cpu_kernel), which Python hands to the
// expert-stream host, and a layer's params as make_layer reads them. Linked into the shared library only; a harness
// executable calls nvfp4_cpu_kernel() itself.
#include "kernel.h"
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::nvfp4_cpu::nvfp4_cpu_kernel()); }

tvm::ffi::Bytes params(int64_t w13_layout, double inv_input_scale13, double inv_input_scale2)
{
    const SglangNvfp4CpuParams p{static_cast<int32_t>(w13_layout), static_cast<float>(inv_input_scale13),
                                 static_cast<float>(inv_input_scale2)};
    return tvm::ffi::Bytes(reinterpret_cast<const char*>(&p), sizeof(p));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cpu_kernel_address, kernel_address);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cpu_params, params);
