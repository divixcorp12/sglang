// The NVFP4 CPU expert library's one tvm-ffi export: its kernel's address (nvfp4_cpu_kernel), which Python hands to the
// expert-stream host. Linked into the shared library only; a harness executable calls nvfp4_cpu_kernel() itself.
#include "kernel.h"
#include <tvm/ffi/function.h>

namespace {
int64_t kernel_address() { return reinterpret_cast<int64_t>(&::sglang::nvfp4_cpu::nvfp4_cpu_kernel()); }
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cpu_kernel_address, kernel_address);
