// Format-independent CPU completion bridge for captured expert-stream waits.
#include <sgl_kernel/tensor.h>

#include <tvm/ffi/function.h>

#include "expert_stream/host/wait_completion.h"
#include "expert_stream/tensor_checks.h"

namespace sglang::expert_stream {

int64_t wait_completion_open(tvm::ffi::TensorView page, tvm::ffi::TensorView mailbox, int64_t lease_address) {
  using namespace host;
  using namespace wire;
  auto cpu = SymbolicDevice{};
  cpu.set_options<kDLCPU, kDLCUDAHost>();
  verify_named("page", TensorMatcher({kPageBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(cpu), page);
  verify_named(
      "wait mailbox",
      TensorMatcher({kWaitCompletionBytes}).with_dtype<uint8_t>().with_device<kDLCPU, kDLCUDAHost>(cpu),
      mailbox);
  RuntimeCheck(lease_address >= 0, "wait completion: lease_address must be nonnegative");
  return WaitCompletionRegistry::open(
      static_cast<uint8_t*>(page.data_ptr()),
      static_cast<uint8_t*>(mailbox.data_ptr()),
      reinterpret_cast<const uint8_t*>(lease_address));
}

void wait_completion_close(int64_t handle) {
  WaitCompletionRegistry::close(handle);
}

void wait_completion_cancel(int64_t handle) {
  WaitCompletionRegistry::cancel(handle);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wait_completion_open, wait_completion_open);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wait_completion_close, wait_completion_close);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(expert_stream_wait_completion_cancel, wait_completion_cancel);

}  // namespace sglang::expert_stream
