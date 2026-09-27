// TensorMatcher::verify never names the tensor it failed on (its message reads "Tensor match failed for
// Tensor<..> at file:line"). verify_named prefixes the tensor's name so a caught exception can be matched by it.
// Host-compilable: no CUDA needed, so the host .cpp instantiation (Task 9) can include this too.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <string>
#include <string_view>
#include <utility>

namespace sglang::expert_stream {

inline void verify_named(std::string_view name, host::TensorMatcher&& matcher, tvm::ffi::TensorView view) {
  try {
    std::move(matcher).verify(view);
  } catch (host::PanicError& e) {
    throw host::PanicError(std::string(name) + ": " + std::string(e.what()));
  }
}

}  // namespace sglang::expert_stream
