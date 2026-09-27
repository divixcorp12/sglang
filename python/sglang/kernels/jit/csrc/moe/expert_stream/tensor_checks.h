// Host-compilable (no CUDA needed): both device launchers and a host .cpp instantiation can include this.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <string>
#include <string_view>
#include <utility>

namespace sglang::expert_stream {

/// \brief Verifies `view` against `matcher`, prefixing `name` onto the exception if it fails.
///
/// `TensorMatcher::verify` never names the tensor it failed on (its message reads "Tensor match failed for
/// Tensor<..> at file:line"); this rethrows with `"<name>: "` prepended, so a caller can match on it.
/// \param name The tensor's name, prefixed onto any raised `host::PanicError`.
/// \param matcher The (moved-from) matcher to run.
/// \param view The tensor to verify.
inline void verify_named(std::string_view name, host::TensorMatcher&& matcher, tvm::ffi::TensorView view) {
  try {
    std::move(matcher).verify(view);
  } catch (host::PanicError& e) {
    throw host::PanicError(std::string(name) + ": " + std::string(e.what()));
  }
}

}  // namespace sglang::expert_stream
