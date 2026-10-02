// Argument checks shared by the device launchers and the host exports.
//
// Host-compilable without CUDA, so both the .cuh launchers and a host .cpp instantiation include it:
//   verify_named          TensorMatcher::verify with the tensor's name in the error
//   checked_row_capacity  a row's slot count as a range-checked kernel argument
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/tensor.h>

#include "lease_layout.h"
#include <cstdint>
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

/// \brief `row_capacity` as a kernel argument, range-checked.
///
/// The row-copy kernels bound every host slot they read by it, and a captured graph freezes it: sound because a
/// row's slab capacity is fixed when the service's tables are built and never changes for the process.
/// \param row_capacity The row's pinned slot count.
/// \return `row_capacity` as uint32_t; throws `host::PanicError` outside (0, 2^32).
inline uint32_t checked_row_capacity(int64_t row_capacity) {
  host::RuntimeCheck(
      row_capacity > 0 && row_capacity <= 0xffffffffLL, "row_capacity: ", row_capacity, " is not a slot count");
  return static_cast<uint32_t>(row_capacity);
}

}  // namespace sglang::expert_stream
