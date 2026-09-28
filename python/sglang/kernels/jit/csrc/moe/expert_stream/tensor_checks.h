// Host-compilable (no CUDA needed): both device launchers and a host .cpp instantiation can include this.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include "lease_layout.h"

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cstring>
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

/// \brief `row_capacity` as a kernel argument, after checking it is the lease block's row-table word for `row`.
///
/// W1 and S take the capacity as an argument, so a captured graph freezes it. That is sound only while it equals the
/// word the service wrote once at construction (the proof is at StreamParams::row_capacity); this check runs on every
/// launch, capture included, and refuses a caller whose value disagrees. Reads the pinned word on the host.
/// \param lease_address The lease block's address, or 0 for a device without one (then only the range is checked).
inline uint32_t checked_row_capacity(int64_t lease_address, int64_t row, int64_t row_capacity) {
  using namespace wire;
  host::RuntimeCheck(
      row_capacity > 0 && row_capacity <= 0xffffffffLL, "row_capacity: ", row_capacity, " is not a slot count");
  if (lease_address != 0) {
    host::RuntimeCheck(
        row >= 0 && row < (kLeaseRowResult - kLeaseRowTable) / kLeaseRowTableBytes,
        "row_capacity: row ", row, " is outside the lease block's row table");
    uint32_t word = 0;
    std::memcpy(&word, reinterpret_cast<const uint8_t*>(lease_address) + kLeaseRowTable + row * kLeaseRowTableBytes + 4, 4);
    host::RuntimeCheck(
        static_cast<int64_t>(word) == row_capacity, "row_capacity: ", row_capacity, " is not row ", row,
        "'s capacity in the lease block's row table (", word, ")");
  }
  return static_cast<uint32_t>(row_capacity);
}

}  // namespace sglang::expert_stream
