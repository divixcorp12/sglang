// The stream's wait on a lease channel's gate (lease_channel_layout.h): cuStreamWaitValue32_v2, captured as a
// memory-op node when the stream is capturing.
#pragma once

#include <cuda_runtime.h>
#include <dlfcn.h>

#include "lease_channel_layout.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sglang::expert_stream {

// cuStreamWaitValue32_v2 from libcuda.so.1, resolved once. Uses the v2 name, never the plain one: that is the v1 API,
// gated by NVreg_EnableStreamMemOPs (host/copy_engine.h). Returns null when the driver lacks it; the launcher then
// refuses.
using StreamWaitValue32 = int (*)(void*, uint64_t, uint32_t, unsigned);
inline StreamWaitValue32 stream_wait_value32() {
  static const StreamWaitValue32 fn = [] {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (lib == nullptr) lib = dlopen("libcuda.so.1", RTLD_NOW);
    return lib == nullptr ? nullptr : reinterpret_cast<StreamWaitValue32>(dlsym(lib, "cuStreamWaitValue32_v2"));
  }();
  return fn;
}
constexpr unsigned kStreamWaitValueGeq = 0;  // CU_STREAM_WAIT_VALUE_GEQ: (int32_t)(*addr - value) >= 0

// Queues the stream's wait for the gate at `gate_address` to read open. The gate is not a counter, so the cyclic GEQ
// never wraps: an open word (29-bit seq << 2 | 1) is in [1, 2^31) and passes, a closed word has bit 31 set and
// blocks. Throws if the driver lacks cuStreamWaitValue32_v2 or the call fails.
inline void enqueue_gate_wait(cudaStream_t stream, uint64_t gate_address) {
  const auto fn = stream_wait_value32();
  if (fn == nullptr)
    throw std::runtime_error("the gate wait needs cuStreamWaitValue32_v2, which libcuda.so.1 does not provide");
  const int r = fn(static_cast<void*>(stream), gate_address, channel::kGateOpen, kStreamWaitValueGeq);
  if (r != 0) throw std::runtime_error("cuStreamWaitValue32_v2 on the gate failed: CUresult " + std::to_string(r));
}

}  // namespace sglang::expert_stream
