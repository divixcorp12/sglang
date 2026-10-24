// The stream's wait on a lease channel's gate (lease_channel_layout.h): cuStreamWaitValue32_v2, captured as a
// memory-op node when the stream is capturing.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "lease_channel_layout.h"
#include <cstdint>
#include <stdexcept>
#include <string>

// With per-thread default streams, cuda.h would rename the v2 names below to their _ptsz variants.
#if defined(CUDA_API_PER_THREAD_DEFAULT_STREAM) || defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
#error "the gate wait calls the legacy-stream driver entry point; do not build it with per-thread default streams"
#endif

namespace sglang::expert_stream {

// Queues the stream's wait for the gate at `gate_address` to read open. The gate is not a counter, so the cyclic GEQ
// never wraps: an open word (29-bit seq << 2 | 1) is in [1, 2^31) and passes, a closed word has bit 31 set and
// blocks. Uses the v2 name, never the plain one: that is the v1 API, gated by NVreg_EnableStreamMemOPs
// (host/copy_engine.h). Throws if the call fails.
inline void enqueue_gate_wait(cudaStream_t stream, uint64_t gate_address) {
  const CUresult r = cuStreamWaitValue32_v2(
      reinterpret_cast<CUstream>(stream), gate_address, channel::kGateOpen, CU_STREAM_WAIT_VALUE_GEQ);
  if (r != CUDA_SUCCESS)
    throw std::runtime_error("cuStreamWaitValue32_v2 on the gate failed: CUresult " + std::to_string(r));
}

}  // namespace sglang::expert_stream
