// The DSpark draft channel's device side (draft_channel.h; LEASE_PROTOCOL.md "The lease channel", its second client):
//
//   post    stages x [M, H] (fp16) and stage `stage`'s CPU-owned routes [M, k] into the draft's pinned areas, writes
//           the record and publishes the head; posts nothing when no route is on the CPU (state[kPending] = 0)
//   finish  closes the gate for the pending G (and opens it itself if done[G] already holds)
//   wait    the stream waits for the gate to read open (a memory-op node under capture)
//   commit  checks done[G] (traps without it) and adds the stage's CPU output into `out`
//
// The protocol is the lease channel's (lease_channel.cuh); only the record and the areas are the draft's.
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "draft_channel.h"
#include "lease_channel.cuh"
#include "stream_wait.h"
#include "tensor_checks.h"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace sglang {
namespace device::expert_stream::draft {

using namespace ::sglang::expert_stream::draft;
namespace ch = ::sglang::device::expert_stream::channel;

struct PostParams {
  const void* x;
  int32_t x_dtype;  // 0 fp16, 1 bf16, 2 fp32
  const int64_t* ids;      // [rows, k], contiguous
  const float* weights;    // [rows, k], contiguous
  const uint8_t* on_cpu;   // [experts]: this stage's CPU-owned experts
  int32_t rows, k, experts, stage;
  int64_t hidden;
  int32_t* state;          // device: the channel's state words
  uint8_t* channel;        // pinned (UVA): page + completion block
  __half* x_dst;           // pinned: this stage's x area, [kMaxRows, hidden]
  int32_t* slots_dst;      // pinned: [kMaxRows, kMaxK], the CPU route's expert id or -1
  float* weights_dst;      // pinned: [kMaxRows, kMaxK], its weight, 0 for a -1 slot
};

SGL_DEVICE float input_value(const void* src, int32_t dtype, int64_t i) {
  if (dtype == 0) return __half2float(static_cast<const __half*>(src)[i]);
  if (dtype == 1) return __bfloat162float(static_cast<const __nv_bfloat16*>(src)[i]);
  return static_cast<const float*>(src)[i];
}

__global__ void draft_post_kernel(const __grid_constant__ PostParams p) {
  int cpu_any = 0;
  for (int r = threadIdx.x; r < p.rows * p.k; r += blockDim.x) {
    const int t = r / p.k, i = r % p.k;
    const int64_t id = p.ids[r];
    const bool cpu = id >= 0 && id < p.experts && p.on_cpu[id] != 0;
    __stwt(p.slots_dst + t * kMaxK + i, cpu ? static_cast<int32_t>(id) : -1);
    __stwt(p.weights_dst + t * kMaxK + i, cpu ? p.weights[r] : 0.0f);
    cpu_any |= cpu ? 1 : 0;
  }
  if (__syncthreads_or(cpu_any) == 0) {
    if (threadIdx.x == 0) p.state[ch::kPending] = 0;
    return;
  }
  // x as fp16, 8 elements per 16-byte store (the launcher checks hidden % 8 and the alignment).
  const int64_t vectors = static_cast<int64_t>(p.rows) * p.hidden / 8;
  for (int64_t v = threadIdx.x; v < vectors; v += blockDim.x) {
    __half h[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = __float2half_rn(input_value(p.x, p.x_dtype, 8 * v + e));
    __stwt(reinterpret_cast<uint4*>(p.x_dst) + v, *reinterpret_cast<const uint4*>(h));
  }
  // Each thread's area stores at system scope before the barrier, so thread 0's release of the record and the head
  // orders all of them (as the target post's stage_cpu_input).
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;
  const uint32_t seq = ch::advance(p.state);
  const uint32_t epoch = static_cast<uint32_t>(p.state[ch::kEpoch]);
  uint8_t* const record = ch::record_at<DraftChannel>(p.channel, seq);
  ch::begin_record<DraftChannel>(record);
  st_relaxed_sys<uint32_t>(
      record + kRecStage,
      (static_cast<uint32_t>(p.stage) & 0xFFFFu) | (static_cast<uint32_t>(p.rows) & 0xFFu) << 16 |
          (static_cast<uint32_t>(p.k) & 0xFFu) << 24);
  st_relaxed_sys<uint32_t>(record + kRecEpoch, epoch);
  ch::end_record<DraftChannel>(record, seq);
  ch::publish_head<DraftChannel>(p.channel, seq);
  p.state[ch::kPending] = static_cast<int32_t>(seq);
  p.state[ch::kPendingEpoch] = static_cast<int32_t>(epoch);
}

struct FinishParams {
  const int32_t* state;
  uint8_t* channel;
};

__global__ void draft_finish_kernel(const __grid_constant__ FinishParams p) {
  if (threadIdx.x != 0) return;
  const uint32_t seq = static_cast<uint32_t>(p.state[ch::kPending]);
  if (seq == 0) return;  // nothing posted: the gate still reads the last open(G), or open(0) from the start
  const uint32_t epoch = static_cast<uint32_t>(p.state[ch::kPendingEpoch]);
  ch::close_gate<DraftChannel>(p.channel, seq, ch::generation(seq, epoch));
}

struct CommitParams {
  const int32_t* state;
  const uint8_t* channel;
  const float* cpu_out;  // pinned: this stage's out area, [kMaxRows, hidden]
  float* out;            // device: [rows, hidden]
  int64_t elements;      // rows * hidden
};

__global__ void draft_commit_kernel(const __grid_constant__ CommitParams p) {
  const uint32_t seq = static_cast<uint32_t>(p.state[ch::kPending]);
  if (seq == 0) return;
  // Every block checks with its own acquire before it reads the host's rows: one block's acquire orders only its own
  // later reads.
  if (threadIdx.x == 0)
    ch::commit_or_trap<DraftChannel>(
        p.channel, seq, ch::generation(seq, static_cast<uint32_t>(p.state[ch::kPendingEpoch])));
  __syncthreads();
  for (int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < p.elements;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x)
    p.out[e] += __ldcv(p.cpu_out + e);
}

}  // namespace device::expert_stream::draft

namespace expert_stream::draft {

/// \brief Checked launchers for the draft channel's kernels. Each queues on the current stream of its tensors'
/// device, so a capture records post, then (in finish) the close, the wait node and the commit, in order.
struct DraftChannelKernels {
  /// `x` [M, H] fp16/bf16/fp32, `ids` [M, k] int64, `weights` [M, k] fp32, `on_cpu` [E] uint8 and `state` [6] int32,
  /// all on one CUDA device; M <= kMaxRows, k <= kMaxK, H % 8 == 0. `channel`, `x_dst`, `slots_dst` and
  /// `weights_dst` are pinned host addresses of the channel buffer and this stage's areas.
  static void post(
      tvm::ffi::TensorView x,
      tvm::ffi::TensorView ids,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView on_cpu,
      tvm::ffi::TensorView state,
      int64_t channel,
      int64_t x_dst,
      int64_t slots_dst,
      int64_t weights_dst,
      int64_t stage) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"rows"};
    auto H = SymbolicSize{"hidden"};
    auto K = SymbolicSize{"k"};
    auto E = SymbolicSize{"experts"};
    auto x_dtype = SymbolicDType{};
    ::sglang::expert_stream::verify_named(
        "x", TensorMatcher({M, H}).with_dtype<fp16_t, bf16_t, fp32_t>(x_dtype).with_device(device), x);
    ::sglang::expert_stream::verify_named("ids", TensorMatcher({M, K}).with_dtype<int64_t>().with_device(device), ids);
    ::sglang::expert_stream::verify_named(
        "weights", TensorMatcher({M, K}).with_dtype<fp32_t>().with_device(device), weights);
    ::sglang::expert_stream::verify_named("on_cpu", TensorMatcher({E}).with_dtype<uint8_t>().with_device(device), on_cpu);
    ::sglang::expert_stream::verify_named("state", TensorMatcher({6}).with_dtype<int32_t>().with_device(device), state);
    RuntimeCheck(x.is_contiguous() && ids.is_contiguous() && weights.is_contiguous(), "x, ids, weights: contiguous");
    RuntimeCheck(M.unwrap() >= 1 && M.unwrap() <= kMaxRows, "draft post: 1-16 rows, not ", M.unwrap());
    RuntimeCheck(K.unwrap() >= 1 && K.unwrap() <= kMaxK, "draft post: 1-8 routes a row, not ", K.unwrap());
    RuntimeCheck(H.unwrap() > 0 && H.unwrap() % 8 == 0, "draft post: the hidden size must be a multiple of 8");
    RuntimeCheck(channel != 0 && channel % 128 == 0, "draft post: the channel must be 128-byte aligned");
    RuntimeCheck(x_dst != 0 && x_dst % 16 == 0 && slots_dst != 0 && weights_dst != 0, "draft post: the areas");
    RuntimeCheck(stage >= 0 && stage <= 0xFFFF, "draft post: stage ", stage);
    const int32_t dtype = x_dtype.is_type<fp16_t>() ? 0 : x_dtype.is_type<bf16_t>() ? 1 : 2;
    const auto params = device::expert_stream::draft::PostParams{
        .x = x.data_ptr(),
        .x_dtype = dtype,
        .ids = static_cast<const int64_t*>(ids.data_ptr()),
        .weights = static_cast<const float*>(weights.data_ptr()),
        .on_cpu = static_cast<const uint8_t*>(on_cpu.data_ptr()),
        .rows = static_cast<int32_t>(M.unwrap()),
        .k = static_cast<int32_t>(K.unwrap()),
        .experts = static_cast<int32_t>(E.unwrap()),
        .stage = static_cast<int32_t>(stage),
        .hidden = H.unwrap(),
        .state = static_cast<int32_t*>(state.data_ptr()),
        .channel = reinterpret_cast<uint8_t*>(channel),
        .x_dst = reinterpret_cast<__half*>(x_dst),
        .slots_dst = reinterpret_cast<int32_t*>(slots_dst),
        .weights_dst = reinterpret_cast<float*>(weights_dst),
    };
    LaunchKernel(1, 256, device.unwrap())(device::expert_stream::draft::draft_post_kernel, params);
  }

  /// Closes the gate for the pending record, queues the stream's wait on it, and adds the host's rows (`cpu_out`, a
  /// pinned [kMaxRows, H] fp32 area) into `out` [M, H] fp32 once done[G] holds. With nothing pending all three pass.
  static void finish(tvm::ffi::TensorView state, int64_t channel, int64_t cpu_out, tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto M = SymbolicSize{"rows"};
    auto H = SymbolicSize{"hidden"};
    ::sglang::expert_stream::verify_named("state", TensorMatcher({6}).with_dtype<int32_t>().with_device(device), state);
    ::sglang::expert_stream::verify_named("out", TensorMatcher({M, H}).with_dtype<fp32_t>().with_device(device), out);
    RuntimeCheck(out.is_contiguous(), "out: contiguous");
    RuntimeCheck(M.unwrap() >= 1 && M.unwrap() <= kMaxRows, "draft finish: 1-16 rows, not ", M.unwrap());
    RuntimeCheck(channel != 0 && channel % 128 == 0 && cpu_out != 0, "draft finish: the channel and the out area");
    const auto stream = LaunchKernel::resolve_device(device.unwrap());
    LaunchKernel(1, 32, stream)(
        device::expert_stream::draft::draft_finish_kernel,
        device::expert_stream::draft::FinishParams{
            .state = static_cast<const int32_t*>(state.data_ptr()), .channel = reinterpret_cast<uint8_t*>(channel)});
    ::sglang::expert_stream::enqueue_gate_wait(stream, static_cast<uint64_t>(channel + DraftChannel::kGate));
    const int64_t elements = M.unwrap() * H.unwrap();
    const auto blocks = static_cast<uint32_t>(std::min<int64_t>((elements + 255) / 256, 64));
    LaunchKernel(blocks, 256, stream)(
        device::expert_stream::draft::draft_commit_kernel,
        device::expert_stream::draft::CommitParams{
            .state = static_cast<const int32_t*>(state.data_ptr()),
            .channel = reinterpret_cast<const uint8_t*>(channel),
            .cpu_out = reinterpret_cast<const float*>(cpu_out),
            .out = static_cast<float*>(out.data_ptr()),
            .elements = elements,
        });
  }
};

}  // namespace expert_stream::draft
}  // namespace sglang
