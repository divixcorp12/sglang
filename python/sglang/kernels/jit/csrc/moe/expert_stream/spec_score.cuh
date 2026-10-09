// The RAM prefetch's GPU scorer (spec docs/superpowers/specs/2026-10-09-dsv41-ram-prefetch-gpu-scorer-design.md):
// two kernels launched right after layer T's post, on its stream and inside its captured graph.
//
//   exl3_ram_prefetch_score_kernel   sqrt(softplus(w_e . x_t)) + b_e of layer T+1's gate for every live token, fp32,
//                                    into a [tokens_max, experts] scratch
//   exl3_ram_prefetch_select_kernel  GateScorer::choose's ranking (host/gate_scorer.h) past the experts VRAM-hot or
//                                    RAM-mapped in layer T+1, published to the record's candidate slot (spec_candidates.h)
//   SpecScoreKernel                  the checked host launchers for both
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "spec_candidates.h"
#include "tensor_checks.h"
#include <cuda_bf16.h>

namespace sglang {
namespace device::expert_stream {

constexpr int kScoreWarps = 8;           // a score block's warps; each scores one expert at a time
constexpr int kScoreTokenTile = 8;       // tokens accumulated per pass over a gate row
constexpr int kSelectThreads = 256;      // one warp per token while ordering, then thread 0 alone
constexpr int kSelectMaxExperts = 1024;  // the select kernel's shared per-expert arrays; 32 per lane while ordering
constexpr int kSelectMaxTokens = 32;     // CpuTokenTable::kMaxTokens
constexpr int kSpecDepth = 12;           // GateScorer::kDepth: each token's ranks walked

SGL_DEVICE float neg_inf() {
  return __uint_as_float(0xFF800000u);
}

SGL_DEVICE float gate_value(const bf16_t* w, int64_t i) {
  return __bfloat162float(w[i]);
}

SGL_DEVICE float gate_value(const float* w, int64_t i) {
  return w[i];
}

// GateScorer's order: the higher score first, the lower id on a tie.
SGL_DEVICE bool ranks_above(float sa, int32_t a, float sb, int32_t b) {
  return sa > sb || (sa == sb && a < b);
}

}  // namespace device::expert_stream

// Arguments of the score kernel, a __grid_constant__ a captured graph freezes.
struct SpecScoreParams {
  const bf16_t* x;    // [tokens, hidden]: layer T's MoE input
  const void* w;      // [experts, hidden]: layer T+1's router weight, bf16 or fp32 (router_fp32)
  const float* bias;  // [experts]: its score-correction bias
  float* scores;      // [tokens_max, experts]; rows past `tokens` keep an earlier record's scores
  int64_t tokens;
  int64_t hidden;
  int64_t experts;
};

// The score kernel: one warp per expert at a time, its lanes striding the hidden size for kScoreTokenTile tokens per
// pass, fp32 sums, then sqrt(softplus(z)) + b with torch's softplus threshold of 20. The sqrt and the add are
// correctly rounded whatever the module's math flags, so integer logits score as the host's do, bit for bit.
template <class W>
__global__ __launch_bounds__(device::expert_stream::kScoreWarps * 32) void exl3_ram_prefetch_score_kernel(
    const __grid_constant__ SpecScoreParams p) {
  using namespace device::expert_stream;
  const int lane = threadIdx.x % 32;
  const W* w = static_cast<const W*>(p.w);
  const int64_t warps = static_cast<int64_t>(gridDim.x) * kScoreWarps;
  for (int64_t e = static_cast<int64_t>(blockIdx.x) * kScoreWarps + threadIdx.x / 32; e < p.experts; e += warps) {
    const W* row = w + e * p.hidden;
    for (int64_t t0 = 0; t0 < p.tokens; t0 += kScoreTokenTile) {
      float acc[kScoreTokenTile] = {};
      for (int64_t h = lane; h < p.hidden; h += 32) {
        const float wv = gate_value(row, h);
#pragma unroll
        for (int j = 0; j < kScoreTokenTile; ++j)
          if (t0 + j < p.tokens) acc[j] = __fmaf_rn(wv, __bfloat162float(p.x[(t0 + j) * p.hidden + h]), acc[j]);
      }
#pragma unroll
      for (int j = 0; j < kScoreTokenTile; ++j) {
        float z = acc[j];
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
          z = __fadd_rn(z, __shfl_xor_sync(0xFFFFFFFFu, z, offset));
        if (lane == 0 && t0 + j < p.tokens) {
          const float softplus = z > 20.0f ? z : log1pf(expf(z));
          const float s = __fadd_rn(__fsqrt_rn(softplus), p.bias[e]);
          p.scores[(t0 + j) * p.experts + e] = isnan(s) ? neg_inf() : s;  // a NaN score ranks below every other
        }
      }
    }
  }
}

// Arguments of the select kernel, a __grid_constant__ a captured graph freezes.
struct SpecSelectParams {
  const float* scores;  // the score kernel's [tokens_max, experts]
  int64_t tokens;       // the record's live tokens; outside 1..tokens_max the slot gets count 0
  int64_t tokens_max;
  int64_t experts;
  int64_t top_k;
  int64_t per_token;
  int64_t top_k_only;
  const int64_t* hot_slots;  // layer T+1's VRAM slots' experts (-1 empty), hot_capacity of them
  int64_t hot_capacity;
  const int32_t* ram_slot;  // layer T+1's row of the device map: >= 0 mapped in RAM
  const int32_t* state;     // the post's state words: kPosted is the record's seq
  uint8_t* candidates;      // the pinned candidate page
};

// The select kernel, one block of kSelectThreads. Warp w orders tokens w, w + 8, ...: kSpecDepth rounds of a warp
// argmax, lane l holding experts l, l + 32, .... Thread 0 then walks each token's order past the skipped experts as
// GateScorer::choose does, merges the picks by best margin, and publishes up to kMaxCandidates to the record's slot:
// seq 0, a release fence, the payload, then the seq with a release (the hot page's order).
__global__ __launch_bounds__(device::expert_stream::kSelectThreads, 1) void exl3_ram_prefetch_select_kernel(
    const __grid_constant__ SpecSelectParams p) {
  using namespace device::expert_stream;
  using Cand = ::sglang::expert_stream::wire::SpecCandidates;
  __shared__ uint8_t skip[kSelectMaxExperts];
  __shared__ uint8_t picked[kSelectMaxExperts];  // 0 never picked, 1 picked, 2 published
  __shared__ uint8_t rank[kSelectMaxExperts];
  __shared__ float best[kSelectMaxExperts];
  __shared__ int16_t order[kSelectMaxTokens][kSpecDepth];
  __shared__ int16_t list[kSelectMaxTokens * kSpecDepth];
  const bool live = p.tokens >= 1 && p.tokens <= p.tokens_max;
  const int64_t depth = min(static_cast<int64_t>(kSpecDepth), p.experts);
  for (int64_t e = threadIdx.x; e < p.experts; e += blockDim.x) {
    skip[e] = p.ram_slot[e] >= 0 ? 1 : 0;
    picked[e] = 0;
  }
  __syncthreads();
  for (int64_t s = threadIdx.x; s < p.hot_capacity; s += blockDim.x) {
    const int64_t e = p.hot_slots[s];
    if (e >= 0 && e < p.experts) skip[e] = 1;
  }
  if (live) {
    const int lane = threadIdx.x % 32;
    for (int64_t t = threadIdx.x / 32; t < p.tokens; t += kSelectThreads / 32) {
      const float* s = p.scores + t * p.experts;
      uint32_t taken = 0;  // bit k: expert lane + 32k is already ordered
      for (int64_t i = 0; i < depth; ++i) {
        float top = neg_inf();
        int32_t id = 0x7FFFFFFF;
        for (int k = 0; k < kSelectMaxExperts / 32 && lane + 32 * k < p.experts; ++k) {
          const int32_t e = lane + 32 * k;
          if ((taken >> k & 1u) == 0 && ranks_above(s[e], e, top, id)) {
            top = s[e];
            id = e;
          }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
          const float other = __shfl_xor_sync(0xFFFFFFFFu, top, offset);
          const int32_t other_id = __shfl_xor_sync(0xFFFFFFFFu, id, offset);
          if (ranks_above(other, other_id, top, id)) {
            top = other;
            id = other_id;
          }
        }
        if (id % 32 == lane) taken |= 1u << (id / 32);
        if (lane == 0) order[t][i] = static_cast<int16_t>(id);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  int n = 0;
  if (live) {
    const int64_t walk = p.top_k_only != 0 ? p.top_k : depth;
    for (int64_t t = 0; t < p.tokens; ++t) {
      const float* s = p.scores + t * p.experts;
      const float kth = s[order[t][p.top_k - 1]];
      int64_t taken = 0;
      for (int64_t i = 0; i < walk && taken < p.per_token; ++i) {
        const int32_t e = order[t][i];
        if (skip[e]) continue;
        // Equal scores (also two infinities) give 0; -inf less a finite score is clamped to the lowest finite float.
        const float margin = s[e] == kth ? 0.0f : fmaxf(__fsub_rn(s[e], kth), -3.40282347e38f);
        if (picked[e] == 0) {
          picked[e] = 1;
          best[e] = margin;
          rank[e] = static_cast<uint8_t>(i);
          list[n++] = static_cast<int16_t>(e);
        } else {
          if (margin > best[e] || (margin == best[e] && i < rank[e])) rank[e] = static_cast<uint8_t>(i);
          best[e] = fmaxf(best[e], margin);
        }
        ++taken;
      }
    }
  }
  const int count = min(n, Cand::kMaxCandidates);
  const uint32_t seq = static_cast<uint32_t>(p.state[kPosted]);
  uint8_t* const slot = p.candidates + static_cast<int64_t>((seq - 1u) % Cand::kCandRecords) * Cand::kCandStride;
  st_relaxed_sys<uint32_t>(slot + Cand::kCandSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(
      slot + Cand::kCandCount, static_cast<uint32_t>(count) | (live ? 0u : Cand::kCandFlagOversize) << 16);
  for (int c = 0; c < count; ++c) {
    int at = -1;
    for (int j = 0; j < n; ++j) {
      const int32_t e = list[j];
      if (picked[e] == 1 && (at < 0 || ranks_above(best[e], e, best[list[at]], list[at]))) at = j;
    }
    const int32_t e = list[at];
    picked[e] = 2;
    uint8_t* const entry = slot + Cand::kCandEntries + c * Cand::kCandEntryBytes;
    st_relaxed_sys<uint32_t>(entry + Cand::kCandExpert, static_cast<uint32_t>(e) | static_cast<uint32_t>(rank[e]) << 16);
    st_relaxed_sys<uint32_t>(entry + Cand::kCandMargin, __float_as_uint(best[e]));
  }
#ifndef EXL3_RAM_MISS_TEST_SPEC_NO_SEQ
  st_release_sys(slot + Cand::kCandSeq, seq);  // orders the payload before the seq
#endif
}

/// \brief Checked host launchers for the GPU scorer's kernels: score, then select, on the stream of the tensors'
/// device and without PDL, so each starts after the post before it completed.
struct SpecScoreKernel {
  /// Launches the score kernel; see SpecScoreParams. `x` bf16 [tokens, hidden], `w` bf16 or fp32 [experts, hidden],
  /// `bias` fp32 [experts], `scores` fp32 [tokens_max, experts] with 1 <= tokens <= tokens_max.
  static void score(
      tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView bias, tvm::ffi::TensorView scores) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto E_ = SymbolicSize{"experts"};
    auto M_ = SymbolicSize{"tokens_max"};
    expert_stream::verify_named("x", TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), x);
    expert_stream::verify_named(
        "w", TensorMatcher({E_, H_}).with_dtype<bf16_t, fp32_t>().with_device<kDLCUDA>(device), w);
    expert_stream::verify_named("bias", TensorMatcher({E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), bias);
    expert_stream::verify_named(
        "scores", TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), scores);
    RuntimeCheck(
        x.is_contiguous() && w.is_contiguous() && bias.is_contiguous() && scores.is_contiguous(),
        "x, w, bias, scores: must be contiguous");
    RuntimeCheck(
        T_.unwrap() >= 1 && T_.unwrap() <= M_.unwrap(), "x: 1..tokens_max rows (a record outside them is not scored)");
    const auto params = SpecScoreParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .w = w.data_ptr(),
        .bias = static_cast<const float*>(bias.data_ptr()),
        .scores = static_cast<float*>(scores.data_ptr()),
        .tokens = T_.unwrap(),
        .hidden = H_.unwrap(),
        .experts = E_.unwrap(),
    };
    const auto stream = LaunchKernel::resolve_device(scores.device());
    const auto blocks = static_cast<unsigned>((E_.unwrap() + kScoreWarps - 1) / kScoreWarps);
    const bool fp32 = w.dtype().code == kDLFloat;
    LaunchKernel(dim3(blocks), dim3(kScoreWarps * 32), stream)(
        fp32 ? exl3_ram_prefetch_score_kernel<float> : exl3_ram_prefetch_score_kernel<bf16_t>, params);
  }

  /// Launches the select kernel; see SpecSelectParams. `ram_slot` is the device map bank's [rows, experts] and
  /// `target` its row; `candidates_address` the pinned candidate page's address.
  static void select(
      tvm::ffi::TensorView scores,
      int64_t tokens,
      int64_t top_k,
      int64_t per_token,
      int64_t top_k_only,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView ram_slot,
      int64_t target,
      tvm::ffi::TensorView state,
      int64_t candidates_address) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto E_ = SymbolicSize{"experts"};
    auto M_ = SymbolicSize{"tokens_max"};
    auto Rows_ = SymbolicSize{"rows"};
    expert_stream::verify_named(
        "scores", TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), scores);
    expert_stream::verify_named(
        "hot_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), hot_slots);
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, E_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "state", TensorMatcher({kStateWords}).with_dtype<int32_t>().with_device<kDLCUDA>(device), state);
    const int64_t experts = E_.unwrap();
    const int64_t depth = std::min<int64_t>(kSpecDepth, experts);
    RuntimeCheck(experts <= kSelectMaxExperts, "experts: the select kernel holds at most ", kSelectMaxExperts);
    RuntimeCheck(M_.unwrap() <= kSelectMaxTokens, "scores: at most ", kSelectMaxTokens, " token rows");
    RuntimeCheck(tokens >= 0, "tokens: must not be negative");
    RuntimeCheck(top_k >= 1 && top_k <= depth, "top_k: must be in 1..", depth);
    RuntimeCheck(per_token >= 1 && per_token <= kSpecDepth, "per_token: must be in 1..", kSpecDepth);
    RuntimeCheck(hot_capacity >= 0 && hot_capacity <= hot_slots.size(0), "hot_capacity: at most hot_slots' size");
    RuntimeCheck(target >= 0 && target < Rows_.unwrap(), "target: outside the map bank");
    RuntimeCheck(
        candidates_address != 0 && candidates_address % 128 == 0, "candidates: a nonzero 128-byte aligned address");
    const auto params = SpecSelectParams{
        .scores = static_cast<const float*>(scores.data_ptr()),
        .tokens = tokens,
        .tokens_max = M_.unwrap(),
        .experts = experts,
        .top_k = top_k,
        .per_token = per_token,
        .top_k_only = top_k_only,
        .hot_slots = static_cast<const int64_t*>(hot_slots.data_ptr()),
        .hot_capacity = hot_capacity,
        .ram_slot = static_cast<const int32_t*>(ram_slot.data_ptr()) + target * experts,
        .state = static_cast<const int32_t*>(state.data_ptr()),
        .candidates = reinterpret_cast<uint8_t*>(candidates_address),
    };
    const auto stream = LaunchKernel::resolve_device(scores.device());
    LaunchKernel(dim3(1), dim3(kSelectThreads), stream)(exl3_ram_prefetch_select_kernel, params);
  }
};

}  // namespace sglang
