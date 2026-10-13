// The fused learned head's tail (the L+2 heads T0 and T3 of analysis/dsv41-drive/prefetch-replay/r1_models.py):
//
//   spec_head_tail_kernel  token logits -> block score per expert (T0: noisy-or; T3: r1_aggregator.block_features and
//                          the aggregator MLP) -> sigmoid -> the hot/RAM skips -> threshold -> cap-1 argmax, published to
//                          a candidate record (spec_candidates.h) in the serving convention
#pragma once

#include "spec_candidates.h"
#include "spec_score.cuh"

namespace sglang {
namespace device::expert_stream {

constexpr int kHeadMaxExperts = 384;   // the tail's one thread per expert
constexpr int kHeadTailThreads = 384;  // the tail's block: one thread per expert
constexpr int kHeadMaxTokens = 8;      // live tokens per record (the learned heads' MAX_WIDTH is 6)
constexpr int kBlockFeatures = 10;     // r1_aggregator.N_BLOCK_FEATURES
constexpr int kAggHidden = 32;         // the aggregator MLP's hidden width

static_assert(kHeadTailThreads % 32 == 0 && kHeadMaxExperts <= kHeadTailThreads, "one thread per expert");

// torch's softplus (beta 1, threshold 20), as the score kernel computes it.
SGL_DEVICE float head_softplus(float z) {
  return z > 20.0f ? z : log1pf(expf(z));
}

// Exact-erf GELU (torch.nn.GELU default).
SGL_DEVICE float head_gelu(float z) {
  return 0.5f * z * (1.0f + erff(z * 0.70710678118654752440f));
}

}  // namespace device::expert_stream

/// \brief Arguments of the tail kernel, a __grid_constant__ a captured graph freezes.
struct SpecHeadTailParams {
  const float* logits;       ///< [tokens, experts]: the head's per-token expert logits
  const int64_t* positions;  ///< [tokens]: 0 is the anchor token, the rest are drafts
  int64_t tokens;            ///< 1..kHeadMaxTokens live rows of `logits`
  int64_t experts;           ///< at most kHeadMaxExperts
  const bf16_t* agg_w1;      ///< [kAggHidden, kBlockFeatures]; nullptr selects T0's noisy-or
  const float* agg_bias;     ///< [kAggHidden]: first-layer bias with the row's embedding contribution folded in
  const bf16_t* agg_w2;      ///< [kAggHidden]
  const float* agg_b2;       ///< [1]
  const int64_t* hot_slots;  ///< the target layer's VRAM slots' experts (-1 empty), hot_capacity of them
  int64_t hot_capacity;
  const int32_t* ram_slot;   ///< the target layer's row of the device map: >= 0 mapped in RAM
  const float* threshold;    ///< [1]: a block scoring below it is not read
  float* block_scores;       ///< [experts]: sigmoid of the block logit, every expert
  float* features;           ///< [experts, kBlockFeatures] block features, or nullptr
  uint8_t* record;           ///< kCandStride bytes: the candidate slot, written as spec_candidates.h lays it out
  const int32_t* state;      ///< serving: the post's state words, kPosted is the record's seq; nullptr offline
  uint8_t* candidates;       ///< serving: the pinned candidate page, slot (seq - 1) % kCandRecords; nullptr offline
};

/// \brief The fused tail: one block of kHeadTailThreads, one thread per expert.
///
/// Computes each expert's block score, then thread 0 publishes the best non-skipped expert at or above the threshold
/// (ties to the lower id, NaN below all) as count 0 or 1, the expert at kCandExpert and its score at kCandMargin. The
/// record's seq is stored last, after a fence, as spec_score.cuh's select kernel does.
__global__ __launch_bounds__(device::expert_stream::kHeadTailThreads, 1) void spec_head_tail_kernel(
    const __grid_constant__ SpecHeadTailParams p) {
  using namespace device::expert_stream;
  using Cand = ::sglang::expert_stream::wire::SpecCandidates;
  constexpr int kWarps = kHeadTailThreads / 32;
  __shared__ uint8_t skip[kHeadMaxExperts];
  __shared__ float w1[kAggHidden * kBlockFeatures];
  __shared__ float bias1[kAggHidden];
  __shared__ float w2[kAggHidden];
  __shared__ float warp_top[kWarps];
  __shared__ int32_t warp_id[kWarps];
  const int tid = threadIdx.x;
  const int lane = tid % 32, warp = tid / 32;
  const bool learned = p.agg_w1 != nullptr;

  for (int64_t e = tid; e < p.experts; e += blockDim.x) skip[e] = p.ram_slot[e] >= 0 ? 1 : 0;
  if (learned) {
    for (int i = tid; i < kAggHidden * kBlockFeatures; i += blockDim.x) w1[i] = __bfloat162float(p.agg_w1[i]);
    if (tid < kAggHidden) {
      bias1[tid] = p.agg_bias[tid];
      w2[tid] = __bfloat162float(p.agg_w2[tid]);
    }
  }
  __syncthreads();
  for (int64_t s = tid; s < p.hot_capacity; s += blockDim.x) {
    const int64_t e = p.hot_slots[s];
    if (e >= 0 && e < p.experts) skip[e] = 1;
  }

  float score = neg_inf();
  const bool mine = tid < p.experts;
  if (mine) {
    float l[kHeadMaxTokens];
    float soft = 0.0f, total = 0.0f, anchor = 0.0f, peak = neg_inf(), draft = neg_inf();
    bool anchor_present = false, draft_present = false;
    for (int t = 0; t < p.tokens; ++t) {
      l[t] = p.logits[t * p.experts + tid];
      soft += head_softplus(l[t]);
      total += l[t];
      peak = fmaxf(peak, l[t]);
      if (p.positions[t] == 0) {
        anchor += l[t];
        anchor_present = true;
      } else {
        draft = fmaxf(draft, l[t]);
        draft_present = true;
      }
    }
    float logit;
    if (!learned) {
      const float s = fmaxf(soft, 1e-12f);
      logit = s + logf(-expm1f(-s));
    } else {
      const float count = static_cast<float>(p.tokens);
      const float mean = total / count;
      float var = 0.0f;
      for (int t = 0; t < p.tokens; ++t) var += (l[t] - mean) * (l[t] - mean);
      var /= count;
      float f[kBlockFeatures] = {anchor,
                                 anchor_present ? 1.0f : 0.0f,
                                 draft_present ? draft : 0.0f,
                                 draft_present ? 1.0f : 0.0f,
                                 peak,
                                 mean,
                                 var > 0.0f ? sqrtf(var) : 0.0f,
                                 total,
                                 -expm1f(-soft),
                                 count};
      if (p.features != nullptr) {
        for (int i = 0; i < kBlockFeatures; ++i) p.features[tid * kBlockFeatures + i] = f[i];
      }
      logit = p.agg_b2[0];
      for (int j = 0; j < kAggHidden; ++j) {
        float z = bias1[j];
        for (int i = 0; i < kBlockFeatures; ++i) z += w1[j * kBlockFeatures + i] * f[i];
        logit += w2[j] * head_gelu(z);
      }
    }
    score = 1.0f / (1.0f + expf(-logit));
    p.block_scores[tid] = score;
  }
  __syncthreads();  // skip is complete

  const float threshold = p.threshold[0];
  float top = neg_inf();
  int32_t id = 0x7FFFFFFF;
  if (mine && skip[tid] == 0 && score >= threshold) {  // a NaN score fails the comparison and never ranks
    top = score;
    id = tid;
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
  if (lane == 0) {
    warp_top[warp] = top;
    warp_id[warp] = id;
  }
  __syncthreads();
  if (tid != 0) return;
  float best = warp_top[0];
  int32_t pick = warp_id[0];
  for (int w = 1; w < kWarps; ++w) {
    if (ranks_above(warp_top[w], warp_id[w], best, pick)) {
      best = warp_top[w];
      pick = warp_id[w];
    }
  }
  const bool found = pick != 0x7FFFFFFF;
  const bool serving = p.candidates != nullptr;
  const uint32_t seq = serving ? static_cast<uint32_t>(p.state[kPosted]) : 1u;
  uint8_t* const slot =
      serving ? p.candidates + static_cast<int64_t>((seq - 1u) % Cand::kCandRecords) * Cand::kCandStride : p.record;
  st_relaxed_sys<uint32_t>(slot + Cand::kCandSeq, 0u);
  __threadfence();
  st_relaxed_sys<uint32_t>(slot + Cand::kCandCount, found ? 1u : 0u);  // flags 0
  uint8_t* const entry = slot + Cand::kCandEntries;
  st_relaxed_sys<uint32_t>(entry + Cand::kCandExpert, found ? static_cast<uint32_t>(pick) : 0xFFFFu);  // rank 0
  st_relaxed_sys<uint32_t>(entry + Cand::kCandMargin, found ? __float_as_uint(best) : 0u);
  __threadfence();
  st_release_sys(slot + Cand::kCandSeq, seq);  // orders the payload before the seq
}

/// \brief Serving's empty publication: one thread writes count 0 with kCandFlagOversize to the post's slot, for a
/// record with no x or tokens outside 1..kHeadMaxTokens, in spec_score.cuh's select order (seq 0, fence, payload, seq).
__global__ void spec_head_empty_kernel(const int32_t* state, uint8_t* candidates) {
  using namespace device::expert_stream;
  using Cand = ::sglang::expert_stream::wire::SpecCandidates;
  const uint32_t seq = static_cast<uint32_t>(state[kPosted]);
  uint8_t* const slot = candidates + static_cast<int64_t>((seq - 1u) % Cand::kCandRecords) * Cand::kCandStride;
  st_relaxed_sys<uint32_t>(slot + Cand::kCandSeq, 0u);
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  st_relaxed_sys<uint32_t>(slot + Cand::kCandCount, Cand::kCandFlagOversize << 16);  // count 0
  st_release_sys(slot + Cand::kCandSeq, seq);
}

}  // namespace sglang
