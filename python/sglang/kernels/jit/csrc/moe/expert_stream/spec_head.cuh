// The L+2 learned head's fused fast path (F123 Stage 2): the folded R1 E5 head (T0 noisy-or, T3 block aggregator)
// over serving's spec_scores, in four launches after the native score kernel, on the tensors' stream:
//
//   spec_head_in_kernel    K1: h = w_pca x (PCA rows, the score kernel's 8-token tile) and the sparse fc_in sum
//                          (route mask and weight, top-12 gate score and margin) plus b_in, row and position bias
//   spec_head_mid_kernel   K2: one 8-block cluster; v = gelu(pre + w_in_pca h), r = w_r2 gelu(w_r1 v + b_r1) + b_r2,
//                          g = sigmoid(w_gate h + b_gate), hid = v + g r; v and gelu(r1) cross blocks through DSMEM
//   spec_head_out_kernel   K3: logits = w_out hid + b_out
//   spec_head_tail_kernel  K4 (spec_head_tail.cuh): block score, sigmoid, skips, threshold, cap-1 record
//   SpecHeadKernel         the checked host launchers
//
// Weights are bf16, accumulation and the workspaces between launches fp32 (D3). Nothing under srt launches these.
#pragma once

#include <sgl_kernel/tensor.h>

#include "spec_candidates.h"
#include "spec_head_tail.cuh"
#include "spec_score.cuh"
#include "tensor_checks.h"
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <initializer_list>

namespace sglang {
namespace device::expert_stream {

constexpr int kHeadThreads = 128;     // K1 and K3 blocks
constexpr int kHeadMaxPca = 128;      // h's columns held in shared memory
constexpr int kHeadMaxHidden = 512;   // the predictor's width held in shared memory by K2 and K3
constexpr int kHeadSparse = 4;        // w_in_sparse channels: route mask, route weight, gate score, gate margin
constexpr int kHeadMarginRank = 6;    // b1_rows.TOPK: a top-12 margin is its score less the 6th-best
constexpr int kMidCluster = 8;        // K2's blocks, one cluster: the portable maximum
constexpr int kMidThreads = 512;
constexpr int kMidMaxSlice = kHeadMaxHidden / kMidCluster;

static_assert(kHeadMaxTokens <= kScoreTokenTile, "K1 covers every token in one tile");
static_assert(kHeadMaxExperts / 32 <= 32, "K1's per-lane taken mask is one word");

SGL_DEVICE float head_sigmoid(float v) {
  return 1.0f / (1.0f + expf(-v));
}

// acc[t] = w[0:n] . in[t * n : t * n + n] for t < tokens, one warp striding n; every lane returns the full sums.
SGL_DEVICE void head_warp_dot(const bf16_t* w, const float* in, int64_t n, int64_t tokens, float (&acc)[kHeadMaxTokens]) {
  const int lane = threadIdx.x % 32;
#pragma unroll
  for (int t = 0; t < kHeadMaxTokens; ++t) acc[t] = 0.0f;
  for (int64_t k = lane; k < n; k += 32) {
    const float wv = __bfloat162float(w[k]);
#pragma unroll
    for (int t = 0; t < kHeadMaxTokens; ++t)
      if (t < tokens) acc[t] = __fmaf_rn(wv, in[t * n + k], acc[t]);
  }
#pragma unroll
  for (int t = 0; t < kHeadMaxTokens; ++t)
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) acc[t] += __shfl_xor_sync(0xFFFFFFFFu, acc[t], offset);
}

}  // namespace device::expert_stream

// Arguments of K1, a __grid_constant__ a captured graph freezes.
struct SpecHeadInParams {
  const bf16_t* x;            // [tokens, hidden]
  const float* scores;        // [tokens_max, experts]: serving's biased fp32 spec_scores, row t for token t
  const int64_t* src_ids;     // [tokens, top_k]: the source layer's routes; ids outside 0..experts are ignored
  const float* src_w;         // [tokens, top_k]: their route weights
  const int64_t* positions;   // [tokens]
  const bf16_t* w_pca;        // [pca, hidden]: the PCA mean is folded into b_in and b_gate
  const float* b_in;          // [dim]
  const bf16_t* w_in_sparse;  // [kHeadSparse, experts, dim]
  const float* row_bias;      // [dim]: the target row's source/target/stratum embedding bias
  const float* pos_bias;      // [widths, dim] of the live width, or nullptr: no position term
  float* h;                   // [kHeadMaxTokens, pca]
  float* pre;                 // [kHeadMaxTokens, dim]: fc_in less its PCA columns
  int64_t tokens;
  int64_t hidden;
  int64_t pca;
  int64_t dim;
  int64_t experts;
  int64_t top_k;
  int64_t widths;      // pos_bias's position extent; positions are clamped into it
  int64_t vectorized;  // x and w_pca 16-byte aligned and hidden a multiple of 8
};

// K1: blocks 0..pca-1 each reduce one PCA row over every token as the score kernel does; the remaining blocks each
// order every token's top kSpecDepth scores (a warp argmax per rank, ties to the lower id) and sum the sparse fc_in
// inputs for kHeadThreads columns of pre.
__global__ __launch_bounds__(device::expert_stream::kHeadThreads) void spec_head_in_kernel(
    const __grid_constant__ SpecHeadInParams p) {
  using namespace device::expert_stream;
  constexpr int N = 8;
  constexpr int kWarps = kHeadThreads / 32;
  __shared__ float partial[kWarps][kHeadMaxTokens];
  __shared__ int16_t top_id[kHeadMaxTokens][kSpecDepth];
  __shared__ float top_score[kHeadMaxTokens][kSpecDepth];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (blockIdx.x < p.pca) {
    const int64_t r = blockIdx.x;
    const bf16_t* row = p.w_pca + r * p.hidden;
    const int64_t vector_end = p.vectorized != 0 ? p.hidden / N * N : 0;
    float acc[kHeadMaxTokens] = {};
#pragma unroll 2
    for (int64_t k = static_cast<int64_t>(threadIdx.x) * N; k < vector_end; k += kHeadThreads * N) {
      float wv[N];
      load_gate<bf16_t>(row + k, wv);
#pragma unroll
      for (int t = 0; t < kHeadMaxTokens; ++t) {
        if (t < p.tokens) {
          float xv[N];
          load_input<N>(p.x + t * p.hidden + k, xv);
#pragma unroll
          for (int i = 0; i < N; ++i) acc[t] = __fmaf_rn(wv[i], xv[i], acc[t]);
        }
      }
    }
    for (int64_t k = vector_end + threadIdx.x; k < p.hidden; k += kHeadThreads) {
      const float wv = __bfloat162float(row[k]);
#pragma unroll
      for (int t = 0; t < kHeadMaxTokens; ++t)
        if (t < p.tokens) acc[t] = __fmaf_rn(wv, __bfloat162float(p.x[t * p.hidden + k]), acc[t]);
    }
#pragma unroll
    for (int t = 0; t < kHeadMaxTokens; ++t) {
      float z = acc[t];
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) z += __shfl_xor_sync(0xFFFFFFFFu, z, offset);
      if (lane == 0) partial[warp][t] = z;
    }
    __syncthreads();
    if (threadIdx.x < p.tokens) {
      float z = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) z += partial[w][threadIdx.x];
      p.h[threadIdx.x * p.pca + r] = z;
    }
    return;
  }
  for (int64_t t = warp; t < p.tokens; t += kWarps) {
    const float* s = p.scores + t * p.experts;
    uint32_t taken = 0;  // bit k: expert lane + 32k is already ordered
    for (int i = 0; i < kSpecDepth; ++i) {
      float top = neg_inf();
      int32_t id = 0x7FFFFFFF;
      for (int k = 0; k < kHeadMaxExperts / 32 && lane + 32 * k < p.experts; ++k) {
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
      if (lane == 0) {
        top_id[t][i] = static_cast<int16_t>(id);
        top_score[t][i] = top;
      }
    }
  }
  __syncthreads();
  const int64_t d = (blockIdx.x - p.pca) * kHeadThreads + threadIdx.x;
  if (d >= p.dim) return;
  const int64_t plane = p.experts * p.dim;
  const bf16_t* w_mask = p.w_in_sparse + d;
  const bf16_t* w_route = w_mask + plane;
  const bf16_t* w_score = w_route + plane;
  const bf16_t* w_margin = w_score + plane;
  const float bias = p.b_in[d] + p.row_bias[d];
  for (int64_t t = 0; t < p.tokens; ++t) {
    float acc = bias;
    if (p.pos_bias != nullptr) {
      const int64_t pos = min(max(p.positions[t], static_cast<int64_t>(0)), p.widths - 1);
      acc += p.pos_bias[pos * p.dim + d];
    }
    for (int64_t k = 0; k < p.top_k; ++k) {
      const int64_t e = p.src_ids[t * p.top_k + k];
      if (e < 0 || e >= p.experts) continue;
      acc += __bfloat162float(w_mask[e * p.dim]);
      acc = __fmaf_rn(__bfloat162float(w_route[e * p.dim]), p.src_w[t * p.top_k + k], acc);
    }
    const float sixth = top_score[t][kHeadMarginRank - 1];
#pragma unroll
    for (int i = 0; i < kSpecDepth; ++i) {
      const int64_t e = top_id[t][i];
      const float s = top_score[t][i];
      acc = __fmaf_rn(__bfloat162float(w_score[e * p.dim]), s, acc);
      acc = __fmaf_rn(__bfloat162float(w_margin[e * p.dim]), __fsub_rn(s, sixth), acc);
    }
    p.pre[t * p.dim + d] = acc;
  }
}

// Arguments of K2, a __grid_constant__ a captured graph freezes.
struct SpecHeadMidParams {
  const float* h;          // [kHeadMaxTokens, pca]
  const float* pre;        // [kHeadMaxTokens, dim]
  const bf16_t* w_in_pca;  // [dim, pca]
  const bf16_t* w_r1;      // [dim, dim]
  const float* b_r1;       // [dim]
  const bf16_t* w_r2;      // [dim, dim]
  const float* b_r2;       // [dim]
  const bf16_t* w_gate;    // [dim, pca]
  const float* b_gate;     // [dim]
  float* hid;              // [kHeadMaxTokens, dim]
  int64_t tokens;
  int64_t pca;
  int64_t dim;
};

// K2: one cluster of kMidCluster blocks, block c owning columns [c * dim / kMidCluster, (c + 1) * dim / kMidCluster).
// Each stage's full input is gathered from the cluster's shared memory after a cluster barrier, so no block reads
// another's slice before it is written; the last barrier keeps every slice alive until all its readers are done.
__global__ __launch_bounds__(device::expert_stream::kMidThreads) void spec_head_mid_kernel(
    const __grid_constant__ SpecHeadMidParams p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();  // thread block clusters need sm_90
#else
  using namespace device::expert_stream;
  namespace cg = cooperative_groups;
  constexpr int kWarps = kMidThreads / 32;
  __shared__ float hs[kHeadMaxTokens * kHeadMaxPca];
  __shared__ float full[kHeadMaxTokens * kHeadMaxHidden];
  __shared__ float vs[kHeadMaxTokens * kMidMaxSlice];
  __shared__ float gs[kHeadMaxTokens * kMidMaxSlice];
  __shared__ float as[kHeadMaxTokens * kMidMaxSlice];
  cg::cluster_group cluster = cg::this_cluster();
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int64_t slice = p.dim / kMidCluster;
  const int64_t c0 = static_cast<int64_t>(cluster.block_rank()) * slice;
  for (int64_t i = threadIdx.x; i < p.tokens * p.pca; i += kMidThreads) hs[i] = p.h[i];
  __syncthreads();
  float acc[kHeadMaxTokens];
  float gacc[kHeadMaxTokens];
  for (int64_t j = warp; j < slice; j += kWarps) {
    const int64_t c = c0 + j;
    head_warp_dot(p.w_in_pca + c * p.pca, hs, p.pca, p.tokens, acc);
    head_warp_dot(p.w_gate + c * p.pca, hs, p.pca, p.tokens, gacc);
#pragma unroll
    for (int t = 0; t < kHeadMaxTokens; ++t) {
      if (t < p.tokens && lane == t) {
        vs[t * slice + j] = head_gelu(p.pre[t * p.dim + c] + acc[t]);
        gs[t * slice + j] = head_sigmoid(gacc[t] + p.b_gate[c]);
      }
    }
  }
  cluster.sync();
  for (int64_t i = threadIdx.x; i < p.tokens * p.dim; i += kMidThreads) {
    const int64_t t = i / p.dim, c = i % p.dim;
    full[i] = cluster.map_shared_rank(&vs[0], static_cast<unsigned>(c / slice))[t * slice + c % slice];
  }
  __syncthreads();
  for (int64_t j = warp; j < slice; j += kWarps) {
    const int64_t c = c0 + j;
    head_warp_dot(p.w_r1 + c * p.dim, full, p.dim, p.tokens, acc);
#pragma unroll
    for (int t = 0; t < kHeadMaxTokens; ++t)
      if (t < p.tokens && lane == t) as[t * slice + j] = head_gelu(acc[t] + p.b_r1[c]);
  }
  cluster.sync();  // also orders this block's reads of `full` before it is rewritten
  for (int64_t i = threadIdx.x; i < p.tokens * p.dim; i += kMidThreads) {
    const int64_t t = i / p.dim, c = i % p.dim;
    full[i] = cluster.map_shared_rank(&as[0], static_cast<unsigned>(c / slice))[t * slice + c % slice];
  }
  __syncthreads();
  for (int64_t j = warp; j < slice; j += kWarps) {
    const int64_t c = c0 + j;
    head_warp_dot(p.w_r2 + c * p.dim, full, p.dim, p.tokens, acc);
#pragma unroll
    for (int t = 0; t < kHeadMaxTokens; ++t) {
      if (t < p.tokens && lane == t) {
        const float r = acc[t] + p.b_r2[c];
        p.hid[t * p.dim + c] = vs[t * slice + j] + gs[t * slice + j] * r;
      }
    }
  }
  cluster.sync();
#endif
}

// Arguments of K3, a __grid_constant__ a captured graph freezes.
struct SpecHeadOutParams {
  const float* hid;     // [kHeadMaxTokens, dim]
  const bf16_t* w_out;  // [experts, dim]
  const float* b_out;   // [experts]
  float* logits;        // [kHeadMaxTokens, experts]
  int64_t tokens;
  int64_t dim;
  int64_t experts;
};

// K3: one warp per expert row, every block holding all of hid.
__global__ __launch_bounds__(device::expert_stream::kHeadThreads) void spec_head_out_kernel(
    const __grid_constant__ SpecHeadOutParams p) {
  using namespace device::expert_stream;
  constexpr int kWarps = kHeadThreads / 32;
  __shared__ float hs[kHeadMaxTokens * kHeadMaxHidden];
  for (int64_t i = threadIdx.x; i < p.tokens * p.dim; i += kHeadThreads) hs[i] = p.hid[i];
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarps + threadIdx.x / 32;
  if (e >= p.experts) return;
  float acc[kHeadMaxTokens];
  head_warp_dot(p.w_out + e * p.dim, hs, p.dim, p.tokens, acc);
#pragma unroll
  for (int t = 0; t < kHeadMaxTokens; ++t)
    if (t < p.tokens && lane == t) p.logits[t * p.experts + e] = acc[t] + p.b_out[e];
}

/// \brief Checked host launchers for the learned L+2 head's fused path: K1-K3 then the tail (spec_head_tail.cuh), on
/// the stream of the tensors' device and without PDL, so the first starts after the native score kernel before it.
///
/// Every input is checked before the first launch and nothing syncs the host, so `run` is graph-capturable.
struct SpecHeadKernel {
  /// \brief Launches K1, K2, K3 and the tail: the head's block scores and its cap-1 read for one record.
  /// \param x bf16 [tokens, hidden], 1 <= tokens <= kHeadMaxTokens: the source layer's MoE input.
  /// \param scores fp32 [tokens_max, experts], tokens <= tokens_max: the native score kernel's biased gate scores.
  /// \param src_ids int64 [tokens, top_k]: the source layer's routes.
  /// \param src_w fp32 [tokens, top_k]: their route weights.
  /// \param positions int64 [tokens]: each token's position in the record.
  /// \param w_pca bf16 [pca, hidden].
  /// \param w_in_pca bf16 [dim, pca]: fc_in's PCA columns.
  /// \param b_in fp32 [dim]: fc_in's bias with the PCA mean folded in.
  /// \param w_in_sparse bf16 [4, experts, dim]: fc_in's route mask, route weight, gate score and margin columns.
  /// \param row_bias fp32 [rows, dim]: each target row's embedding bias.
  /// \param pos_bias fp32 [widths, widths, dim] by (width - 1, position), or empty: no position term.
  /// \param w_r1 bf16 [dim, dim]. \param b_r1 fp32 [dim]. \param w_r2 bf16 [dim, dim]. \param b_r2 fp32 [dim].
  /// \param w_gate bf16 [dim, pca]. \param b_gate fp32 [dim]: with the PCA mean folded in.
  /// \param w_out bf16 [experts, dim]. \param b_out fp32 [experts].
  /// \param agg_w1 bf16 [kAggHidden, kBlockFeatures], or empty: the noisy-or block score.
  /// \param agg_row_bias fp32 [rows, kAggHidden], or empty with agg_w1.
  /// \param agg_w2 bf16 [kAggHidden], or empty with agg_w1. \param agg_b2 fp32 [1], or empty with agg_w1.
  /// \param row The target row of row_bias and agg_row_bias.
  /// \param hot_slots int64 [slots]: the target layer's VRAM slots' experts (-1 empty), as run_spec_select takes.
  /// \param hot_capacity The live prefix of hot_slots.
  /// \param ram_slot int32 [map rows, experts]: the device map bank; >= 0 is mapped in RAM.
  /// \param target The target layer's row of ram_slot.
  /// \param threshold fp32 [1]: the frozen block-score threshold.
  /// \param h fp32 [kHeadMaxTokens, pca] workspace. \param pre fp32 [kHeadMaxTokens, dim] workspace.
  /// \param hid fp32 [kHeadMaxTokens, dim] workspace. \param logits fp32 [kHeadMaxTokens, experts] workspace.
  /// \param block_scores fp32 [experts]: the sigmoid block scores.
  /// \param features fp32 [experts, kBlockFeatures], or empty: the aggregator's inputs, for tests.
  /// \param record uint8 [>= kCandStride], device or pinned: count, expert and block score of the read.
  static void
  run(tvm::ffi::TensorView x,
      tvm::ffi::TensorView scores,
      tvm::ffi::TensorView src_ids,
      tvm::ffi::TensorView src_w,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView w_pca,
      tvm::ffi::TensorView w_in_pca,
      tvm::ffi::TensorView b_in,
      tvm::ffi::TensorView w_in_sparse,
      tvm::ffi::TensorView row_bias,
      tvm::ffi::TensorView pos_bias,
      tvm::ffi::TensorView w_r1,
      tvm::ffi::TensorView b_r1,
      tvm::ffi::TensorView w_r2,
      tvm::ffi::TensorView b_r2,
      tvm::ffi::TensorView w_gate,
      tvm::ffi::TensorView b_gate,
      tvm::ffi::TensorView w_out,
      tvm::ffi::TensorView b_out,
      tvm::ffi::TensorView agg_w1,
      tvm::ffi::TensorView agg_row_bias,
      tvm::ffi::TensorView agg_w2,
      tvm::ffi::TensorView agg_b2,
      int64_t row,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView ram_slot,
      int64_t target,
      tvm::ffi::TensorView threshold,
      tvm::ffi::TensorView h,
      tvm::ffi::TensorView pre,
      tvm::ffi::TensorView hid,
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView block_scores,
      tvm::ffi::TensorView features,
      tvm::ffi::TensorView record) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"hidden"};
    auto M_ = SymbolicSize{"tokens_max"};
    auto E_ = SymbolicSize{"experts"};
    auto K_ = SymbolicSize{"top_k"};
    auto P_ = SymbolicSize{"pca"};
    auto D_ = SymbolicSize{"dim"};
    auto R_ = SymbolicSize{"rows"};
    auto W_ = SymbolicSize{"widths"};
    expert_stream::verify_named("x", TensorMatcher({T_, H_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), x);
    expert_stream::verify_named(
        "scores", TensorMatcher({M_, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), scores);
    expert_stream::verify_named(
        "src_ids", TensorMatcher({T_, K_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), src_ids);
    expert_stream::verify_named(
        "src_w", TensorMatcher({T_, K_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), src_w);
    expert_stream::verify_named(
        "w_pca", TensorMatcher({P_, H_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_pca);
    expert_stream::verify_named(
        "w_in_pca", TensorMatcher({D_, P_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_in_pca);
    expert_stream::verify_named("b_in", TensorMatcher({D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), b_in);
    expert_stream::verify_named(
        "w_in_sparse",
        TensorMatcher({kHeadSparse, E_, D_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device),
        w_in_sparse);
    expert_stream::verify_named(
        "row_bias", TensorMatcher({R_, D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), row_bias);
    const bool position = !is_empty(pos_bias);
    if (position)
      expert_stream::verify_named(
          "pos_bias", TensorMatcher({W_, W_, D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), pos_bias);
    expert_stream::verify_named(
        "w_r1", TensorMatcher({D_, D_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_r1);
    expert_stream::verify_named("b_r1", TensorMatcher({D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), b_r1);
    expert_stream::verify_named(
        "w_r2", TensorMatcher({D_, D_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_r2);
    expert_stream::verify_named("b_r2", TensorMatcher({D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), b_r2);
    expert_stream::verify_named(
        "w_gate", TensorMatcher({D_, P_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_gate);
    expert_stream::verify_named(
        "b_gate", TensorMatcher({D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), b_gate);
    expert_stream::verify_named(
        "w_out", TensorMatcher({E_, D_}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), w_out);
    expert_stream::verify_named(
        "b_out", TensorMatcher({E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), b_out);
    expert_stream::verify_named(
        "h", TensorMatcher({kHeadMaxTokens, P_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), h);
    expert_stream::verify_named(
        "pre", TensorMatcher({kHeadMaxTokens, D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), pre);
    expert_stream::verify_named(
        "hid", TensorMatcher({kHeadMaxTokens, D_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), hid);
    RuntimeCheck(
        all_contiguous({x, scores, src_ids, src_w, w_pca, w_in_pca, b_in, w_in_sparse, row_bias, w_r1, b_r1, w_r2,
                        b_r2, w_gate, b_gate, w_out, b_out, h, pre, hid}) &&
            (!position || pos_bias.is_contiguous()),
        "spec_head: every tensor must be contiguous");
    const int64_t tokens = T_.unwrap(), pca = P_.unwrap(), dim = D_.unwrap(), rows = R_.unwrap();
    RuntimeCheck(tokens >= 1 && tokens <= kHeadMaxTokens, "x: 1..", kHeadMaxTokens, " tokens");
    RuntimeCheck(tokens <= M_.unwrap(), "scores: fewer rows than x's tokens");
    RuntimeCheck(pca >= 1 && pca <= kHeadMaxPca, "w_pca: 1..", kHeadMaxPca, " PCA rows");
    RuntimeCheck(
        dim >= kMidCluster && dim <= kHeadMaxHidden && dim % kMidCluster == 0,
        "dim: a multiple of ",
        kMidCluster,
        " up to ",
        kHeadMaxHidden);
    RuntimeCheck(row >= 0 && row < rows, "row: outside row_bias");
    RuntimeCheck(!position || tokens <= W_.unwrap(), "pos_bias: no entry for this width");
    const auto tail = tail_params(
        logits, positions, tokens, agg_w1, agg_row_bias, agg_w2, agg_b2, row, hot_slots, hot_capacity, ram_slot,
        target, threshold, block_scores, features, record, device, E_, T_);
    RuntimeCheck(!has_aggregator(agg_w1) || agg_row_bias.size(0) == rows, "agg_row_bias: not row_bias's rows");
    const int64_t experts = E_.unwrap();
    const int64_t widths = position ? W_.unwrap() : 0;
    const bool vectorized = H_.unwrap() % 8 == 0 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                            reinterpret_cast<uintptr_t>(w_pca.data_ptr()) % 16 == 0;
    const auto in = SpecHeadInParams{
        .x = static_cast<const bf16_t*>(x.data_ptr()),
        .scores = static_cast<const float*>(scores.data_ptr()),
        .src_ids = static_cast<const int64_t*>(src_ids.data_ptr()),
        .src_w = static_cast<const float*>(src_w.data_ptr()),
        .positions = static_cast<const int64_t*>(positions.data_ptr()),
        .w_pca = static_cast<const bf16_t*>(w_pca.data_ptr()),
        .b_in = static_cast<const float*>(b_in.data_ptr()),
        .w_in_sparse = static_cast<const bf16_t*>(w_in_sparse.data_ptr()),
        .row_bias = static_cast<const float*>(row_bias.data_ptr()) + row * dim,
        .pos_bias = position ? static_cast<const float*>(pos_bias.data_ptr()) + (tokens - 1) * widths * dim : nullptr,
        .h = static_cast<float*>(h.data_ptr()),
        .pre = static_cast<float*>(pre.data_ptr()),
        .tokens = tokens,
        .hidden = H_.unwrap(),
        .pca = pca,
        .dim = dim,
        .experts = experts,
        .top_k = K_.unwrap(),
        .widths = widths,
        .vectorized = vectorized,
    };
    const auto mid = SpecHeadMidParams{
        .h = static_cast<const float*>(h.data_ptr()),
        .pre = static_cast<const float*>(pre.data_ptr()),
        .w_in_pca = static_cast<const bf16_t*>(w_in_pca.data_ptr()),
        .w_r1 = static_cast<const bf16_t*>(w_r1.data_ptr()),
        .b_r1 = static_cast<const float*>(b_r1.data_ptr()),
        .w_r2 = static_cast<const bf16_t*>(w_r2.data_ptr()),
        .b_r2 = static_cast<const float*>(b_r2.data_ptr()),
        .w_gate = static_cast<const bf16_t*>(w_gate.data_ptr()),
        .b_gate = static_cast<const float*>(b_gate.data_ptr()),
        .hid = static_cast<float*>(hid.data_ptr()),
        .tokens = tokens,
        .pca = pca,
        .dim = dim,
    };
    const auto out = SpecHeadOutParams{
        .hid = static_cast<const float*>(hid.data_ptr()),
        .w_out = static_cast<const bf16_t*>(w_out.data_ptr()),
        .b_out = static_cast<const float*>(b_out.data_ptr()),
        .logits = static_cast<float*>(logits.data_ptr()),
        .tokens = tokens,
        .dim = dim,
        .experts = experts,
    };
    const auto stream = LaunchKernel::resolve_device(scores.device());
    const auto in_blocks = static_cast<unsigned>(pca + (dim + kHeadThreads - 1) / kHeadThreads);
    LaunchKernel(dim3(in_blocks), dim3(kHeadThreads), stream)(spec_head_in_kernel, in);
    LaunchKernel(dim3(kMidCluster), dim3(kMidThreads), stream)
        .enable_cluster(dim3(kMidCluster))(spec_head_mid_kernel, mid);
    const auto out_blocks = static_cast<unsigned>((experts + kHeadThreads / 32 - 1) / (kHeadThreads / 32));
    LaunchKernel(dim3(out_blocks), dim3(kHeadThreads), stream)(spec_head_out_kernel, out);
    LaunchKernel(dim3(1), dim3(kHeadTailThreads), stream)(spec_head_tail_kernel, tail);
  }

  /// \brief Launches the tail alone over precomputed token logits: the block score, sigmoid, skips, threshold and
  /// the cap-1 record, as `run`'s last launch.
  /// \param logits fp32 [kHeadMaxTokens, experts]: token logits, rows 0..tokens-1 live.
  /// \param positions int64 [tokens], 1 <= tokens <= kHeadMaxTokens: the live tokens' positions.
  /// \param agg_w1 bf16 [kAggHidden, kBlockFeatures], or empty: the noisy-or block score.
  /// \param agg_row_bias fp32 [rows, kAggHidden], or empty with agg_w1.
  /// \param agg_w2 bf16 [kAggHidden], or empty with agg_w1. \param agg_b2 fp32 [1], or empty with agg_w1.
  /// \param row The target row of agg_row_bias.
  /// \param hot_slots int64 [slots]: the target layer's VRAM slots' experts (-1 empty).
  /// \param hot_capacity The live prefix of hot_slots.
  /// \param ram_slot int32 [map rows, experts]: the device map bank; >= 0 is mapped in RAM.
  /// \param target The target layer's row of ram_slot.
  /// \param threshold fp32 [1]: the frozen block-score threshold.
  /// \param block_scores fp32 [experts]: the sigmoid block scores.
  /// \param features fp32 [experts, kBlockFeatures], or empty.
  /// \param record uint8 [>= kCandStride], device or pinned: count, expert and block score of the read.
  static void tail(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView positions,
      tvm::ffi::TensorView agg_w1,
      tvm::ffi::TensorView agg_row_bias,
      tvm::ffi::TensorView agg_w2,
      tvm::ffi::TensorView agg_b2,
      int64_t row,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView ram_slot,
      int64_t target,
      tvm::ffi::TensorView threshold,
      tvm::ffi::TensorView block_scores,
      tvm::ffi::TensorView features,
      tvm::ffi::TensorView record) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto E_ = SymbolicSize{"experts"};
    auto T_ = SymbolicSize{"tokens"};
    const int64_t tokens = positions.ndim() == 1 ? positions.size(0) : 0;
    RuntimeCheck(tokens >= 1 && tokens <= kHeadMaxTokens, "positions: 1..", kHeadMaxTokens, " tokens");
    const auto params = tail_params(
        logits, positions, tokens, agg_w1, agg_row_bias, agg_w2, agg_b2, row, hot_slots, hot_capacity, ram_slot,
        target, threshold, block_scores, features, record, device, E_, T_);
    const auto stream = LaunchKernel::resolve_device(logits.device());
    LaunchKernel(dim3(1), dim3(kHeadTailThreads), stream)(spec_head_tail_kernel, params);
  }

 private:
  static bool is_empty(tvm::ffi::TensorView v) {
    return v.ndim() == 1 && v.size(0) == 0;
  }

  static bool has_aggregator(tvm::ffi::TensorView agg_w1) {
    return !is_empty(agg_w1);
  }

  static bool all_contiguous(std::initializer_list<tvm::ffi::TensorView> views) {
    for (const auto& v : views)
      if (!v.is_contiguous()) return false;
    return true;
  }

  // Checks the tail's tensors (E_ and T_ shared with the caller's) and builds its params; row-offset pointers.
  static SpecHeadTailParams tail_params(
      tvm::ffi::TensorView logits,
      tvm::ffi::TensorView positions,
      int64_t tokens,
      tvm::ffi::TensorView agg_w1,
      tvm::ffi::TensorView agg_row_bias,
      tvm::ffi::TensorView agg_w2,
      tvm::ffi::TensorView agg_b2,
      int64_t row,
      tvm::ffi::TensorView hot_slots,
      int64_t hot_capacity,
      tvm::ffi::TensorView ram_slot,
      int64_t target,
      tvm::ffi::TensorView threshold,
      tvm::ffi::TensorView block_scores,
      tvm::ffi::TensorView features,
      tvm::ffi::TensorView record,
      host::SymbolicDevice& device,
      host::SymbolicSize& E_,
      host::SymbolicSize& T_) {
    using namespace host;
    using namespace device::expert_stream;
    using Cand = ::sglang::expert_stream::wire::SpecCandidates;
    auto RA_ = SymbolicSize{"agg_rows"};
    auto Rows_ = SymbolicSize{"map_rows"};
    auto record_mem = SymbolicDevice{};
    record_mem.set_options<kDLCUDA, kDLCUDAHost>();
    expert_stream::verify_named(
        "logits", TensorMatcher({kHeadMaxTokens, E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), logits);
    expert_stream::verify_named(
        "positions", TensorMatcher({T_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), positions);
    const bool aggregator = has_aggregator(agg_w1);
    RuntimeCheck(
        is_empty(agg_row_bias) == !aggregator && is_empty(agg_w2) == !aggregator && is_empty(agg_b2) == !aggregator,
        "agg_w1, agg_row_bias, agg_w2, agg_b2: all present (aggregator) or all empty (noisy-or)");
    if (aggregator) {
      expert_stream::verify_named(
          "agg_w1",
          TensorMatcher({kAggHidden, kBlockFeatures}).with_dtype<bf16_t>().with_device<kDLCUDA>(device),
          agg_w1);
      expert_stream::verify_named(
          "agg_row_bias",
          TensorMatcher({RA_, kAggHidden}).with_dtype<fp32_t>().with_device<kDLCUDA>(device),
          agg_row_bias);
      expert_stream::verify_named(
          "agg_w2", TensorMatcher({kAggHidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(device), agg_w2);
      expert_stream::verify_named(
          "agg_b2", TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), agg_b2);
      RuntimeCheck(
          all_contiguous({agg_w1, agg_row_bias, agg_w2, agg_b2}), "agg_w1, agg_row_bias, agg_w2, agg_b2: contiguous");
      RuntimeCheck(row >= 0 && row < RA_.unwrap(), "row: outside agg_row_bias");
    }
    RuntimeCheck(row >= 0, "row: must not be negative");
    expert_stream::verify_named(
        "hot_slots", TensorMatcher({-1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), hot_slots);
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, E_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "threshold", TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), threshold);
    expert_stream::verify_named(
        "block_scores", TensorMatcher({E_}).with_dtype<fp32_t>().with_device<kDLCUDA>(device), block_scores);
    const bool with_features = !is_empty(features);
    if (with_features)
      expert_stream::verify_named(
          "features",
          TensorMatcher({E_, kBlockFeatures}).with_dtype<fp32_t>().with_device<kDLCUDA>(device),
          features);
    expert_stream::verify_named(
        "record", TensorMatcher({-1}).with_dtype<uint8_t>().with_device<kDLCUDA, kDLCUDAHost>(record_mem), record);
    RuntimeCheck(
        all_contiguous({logits, positions, hot_slots, ram_slot, block_scores, record}) &&
            (!with_features || features.is_contiguous()),
        "spec_head tail: every tensor must be contiguous");
    const int64_t experts = E_.unwrap();
    RuntimeCheck(experts >= kSpecDepth && experts <= kHeadMaxExperts, "experts: ", kSpecDepth, "..", kHeadMaxExperts);
    RuntimeCheck(hot_capacity >= 0 && hot_capacity <= hot_slots.size(0), "hot_capacity: at most hot_slots' size");
    RuntimeCheck(target >= 0 && target < Rows_.unwrap(), "target: outside the map bank");
    RuntimeCheck(record.size(0) >= Cand::kCandStride, "record: at least ", Cand::kCandStride, " bytes");
    SpecHeadTailParams p{};
    p.logits = static_cast<decltype(p.logits)>(logits.data_ptr());
    p.positions = static_cast<decltype(p.positions)>(positions.data_ptr());
    p.tokens = static_cast<decltype(p.tokens)>(tokens);
    p.experts = static_cast<decltype(p.experts)>(experts);
    p.agg_w1 = aggregator ? static_cast<decltype(p.agg_w1)>(agg_w1.data_ptr()) : nullptr;
    const float* agg_bias = static_cast<const float*>(agg_row_bias.data_ptr());
    p.agg_bias = aggregator ? static_cast<decltype(p.agg_bias)>(agg_bias + row * kAggHidden) : nullptr;
    p.agg_w2 = aggregator ? static_cast<decltype(p.agg_w2)>(agg_w2.data_ptr()) : nullptr;
    p.agg_b2 = aggregator ? static_cast<decltype(p.agg_b2)>(agg_b2.data_ptr()) : nullptr;
    p.hot_slots = static_cast<decltype(p.hot_slots)>(hot_slots.data_ptr());
    p.hot_capacity = static_cast<decltype(p.hot_capacity)>(hot_capacity);
    p.ram_slot = static_cast<decltype(p.ram_slot)>(static_cast<const int32_t*>(ram_slot.data_ptr()) + target * experts);
    p.threshold = static_cast<decltype(p.threshold)>(threshold.data_ptr());
    p.block_scores = static_cast<decltype(p.block_scores)>(block_scores.data_ptr());
    p.features = with_features ? static_cast<decltype(p.features)>(features.data_ptr()) : nullptr;
    p.record = static_cast<decltype(p.record)>(record.data_ptr());
    return p;
  }
};

}  // namespace sglang
