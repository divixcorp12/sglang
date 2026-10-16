// The gather-time lane head (srt/layers/moe/lane_head.py): two kernels per layer gather, between the router's
// top-k and the fused route plan. The first projects each token's gate input onto the layer's PCA basis, in partial
// sums over a split of the hidden dimension; the second scores every routed expert, writes its int64 plan key, and
// then advances the layer's per-request state (score and last verify) as next_verify_head's extraction did.
// lane_head.LaneHeadReference is the definition these kernels reproduce and the test oracle.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <stdint.h>

namespace sglang {

constexpr int kLaneHeadFeatures = 17;      // lane_head.FEATURES
constexpr int kLaneHeadPcaThreads = 128;
constexpr int kLaneHeadRanksPerBlock = 16;
constexpr int kLaneHeadKeyThreads = 512;   // >= experts and >= routes
constexpr int kLaneHeadMaxTokens = 16;     // lane_head.MAX_TOKENS
constexpr int kLaneHeadMaxRoutes = 64;     // lane_head.MAX_ROUTES

SGL_DEVICE fp32_t lane_head_f32(fp32_t v) {
  return v;
}

SGL_DEVICE fp32_t lane_head_f32(bf16_t v) {
  return __bfloat162float(v);
}

/**
 * \brief partial[split, tok, r0 + i] = sum over the split's h of (x[tok, h] - mean[h]) * proj[h, r0 + i].
 *
 * Grid (tokens, rank / 16, splits); the key kernel sums the splits in order.
 */
template <typename ProjT>
__global__ __launch_bounds__(kLaneHeadPcaThreads) void lane_head_pca_kernel(
    const bf16_t* __restrict__ x,
    const fp32_t* __restrict__ mean,
    const ProjT* __restrict__ proj,
    fp32_t* __restrict__ partial,
    int32_t hidden,
    int32_t rank,
    int32_t max_tokens) {
  const int32_t tok = static_cast<int32_t>(blockIdx.x);
  const int32_t r0 = static_cast<int32_t>(blockIdx.y) * kLaneHeadRanksPerBlock;
  const int32_t split = static_cast<int32_t>(blockIdx.z);
  const int32_t chunk = (hidden + static_cast<int32_t>(gridDim.z) - 1) / static_cast<int32_t>(gridDim.z);
  const int32_t h0 = split * chunk;
  const int32_t h1 = h0 + chunk < hidden ? h0 + chunk : hidden;
  fp32_t acc[kLaneHeadRanksPerBlock];
#pragma unroll
  for (int i = 0; i < kLaneHeadRanksPerBlock; ++i) acc[i] = 0.0f;
  const bf16_t* xrow = x + static_cast<int64_t>(tok) * hidden;
  for (int32_t h = h0 + static_cast<int32_t>(threadIdx.x); h < h1; h += kLaneHeadPcaThreads) {
    const fp32_t v = __bfloat162float(xrow[h]) - mean[h];
    const ProjT* prow = proj + static_cast<int64_t>(h) * rank + r0;
#pragma unroll
    for (int i = 0; i < kLaneHeadRanksPerBlock; ++i) acc[i] += v * lane_head_f32(prow[i]);
  }
  constexpr int kWarps = kLaneHeadPcaThreads / device::kWarpThreads;
  __shared__ fp32_t sums[kWarps][kLaneHeadRanksPerBlock];
  const int lane = static_cast<int>(threadIdx.x) % device::kWarpThreads;
  const int warp = static_cast<int>(threadIdx.x) / device::kWarpThreads;
#pragma unroll
  for (int i = 0; i < kLaneHeadRanksPerBlock; ++i) {
    const fp32_t s = device::warp::reduce_sum(acc[i]);
    if (lane == 0) sums[warp][i] = s;
  }
  __syncthreads();
  if (threadIdx.x < kLaneHeadRanksPerBlock) {
    fp32_t s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) s += sums[w][threadIdx.x];
    partial[(static_cast<int64_t>(split) * max_tokens + tok) * rank + r0 + threadIdx.x] = s;
  }
}

/**
 * \brief One block. Each expert's first live route scores it and writes its key; then every expert's score decays
 * by decay_pow[live] and adds its route count, the routed ones' last verify becomes this one, and the counter steps.
 *
 * Features (lane_head.FEATURES, raw, standardized here as next_verify_head.predict does): token count one-hot
 * (1, 2, 3+), first live position one-hot (0..5), the largest gate weight (earliest on ties), log1p(score),
 * score == 0, log1p(age), never routed, resident, live / 6, odds (the value training saw). A dead row (a verify's
 * rows past the DSpark gate's cut) is not live and its routes are not counted.
 */
template <typename IdT, typename WT>
__global__ __launch_bounds__(kLaneHeadKeyThreads) void lane_head_keys_kernel(
    const IdT* __restrict__ topk_ids,
    const WT* __restrict__ topk_weights,
    const bool* __restrict__ dead,
    const fp32_t* __restrict__ partial,
    const fp32_t* __restrict__ feature_mean,
    const fp32_t* __restrict__ feature_scale,
    const fp32_t* __restrict__ z_scale,
    const fp32_t* __restrict__ expert_bias,
    const fp32_t* __restrict__ dense_w,
    const fp32_t* __restrict__ dense_b,
    const fp32_t* __restrict__ z_w,
    const fp32_t* __restrict__ expert_u,
    const int64_t* __restrict__ expert_to_slot,
    fp32_t* __restrict__ score,
    int32_t* __restrict__ last,
    int32_t* __restrict__ verify_ct,
    const fp32_t* __restrict__ decay_pow,
    int64_t* __restrict__ keys,
    int32_t tokens,
    int32_t top_k,
    int32_t rank,
    int32_t experts,
    int32_t splits,
    int32_t max_tokens,
    fp32_t odds) {
  __shared__ int32_t s_cnt[kLaneHeadKeyThreads];
  __shared__ bool s_live_row[kLaneHeadMaxTokens];
  __shared__ int32_t s_live;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t routes = tokens * top_k;
  const int32_t t = verify_ct[0];
  if (tid < experts) s_cnt[tid] = 0;
  if (tid == 0) s_live = 0;
  __syncthreads();
  if (tid < tokens) {
    bool live = dead == nullptr || !dead[tid];
    bool any = false;
    for (int32_t k = 0; k < top_k; ++k) any |= static_cast<int64_t>(topk_ids[tid * top_k + k]) >= 0;
    live = live && any;
    s_live_row[tid] = live;
    if (live) atomicAdd(&s_live, 1);
  }
  __syncthreads();
  const int32_t live_count = s_live;
  if (tid < routes) {
    const int64_t e = static_cast<int64_t>(topk_ids[tid]);
    bool first = s_live_row[tid / top_k] && e >= 0 && e < experts;
    for (int32_t j = 0; first && j < tid; ++j)
      if (s_live_row[j / top_k] && static_cast<int64_t>(topk_ids[j]) == e) first = false;
    if (first) {
      int32_t cnt = 0;
      int32_t best_tok = tid / top_k;
      fp32_t best_w = static_cast<fp32_t>(topk_weights[tid]);
      for (int32_t j = tid; j < routes; ++j) {
        if (!s_live_row[j / top_k] || static_cast<int64_t>(topk_ids[j]) != e) continue;
        ++cnt;
        const fp32_t w = static_cast<fp32_t>(topk_weights[j]);
        if (w > best_w) {
          best_w = w;
          best_tok = j / top_k;
        }
      }
      const int32_t first_pos = tid / top_k;
      const fp32_t sc = score[e];
      const int32_t prev = last[e];
      fp32_t f[kLaneHeadFeatures];
#pragma unroll
      for (int i = 0; i < kLaneHeadFeatures; ++i) f[i] = 0.0f;
      f[cnt == 1 ? 0 : cnt == 2 ? 1 : 2] = 1.0f;
      f[3 + (first_pos < 5 ? first_pos : 5)] = 1.0f;
      f[9] = best_w;
      f[10] = log1pf(sc);
      f[11] = sc <= 0.0f ? 1.0f : 0.0f;
      f[12] = prev >= 0 ? log1pf(static_cast<fp32_t>(t - prev)) : 0.0f;
      f[13] = prev >= 0 ? 0.0f : 1.0f;
      f[14] = expert_to_slot[e] >= 0 ? 1.0f : 0.0f;
      f[15] = static_cast<fp32_t>(live_count) / 6.0f;
      f[16] = odds;
      fp32_t dense = 0.0f;
#pragma unroll
      for (int i = 0; i < kLaneHeadFeatures; ++i) dense += (f[i] - feature_mean[i]) / feature_scale[i] * dense_w[i];
      fp32_t zdot = 0.0f;
      fp32_t udot = 0.0f;
      const fp32_t* urow = expert_u + e * rank;
      for (int32_t r = 0; r < rank; ++r) {
        fp32_t z = 0.0f;
        for (int32_t s = 0; s < splits; ++s) z += partial[(static_cast<int64_t>(s) * max_tokens + best_tok) * rank + r];
        const fp32_t zs = z / z_scale[r];
        zdot += zs * z_w[r];
        udot += urow[r] * zs;
      }
      const fp32_t logit = expert_bias[e] + (dense + dense_b[0]) + zdot + udot;
      const fp32_t p = 1.0f / (1.0f + expf(-logit));
      // P >= 0 orders like its bit pattern (residency_rank_keys' shape); ties go to the lower expert id.
      keys[e] = static_cast<int64_t>(__float_as_int(p + 0.0f)) * (static_cast<int64_t>(1) << 16) +
                (static_cast<int64_t>(experts) - 1 - e);
      s_cnt[e] = cnt;
    }
  }
  __syncthreads();
  if (tid < experts) {
    const int32_t c = s_cnt[tid];
    score[tid] = score[tid] * decay_pow[live_count] + static_cast<fp32_t>(c);
    if (c > 0) last[tid] = t;
  }
  if (tid == 0) verify_ct[0] = t + 1;
}

/**
 * \brief Validate, then launch the PCA and the key kernels on the tensors' stream.
 *
 * \tparam IdT   int32_t or int64_t route ids
 * \tparam WT    fp32_t or bf16_t routing weights
 * \tparam ProjT bf16_t or fp32_t PCA projection
 */
template <typename IdT, typename WT, typename ProjT>
void lane_head_keys(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView x,
    tvm::ffi::Optional<tvm::ffi::TensorView> dead,
    tvm::ffi::TensorView pca_mean,
    tvm::ffi::TensorView pca_proj,
    tvm::ffi::TensorView feature_mean,
    tvm::ffi::TensorView feature_scale,
    tvm::ffi::TensorView z_scale,
    tvm::ffi::TensorView expert_bias,
    tvm::ffi::TensorView dense_w,
    tvm::ffi::TensorView dense_b,
    tvm::ffi::TensorView z_w,
    tvm::ffi::TensorView expert_u,
    tvm::ffi::TensorView expert_to_slot,
    tvm::ffi::TensorView score,
    tvm::ffi::TensorView last,
    tvm::ffi::TensorView verify_ct,
    tvm::ffi::TensorView z_partial,
    tvm::ffi::TensorView decay_pow,
    tvm::ffi::TensorView keys,
    double odds) {
  using namespace host;
  SymbolicSize tokens{"tokens"}, top_k{"top_k"}, hidden{"hidden"}, rank{"rank"}, experts{"experts"};
  SymbolicSize features{"features"}, splits{"splits"}, max_tokens{"max_tokens"}, decay_len{"decay_len"};
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  TensorMatcher({tokens, top_k}).with_dtype<IdT>().with_device(dev).verify(topk_ids);
  TensorMatcher({tokens, top_k}).with_dtype<WT>().with_device(dev).verify(topk_weights);
  TensorMatcher({tokens, hidden}).with_dtype<bf16_t>().with_device(dev).verify(x);
  TensorMatcher({hidden}).with_dtype<fp32_t>().with_device(dev).verify(pca_mean);
  TensorMatcher({hidden, rank}).with_dtype<ProjT>().with_device(dev).verify(pca_proj);
  TensorMatcher({features}).with_dtype<fp32_t>().with_device(dev).verify(feature_mean).verify(feature_scale).verify(dense_w);
  TensorMatcher({rank}).with_dtype<fp32_t>().with_device(dev).verify(z_scale).verify(z_w);
  TensorMatcher({experts}).with_dtype<fp32_t>().with_device(dev).verify(expert_bias).verify(score);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device(dev).verify(dense_b);
  TensorMatcher({experts, rank}).with_dtype<fp32_t>().with_device(dev).verify(expert_u);
  TensorMatcher({experts}).with_dtype<int64_t>().with_device(dev).verify(expert_to_slot).verify(keys);
  TensorMatcher({experts}).with_dtype<int32_t>().with_device(dev).verify(last);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(dev).verify(verify_ct);
  TensorMatcher({splits, max_tokens, rank}).with_dtype<fp32_t>().with_device(dev).verify(z_partial);
  TensorMatcher({decay_len}).with_dtype<fp32_t>().with_device(dev).verify(decay_pow);
  const int64_t n_tokens = tokens.unwrap(), k = top_k.unwrap(), h = hidden.unwrap(), r = rank.unwrap();
  const int64_t n_experts = experts.unwrap(), n_splits = splits.unwrap(), n_max_tokens = max_tokens.unwrap();
  RuntimeCheck(features.unwrap() == kLaneHeadFeatures, "lane head: ", features.unwrap(), " features, the kernel computes ",
               kLaneHeadFeatures);
  RuntimeCheck(n_tokens >= 1 && n_tokens <= n_max_tokens && n_max_tokens <= kLaneHeadMaxTokens, "lane head: ", n_tokens,
               " tokens, the scratch holds ", n_max_tokens, " (at most ", kLaneHeadMaxTokens, ")");
  RuntimeCheck(n_tokens * k <= kLaneHeadMaxRoutes, "lane head: ", n_tokens * k, " routes, at most ", kLaneHeadMaxRoutes);
  RuntimeCheck(decay_len.unwrap() > n_max_tokens, "lane head: decay powers for ", decay_len.unwrap() - 1,
               " live tokens, the scratch holds ", n_max_tokens);
  RuntimeCheck(r % kLaneHeadRanksPerBlock == 0 && r > 0, "lane head: rank ", r, " must be a multiple of ",
               kLaneHeadRanksPerBlock);
  RuntimeCheck(n_experts <= kLaneHeadKeyThreads, "lane head: ", n_experts, " experts, at most ", kLaneHeadKeyThreads);
  RuntimeCheck(n_splits >= 1 && n_splits <= h, "lane head: ", n_splits, " PCA splits over hidden ", h);
  const bool* dead_ptr = nullptr;
  if (dead.has_value()) {
    const tvm::ffi::TensorView d = dead.value();
    RuntimeCheck(d.dtype().code == kDLBool && d.dtype().bits == 8, "lane head: dead must be a bool tensor");
    RuntimeCheck(d.is_contiguous() && d.numel() >= n_tokens, "lane head: dead must be contiguous with a row per token");
    RuntimeCheck(d.device() == dev.unwrap(), "lane head: dead is on another device");
    dead_ptr = static_cast<const bool*>(d.data_ptr());
  }
  const DLDevice device = dev.unwrap();
  LaunchKernel(
      dim3(static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(r / kLaneHeadRanksPerBlock),
           static_cast<uint32_t>(n_splits)),
      kLaneHeadPcaThreads,
      device)(
      lane_head_pca_kernel<ProjT>,
      static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const fp32_t*>(pca_mean.data_ptr()),
      static_cast<const ProjT*>(pca_proj.data_ptr()),
      static_cast<fp32_t*>(z_partial.data_ptr()),
      static_cast<int32_t>(h),
      static_cast<int32_t>(r),
      static_cast<int32_t>(n_max_tokens));
  LaunchKernel(1, kLaneHeadKeyThreads, device)(
      lane_head_keys_kernel<IdT, WT>,
      static_cast<const IdT*>(topk_ids.data_ptr()),
      static_cast<const WT*>(topk_weights.data_ptr()),
      dead_ptr,
      static_cast<const fp32_t*>(z_partial.data_ptr()),
      static_cast<const fp32_t*>(feature_mean.data_ptr()),
      static_cast<const fp32_t*>(feature_scale.data_ptr()),
      static_cast<const fp32_t*>(z_scale.data_ptr()),
      static_cast<const fp32_t*>(expert_bias.data_ptr()),
      static_cast<const fp32_t*>(dense_w.data_ptr()),
      static_cast<const fp32_t*>(dense_b.data_ptr()),
      static_cast<const fp32_t*>(z_w.data_ptr()),
      static_cast<const fp32_t*>(expert_u.data_ptr()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<fp32_t*>(score.data_ptr()),
      static_cast<int32_t*>(last.data_ptr()),
      static_cast<int32_t*>(verify_ct.data_ptr()),
      static_cast<const fp32_t*>(decay_pow.data_ptr()),
      static_cast<int64_t*>(keys.data_ptr()),
      static_cast<int32_t>(n_tokens),
      static_cast<int32_t>(k),
      static_cast<int32_t>(r),
      static_cast<int32_t>(n_experts),
      static_cast<int32_t>(n_splits),
      static_cast<int32_t>(n_max_tokens),
      static_cast<fp32_t>(odds));
}

}  // namespace sglang
