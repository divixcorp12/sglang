// The gather-time lane head (srt/layers/moe/lane_head.py): two kernels per layer gather, between the router's
// top-k and the fused route plan. The first projects each live token's gate input onto the layer's PCA basis; the
// second scores every routed expert and writes its int64 plan key. See lane_head.py for the definition the kernels
// reproduce; the Python reference there is the test oracle.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

namespace sglang {

constexpr int kLaneHeadFeatures = 17;
constexpr int kLaneHeadPcaThreads = 256;
constexpr int kLaneHeadRanksPerBlock = 16;
constexpr int kLaneHeadKeyThreads = 128;  // >= tokens * top_k (at most 64 routes with dedup)
constexpr float kLaneHeadKeyScale = 2147483648.0f;  // 2^31

/**
 * \brief z[tok, r0 .. r0+15] = sum_h (x[tok, h] - mean[h]) * proj[h, r0 + i]: one block per (token, 16 ranks).
 *
 * \param x      bf16 [tokens, hidden] gate input
 * \param mean   fp32 [hidden]
 * \param proj   bf16 [hidden, rank], rank a multiple of 16
 * \param z      fp32 [max_tokens, rank] out
 */
template <bool kUsePDL>
__global__ __launch_bounds__(kLaneHeadPcaThreads) void lane_head_pca_kernel(
    const bf16_t* __restrict__ x,
    const fp32_t* __restrict__ mean,
    const bf16_t* __restrict__ proj,
    fp32_t* __restrict__ z,
    int32_t hidden,
    int32_t rank) {
  using vec8 = device::AlignedVector<bf16_t, 8>;
  const int32_t tok = static_cast<int32_t>(blockIdx.x);
  const int32_t r0 = static_cast<int32_t>(blockIdx.y) * kLaneHeadRanksPerBlock;
  device::PDLWaitPrimary<kUsePDL>();
  fp32_t acc[kLaneHeadRanksPerBlock];
#pragma unroll
  for (int i = 0; i < kLaneHeadRanksPerBlock; ++i) acc[i] = 0.0f;
  const bf16_t* xrow = x + static_cast<int64_t>(tok) * hidden;
  for (int32_t h = static_cast<int32_t>(threadIdx.x); h < hidden; h += kLaneHeadPcaThreads) {
    const fp32_t v = static_cast<fp32_t>(xrow[h]) - mean[h];
    const bf16_t* prow = proj + static_cast<int64_t>(h) * rank + r0;
    vec8 lo, hi;
    lo.load(prow, 0);
    hi.load(prow, 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      acc[i] += v * static_cast<fp32_t>(lo[i]);
      acc[8 + i] += v * static_cast<fp32_t>(hi[i]);
    }
  }
  __shared__ fp32_t partial[kLaneHeadPcaThreads / device::kWarpThreads][kLaneHeadRanksPerBlock];
  const int lane = device::get_lane_id();
  const int warp = static_cast<int>(threadIdx.x) / device::kWarpThreads;
#pragma unroll
  for (int i = 0; i < kLaneHeadRanksPerBlock; ++i) {
    const fp32_t s = warp::reduce_sum(acc[i]);
    if (lane == 0) partial[warp][i] = s;
  }
  __syncthreads();
  if (threadIdx.x < kLaneHeadRanksPerBlock) {
    fp32_t s = 0.0f;
#pragma unroll
    for (int w = 0; w < kLaneHeadPcaThreads / device::kWarpThreads; ++w) s += partial[w][threadIdx.x];
    z[static_cast<int64_t>(tok) * rank + r0 + threadIdx.x] = s;
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

/**
 * \brief One thread per route; the first route of an expert scores it and writes its key.
 *
 * Features (lane_head.FEATURES): token count one-hot (1, 2, 3+), first position one-hot (0..5), the largest gate
 * weight, log1p(score), score == 0, log1p(age), never routed, resident, live / 6, odds (1). z is the PCA of the
 * token with the largest gate weight (earliest on ties).
 */
template <typename IdT, typename WT>
__global__ __launch_bounds__(kLaneHeadKeyThreads) void lane_head_keys_kernel(
    const IdT* __restrict__ topk_ids,
    const WT* __restrict__ topk_weights,
    const fp32_t* __restrict__ z,
    const fp32_t* __restrict__ expert_bias,
    const fp32_t* __restrict__ expert_u,
    const fp32_t* __restrict__ bias,
    const fp32_t* __restrict__ dense_w,
    const fp32_t* __restrict__ z_w,
    const fp32_t* __restrict__ insert_scores,
    const int32_t* __restrict__ last_routed,
    const int64_t* __restrict__ expert_to_slot,
    const int64_t* __restrict__ forward,
    int64_t* __restrict__ keys,
    int64_t* __restrict__ launches,
    int32_t tokens,
    int32_t top_k,
    int32_t rank,
    int32_t experts,
    fp32_t odds) {
  __shared__ int32_t live_count;
  const int32_t routes = tokens * top_k;
  const int32_t j = static_cast<int32_t>(threadIdx.x);
  if (j == 0) live_count = 0;
  __syncthreads();
  // live tokens: those with any route >= 0
  if (j < tokens) {
    bool any = false;
    for (int32_t k = 0; k < top_k; ++k) any |= static_cast<int64_t>(topk_ids[j * top_k + k]) >= 0;
    if (any) atomicAdd(&live_count, 1);
  }
  __syncthreads();
  if (j == 0) atomicAdd(reinterpret_cast<unsigned long long*>(launches), 1ull);
  if (j >= routes) return;
  const int64_t e = static_cast<int64_t>(topk_ids[j]);
  if (e < 0 || e >= experts) return;
  for (int32_t jj = 0; jj < j; ++jj)
    if (static_cast<int64_t>(topk_ids[jj]) == e) return;  // not the first route of e
  int32_t cnt = 0;
  int32_t best_tok = -1;
  fp32_t best_w = -1.0f;
  for (int32_t jj = j; jj < routes; ++jj) {
    if (static_cast<int64_t>(topk_ids[jj]) != e) continue;
    ++cnt;
    const fp32_t w = static_cast<fp32_t>(topk_weights[jj]);
    const int32_t tok = jj / top_k;
    if (w > best_w) {
      best_w = w;
      best_tok = tok;
    }
  }
  const int32_t first_pos = j / top_k;
  fp32_t f[kLaneHeadFeatures];
#pragma unroll
  for (int i = 0; i < kLaneHeadFeatures; ++i) f[i] = 0.0f;
  f[cnt == 1 ? 0 : cnt == 2 ? 1 : 2] = 1.0f;
  f[3 + (first_pos < 5 ? first_pos : 5)] = 1.0f;
  f[9] = best_w;
  const fp32_t score = insert_scores[e] > 0.0f ? insert_scores[e] : 0.0f;
  f[10] = log1pf(score);
  f[11] = insert_scores[e] <= 0.0f ? 1.0f : 0.0f;
  const int32_t last = last_routed[e];
  const int64_t age = last >= 0 ? forward[0] - static_cast<int64_t>(last) : -1;
  f[12] = age >= 0 ? log1pf(static_cast<fp32_t>(age)) : 0.0f;
  f[13] = age < 0 ? 1.0f : 0.0f;
  f[14] = expert_to_slot[e] >= 0 ? 1.0f : 0.0f;
  f[15] = static_cast<fp32_t>(live_count) / 6.0f;
  f[16] = odds;  // the value training saw (lane_head.LaneHeadMeta.odds)
  fp32_t logit = expert_bias[e] + bias[0];
#pragma unroll
  for (int i = 0; i < kLaneHeadFeatures; ++i) logit += dense_w[i] * f[i];
  const fp32_t* zrow = z + static_cast<int64_t>(best_tok) * rank;
  const fp32_t* urow = expert_u + e * rank;
  fp32_t zsum = 0.0f;
  fp32_t usum = 0.0f;
  for (int32_t r = 0; r < rank; ++r) {
    zsum += z_w[r] * zrow[r];
    usum += urow[r] * zrow[r];
  }
  logit += zsum + usum;
  const fp32_t p = 1.0f / (1.0f + expf(-logit));
  keys[e] = static_cast<int64_t>(p * kLaneHeadKeyScale) * (static_cast<int64_t>(1) << 16) +
            (static_cast<int64_t>(experts) - 1 - e);
}

/**
 * \brief Validate, then launch the PCA and the key kernels on the tensors' stream.
 *
 * \tparam IdT     int32_t or int64_t route ids
 * \tparam WT      fp32_t or bf16_t routing weights
 * \tparam kUsePDL PDL between the two kernels
 */
template <typename IdT, typename WT, bool kUsePDL>
void lane_head_keys(
    tvm::ffi::TensorView topk_ids,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView pca_mean,
    tvm::ffi::TensorView pca_proj,
    tvm::ffi::TensorView expert_bias,
    tvm::ffi::TensorView expert_u,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView dense_w,
    tvm::ffi::TensorView z_w,
    tvm::ffi::TensorView insert_scores,
    tvm::ffi::TensorView last_routed,
    tvm::ffi::TensorView expert_to_slot,
    tvm::ffi::TensorView forward,
    tvm::ffi::TensorView z,
    tvm::ffi::TensorView keys,
    tvm::ffi::TensorView launches,
    double odds) {
  using namespace host;
  SymbolicSize tokens{"tokens"}, top_k{"top_k"}, hidden{"hidden"}, rank{"rank"}, experts{"experts"}, max_tokens{"max_tokens"};
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  TensorMatcher({tokens, top_k}).with_dtype<IdT>().with_device<kDLCUDA>(dev).verify(topk_ids);
  TensorMatcher({tokens, top_k}).with_dtype<WT>().with_device<kDLCUDA>(dev).verify(topk_weights);
  TensorMatcher({tokens, hidden}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(x);
  TensorMatcher({hidden}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(pca_mean);
  TensorMatcher({hidden, rank}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).ensure_alignment(16).verify(pca_proj);
  TensorMatcher({experts}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(expert_bias).verify(insert_scores);
  TensorMatcher({experts, rank}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(expert_u);
  TensorMatcher({1}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(bias);
  TensorMatcher({kLaneHeadFeatures}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(dense_w);
  TensorMatcher({rank}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(z_w);
  TensorMatcher({experts}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(last_routed);
  TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(forward).verify(launches);
  TensorMatcher({max_tokens, rank}).with_dtype<fp32_t>().with_device<kDLCUDA>(dev).verify(z);
  TensorMatcher({experts}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(keys).verify(expert_to_slot);
  const int64_t n_tokens = tokens.unwrap(), k = top_k.unwrap(), h = hidden.unwrap(), r = rank.unwrap(),
                n_experts = experts.unwrap();
  CHECK_HOST(n_tokens >= 1 && n_tokens <= max_tokens.unwrap()) << "lane head: tokens " << n_tokens << " exceed the z scratch " << max_tokens.unwrap();
  CHECK_HOST(n_tokens * k <= kLaneHeadKeyThreads) << "lane head: " << n_tokens * k << " routes, at most " << kLaneHeadKeyThreads;
  CHECK_HOST(r % kLaneHeadRanksPerBlock == 0 && r > 0) << "lane head: rank " << r << " must be a multiple of " << kLaneHeadRanksPerBlock;
  CHECK_HOST(n_experts <= 65536) << "lane head: the key's tie-break holds at most 65536 experts";
  const DLDevice device = dev.unwrap();
  LaunchKernel(dim3(static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(r / kLaneHeadRanksPerBlock)), kLaneHeadPcaThreads, device)
      .enable_pdl(kUsePDL)(
          lane_head_pca_kernel<kUsePDL>,
          static_cast<const bf16_t*>(x.data_ptr()),
          static_cast<const fp32_t*>(pca_mean.data_ptr()),
          static_cast<const bf16_t*>(pca_proj.data_ptr()),
          static_cast<fp32_t*>(z.data_ptr()),
          static_cast<int32_t>(h),
          static_cast<int32_t>(r));
  LaunchKernel(1, kLaneHeadKeyThreads, device)(
      lane_head_keys_kernel<IdT, WT>,
      static_cast<const IdT*>(topk_ids.data_ptr()),
      static_cast<const WT*>(topk_weights.data_ptr()),
      static_cast<const fp32_t*>(z.data_ptr()),
      static_cast<const fp32_t*>(expert_bias.data_ptr()),
      static_cast<const fp32_t*>(expert_u.data_ptr()),
      static_cast<const fp32_t*>(bias.data_ptr()),
      static_cast<const fp32_t*>(dense_w.data_ptr()),
      static_cast<const fp32_t*>(z_w.data_ptr()),
      static_cast<const fp32_t*>(insert_scores.data_ptr()),
      static_cast<const int32_t*>(last_routed.data_ptr()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<const int64_t*>(forward.data_ptr()),
      static_cast<int64_t*>(keys.data_ptr()),
      static_cast<int64_t*>(launches.data_ptr()),
      static_cast<int32_t>(n_tokens),
      static_cast<int32_t>(k),
      static_cast<int32_t>(r),
      static_cast<int32_t>(n_experts),
      static_cast<fp32_t>(odds));
}

}  // namespace sglang
