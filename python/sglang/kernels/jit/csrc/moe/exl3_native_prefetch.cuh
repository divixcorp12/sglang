// Native next-layer prefetch for DSV4.1 EXL3 decode (plan 2026-09-25-dsv41-native-prefetch).
//
// plan (in layer T-1's captured forward, after its gather, before its fused MoE): layer T's router logits of layer
// T-1's router input are already in `logits` (tiny_gemm_bf16). Score them as T's router does (sqrt(softplus) through
// log1p, plus T's correction bias), take the top 6, and pick the best one that is neither resident in T's hot cache
// (`mapping`) nor missing from the pinned tier (the service's published host map `ram_map`, read as a hint: the
// service re-checks). Its victim is the first of T's ranked slots past the demand shortlist (`pvictims`, DIRECT's own
// order) that does not hold one of the six. Then post one request on the prefetch page and remember it in `pending`.
// Nothing about T's residency changes here.
//
// commit (at the start of layer T's forward, before its gather reads residency): if T has a pending request, wait
// for its done word. COPIED moves the victim slot to the new expert (unmap the old expert, map the new one, bump the
// slot generation); SKIPPED changes nothing. A timeout, the fatal word or the lease block's shutdown word raises the
// page's fatal word and unmaps the victim, which is then neither free nor evictable: a copy may still land there.
//
// The page layout mirrors exl3_ram_miss_host.cpp (kPf*) and ops/moe/exl3_ram_miss.py (PREFETCH_FIELDS).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace sglang {

namespace exl3_native_prefetch_device {

constexpr int kPlanBlock = 128;
constexpr int kTopK = 6;
constexpr int kMaxExperts = 512;
constexpr int kMaxVictims = 16;
constexpr int64_t kFatal = 8;                // request page: the fatal word
constexpr int64_t kLeaseHeaderShutdown = 20;  // lease block header: the shutdown word
constexpr int64_t kPfReqGen = 0;
constexpr int64_t kPfReqRow = 8;
constexpr int64_t kPfReqExpert = 12;
constexpr int64_t kPfReqDst = 16;
constexpr int64_t kPfDoneGen = 128;
constexpr uint64_t kPfTagRequest = 1;
constexpr uint64_t kPfTagCopied = 1;
constexpr uint64_t kPfTagSkipped = 2;
constexpr uint64_t kGenerationMask = (1ull << 56) - 1;

// Device counters, int64, cumulative (NATIVE_PREFETCH_COUNTERS in exl3_native_prefetch.py).
constexpr int kPosted = 0;       // requests posted
constexpr int kNoCandidate = 1;  // plans with no eligible expert among the top 6
constexpr int kRamFiltered = 2;  // top-6 experts dropped because the pinned tier lacked them
constexpr int kNoVictim = 3;     // plans with a candidate but no victim slot
constexpr int kCopied = 4;       // commits of a COPIED request
constexpr int kSkipped = 5;      // commits of a SKIPPED request
constexpr int kUsed = 6;         // COPIED experts the target layer routed
constexpr int kAborted = 7;      // commits that timed out or saw the fatal / shutdown word
constexpr int kWindowNs = 8;     // %globaltimer ns from each post to its commit's entry: the compute the copy overlaps
constexpr int kWaitNs = 9;       // ns each commit spent waiting for its done word: the copy time left exposed
constexpr int kCounters = 10;
constexpr int kPending = 6;      // pending record: {valid, expert, slot, generation, post ns, unused}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint8_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys(uint8_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint64_t ld_acquire_sys64(const uint8_t* address) {
  uint64_t value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void st_release_sys64(uint8_t* address, uint64_t value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// The router's key (moe_fused_gate, SQRTSOFTPLUS_LOG1P): sqrt(x > 20 ? x : log1p(exp(x))) + bias; NaN ranks first.
__device__ __forceinline__ float biased_score(float logit, float bias) {
  const float sp = logit > 20.0f ? logit : log1pf(expf(logit));
  const float key = sqrtf(sp) + bias;
  return isnan(key) ? INFINITY : key;
}

// (key, expert) ordering: larger key first, then lower expert.
__device__ __forceinline__ bool better(float a, int ea, float b, int eb) {
  return a > b || (a == b && ea < eb);
}

}  // namespace exl3_native_prefetch_device

template <bool kBiasBf16>
__global__ __launch_bounds__(exl3_native_prefetch_device::kPlanBlock, 1) void exl3_native_prefetch_plan_kernel(
    const float* __restrict__ logits,
    const void* __restrict__ bias,
    int64_t experts,
    const int64_t* __restrict__ mapping,
    const int64_t* __restrict__ slot_to_expert,
    const int64_t* __restrict__ pvictims,
    const bool* __restrict__ pvalid,
    int64_t victims,
    const int32_t* __restrict__ ram_map,
    const uint8_t* __restrict__ page,
    uint8_t* __restrict__ pf_page,
    int64_t row,
    int64_t* __restrict__ pending,
    int64_t* __restrict__ gen_counter,
    int64_t* __restrict__ counters) {
  using namespace exl3_native_prefetch_device;
  __shared__ float keys[kMaxExperts];
  __shared__ float warp_key[kPlanBlock / 32];
  __shared__ int warp_expert[kPlanBlock / 32];
  __shared__ int top[kTopK];
  const int tid = threadIdx.x;
  for (int e = tid; e < experts; e += kPlanBlock) {
    const float b = kBiasBf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[e])
                              : static_cast<const float*>(bias)[e];
    keys[e] = biased_score(logits[e], b);
  }
  __syncthreads();
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int best_e = 0x7FFFFFFF;
    for (int e = tid; e < experts; e += kPlanBlock) {
      if (better(keys[e], e, best, best_e)) {
        best = keys[e];
        best_e = e;
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(0xFFFFFFFFu, best, offset);
      const int other_e = __shfl_down_sync(0xFFFFFFFFu, best_e, offset);
      if (better(other, other_e, best, best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    if ((tid & 31) == 0) {
      warp_key[tid >> 5] = best;
      warp_expert[tid >> 5] = best_e;
    }
    __syncthreads();
    if (tid == 0) {
      float b = warp_key[0];
      int be = warp_expert[0];
      for (int w = 1; w < kPlanBlock / 32; ++w) {
        if (better(warp_key[w], warp_expert[w], b, be)) {
          b = warp_key[w];
          be = warp_expert[w];
        }
      }
      top[k] = be;
      keys[be] = -INFINITY;  // taken; an expert whose key is -inf is never chosen before a finite one
    }
    __syncthreads();
  }
  if (tid != 0) return;
  pending[0] = 0;
  if (ld_acquire_sys(page + kFatal) != 0) return;  // the service is failing stop: post nothing
  int candidate = -1;
  for (int k = 0; k < kTopK; ++k) {
    const int e = top[k];
    if (e < 0 || e >= experts || mapping[e] >= 0) continue;  // resident in T already
    if (*reinterpret_cast<const volatile int32_t*>(ram_map + e) < 0) {
      counters[kRamFiltered] += 1;  // not in the pinned tier: at h=1 there is no lead for an NVMe read
      continue;
    }
    candidate = e;
    break;
  }
  if (candidate < 0) {
    counters[kNoCandidate] += 1;
    return;
  }
  int64_t victim = -1;
  for (int64_t i = 0; i < victims && i < kMaxVictims; ++i) {
    if (!pvalid[i]) continue;
    const int64_t slot = pvictims[i];
    const int64_t held = slot_to_expert[slot];
    bool predicted = false;
    for (int k = 0; k < kTopK; ++k) predicted = predicted || held == top[k];
    if (predicted) continue;
    victim = slot;
    break;
  }
  if (victim < 0) {
    counters[kNoVictim] += 1;
    return;
  }
  const int64_t gen = (gen_counter[0] + 1) & static_cast<int64_t>(kGenerationMask);
  gen_counter[0] = gen == 0 ? 1 : gen;
  const uint64_t generation = static_cast<uint64_t>(gen_counter[0]);
  volatile int32_t* words = reinterpret_cast<volatile int32_t*>(pf_page);
  words[kPfReqRow / 4] = static_cast<int32_t>(row);
  words[kPfReqExpert / 4] = candidate;
  words[kPfReqDst / 4] = static_cast<int32_t>(victim);
  __threadfence_system();
  st_release_sys64(pf_page + kPfReqGen, (kPfTagRequest << 56) | generation);
  pending[0] = 1;
  pending[1] = candidate;
  pending[2] = victim;
  pending[3] = static_cast<int64_t>(generation);
  pending[4] = static_cast<int64_t>(global_ns());
  counters[kPosted] += 1;
}

__global__ __launch_bounds__(32, 1) void exl3_native_prefetch_commit_kernel(
    int64_t* __restrict__ pending,
    const uint8_t* __restrict__ pf_page,
    uint8_t* __restrict__ page,
    const uint8_t* __restrict__ lease,
    int64_t timeout_ns,
    int64_t* __restrict__ mapping,
    int64_t* __restrict__ slot_to_expert,
    uint8_t* __restrict__ slot_state,
    int64_t* __restrict__ slot_generations,
    uint8_t ready_state,
    const int64_t* __restrict__ routes,
    int64_t route_count,
    int64_t* __restrict__ counters) {
  using namespace exl3_native_prefetch_device;
  if (threadIdx.x != 0 || pending[0] == 0) return;
  const int64_t expert = pending[1];
  const int64_t slot = pending[2];
  const uint64_t generation = static_cast<uint64_t>(pending[3]);
  pending[0] = 0;
  const uint64_t entry = global_ns();
  counters[kWindowNs] += static_cast<int64_t>(entry - static_cast<uint64_t>(pending[4]));
  const uint64_t deadline = entry + static_cast<uint64_t>(timeout_ns);
  uint64_t word = ld_acquire_sys64(pf_page + kPfDoneGen);
  bool aborted = false;
  while ((word & kGenerationMask) != generation) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0 || ld_acquire_sys(page + kFatal) != 0 ||
        (lease != nullptr && ld_acquire_sys(lease + kLeaseHeaderShutdown) != 0)) {
      aborted = true;
      break;
    }
    __nanosleep(256);
    word = ld_acquire_sys64(pf_page + kPfDoneGen);
  }
  counters[kWaitNs] += static_cast<int64_t>(global_ns() - entry);
  if (aborted) {
    // The copy may still land in the victim slot: take the slot out of residency for good and fail stop.
    const int64_t old = slot_to_expert[slot];
    if (old >= 0 && mapping[old] == slot) mapping[old] = -1;
    slot_to_expert[slot] = -1;
    counters[kAborted] += 1;
    if (ld_acquire_sys(page + kFatal) == 0) st_release_sys(page + kFatal, 0xFFFFFFFFu);
    return;
  }
  if ((word >> 56) != kPfTagCopied) {
    counters[kSkipped] += 1;
    return;
  }
  const int64_t old = slot_to_expert[slot];
  if (old >= 0 && mapping[old] == slot) mapping[old] = -1;
  mapping[expert] = slot;
  slot_to_expert[slot] = expert;
  slot_state[slot] = ready_state;
  slot_generations[slot] += 1;
  counters[kCopied] += 1;
  bool used = false;
  for (int64_t i = 0; i < route_count; ++i) used = used || routes[i] == expert;
  if (used) counters[kUsed] += 1;
}

void exl3_native_prefetch_plan(
    tvm::ffi::TensorView logits,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView mapping,
    tvm::ffi::TensorView slot_to_expert,
    tvm::ffi::TensorView pvictims,
    tvm::ffi::TensorView pvalid,
    int64_t ram_map_address,
    int64_t page_address,
    int64_t pf_page_address,
    int64_t row,
    tvm::ffi::TensorView pending,
    tvm::ffi::TensorView gen_counter,
    tvm::ffi::TensorView counters) {
  const int64_t experts = logits.size(logits.dim() - 1);
  if (experts > exl3_native_prefetch_device::kMaxExperts) throw std::runtime_error("native prefetch: too many experts");
  const bool bf16 = bias.dtype().code == kDLBfloat;
  const auto stream = host::LaunchKernel::resolve_device(logits.device());
  auto launch = [&](auto kernel) {
    host::LaunchKernel(1, exl3_native_prefetch_device::kPlanBlock, stream)(
        kernel,
        static_cast<const float*>(logits.data_ptr()),
        static_cast<const void*>(bias.data_ptr()),
        experts,
        static_cast<const int64_t*>(mapping.data_ptr()),
        static_cast<const int64_t*>(slot_to_expert.data_ptr()),
        static_cast<const int64_t*>(pvictims.data_ptr()),
        static_cast<const bool*>(pvalid.data_ptr()),
        static_cast<int64_t>(pvictims.size(0)),
        reinterpret_cast<const int32_t*>(ram_map_address),
        reinterpret_cast<const uint8_t*>(page_address),
        reinterpret_cast<uint8_t*>(pf_page_address),
        row,
        static_cast<int64_t*>(pending.data_ptr()),
        static_cast<int64_t*>(gen_counter.data_ptr()),
        static_cast<int64_t*>(counters.data_ptr()));
  };
  if (bf16) {
    launch(exl3_native_prefetch_plan_kernel<true>);
  } else {
    launch(exl3_native_prefetch_plan_kernel<false>);
  }
}

void exl3_native_prefetch_commit(
    tvm::ffi::TensorView pending,
    int64_t pf_page_address,
    int64_t page_address,
    int64_t lease_address,
    int64_t timeout_ns,
    tvm::ffi::TensorView mapping,
    tvm::ffi::TensorView slot_to_expert,
    tvm::ffi::TensorView slot_state,
    tvm::ffi::TensorView slot_generations,
    int64_t ready_state,
    tvm::ffi::TensorView routes,
    tvm::ffi::TensorView counters) {
  const auto stream = host::LaunchKernel::resolve_device(pending.device());
  host::LaunchKernel(1, 32, stream)(
      exl3_native_prefetch_commit_kernel,
      static_cast<int64_t*>(pending.data_ptr()),
      reinterpret_cast<const uint8_t*>(pf_page_address),
      reinterpret_cast<uint8_t*>(page_address),
      reinterpret_cast<const uint8_t*>(lease_address),
      timeout_ns,
      static_cast<int64_t*>(mapping.data_ptr()),
      static_cast<int64_t*>(slot_to_expert.data_ptr()),
      static_cast<uint8_t*>(slot_state.data_ptr()),
      static_cast<int64_t*>(slot_generations.data_ptr()),
      static_cast<uint8_t>(ready_state),
      static_cast<const int64_t*>(routes.data_ptr()),
      static_cast<int64_t>(routes.size(0)),
      static_cast<int64_t*>(counters.data_ptr()));
}

}  // namespace sglang
