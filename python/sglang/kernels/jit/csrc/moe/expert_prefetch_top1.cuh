#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cmath>
#include <limits>
#include <stdint.h>

namespace sglang {

constexpr int kExpertPrefetchTop1Threads = 256;

// The serving specialization intentionally accepts one score row only.  That
// makes its reduction identical to the reference ``scores.sum(dim=0)`` (the
// value is each input score) while avoiding a second, differently-rounded
// reduction implementation.  Wider batches retain the reference candidate
// bank in Python.
__global__ __launch_bounds__(kExpertPrefetchTop1Threads, 1) void select_prefetch_top1_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ expert_to_slot,
    int experts,
    int64_t* __restrict__ expert_id_out,
    bool* __restrict__ valid_out,
    int32_t* __restrict__ count_out) {
  float best_score = -std::numeric_limits<float>::infinity();
  int best_id = INT_MAX;
  for (int expert = threadIdx.x; expert < experts; expert += blockDim.x) {
    const float score = scores[expert];
    const bool eligible = ::isfinite(score) && expert_to_slot[expert] < 0;
    // Ascending id is the reference stable-descending-sort tie break.
    if (eligible && (score > best_score || (score == best_score && expert < best_id))) {
      best_score = score;
      best_id = expert;
    }
  }

  __shared__ float scores_shared[kExpertPrefetchTop1Threads];
  __shared__ int ids_shared[kExpertPrefetchTop1Threads];
  scores_shared[threadIdx.x] = best_score;
  ids_shared[threadIdx.x] = best_id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      const float other_score = scores_shared[threadIdx.x + stride];
      const int other_id = ids_shared[threadIdx.x + stride];
      if (other_score > scores_shared[threadIdx.x] ||
          (other_score == scores_shared[threadIdx.x] && other_id < ids_shared[threadIdx.x])) {
        scores_shared[threadIdx.x] = other_score;
        ids_shared[threadIdx.x] = other_id;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const bool valid = ids_shared[0] != INT_MAX;
    // -1 is safe with a zero count: the pull copy kernel gates every id load
    // on count, while consumers receive an explicit no-offer state.
    expert_id_out[0] = valid ? static_cast<int64_t>(ids_shared[0]) : -1;
    valid_out[0] = valid;
    count_out[0] = valid ? 1 : 0;
  }
}

void select_prefetch_top1_gpu(
    tvm::ffi::TensorView scores,
    tvm::ffi::TensorView expert_to_slot,
    tvm::ffi::TensorView expert_id_out,
    tvm::ffi::TensorView valid_out,
    tvm::ffi::TensorView count_out) {
  const auto stream = host::LaunchKernel::resolve_device(scores.device());
  host::LaunchKernel(1, kExpertPrefetchTop1Threads, stream)(
      select_prefetch_top1_kernel,
      static_cast<const float*>(scores.data_ptr()),
      static_cast<const int64_t*>(expert_to_slot.data_ptr()),
      static_cast<int>(scores.numel()),
      static_cast<int64_t*>(expert_id_out.data_ptr()),
      static_cast<bool*>(valid_out.data_ptr()),
      static_cast<int32_t*>(count_out.data_ptr()));
}

}  // namespace sglang
