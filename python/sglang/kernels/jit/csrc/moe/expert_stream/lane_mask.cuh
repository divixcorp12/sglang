// Lane masks of the RAM-miss wire (lease_layout.h LaneMask): one u32 up to 32 lanes, one u64 above. The u32 overloads
// are the intrinsics the narrow code always used, so a narrow build compiles to what it did.
#pragma once

#include <cstdint>

namespace sglang::expert_stream {

__device__ __forceinline__ int lowest_lane(uint32_t mask) {
  return __ffs(mask) - 1;
}
__device__ __forceinline__ int lowest_lane(uint64_t mask) {
  return __ffsll(static_cast<long long>(mask)) - 1;
}
__device__ __forceinline__ int lane_count(uint32_t mask) {
  return __popc(mask);
}
__device__ __forceinline__ int lane_count(uint64_t mask) {
  return __popcll(mask);
}

// A lane mask from its words: `lo`, and for a u64 mask `hi` as the high half.
template <typename MaskT>
__device__ __forceinline__ MaskT load_lane_mask(const int32_t* lo, const int32_t* hi) {
  if constexpr (sizeof(MaskT) == 8) {
    return static_cast<MaskT>(static_cast<uint32_t>(*lo)) | static_cast<MaskT>(static_cast<uint32_t>(*hi)) << 32;
  } else {
    return static_cast<uint32_t>(*lo);
  }
}

}  // namespace sglang::expert_stream
