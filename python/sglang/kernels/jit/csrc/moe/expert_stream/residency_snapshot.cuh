// The residency capture's kernels (SGLANG_DSV41_RESIDENCY_CAPTURE), launched right after layer T's post on its stream
// and inside its captured graph. Their record layouts are the host reference's
// (python/sglang/srt/layers/moe/residency_capture_ref.py, module docstring); the kernels transcribe it.
//
//   exl3_residency_snapshot_kernel  the RAM map of the next kResidencyHorizons rows at T's prediction point
//   exl3_route_labels_kernel        each route's lane kind after the post typed the lanes
//   ResidencySnapshotKernel         the checked host launchers for both
#pragma once

#include <sgl_kernel/tensor.h>

#include "lease_device.cuh"
#include "tensor_checks.h"

namespace sglang {
namespace device::expert_stream {

constexpr int kResidencyHorizons = 4;       // residency_capture_ref.HORIZONS
constexpr int kResidencyMaskWords = 6;      // residency_capture_ref.MASK_WORDS: 384 experts, 64 per word
constexpr int kResidencyMaxExperts = kResidencyMaskWords * 64;
constexpr int kResidencyThreads = kResidencyHorizons * 32;  // one warp per horizon
constexpr int kLabelThreads = 256;
constexpr uint8_t kKindVramHot = 0;  // residency_capture_ref.KIND_VRAM_HOT

}  // namespace device::expert_stream

// Arguments of the snapshot kernel, a __grid_constant__ a captured graph freezes.
struct ResidencySnapshotParams {
  const int32_t* ram_slot;  // the device map bank's [rows, experts]: >= 0 mapped in RAM
  int64_t rows;
  int64_t experts;
  const int32_t* targets;  // [kResidencyHorizons]: each horizon's map row, -1 for none (host-precomputed)
  const uint8_t* valid;    // [kResidencyHorizons]: host-precomputed, copied as is
  uint64_t* masks;         // [kResidencyHorizons, kResidencyMaskWords]
  uint8_t* out_valid;      // [kResidencyHorizons]
  uint64_t* stamp;         // [1]: %globaltimer at the snapshot
};

// The snapshot kernel, one block of kResidencyThreads: warp h ballots horizon h's row 32 experts at a time into its
// mask words (all zero for target -1), lane 0 stores them and copies valid[h]. A target past the bank traps.
__global__ __launch_bounds__(device::expert_stream::kResidencyThreads, 1) void exl3_residency_snapshot_kernel(
    const __grid_constant__ ResidencySnapshotParams p) {
  using namespace device::expert_stream;
  if (threadIdx.x == 0) *p.stamp = global_ns();
  const int lane = threadIdx.x % 32, h = threadIdx.x / 32;
  const int32_t target = p.targets[h];
  if (target >= p.rows) __trap();
  const int32_t* row = target >= 0 ? p.ram_slot + static_cast<int64_t>(target) * p.experts : nullptr;
#pragma unroll
  for (int w = 0; w < kResidencyMaskWords; ++w) {
    uint32_t half[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int64_t e = static_cast<int64_t>(2 * w + k) * 32 + lane;
      half[k] = __ballot_sync(0xFFFFFFFFu, row != nullptr && e < p.experts && row[e] >= 0);
    }
    if (lane == 0) p.masks[h * kResidencyMaskWords + w] = static_cast<uint64_t>(half[0]) | uint64_t{half[1]} << 32;
  }
  if (lane == 0) p.out_valid[h] = p.valid[h];
}

// Arguments of the label kernel, a __grid_constant__ a captured graph freezes.
struct RouteLabelParams {
  const int64_t* routes;     // [width]: the layer's routed experts in route order
  int64_t width;
  const int64_t* planned;    // [lanes]: the post's plan, an expert per lane
  int64_t lanes;
  const int32_t* count;      // [1]: the plan's live lanes, clamped to 0..lanes
  const int32_t* lane_kind;  // [lanes]: the post's ram_slot_map.LaneKind per lane
  uint8_t* kinds;            // [width]
  uint64_t* stamp;           // [1]: %globaltimer at the labelling
};

// The label kernel: a thread per route finds its lane among the first `count` planned ones (the last match, as the
// reference's dict keeps) and writes that lane's kind, or kKindVramHot for a route outside the plan.
__global__ __launch_bounds__(device::expert_stream::kLabelThreads) void exl3_route_labels_kernel(
    const __grid_constant__ RouteLabelParams p) {
  using namespace device::expert_stream;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) *p.stamp = global_ns();
  if (i >= p.width) return;
  const int64_t count = *p.count;
  const int64_t live = count < 0 ? 0 : (count > p.lanes ? p.lanes : count);
  const int64_t e = p.routes[i];
  int32_t kind = kKindVramHot;
  for (int64_t j = 0; j < live; ++j)
    if (p.planned[j] == e) kind = p.lane_kind[j];
  p.kinds[i] = static_cast<uint8_t>(kind);
}

/// \brief Checked host launchers for the residency capture's kernels, on the stream of the tensors' device and without
/// PDL, so each starts after the post before it completed. Record layout: residency_capture_ref's module docstring.
struct ResidencySnapshotKernel {
  /// \brief Launches the snapshot kernel; see ResidencySnapshotParams.
  /// \param ram_slot int32 [rows, experts], the device map bank, experts <= 384.
  /// \param targets int32 [4], each horizon's map row or -1.
  /// \param valid uint8 [4], each horizon's precomputed valid flag.
  /// \param masks int64 [4, 6], the horizons' residency masks (uint64 bits).
  /// \param out_valid uint8 [4], valid copied.
  /// \param stamp int64 [1], the %globaltimer stamp.
  static void snapshot(
      tvm::ffi::TensorView ram_slot,
      tvm::ffi::TensorView targets,
      tvm::ffi::TensorView valid,
      tvm::ffi::TensorView masks,
      tvm::ffi::TensorView out_valid,
      tvm::ffi::TensorView stamp) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto Rows_ = SymbolicSize{"rows"};
    auto E_ = SymbolicSize{"experts"};
    expert_stream::verify_named(
        "ram_slot", TensorMatcher({Rows_, E_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), ram_slot);
    expert_stream::verify_named(
        "targets",
        TensorMatcher({kResidencyHorizons}).with_dtype<int32_t>().with_device<kDLCUDA>(device),
        targets);
    expert_stream::verify_named(
        "valid", TensorMatcher({kResidencyHorizons}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), valid);
    expert_stream::verify_named(
        "masks",
        TensorMatcher({kResidencyHorizons, kResidencyMaskWords}).with_dtype<int64_t>().with_device<kDLCUDA>(device),
        masks);
    expert_stream::verify_named(
        "out_valid",
        TensorMatcher({kResidencyHorizons}).with_dtype<uint8_t>().with_device<kDLCUDA>(device),
        out_valid);
    expert_stream::verify_named(
        "stamp", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), stamp);
    RuntimeCheck(
        ram_slot.is_contiguous() && targets.is_contiguous() && valid.is_contiguous() && masks.is_contiguous() &&
            out_valid.is_contiguous() && stamp.is_contiguous(),
        "ram_slot, targets, valid, masks, out_valid, stamp: must be contiguous");
    RuntimeCheck(
        E_.unwrap() >= 1 && E_.unwrap() <= kResidencyMaxExperts, "ram_slot: 1..", kResidencyMaxExperts, " experts");
    const auto params = ResidencySnapshotParams{
        .ram_slot = static_cast<const int32_t*>(ram_slot.data_ptr()),
        .rows = Rows_.unwrap(),
        .experts = E_.unwrap(),
        .targets = static_cast<const int32_t*>(targets.data_ptr()),
        .valid = static_cast<const uint8_t*>(valid.data_ptr()),
        .masks = static_cast<uint64_t*>(masks.data_ptr()),
        .out_valid = static_cast<uint8_t*>(out_valid.data_ptr()),
        .stamp = static_cast<uint64_t*>(stamp.data_ptr()),
    };
    const auto stream = LaunchKernel::resolve_device(masks.device());
    LaunchKernel(dim3(1), dim3(kResidencyThreads), stream)(exl3_residency_snapshot_kernel, params);
  }

  /// \brief Launches the label kernel; see RouteLabelParams.
  /// \param routes int64 [width], the routed experts in route order.
  /// \param planned int64 [lanes], the post's plan.
  /// \param count int32 [1], the plan's live lanes.
  /// \param lane_kind int32 [lanes], the post's lane kinds.
  /// \param kinds uint8 [width], each route's kind.
  /// \param stamp int64 [1], the %globaltimer stamp.
  static void labels(
      tvm::ffi::TensorView routes,
      tvm::ffi::TensorView planned,
      tvm::ffi::TensorView count,
      tvm::ffi::TensorView lane_kind,
      tvm::ffi::TensorView kinds,
      tvm::ffi::TensorView stamp) {
    using namespace host;
    using namespace device::expert_stream;
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto W_ = SymbolicSize{"width"};
    auto L_ = SymbolicSize{"lanes"};
    expert_stream::verify_named(
        "routes", TensorMatcher({W_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), routes);
    expert_stream::verify_named(
        "planned", TensorMatcher({L_}).with_dtype<int64_t>().with_device<kDLCUDA>(device), planned);
    expert_stream::verify_named(
        "count", TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device), count);
    expert_stream::verify_named(
        "lane_kind", TensorMatcher({L_}).with_dtype<int32_t>().with_device<kDLCUDA>(device), lane_kind);
    expert_stream::verify_named(
        "kinds", TensorMatcher({W_}).with_dtype<uint8_t>().with_device<kDLCUDA>(device), kinds);
    expert_stream::verify_named(
        "stamp", TensorMatcher({1}).with_dtype<int64_t>().with_device<kDLCUDA>(device), stamp);
    RuntimeCheck(
        routes.is_contiguous() && planned.is_contiguous() && count.is_contiguous() && lane_kind.is_contiguous() &&
            kinds.is_contiguous() && stamp.is_contiguous(),
        "routes, planned, count, lane_kind, kinds, stamp: must be contiguous");
    RuntimeCheck(W_.unwrap() >= 1 && L_.unwrap() >= 1, "routes, planned: must not be empty");
    const auto params = RouteLabelParams{
        .routes = static_cast<const int64_t*>(routes.data_ptr()),
        .width = W_.unwrap(),
        .planned = static_cast<const int64_t*>(planned.data_ptr()),
        .lanes = L_.unwrap(),
        .count = static_cast<const int32_t*>(count.data_ptr()),
        .lane_kind = static_cast<const int32_t*>(lane_kind.data_ptr()),
        .kinds = static_cast<uint8_t*>(kinds.data_ptr()),
        .stamp = static_cast<uint64_t*>(stamp.data_ptr()),
    };
    const auto blocks = static_cast<unsigned>((W_.unwrap() + kLabelThreads - 1) / kLabelThreads);
    const auto stream = LaunchKernel::resolve_device(kinds.device());
    LaunchKernel(dim3(blocks), dim3(kLabelThreads), stream)(exl3_route_labels_kernel, params);
  }
};

}  // namespace sglang
