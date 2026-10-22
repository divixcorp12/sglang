// The NVFP4 streamed expert row as the expert-stream transport sees it: six tensors in NVFP4_STREAM_TENSORS order
// (srt/layers/moe/expert_stream.py), which is also the slab order of the NVFP4 CPU kernel (csrc/nvfp4/optimized,
// quant.hpp's SlabName; its optional up-alpha slab is not streamed). Per expert, with hidden H and intermediate I:
// w13_weight packed E2M1 [2I, H/2], w2_weight [H, I/2], the two block-scale tensors E4M3 in the GPU's 128x4 swizzle,
// and g1/g2 one fp32 GEMM alpha each.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sglang::nvfp4 {

struct Nvfp4RowLayout {
  static constexpr std::string_view kName = "nvfp4";
  static constexpr std::array<std::string_view, 6> kNames = {
      "w13_weight", "w2_weight", "w13_blockscale_swizzled", "w2_blockscale_swizzled", "g1_alphas", "g2_alphas"};
  // g1/g2: 4 bytes of a multi-MB row, below what a DMA descriptor pays for, so the copy wait's SMs read them (as EXL3's
  // suh/svh); the block scales are 1/8 of the weight bytes and stay with the DMA engine.
  static constexpr uint32_t kSmallMask = 0b110000;
  // w2_weight and w2_blockscale_swizzled: the down projection's operands, what a CPU forward reads after its gate/up
  // GEMVs. g2_alphas is not in it: the NVFP4 kernel reads the down alpha when it builds the expert, before any GEMV.
  static constexpr uint32_t kSecondStageMask = 0b001010;
};

}  // namespace sglang::nvfp4
