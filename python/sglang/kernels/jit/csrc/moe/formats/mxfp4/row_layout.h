// The MXFP4 streamed expert row (MiMo V2.6 Pro, Mxfp4FlashinferCutlassMoEMethod) as the expert-stream transport sees it:
// the four weight tensors of the FlashInfer CUTLASS MXFP4 MoE (packed E2M1 weights and their E8M0 scales, after the host
// scale interleave) and the two per-expert side tensors the runner indexes per expert, one fp32 each
// (mxfp4_weight_global_scale, swiglu_limit; srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py). They are streamed
// as row tensors so a hot-cache slot id selects them together with the weights.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sglang::mxfp4 {

struct Mxfp4RowLayout {
  static constexpr std::string_view kName = "mxfp4";
  static constexpr std::array<std::string_view, 6> kNames = {
      "w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale", "mxfp4_weight_global_scale", "swiglu_limit"};
  // The two side tensors: 4 bytes each of a multi-MB row, read by the copy wait's SMs like EXL3's suh/svh; the E8M0
  // scales (one byte per 32 weights) stay with the DMA engine.
  static constexpr uint32_t kSmallMask = 0b110000;
  // w2_weight and w2_weight_scale: the down projection, what a two-stage CPU miss would read second. MiMo has no CPU
  // kernel today, so this only fixes the cut a later MXFP4 CPU kernel gets; the side tensors feed gemm1 and stay first.
  static constexpr uint32_t kSecondStageMask = 0b001100;
};

}  // namespace sglang::mxfp4
