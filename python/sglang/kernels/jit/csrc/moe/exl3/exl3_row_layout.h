// The EXL3 streamed expert row as the expert-stream transport sees it: six tensors in EXL3_STREAMED_NAMES order
// (srt/layers/moe/exl3_expert_format.py); test_exl3_ram_miss_copy_engine checks the two agree.
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sglang::exl3 {

struct Exl3RowLayout {
  static constexpr std::string_view kName = "exl3";
  static constexpr std::array<std::string_view, 6> kNames = {
      "w13_trellis", "w13_suh", "w13_svh", "w2_trellis", "w2_suh", "w2_svh"};
  // suh/svh: 44.5 KB of a 13.3 MB DSV4.1 row, read by the copy wait's SMs rather than the DMA engine.
  static constexpr uint32_t kSmallMask = 0b110110;
  // w2_*: what a CPU forward first reads after its gate/up GEMVs (the down input's suh, then the down GEMVs). A
  // two-stage CPU miss (SGLANG_DSV41_CPU_TWO_STAGE) reads the image before them first.
  static constexpr uint32_t kSecondStageMask = 0b111000;
};

}  // namespace sglang::exl3
