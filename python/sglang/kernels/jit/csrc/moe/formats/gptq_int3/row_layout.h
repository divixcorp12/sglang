#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace sglang::gptq_int3 {
struct RowLayout {
  static constexpr std::string_view kName = "gptq_int3";
  static constexpr std::array<std::string_view, 6> kNames = {
      "w13_low2", "w13_high1", "w13_scales", "w2_low2", "w2_high1", "w2_scales"};
  static constexpr uint32_t kSmallMask = 0, kSecondStageMask = 0b111000;
};
}  // namespace sglang::gptq_int3
