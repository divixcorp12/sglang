// Test only: a second row layout, two names with only the second SM-readable, to show a format is a trait plus
// two bindings files (test_expert_stream_second_layout.py).
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace sglang::expert_stream::testing {

struct TwoNameLayout {
  static constexpr std::string_view kName = "two";
  static constexpr std::array<std::string_view, 2> kNames = {"a", "b"};
  static constexpr uint32_t kSmallMask = 0b10;
};

}  // namespace sglang::expert_stream::testing
