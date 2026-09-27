// What the transport needs to know about a streamed row's format at compile time; everything else is a runtime table.
#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

template <typename L>
concept ExpertRowLayout = requires {
  { L::kName } -> std::convertible_to<std::string_view>;
  { L::kSmallMask } -> std::convertible_to<uint32_t>;
  L::kNames.size();
} && (L::kNames.size() >= 1) && (L::kNames.size() <= 32) && ((uint64_t{L::kSmallMask} >> L::kNames.size()) == 0);

template <ExpertRowLayout L>
inline constexpr int64_t kNumNames = static_cast<int64_t>(L::kNames.size());

// "<name> RAM miss: ": the prefix every service error carries, so EXL3's messages are unchanged.
template <ExpertRowLayout L>
std::string error_prefix() {
  return std::string(L::kName) + " RAM miss: ";
}

}  // namespace sglang::expert_stream
