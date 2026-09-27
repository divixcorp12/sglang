// What the transport needs to know about a streamed row's format at compile time; everything else is a runtime table.
#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

// Precondition on every instantiation, not checked by the concept itself: bit i of kSmallMask names layout
// name i (kNames[i]), and the transport's copy table (RamTier::set_copy_table) is required to list exactly
// one entry per layout name, in kNames order -- so a copy-table entry index and a layout-name index are the
// same number. EXL3's copy table satisfies this (Ruling 15); a layout that builds its copy table in a
// different order, or with a different entry count than kNames.size(), breaks the sm_mask check below.
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
