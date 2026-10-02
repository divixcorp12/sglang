// ExpertRowLayout: what the transport needs to know about a streamed row's format at compile time.
//
// Everything else about a row is a runtime table. A layout supplies its name, its tensor names in copy-table order and
// the mask of names the copy wait kernel may read with SMs; error_prefix() derives the prefix every service error
// carries.
#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>

namespace sglang::expert_stream {

// A compile-time description of one streamed row format.
//
// Requires `kName`, `kSmallMask` (bit i marks kNames[i] as readable by the copy wait's SMs) and `kNames`, between 1
// and 32 names with no kSmallMask bit past the last.
//
// Precondition on every instantiation, which the concept cannot check: the transport's copy table
// (RamTier::set_copy_table) lists exactly one entry per layout name, in kNames order, so a copy-table entry index and
// a layout-name index are the same number. EXL3's copy table satisfies this. A layout whose copy table is built in a
// different order, or has a different entry count than kNames.size(), breaks the sm_mask check.
template <typename L>
concept ExpertRowLayout = requires {
  { L::kName } -> std::convertible_to<std::string_view>;
  { L::kSmallMask } -> std::convertible_to<uint32_t>;
  L::kNames.size();
} && (L::kNames.size() >= 1) && (L::kNames.size() <= 32) && ((uint64_t{L::kSmallMask} >> L::kNames.size()) == 0);

template <ExpertRowLayout L>
inline constexpr int64_t kNumNames = static_cast<int64_t>(L::kNames.size());

// Returns "<name> RAM miss: ", the prefix every service error carries (EXL3's messages are unchanged).
template <ExpertRowLayout L>
std::string error_prefix() {
  return std::string(L::kName) + " RAM miss: ";
}

}  // namespace sglang::expert_stream
