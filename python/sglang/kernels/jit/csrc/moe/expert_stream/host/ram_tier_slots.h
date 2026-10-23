// The slot state of one streamed row (Tier) and the census of what a request could take from it (VictimCensus).
#pragma once

#include <cstdint>
#include <vector>

namespace sglang {
namespace expert_stream {

// The slot state of one streamed row. Every member is the tier owner's, except `rows_demand`, which other threads read.
//
// Slots are indexed 0..capacity-1 and are kFree, kReady (holds an expert's bytes, mapped), kStaging (a landing
// place for a miss, never mapped), or kSpec (a speculative pool slot, never mapped; RamTier::reserve_spec_pool).
// `slot_to_expert` and `expert_slot` are inverse maps, -1 where unset. `stamp` is the slot's last-use tick, the LRU
// order.
//
// Prefill share: `prefill_owned[slot]` is 1 while a row a prefill admitted has not been used by decode, and the
// owning group's GroupRow::owned counts them. Only take_admit_slot_locked sets it, so both stay zero with the share
// off. The row's staging slots and map chain are its groups' (GroupRow).
struct Tier {
  int64_t capacity = 0;
  std::vector<int32_t> slot_to_expert;
  std::vector<uint8_t> state;
  std::vector<uint64_t> stamp;
  std::vector<int32_t> expert_slot;  // assigned slot (READY) or -1
  std::vector<uint8_t> hot;
  std::vector<uint8_t> filling;  // a prefill fill is still writing this kReady slot: never a victim, never released
  std::vector<uint8_t> prefill_owned;
  // Added to by any group's serve() (a relaxed fetch_add through std::atomic_ref), read relaxed by layer_rows from
  // any thread.
  int64_t rows_demand = 0;
};

// What a request could take from a row, counted without taking anything.
struct VictimCensus {
  int64_t free = 0;       // FREE slots
  int64_t evictable = 0;  // READY, not hot, not requested, not filling
};

}  // namespace expert_stream
}  // namespace sglang
