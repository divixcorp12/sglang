// The NVMe-to-RAM prefetch's shared state (spec docs/superpowers/specs/2026-10-08-dsv41-ram-prefetch-design.md,
// Phase 1).
//
//   SpecPool   per streamed row and NUMA group, `share` RAM-tier slots in state kSpec that the device never maps; an
//              entry is empty, reading expert e, landed with expert e, or swapped: holding a swap's victim until the
//              row's delta that evicts it is published
//
// Threads. An entry's word, for_seq and landed are atomics any thread may load. Its slot, freed_at, and every
// transition of its word out of empty, landed or swapped, is under its group's mutex: the group's speculative thread
// claims entries, the group's service thread swaps them (RamTier::take_pooled_locked), and the row's publisher, any
// group's service thread, releases swapped ones (RamTier::release_swapped). Only the claimer stores a reading word's
// successor.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sglang::expert_stream {

enum : uint32_t { kPoolEmpty = 0, kPoolReading = 1, kPoolLanded = 2, kPoolSwapped = 3 };

// A swapped entry's expert bits: 0xFFFF names no expert, since reserve_spec_pool refuses 65536 experts or more.
constexpr int32_t kPoolNoExpert = 0xFFFF;

// An entry's word: the state above bit 16, the expert below (RamTier::reserve_spec_pool refuses 65536 experts or more).
inline uint32_t pool_word(uint32_t state, int32_t expert) {
  return state << 16 | (static_cast<uint32_t>(expert) & 0xFFFFu);
}
inline uint32_t pool_state(uint32_t word) {
  return word >> 16;
}
inline int32_t pool_expert(uint32_t word) {
  return static_cast<int32_t>(word & 0xFFFFu);
}

struct PoolEntry {
  int32_t slot = -1;  // under the group's mutex
  std::atomic<uint32_t> word{kPoolEmpty};
  std::atomic<uint32_t> for_seq{0};  // the source record whose scoring issued the read
  std::atomic<uint64_t> landed{0};   // landing order: with no empty entry, the oldest landed one is reclaimed
  uint64_t freed_at = 0;             // swapped: the map chain whose delta evicts the victim; under the group's mutex
};

class SpecPool {
 public:
  static constexpr int kMaxShare = 4;  // Python mirror: ram_prefetch.MAX_SPEC_SHARE

  SpecPool(int64_t rows, int groups, int share)
      : groups_(groups),
        share_(share),
        entries_(std::make_unique<PoolEntry[]>(static_cast<size_t>(rows * groups * share))),
        mutexes_(std::make_unique<std::mutex[]>(static_cast<size_t>(groups))) {}

  int share() const {
    return share_;
  }
  PoolEntry& entry(int64_t row, int g, int i) {
    return entries_[(row * groups_ + g) * share_ + i];
  }
  const PoolEntry& entry(int64_t row, int g, int i) const {
    return entries_[(row * groups_ + g) * share_ + i];
  }
  std::mutex& mutex(int g) {
    return mutexes_[g];
  }

  // Group g's entry of `row` holding `expert`, reading or landed, or -1. A kPoolSwapped entry (its slot went into the
  // row's mapping) carries kPoolNoExpert, so it never matches. Atomic loads: any thread.
  int find(int64_t row, int g, int32_t expert) const {
    for (int i = 0; i < share_; ++i) {
      const uint32_t word = entry(row, g, i).word.load(std::memory_order_acquire);
      if (word != kPoolEmpty && pool_expert(word) == expert) return i;
    }
    return -1;
  }

  // True when some group's entry of `row` holds `expert` for a record other than `seq`. The scorer skips such an
  // expert; one issued for `seq` itself stays ranked, so both groups' lists agree whichever group read first.
  bool pooled_before(int64_t row, int32_t expert, uint32_t seq) const {
    for (int g = 0; g < groups_; ++g) {
      const int i = find(row, g, expert);
      if (i >= 0 && entry(row, g, i).for_seq.load(std::memory_order_relaxed) != seq) return true;
    }
    return false;
  }

  // Under mutex(g): an empty entry of `row`, else its oldest landed one not issued from `seq`, else -1. A job never
  // reclaims its own pick.
  int claimable_locked(int64_t row, int g, uint32_t seq) const {
    int oldest = -1;
    for (int i = 0; i < share_; ++i) {
      const PoolEntry& e = entry(row, g, i);
      const uint32_t state = pool_state(e.word.load(std::memory_order_relaxed));
      if (state == kPoolEmpty) return i;
      if (state == kPoolLanded && e.for_seq.load(std::memory_order_relaxed) != seq &&
          (oldest < 0 || e.landed.load(std::memory_order_relaxed) <
                             entry(row, g, oldest).landed.load(std::memory_order_relaxed)))
        oldest = i;
    }
    return oldest;
  }

  uint64_t next_landing() {
    return landings_.fetch_add(1, std::memory_order_relaxed) + 1;
  }

 private:
  int groups_;
  int share_;
  std::unique_ptr<PoolEntry[]> entries_;
  std::unique_ptr<std::mutex[]> mutexes_;
  std::atomic<uint64_t> landings_{0};
};

// The RAM prefetch's settings (RamTier::enable_ram_prefetch).
struct RamPrefetchConfig {
  std::vector<int32_t> target;  // per source row: the next streamed layer's row, or -1 (none, or no biased gate)
  std::vector<int32_t> gate;    // per source row: the target's index in gates and bias, or -1
  const uint16_t* gates = nullptr;  // bf16 [gate_count, experts, hidden], host memory the caller keeps alive
  const float* bias = nullptr;      // fp32 [gate_count, experts]
  int64_t gate_count = 0;
  int64_t hidden = 0;
  int top_k = 0;
  int per_token = 0;
  int per_layer = 0;
  std::vector<std::vector<int>> cores;  // per group: its speculative thread's cores; empty inherits the caller's
};

// One record handed from a group's service thread to its speculative thread: a record of `row` that staged `tokens`
// live inputs.
struct SpecJob {
  uint32_t seq = 0;
  int64_t row = 0;
  int64_t tokens = 1;
};

}  // namespace sglang::expert_stream
