// One NUMA node's share of the RAM tier (spec 2026-10-03-numa-node-distributor-design, Part 3): its slots of every
// row, and the serve state its RAM thread owns. NumaNodeDistributor holds one per node of the build.
#pragma once

#include "copy_engine.h"

namespace sglang::expert_stream {

// A group's part of one row.
struct GroupRow {
  int64_t lo = 0;  // the group's slots of the row are [lo, hi)
  int64_t hi = 0;
  // Its staging slots, in the order the device gives them to this node's misses.
  FixedVec<int32_t, Wire::kLanes> staging;
  bool staging_reserved = false;  // reserve_staging has run; until then the row serves no miss
  int64_t owned = 0;  // prefill-owned slots of the range (Tier::prefill_owned)
};

// Everything one group's service thread owns. The tier's single-owner rule holds per group: its service thread while
// it runs, and the caller that paused every group (or called pump()) otherwise.
template <class Source>
struct NumaGroup {
  using CpuExpertEngine = BasicCpuExpertEngine<typename Source::BuildType>;
  NumaGroup(int index, int sq_thread_cpu, Tables tables, bool direct, std::vector<GroupRow> rows,
            const std::string& trace_name)
      : index(index),
        sq_thread_cpu(sq_thread_cpu),
        reader(std::move(tables), direct),
        rows(std::move(rows)),
        trace(trace_name) {}

  bool owns(int64_t row, int64_t slot) const {
    return slot >= rows[row].lo && slot < rows[row].hi;
  }

  const int index;          // the wire's node axis: Wire::home(expert) == index for every expert it serves
  const int sq_thread_cpu;  // its ring's SQPOLL core; -1 unpinned, -2 the env's (BasicUringReader::kEnvSqThreadCpu)
  Source reader;            // its own io_uring ring
  std::vector<GroupRow> rows;
  std::unique_ptr<CpuExpertEngine> cpu;  // CPU experts on this node's cores, when enabled
  uint64_t tick = 0;                     // its LRU clock: stamps compare only within its own slots
  uint32_t next_demand = 1;
  std::atomic<uint32_t> handled{0};  // the last seq this group finished
  std::atomic<uint64_t> busy{0};     // RamTier::busy_episode of this group
  uint64_t episodes = 0;
  int64_t demands_read = 0;
  int64_t publish_refused_seen = 0;  // the reader's refusals already added to the tier's kPiecePublishRefused
  std::vector<uint8_t> packed;
  // Two-stage CPU misses (TwoSpanRows): which rows of a demand read go in two spans, and their first spans' landing.
  std::vector<uint8_t> two_span;
  std::vector<uint8_t> prefix;
  std::vector<PieceTarget> piece_targets;
  PiecePublish piece_publish;
  std::vector<uint8_t> hot_scratch;
  LineCounters<kCounterCount> core;  // RamTier::count's block for this group's thread
  // InstrBuild only: the record's demand events (miss_expert); empty in production.
  [[no_unique_address]] JobTrace<Source::BuildType::kMetrics> trace;
};

}  // namespace sglang::expert_stream
