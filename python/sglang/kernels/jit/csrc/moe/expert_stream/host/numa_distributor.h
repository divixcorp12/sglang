// The NUMA groups of one RAM tier, the home rule, and the combiner that merges the groups' map deltas
// (spec 2026-10-03-numa-node-distributor-design, Part 3). At one node: one group, and every report completes.
#pragma once

#include "numa_group.h"

namespace sglang::expert_stream {

// One group's part of a record's map delta: its entries ({expert, slot}, slot -1 an eviction) and its staging list
// after its victims.
struct DeltaReport {
  int count = 0;
  int32_t entries[Wire::kDeltaMaxEntries][2];
  FixedVec<int32_t, Wire::kLanes> staging;
};

template <class Source>
class NumaNodeDistributor {
 public:
  using Group = NumaGroup<Source>;

  NumaNodeDistributor(std::vector<std::unique_ptr<Group>> groups, int64_t rows)
      : groups_(std::move(groups)), rows_(static_cast<size_t>(rows)) {}

  static int home(int64_t expert) {
    return Wire::home(expert);
  }
  int size() const {
    return static_cast<int>(groups_.size());
  }
  Group& group(int g) {
    return *groups_[g];
  }
  const Group& group(int g) const {
    return *groups_[g];
  }
  Group& home_group(int64_t expert) {
    return *groups_[home(expert)];
  }

  // The nodes with a miss lane in `request`: the groups that report its delta. Every group computes it from the
  // whole record it read.
  static uint32_t miss_nodes(const Request& request) {
    uint32_t nodes = 0;
    for (const Lane& lane : request.lanes)
      if (is_miss(lane.kind)) nodes |= 1u << home(lane.expert);
    return nodes;
  }

  // The nodes with a host lane (HIT_COPY, HIT_CPU, MISS_CPU): the groups that send the copy engine a part.
  static uint32_t host_nodes(const Request& request) {
    uint32_t nodes = 0;
    for (const Lane& lane : request.lanes)
      if (lane.kind == Wire::kKindHitCopy || lane.kind == Wire::kKindHitCpu || lane.kind == Wire::kKindMissCpu)
        nodes |= 1u << home(lane.expert);
    return nodes;
  }

  // The chain a miss record of `row` must carry: the one after the row's last published delta. Exact for a group with
  // a miss in the record being served, since that delta cannot publish without this group's report.
  uint64_t expected_chain(int64_t row) const {
    return rows_[row].written.load(std::memory_order_acquire) + 1;
  }

  // reserve_staging's lists: group g's slots before any record. The owner, before any thread runs.
  void seed(int64_t row, int g, const FixedVec<int32_t, Wire::kLanes>& staging) {
    rows_[row].staging[g] = staging;
  }

  // Group g reports its part of `request`'s delta; `expected` is miss_nodes(request). Returns true for exactly one
  // reporter, the last, which then publishes. The state word is (chain << 32 | groups reported), so a report of the
  // row's next chain never counts this one's bits. Every write of the word is a read-modify-write, so the publisher's
  // acquire sees every earlier report of this row, this record's and the reports that left the other nodes' lists.
  bool report(const Request& request, int g, const DeltaReport& part, uint32_t expected) {
    RowCombine& row = rows_[request.row];
    row.parts[g] = part;
    row.staging[g] = part.staging;
    const uint64_t chain = (request.chain & 0xFFFFFFFFull) << 32;
    uint64_t seen = row.reported.load(std::memory_order_relaxed);
    uint64_t next;
    do {
      const uint64_t mask = (seen & ~0xFFFFFFFFull) == chain ? seen & 0xFFFFFFFFull : 0;
      next = chain | mask | (1ull << g);
    } while (!row.reported.compare_exchange_weak(seen, next, std::memory_order_acq_rel, std::memory_order_relaxed));
    return (next & 0xFFFFFFFFull) == expected;
  }

  // The last reporter writes the row's one delta: every node's staging list, the reporters' entries in node order,
  // then the tag with a release (entries before the tag, as the device reads them). False when the row's previous
  // delta is not chain - 1, which the device's post order rules out.
  bool publish(uint8_t* lease, const Request& request, uint32_t reporters) {
    RowCombine& row = rows_[request.row];
    if (row.written.load(std::memory_order_acquire) + 1 != request.chain) return false;
    int32_t entries[Wire::kDeltaMaxEntries][2];
    int count = 0;
    for (int g = 0; g < size(); ++g) {
      if ((reporters >> g & 1u) == 0) continue;
      for (int i = 0; i < row.parts[g].count; ++i) {
        entries[count][0] = row.parts[g].entries[i][0];
        entries[count][1] = row.parts[g].entries[i][1];
        ++count;
      }
    }
    write(lease, request.row, request.chain, entries, count);
    return true;
  }

  // reserve_staging's tag-1 delta: every node's seeded list, no entries.
  void publish_seed(uint8_t* lease, int64_t row) {
    write(lease, row, 1, nullptr, 0);
  }

 private:
  struct RowCombine {
    std::atomic<uint64_t> reported{0};
    std::atomic<uint64_t> written{0};  // the chain of the row's last delta
    std::array<FixedVec<int32_t, Wire::kLanes>, Wire::kNodes> staging;
    std::array<DeltaReport, Wire::kNodes> parts;
  };

  // The payload with plain stores, an sfence, then the tag with a release (the device acquires the tag before it
  // reads the rest). The device read the row's previous delta in its last post, which came before this record, so
  // nothing reads the record while it is rewritten. `written` is stored first: once the tag is out the device may
  // post the row's next record, whose publisher reads it.
  void write(uint8_t* lease, int64_t row_index, uint64_t tag, const int32_t (*entries)[2], int count) {
    RowCombine& row = rows_[row_index];
    uint8_t* d = lease + Wire::kDeltaBase + row_index * Wire::kDeltaStride;
    const uint32_t n = static_cast<uint32_t>(count);
    std::memcpy(d + Wire::kDeltaCount, &n, 4);
    for (int node = 0; node < Wire::kNodes; ++node) {
      const FixedVec<int32_t, Wire::kLanes>& list = row.staging[node];
      for (int k = 0; k < Wire::kLanes; ++k) {
        const int16_t slot = static_cast<int16_t>(k < static_cast<int>(list.size()) ? list[k] : -1);
        std::memcpy(d + Wire::kDeltaStaging + 2 * (node * Wire::kLanes + k), &slot, 2);
      }
    }
    for (int i = 0; i < count; ++i) {
      const int16_t entry[2] = {static_cast<int16_t>(entries[i][0]), static_cast<int16_t>(entries[i][1])};
      std::memcpy(d + Wire::kDeltaEntries + 4 * i, entry, 4);
    }
    _mm_sfence();
    row.written.store(tag, std::memory_order_release);
    store_release64(d + Wire::kDeltaTag, tag);
  }

  std::vector<std::unique_ptr<Group>> groups_;
  std::vector<RowCombine> rows_;
};

}  // namespace sglang::expert_stream
