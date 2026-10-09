// Device-side constants and helpers shared by the slot-map protocol kernels and the row-copy kernels.
//
// The pieces, in file order:
//   state words        the device's own per-layer state (kPosted ... kDeadlineHi), never on the wire
//   memory accessors   ld/st wrappers that make each access's ordering explicit at its call site
//   record writer      write_record: the seqlock writer for a demand record (lease_layout.h)
//   map delta          RowMap, MapDelta and the await/load/apply steps that mirror the host's slot map on the device
//   lane typing        type_lanes: classifies each planned lane as a hit, a miss or a CPU lane
//
// See analysis/dsv41-drive/LEASE_PROTOCOL.md, "The chain".
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/distributed/ptx.cuh>

#include <cuda/atomic>
#include <cuda/ptx>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "lease_channel.cuh"
#include "lease_layout.h"
#include "lease_primitives.cuh"
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace sglang {
namespace device::expert_stream {

using namespace ::sglang::expert_stream::wire;

constexpr int kBlock = 32;
constexpr int kCopyWaitThreads = 256;  // the copy wait's block when it also reads the small tensors

// The device's own words (`state`, int32, device memory): never on the wire. The first four are the lease channel's
// (lease_channel.cuh); kPending is the seq of the request this layer's chain serves, 0 for an unarmed post.
using channel::kEpoch;
using channel::kPending;
using channel::kPendingEpoch;
using channel::kPosted;
// The stream kernel's absolute deadline, written by the post as two int32 halves. It bounds the one device spin that
// nothing else bounds.
constexpr int kDeadlineLo = 4;
constexpr int kDeadlineHi = 5;
constexpr int kStateWords = 6;

constexpr uint64_t kGenerationMask = (1ull << 56) - 1;

// The copy wait's gate word for request `seq`: `low` is Wire::kLeaseGateClosed or Wire::kLeaseGateOpen.
SGL_DEVICE uint32_t copy_gate_word(uint32_t seq, uint32_t low) {
  return channel::gate_word(seq, low);
}

// The device's global timer in nanoseconds, the clock every deadline is measured on.
SGL_DEVICE uint64_t global_ns() {
  return cuda::ptx::get_sreg_globaltimer();
}

SGL_DEVICE void store_deadline(int32_t* state, uint64_t deadline) {
  state[kDeadlineLo] = static_cast<int32_t>(static_cast<uint32_t>(deadline & 0xFFFFFFFFull));
  state[kDeadlineHi] = static_cast<int32_t>(static_cast<uint32_t>(deadline >> 32));
}

SGL_DEVICE uint64_t load_deadline(const int32_t* state) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineHi])) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(state[kDeadlineLo]));
}

// Whether `observed` has reached `seq`, comparing cyclically so a wrapped 32-bit seq still orders correctly.
SGL_DEVICE bool reached(uint32_t observed, uint32_t seq) {
  return static_cast<int32_t>(observed - seq) >= 0;
}

SGL_DEVICE bool listed(const int32_t* ids, int count, int32_t id) {
  for (int i = 0; i < count; ++i) {
    if (ids[i] == id) return true;
  }
  return false;
}

// The generation G of the request the chain serves. kPending and kPendingEpoch change only at the next post,
// stream-ordered after the chain's last kernel, so every kernel of the chain reads the same value.
SGL_DEVICE uint64_t pending_generation(const int32_t* state) {
  return channel::generation(static_cast<uint32_t>(state[kPending]), static_cast<uint32_t>(state[kPendingEpoch]));
}

SGL_DEVICE int64_t ring_index(uint32_t seq) {
  return channel::ring_index<TargetChannel>(seq);
}

// One request's typed lanes, as type_lanes produces them and the post writes them into its record (lease_layout.h
// Wire::kRecLaneSlot, Wire::kRecKinds).
struct TypedLanes {
  int32_t slot[Wire::kLanes];  // the RAM slot of a hit, the staging slot of a miss
  uint8_t kind[Wire::kLanes];  // kKind*
  int32_t node[Wire::kLanes];  // the lane's home node (Wire::home of its expert)
};

SGL_DEVICE bool is_cpu_kind(uint32_t kind) {
  return kind == Wire::kKindHitCpu || kind == Wire::kKindMissCpu;
}

// One request's record as the post knows it, in registers and device memory; write_record narrows and packs it into
// the wire layout. The pointers address `count` (or `protect_count`) entries.
struct RecordFields {
  int64_t row;
  uint32_t flags;          // kRecFlag*
  uint64_t chain;          // the row's map-chain number, 0 when no lane misses
  uint32_t epoch;          // so G = epoch << 32 | seq
  const int32_t* protect;  // protect_count routed experts
  int protect_count;
  int64_t count;            // lanes
  const int64_t* planned;   // lane experts
  const int32_t* dst;       // VRAM destination slots
  const float* weight;      // routing weights
  const TypedLanes* lanes;  // kinds and source slots
};

// Two record ids as one i16 pair, -1 for none; traps on an id the record cannot carry rather than wrap it.
SGL_DEVICE uint32_t pack_ids(int64_t lo, int64_t hi) {
  if (lo < -1 || lo > Wire::kRecIdMax || hi < -1 || hi > Wire::kRecIdMax) __trap();
  return (static_cast<uint32_t>(lo) & 0xFFFFu) | (static_cast<uint32_t>(hi) << 16);
}

// Writes one demand record as a seqlock: seq = 0, a release fence, the payload, then `seq` with a release store.
// A record of SM hits only may lap the ring before the service reads it, so a half-rewritten record must never pass
// the reader's seq re-check (read_record). Traps when the lane or protect count exceeds Wire::kLanes.
SGL_DEVICE void write_record(uint8_t* record, uint32_t seq, const RecordFields& f) {
  constexpr int L = Wire::kLanes;
  constexpr int kHeaderWords = static_cast<int>((Wire::kRecHeaderBytes - Wire::kRecEpoch) / 4);
  static_assert(kHeaderWords % 4 == 0 && L % 8 == 0, "the header and lane arrays are whole 16-byte stores");
  static_assert(
      Wire::kRecCounts == Wire::kRecRow + 2 && Wire::kRecFlags == Wire::kRecRow + 3,
      "row, counts and flags are one u32 store");
  static_assert(
      Wire::kRecChain % 8 == 0 && Wire::kRecKinds == Wire::kRecEpoch + 4,
      "chain is one v2 store; epoch and kinds one v4");
  static_assert(
      Wire::kDemandRing % 128 == 0 && Wire::kRecordBytes % 128 == 0,
      "a record's lines are whole 128-byte prefetch pairs");
  if (f.count > L || f.protect_count > L) __trap();
  uint32_t protect_w[L / 2], expert_w[L / 2], slot_w[L / 2], dst_w[L / 2], weight_w[L];
  uint32_t header_w[kHeaderWords] = {};
#pragma unroll
  for (int i = 0; i < L; i += 2) {
    const bool a = i < f.count, b = i + 1 < f.count;
    protect_w[i / 2] =
        pack_ids(i < f.protect_count ? f.protect[i] : -1, i + 1 < f.protect_count ? f.protect[i + 1] : -1);
    expert_w[i / 2] = pack_ids(a ? f.planned[i] : -1, b ? f.planned[i + 1] : -1);
    slot_w[i / 2] = pack_ids(a ? f.lanes->slot[i] : -1, b ? f.lanes->slot[i + 1] : -1);
    dst_w[i / 2] = pack_ids(a ? f.dst[i] : -1, b ? f.dst[i + 1] : -1);
  }
  header_w[0] = f.epoch;
#pragma unroll
  for (int i = 0; i < L; ++i) {
    const bool used = i < f.count;
    weight_w[i] = __float_as_uint(used ? f.weight[i] : 0.0f);
    header_w[1 + i / 8] |= (used ? static_cast<uint32_t>(f.lanes->kind[i]) : 0u) << (4 * (i % 8));
  }
  uint32_t counts = static_cast<uint32_t>(f.count);
  if constexpr (Wire::kPackedCounts) {
    counts |= static_cast<uint32_t>(f.protect_count) << 4;
  } else {
    constexpr int b = static_cast<int>(Wire::kRecProtectCount - Wire::kRecEpoch);
    header_w[b / 4] |= static_cast<uint32_t>(f.protect_count) << (8 * (b % 4));
  }
  const uint32_t head = (static_cast<uint32_t>(f.row) & 0xFFFFu) | counts << 16 | (f.flags & 0xFFu) << 24;
  channel::begin_record<TargetChannel>(record);
  st_relaxed_sys<uint32_t>(record + Wire::kRecRow, head);
  st_relaxed_sys_v2(
      record + Wire::kRecChain, static_cast<uint32_t>(f.chain & 0xFFFFFFFFull), static_cast<uint32_t>(f.chain >> 32));
#pragma unroll
  for (int w = 0; w < kHeaderWords; w += 4)
    st_relaxed_sys_v4(record + Wire::kRecEpoch + 4 * w, header_w[w], header_w[w + 1], header_w[w + 2], header_w[w + 3]);
#pragma unroll
  for (int w = 0; w < L / 2; w += 4) {
    st_relaxed_sys_v4(
        record + Wire::kRecProtect + 4 * w, protect_w[w], protect_w[w + 1], protect_w[w + 2], protect_w[w + 3]);
    st_relaxed_sys_v4(
        record + Wire::kRecLaneExpert + 4 * w, expert_w[w], expert_w[w + 1], expert_w[w + 2], expert_w[w + 3]);
    st_relaxed_sys_v4(record + Wire::kRecLaneSlot + 4 * w, slot_w[w], slot_w[w + 1], slot_w[w + 2], slot_w[w + 3]);
    st_relaxed_sys_v4(record + Wire::kRecLaneDst + 4 * w, dst_w[w], dst_w[w + 1], dst_w[w + 2], dst_w[w + 3]);
  }
#pragma unroll
  for (int w = 0; w < L; w += 4)
    st_relaxed_sys_v4(
        record + Wire::kRecLaneWeight + 4 * w, weight_w[w], weight_w[w + 1], weight_w[w + 2], weight_w[w + 3]);
  channel::end_record<TargetChannel>(record, seq);
}

// One row of the device's map bank (ExpertStreamDevice.map_bank), in device memory: the device's copy of the row's
// RAM-tier map. The post applies the host's deltas to it and types lanes from it.
struct RowMap {
  int32_t* ram_slot;     // [experts]: the expert's RAM slot, -1 when not resident
  int32_t* staging;      // [Wire::kNodes * Wire::kLanes], node-major: the row's staging slots per node
  int64_t* map_chain;    // the row's chain word
  int64_t* map_applied;  // the chain number of the row's last applied delta
  int64_t experts;
  uint32_t row_capacity;
};

// A row's map delta in registers. load_map_delta issues every load before it uses any, so their round trips overlap.
struct MapDelta {
  uint32_t count;
  int32_t staging[Wire::kNodes * Wire::kLanes];
  int32_t expert[Wire::kDeltaMaxEntries];
  int32_t slot[Wire::kDeltaMaxEntries];
};

// Spins until the host has published the delta that follows the row's last map chain, and returns true when that delta
// is not applied yet. Traps at `deadline` if the host never publishes it. The host publishes a chain's delta before it
// reads that chain's misses, so the spin runs only when the host fell a whole token behind.
SGL_DEVICE bool await_map_delta(const uint8_t* delta, const RowMap& map, uint64_t deadline) {
  const uint64_t want = static_cast<uint64_t>(*map.map_chain);
  while (ld_acquire_sys64(delta + Wire::kDeltaTag) != want) {
    if (static_cast<int64_t>(global_ns() - deadline) >= 0) __trap();  // the host never published it
    __nanosleep(256);
  }
  return static_cast<uint64_t>(*map.map_applied) != want;
}

// Loads the delta's payload. Call only after await_map_delta's acquire of the tag.
SGL_DEVICE MapDelta load_map_delta(const uint8_t* delta) {
  constexpr int kStagingLoads = Wire::kNodes * Wire::kLanes / 8;
  constexpr int kEntryLoads = static_cast<int>(Wire::kDeltaMaxEntries / 4);
  static_assert(Wire::kDeltaStaging % 16 == 0 && Wire::kDeltaEntries % 16 == 0, "16-byte delta loads");
  MapDelta d;
  d.count = ld_relaxed_sys<uint32_t>(delta + Wire::kDeltaCount);
  uint4 v[kStagingLoads + kEntryLoads];
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i)
    v[i] = ld_relaxed_sys_v4(delta + Wire::kDeltaStaging + 16 * i);
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i)
    v[kStagingLoads + i] = ld_relaxed_sys_v4(delta + Wire::kDeltaEntries + 16 * i);
  const auto lo = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)); };
  const auto hi = [](uint32_t w) { return static_cast<int32_t>(static_cast<int16_t>(w >> 16)); };
#pragma unroll
  for (int i = 0; i < kStagingLoads; ++i) {
    const uint32_t words[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      d.staging[8 * i + 2 * k] = lo(words[k]);
      d.staging[8 * i + 2 * k + 1] = hi(words[k]);
    }
  }
#pragma unroll
  for (int i = 0; i < kEntryLoads; ++i) {
    const uint4 e = v[kStagingLoads + i];
    const uint32_t words[4] = {e.x, e.y, e.z, e.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      d.expert[4 * i + w] = lo(words[w]);
      d.slot[4 * i + w] = hi(words[w]);
    }
  }
  return d;
}

// Validates a loaded delta and applies it to the row, then sets map_applied to the row's chain number so the delta is
// applied exactly once. Traps on an entry the row cannot hold.
SGL_DEVICE void apply_map_delta(const MapDelta& d, const RowMap& map) {
  if (d.count > static_cast<uint32_t>(Wire::kDeltaMaxEntries)) __trap();
#pragma unroll
  for (int i = 0; i < Wire::kDeltaMaxEntries; ++i) {
    if (static_cast<uint32_t>(i) < d.count) {
      const int32_t expert = d.expert[i];
      const int32_t slot = d.slot[i];
      if (expert < 0 || expert >= map.experts || slot < -1 || slot >= static_cast<int32_t>(map.row_capacity)) __trap();
      map.ram_slot[expert] = slot;
    }
  }
#pragma unroll
  for (int k = 0; k < Wire::kNodes * Wire::kLanes; ++k)
    map.staging[k] = d.staging[k];
  *map.map_applied = *map.map_chain;
}

// The plan as the post reads it, in device memory: `count` planned experts and their VRAM destination slots. Lanes
// from forced_from on found no VRAM victim (spill: DIRECT's narrowed verify with CPU experts) and must be CPU lanes;
// forced_from == count forces none.
struct LanePlan {
  const int64_t* planned;
  const int32_t* dst;
  int64_t count;
  int64_t forced_from;
};

// Everything besides the map that decides a lane's kind, loaded before type_lanes runs.
struct LanePolicy {
  bool host_lanes;   // a captured post while the copy engine is armed (Wire::kCopyArmed)
  bool hit_copy_ce;  // SGLANG_DSV41_RAM_HIT_COPY=ce
  bool cpu_on;       // CPU experts are enabled
  bool cpu_misses;   // a miss may also be a CPU lane (SGLANG_DSV41_CPU_EXPERTS_MISSES)
  bool ce_ok;        // the row's copy table is set
  bool cpu_ok;       // the row's CPU layer is registered
  int32_t dst_rows;
  // Experiment (SGLANG_DSV41_CPU_SPLIT_MISS_CUT / _MAX): a node with 1..miss_cut_max forced CPU misses takes miss_cut
  // fewer of its split's lanes. 0: no cut.
  int32_t miss_cut;
  int32_t miss_cut_max;
  int32_t split[Wire::kNodes][Wire::kLanes + 1];  // Wire::kSplit: per node, CPU lanes per n eligible lanes
};

// Loads every node's Wire::kSplit table in whole 16-byte loads; words past a table are dropped.
SGL_DEVICE void load_split(const uint8_t* split, int32_t (&out)[Wire::kNodes][Wire::kLanes + 1]) {
  constexpr int kLoads = static_cast<int>(Wire::kSplitStride / 16);
  static_assert(
      Wire::kSplit % 16 == 0 && Wire::kSplit + Wire::kNodes * Wire::kSplitStride <= Wire::kLeaseBlockBytes,
      "split loads");
#pragma unroll
  for (int node = 0; node < Wire::kNodes; ++node) {
    uint4 v[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i)
      v[i] = ld_relaxed_sys_v4(split + node * Wire::kSplitStride + 16 * i);
#pragma unroll
    for (int n = 0; n <= Wire::kLanes; ++n) {
      const uint4 q = v[n / 4];
      const uint32_t words[4] = {q.x, q.y, q.z, q.w};
      out[node][n] = static_cast<int32_t>(words[n % 4]);
    }
  }
}

// Types each lane of the plan: its kind and source slot. Transcribes ram_slot_map.type_lanes, the host reference.
//
// A hit takes its RAM slot, and an unforced miss the next slot of its home node's staging list. Node n's CPU takes the
// last split[n][k] of its k eligible unforced lanes in plan order. A forced lane (j >= plan.forced_from) is a CPU lane
// whatever the split: a hit at its RAM slot, a miss with slot -1 and no staging slot, which the host reads into a RAM
// victim (RamTier::reserve_victims_locked). Traps where the reference raises ValueError: a plan wider than
// Wire::kLanes, an expert out of range or repeated, a hit slot past the row's capacity, an unforced miss with no
// staging slot on its node (live misses <= victim lanes = staging per node: an assert), a split entry above its n.
// With policy.miss_cut > 0, a node with 1..policy.miss_cut_max forced misses (forced lanes that are not RAM hits)
// takes policy.miss_cut fewer of its split's lanes, at least 0; forced lanes stay CPU lanes.
// Returns false where the reference raises LaneOverflow: forced lanes with no host lanes or no CPU layer; `out` is then
// partial. Reads no host memory; the caller loads the split table into the policy.
SGL_DEVICE bool type_lanes(const LanePlan& plan, const RowMap& map, const LanePolicy& policy, TypedLanes& out) {
  if (plan.count > Wire::kLanes) __trap();
  int64_t expert[Wire::kLanes];
  int32_t dst[Wire::kLanes];
  int32_t ram[Wire::kLanes];
  int32_t staging[Wire::kNodes * Wire::kLanes];
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    expert[j] = j < plan.count ? plan.planned[j] : 0;
    dst[j] = j < plan.count ? plan.dst[j] : -1;
  }
#pragma unroll
  for (int k = 0; k < Wire::kNodes * Wire::kLanes; ++k)
    staging[k] = map.staging[k];
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j)
    if (j < plan.count && (expert[j] < 0 || expert[j] >= map.experts)) __trap();
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j)
    ram[j] = j < plan.count ? map.ram_slot[expert[j]] : -1;
  bool hit[Wire::kLanes];
  bool eligible[Wire::kLanes];
  int m[Wire::kNodes] = {};
  int n[Wire::kNodes] = {};
  int forced_misses[Wire::kNodes] = {};
  const bool can_cpu = policy.host_lanes && policy.cpu_on && policy.cpu_ok;
#pragma unroll
  for (int j = 0; j < Wire::kLanes; ++j) {
    if (j >= plan.count) break;
    for (int i = 0; i < j; ++i)
      if (expert[i] == expert[j]) __trap();
    const int node = Wire::home(expert[j]);
    const bool forced = j >= plan.forced_from;
    out.node[j] = node;
    hit[j] = ram[j] >= 0;
    if (hit[j]) {
      if (static_cast<uint32_t>(ram[j]) >= map.row_capacity) __trap();
      out.slot[j] = ram[j];
    } else if (forced) {
      out.slot[j] = -1;  // host-placed: the host reads it into a RAM victim of its node
    } else {
      if (m[node] >= Wire::kLanes || staging[node * Wire::kLanes + m[node]] < 0) __trap();
      out.slot[j] = staging[node * Wire::kLanes + m[node]++];
    }
    if (forced && !can_cpu) return false;
    eligible[j] = !forced && can_cpu && (hit[j] || policy.cpu_misses);
    n[node] += eligible[j] ? 1 : 0;
    forced_misses[node] += forced && !hit[j] ? 1 : 0;
  }
  int take[Wire::kNodes];
#pragma unroll
  for (int node = 0; node < Wire::kNodes; ++node) {
    take[node] = n[node] > 0 ? policy.split[node][n[node]] : 0;
    if (take[node] < 0 || take[node] > n[node]) __trap();
    if (policy.miss_cut > 0 && forced_misses[node] >= 1 && forced_misses[node] <= policy.miss_cut_max)
      take[node] = take[node] > policy.miss_cut ? take[node] - policy.miss_cut : 0;
  }
  const bool copy_ok = policy.host_lanes && policy.hit_copy_ce && policy.ce_ok;
  for (int64_t j = plan.count - 1; j >= 0; --j) {
    const int node = out.node[j];
    const bool forced = j >= plan.forced_from;
    const bool cpu = forced || (take[node] > 0 && eligible[j]);
    if (cpu && !forced) --take[node];
    uint8_t kind;
    if (cpu) {
      kind = hit[j] ? Wire::kKindHitCpu : Wire::kKindMissCpu;
    } else if (hit[j]) {
      kind = copy_ok && dst[j] >= 0 && dst[j] < policy.dst_rows ? Wire::kKindHitCopy : Wire::kKindHitSm;
    } else {
      kind = Wire::kKindMissGpu;
    }
    out.kind[j] = kind;
  }
  return true;
}

}  // namespace device::expert_stream
}  // namespace sglang
