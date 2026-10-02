// The device's side of the expert-stream lease protocol, for the bench: a C++ port of python/sglang/test/
// dsv41_chain_sim.py's ChainSim (analysis/dsv41-drive/LEASE_PROTOCOL.md). It stands in for the post kernel, S and CW;
// it is not evidence about them. Lanes are typed as ram_slot_map.type_lanes types them, with no copy table
// (hit_copy="sm": a hit the CPU does not take is kKindHitSm) and CPU hits only (cpu_misses=false).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace fullstack {

constexpr int kLanes = 8;  // wire::kLeaseLanes

int64_t monotonic_ns();  // CLOCK_MONOTONIC: the host's now_ns() and its stage trace read the same clock

struct SimRequest {
  uint32_t seq = 0;
  uint64_t gen = 0;  // epoch << 32 | seq
  int64_t idx = 0;   // (seq - 1) % 16: the ring slot, the CopyDone and PieceMask index
  int64_t row = 0;
  int count = 0;
  std::array<int32_t, kLanes> experts{};
  std::array<int32_t, kLanes> kinds{};  // wire::kKind*
  std::array<int32_t, kLanes> slots{};  // a hit's RAM slot, a miss's staging slot
  uint64_t chain = 0;                   // the row's map-chain number when a lane misses, else 0
};

// Called with the record after its payload is written and before its seq is published (the self-test's seqlock check).
using PostHook = std::function<void(const uint8_t* record)>;

class DeviceSim {
 public:
  DeviceSim(uint8_t* page, uint8_t* lease, int64_t rows, int64_t experts, uint32_t epoch = 0);

  // The row's CPU layer is registered: its hits become CPU-eligible (the device side's set_row_cpu).
  void set_row_cpu(int64_t row);

  // The post kernel: apply the row's pending delta (waiting for it until deadline_ns, then throwing), type the lanes,
  // write the record (seq = 0, payload, seq with a release) and demand_head (release). Protect ids are the experts;
  // destinations are 0..count-1.
  SimRequest post(int64_t row, std::span<const int32_t> experts, std::span<const float> weights, bool captured,
                  int64_t deadline_ns, const PostHook& before_publish = {});

  // Apply the row's delta now, waiting for the host to publish it; throws at deadline_ns.
  void sync_row(int64_t row, int64_t deadline_ns);

  // CW: close the gate for G (seq_cst), spin until CopyDone == G, then open the gate if it is still closed for G
  // (CW's own open, for a CopyDone that landed before the close). False at deadline_ns, the gate left closed.
  // True at once for a request with no host lane.
  bool copy_wait(const SimRequest& request, int64_t deadline_ns);

  // S: lane `lane`'s PieceMask word reads piece_word(G) with all 8 piece bits. False at deadline_ns.
  bool wait_pieces(const SimRequest& request, int lane, int64_t deadline_ns) const;

  static bool needs_copy_wait(const SimRequest& request);  // a kKindHitCopy, kKindHitCpu or kKindMissCpu lane

  int32_t ram_slot(int64_t row, int32_t expert) const;
  std::array<int32_t, kLanes> staging(int64_t row) const;
  uint64_t map_chain(int64_t row) const;
  uint64_t copy_done(const SimRequest& request) const;
  uint32_t copy_gate() const;
  uint32_t epoch() const;

 private:
  bool apply_pending(int64_t row);

  uint8_t* page_;
  uint8_t* lease_;
  int64_t rows_;
  int64_t experts_;
  uint32_t epoch_;
  std::vector<int32_t> ram_slot_;                    // [rows][experts]
  std::vector<std::array<int32_t, kLanes>> staging_;  // [rows]
  std::vector<uint64_t> map_chain_;                  // starts at 1: the attach delta's tag
  std::vector<uint64_t> map_applied_;
  std::vector<uint8_t> row_cpu_;
};

// Make `experts` resident in `row` through uncaptured posts, as a plain request does. A post misses at most `staging`
// experts (the row's staging slots), so they go in groups; every lane's pieces are awaited, then the row's delta is
// applied. Throws on a lane that is not a miss, a late piece, or an expert left unmapped. Returns the last post's seq.
uint32_t load_experts(DeviceSim& sim, int64_t row, std::span<const int32_t> experts, int staging, int64_t timeout_ns);

}  // namespace fullstack
