// The bench's --self-test: checks the harness itself, so a bench result is never the harness's fault.
//
// Needs no fixture file and no GPU. In order:
//   test_placement          the placement rules, against a fake topology of the reference machine
//   test_record_bytes ...   DeviceSim's records, seqlock order, delta handling, lane typing, epoch wrap and copy-wait
//                           gate, on a standalone page and lease block with the host's words written by hand
//   test_image_stamp        row-image layout and stamp reuse
//   test_stack              the real stack (this binary's build) on synthetic rows with a fake kernel
//   test_two_groups         two NUMA groups' stacks, each with its own cores and slots (a Wire::kNodes == 2 build)
//   test_groups_must_share_one_kernel  a group naming a second kernel is refused (a Wire::kNodes == 2 build)
// Each failed check prints "FAIL file:line" and counts toward run_self_test's return value.
#include "self_test.h"

#include "aligned.h"
#include "device_sim.h"
#include "expert_stream/host/tier_protocol.h"
#include "row_images.h"
#include "stack.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fullstack {
namespace {

int checks = 0;
int failures = 0;
namespace w = ::sglang::expert_stream::wire;
namespace es = ::sglang::expert_stream;
namespace ce = ::sglang::cpu_experts;

// Records one check and reports it on stderr when it fails.
void check(bool ok, const char* what, const char* file, int line) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  }
}

// Passes when `f` throws an exception whose message contains `needle`.
template <class F>
void check_throws(F&& f, const std::string& needle, const char* what, const char* file, int line) {
  try {
    f();
  } catch (const std::exception& error) {
    const bool found = std::string(error.what()).find(needle) != std::string::npos;
    check(found, what, file, line);
    if (!found) std::fprintf(stderr, "  message: %s\n", error.what());
    return;
  }
  check(false, what, file, line);
}

#define CHECK(cond) check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_THROWS(expr, needle) check_throws([&] { expr; }, needle, #expr " throws " needle, __FILE__, __LINE__)

// ---- placement ----

// The reference machine (72 CPUs, two NUMA nodes): node 0 = 0-17,36-53; node 1 = 18-35,54-71; c and c + 36 are SMT
// siblings. Allowed: the bench's partition, 16-33 and 52-69.
Topology fake_topology() {
  Topology t;
  t.node_of = [](int cpu) { return cpu % 36 < 18 ? 0 : 1; };
  t.siblings_of = [](int cpu) { return std::vector<int>{cpu % 36, cpu % 36 + 36}; };
  CPU_ZERO(&t.allowed);
  for (int cpu = 16; cpu <= 33; ++cpu) {
    CPU_SET(cpu, &t.allowed);
    CPU_SET(cpu + 36, &t.allowed);
  }
  return t;
}

// The bench's default placement, which the fake topology accepts.
Placement production_placement() {
  Placement p;
  p.writer = 16;
  p.copy = 52;
  p.groups.push_back({17, parse_cpus("18-33"), 1});
  return p;
}

// True when validate_placement accepts the placement; prints the refusal otherwise.
bool passes(const Placement& p, const Topology& t, bool check_nodes) {
  try {
    validate_placement(p, t, check_nodes);
    return true;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "  refused: %s\n", error.what());
    return false;
  }
}

void test_placement() {
  CHECK(parse_cpus("16-17,52") == std::vector<int32_t>({16, 17, 52}));
  CHECK_THROWS(parse_cpus("3,3"), "Duplicate CPU");
  CHECK_THROWS(parse_cpus("5-3"), "Invalid CPU range");
  CHECK_THROWS(parse_cpus("x"), "Invalid integer");
  const Topology t = fake_topology();
  CHECK(passes(production_placement(), t, true));

  Placement twice = production_placement();
  twice.groups[0].service = 16;
  CHECK_THROWS(validate_placement(twice, t, true), "two roles");

  Placement sibling = production_placement();
  sibling.copy = 53;  // 17's SMT sibling
  CHECK_THROWS(validate_placement(sibling, t, true), "physical core of the service CPU 17");

  Placement outside = production_placement();
  outside.copy = 34;
  CHECK_THROWS(validate_placement(outside, t, true), "outside the process's allowed CPUs");

  Placement writer_node = production_placement();
  writer_node.writer = 19;
  writer_node.groups[0].workers = parse_cpus("18,20-33");
  CHECK_THROWS(validate_placement(writer_node, t, true), "writer CPU 19 is on NUMA node 1, not node 0");
  CHECK(passes(writer_node, t, false));  // the self-test's mode: no node rules

  Topology wider = fake_topology();
  CPU_SET(10, &wider.allowed);
  Placement worker_node = production_placement();
  worker_node.groups[0].workers.back() = 10;
  CHECK_THROWS(validate_placement(worker_node, wider, true), "worker CPU 10 is on NUMA node 0, not node 1");

  std::vector<int> expected = expected_threads(production_placement());
  std::vector<int> want = {17};  // sorted: service, workers 18-33, then the copy CPU twice (copy thread, watchdog)
  for (int cpu = 18; cpu <= 33; ++cpu)
    want.push_back(cpu);
  want.push_back(52);
  want.push_back(52);
  CHECK(expected == want);
  CHECK(cpu_list({52, 16, 17}) == "16,17,52");
}

// The reference machine with every CPU allowed: room for two groups' services and workers on their own nodes.
Topology wide_topology() {
  Topology t = fake_topology();
  CPU_ZERO(&t.allowed);
  for (int cpu = 0; cpu < 72; ++cpu)
    CPU_SET(cpu, &t.allowed);
  return t;
}

// Above one group a group's service and workers sit on its own node; the writer and the copy thread stay on host_node.
void test_two_group_placement() {
  const Topology t = wide_topology();
  Placement p;
  p.writer = 16;
  p.copy = 52;
  p.groups.push_back({17, parse_cpus("8-15"), 0});
  p.groups.push_back({35, parse_cpus("18-33"), 1});
  CHECK(passes(p, t, true));
  CHECK(expected_threads(p) == std::vector<int>({8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 52, 52}));

  Placement shared = p;
  shared.groups[1].workers = parse_cpus("17-32");  // 17 is group 0's service
  CHECK_THROWS(validate_placement(shared, t, true), "two roles");

  Placement sibling = p;
  sibling.copy = 53;  // 17's SMT sibling
  CHECK_THROWS(validate_placement(sibling, t, true), "physical core of the service CPU 17");

  Placement own_node = p;
  own_node.groups[1].service = 19;
  own_node.groups[1].workers = parse_cpus("18,20-33");
  CHECK(passes(own_node, t, true));  // 19 is on node 1, group 1's node
  own_node.groups[1].service = 0;  // node 0: not group 1's node
  CHECK_THROWS(validate_placement(own_node, t, true), "service CPU 0 is on NUMA node 0, not node 1");

  Placement wrong_workers = p;
  wrong_workers.groups[0].workers = parse_cpus("34");  // a node-1 CPU for a node-0 group
  CHECK_THROWS(validate_placement(wrong_workers, t, true), "worker CPU 34 is on NUMA node 1, not node 0");
}

// ---- DeviceSim, on a standalone page and lease block (no service: the host's words are written by hand) ----

uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & w::Wire::kLeaseGateSeqMask) << w::Wire::kLeaseGateSeqShift) | low;
}

int64_t soon() {
  return monotonic_ns() + 100'000'000;
}

constexpr int kWireLanes = w::Wire::kLanes;
using Staging = std::array<int16_t, kWireLanes>;
using SplitTable = std::array<int32_t, kWireLanes + 1>;

constexpr Staging kStaging012 = [] {
  Staging s{};
  s.fill(-1);
  for (int i = 0; i < 3; ++i)
    s[i] = static_cast<int16_t>(i);
  return s;
}();
constexpr SplitTable kAllToCpu = [] {
  SplitTable t{};
  for (int i = 0; i <= kWireLanes; ++i)
    t[i] = i;
  return t;
}();

// A standalone request page and lease block, with helpers that write the host's words (deltas, the split table, the
// armed flag) the way the host does, and read the device's words back.
struct Blocks {
  explicit Blocks(int64_t rows)
      : page(aligned_zeroed(w::Wire::kPageBytes)),
        lease_bytes(w::Wire::kLeaseBlockBytes + round_up(rows * w::Wire::kDeltaStride, 4096)),
        lease(aligned_zeroed(lease_bytes)) {}

  // The host's delta record for `row`: payload, then the tag with a release (RamTier::publish_delta_locked).
  void
  delta(int64_t row, uint64_t tag, Staging staging, std::vector<std::pair<int16_t, int16_t>> entries) {
    uint8_t* d = lease.get() + w::Wire::kDeltaBase + row * w::Wire::kDeltaStride;
    const auto count = static_cast<uint32_t>(entries.size());
    std::memcpy(d + w::Wire::kDeltaCount, &count, 4);
    for (int node = 0; node < w::Wire::kNodes; ++node)  // every node's list is the same: slots are not under test here
      std::memcpy(d + w::Wire::kDeltaStaging + node * sizeof(staging), staging.data(), sizeof(staging));
    for (size_t i = 0; i < entries.size(); ++i) {
      const int16_t entry[2] = {entries[i].first, entries[i].second};
      std::memcpy(d + w::Wire::kDeltaEntries + 4 * i, entry, 4);
    }
    __atomic_store_n(reinterpret_cast<uint64_t*>(d + w::Wire::kDeltaTag), tag, __ATOMIC_RELEASE);
  }
  void split(SplitTable table) {
    for (int node = 0; node < w::Wire::kNodes; ++node)
      std::memcpy(lease.get() + w::Wire::kSplit + node * w::Wire::kSplitStride, table.data(), sizeof(table));
  }
  void armed(bool on) {
    const uint32_t value = on ? 1 : 0;
    std::memcpy(lease.get() + w::Wire::kCopyArmed, &value, 4);
  }
  template <class T>
  T at(int64_t offset) const {
    T value;
    std::memcpy(&value, page.get() + offset, sizeof(value));
    return value;
  }

  AlignedBuffer page;
  int64_t lease_bytes;
  AlignedBuffer lease;
};

void test_record_bytes() {
  Blocks b(2);
  b.delta(0, 1, kStaging012, {{3, 4}});  // expert 3 resident in slot 4
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 2, 8);
  sim.set_row_cpu(0);
  const int32_t experts[] = {3, 5};
  const float weights[] = {0.5f, 0.25f};
  const SimRequest r = sim.post(0, experts, weights, true, soon());
  // type_lanes: expert 3 hits slot 4 and is the CPU's (split[1] = 1); expert 5 misses into staging[0] = 0.
  CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitCpu) && r.kinds[1] == int32_t(w::Wire::kKindMissGpu));
  CHECK(r.slots[0] == 4 && r.slots[1] == 0);
  CHECK(r.seq == 1 && r.gen == 1 && r.idx == 0 && r.chain == 2 && sim.map_chain(0) == 2);
  const int64_t rec = w::Wire::kDemandRing;
  CHECK(b.at<uint32_t>(w::Wire::kDemandHead) == 1);
  CHECK(b.at<uint32_t>(rec + w::Wire::kRecSeq) == 1);
  CHECK(b.at<uint16_t>(rec + w::Wire::kRecRow) == 0);
  if constexpr (w::Wire::kPackedCounts) {
    CHECK(b.at<uint8_t>(rec + w::Wire::kRecCounts) == (2 | 2 << 4));
  } else {
    CHECK(b.at<uint8_t>(rec + w::Wire::kRecCounts) == 2);
    CHECK(b.at<uint8_t>(rec + w::Wire::kRecProtectCount) == 2);
  }
  CHECK(b.at<uint8_t>(rec + w::Wire::kRecFlags) == w::Wire::kRecFlagCaptured);
  CHECK(b.at<uint64_t>(rec + w::Wire::kRecChain) == 2);
  CHECK(b.at<uint32_t>(rec + w::Wire::kRecEpoch) == 0);
  CHECK(b.at<uint32_t>(rec + w::Wire::kRecKinds) == (3u | 4u << 4));
  for (int i = 1; i < w::Wire::kKindWords; ++i)
    CHECK(b.at<uint32_t>(rec + w::Wire::kRecKinds + 4 * i) == 0);
  std::array<int16_t, kWireLanes> ids, slots, dst;
  std::array<float, kWireLanes> lane_weights{};
  ids.fill(-1);
  slots.fill(-1);
  dst.fill(-1);
  ids[0] = 3, ids[1] = 5;
  slots[0] = 4, slots[1] = 0;
  dst[0] = 0, dst[1] = 1;
  lane_weights[0] = 0.5f, lane_weights[1] = 0.25f;
  CHECK(std::memcmp(b.page.get() + rec + w::Wire::kRecProtect, ids.data(), 2 * kWireLanes) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::Wire::kRecLaneExpert, ids.data(), 2 * kWireLanes) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::Wire::kRecLaneSlot, slots.data(), 2 * kWireLanes) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::Wire::kRecLaneDst, dst.data(), 2 * kWireLanes) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::Wire::kRecLaneWeight, lane_weights.data(), 4 * kWireLanes) == 0);
  // The production parser reads it back as the device meant it.
  es::Request req;
  CHECK(es::read_record(b.page.get() + rec, 1, &req) == es::RecordRead::kOk);
  CHECK(req.gen == 1 && req.row == 0 && req.captured && req.chain == 2);
  CHECK(req.lanes.size() == 2 && req.protect.size() == 2);
  CHECK(
      req.lanes[0].expert == 3 && req.lanes[0].slot == 4 && req.lanes[0].dst == 0 && req.lanes[0].weight == 0.5f &&
      req.lanes[0].kind == w::Wire::kKindHitCpu);
  CHECK(req.lanes[1].expert == 5 && req.lanes[1].slot == 0 && req.lanes[1].kind == w::Wire::kKindMissGpu);
}

// A request of every lane a hit: the record's counts, kinds and per-lane arrays decode through read_record at any width.
void test_full_width_record() {
  constexpr int n = kWireLanes;
  Blocks b(1);
  std::vector<std::pair<int16_t, int16_t>> entries;
  for (int j = 0; j < n; ++j)
    entries.emplace_back(static_cast<int16_t>(j), static_cast<int16_t>(100 + j));
  b.delta(0, 1, kStaging012, entries);
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, n);
  sim.set_row_cpu(0);
  std::vector<int32_t> experts(n);
  std::vector<float> weights(n);
  for (int j = 0; j < n; ++j) {
    experts[j] = n - 1 - j;
    weights[j] = 0.5f + static_cast<float>(j);
  }
  const SimRequest r = sim.post(0, experts, weights, true, soon());
  es::Request req;
  CHECK(es::read_record(b.page.get() + w::Wire::kDemandRing + r.idx * w::Wire::kRecordBytes, r.seq, &req) ==
        es::RecordRead::kOk);
  CHECK(req.gen == r.gen && req.row == 0 && req.captured);
  CHECK(static_cast<int>(req.lanes.size()) == n && static_cast<int>(req.protect.size()) == n);
  for (int j = 0; j < n; ++j) {
    CHECK(req.protect[j] == experts[j]);
    CHECK(req.lanes[j].expert == experts[j]);
    CHECK(req.lanes[j].slot == 100 + experts[j]);
    CHECK(req.lanes[j].dst == j);
    CHECK(req.lanes[j].weight == weights[j]);
    CHECK(req.lanes[j].kind == static_cast<uint32_t>(r.kinds[j]));
    CHECK(req.lanes[j].kind == w::Wire::kKindHitCpu);  // every lane eligible, split sends all to the CPU
  }
}

void test_seqlock_order() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 5}});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {0.75f};
  bool hooked = false;
  sim.post(0, e, wt, true, soon(), [&](const uint8_t* record) {
    hooked = true;
    uint32_t seq, head, kinds;
    float weight;
    std::memcpy(&seq, record + w::Wire::kRecSeq, 4);
    std::memcpy(&head, b.page.get() + w::Wire::kDemandHead, 4);
    std::memcpy(&kinds, record + w::Wire::kRecKinds, 4);
    std::memcpy(&weight, record + w::Wire::kRecLaneWeight, 4);
    CHECK(seq == 0);   // the seqlock word is 0 while the payload is in place
    CHECK(head == 0);  // demand_head moves only after the seq
    CHECK(kinds == w::Wire::kKindHitCpu && weight == 0.75f);
  });
  CHECK(hooked);
  CHECK(b.at<uint32_t>(w::Wire::kDemandRing + w::Wire::kRecSeq) == 1 && b.at<uint32_t>(w::Wire::kDemandHead) == 1);
  // A rewrite of a ring slot that held an earlier record also zeroes its seq first: seq 17 reuses slot 0.
  for (int i = 0; i < 15; ++i)
    sim.post(0, e, wt, true, soon());
  bool rehooked = false;
  sim.post(0, e, wt, true, soon(), [&](const uint8_t* record) {
    rehooked = true;
    uint32_t seq;
    std::memcpy(&seq, record + w::Wire::kRecSeq, 4);
    CHECK(record == b.page.get() + w::Wire::kDemandRing);
    CHECK(seq == 0);
  });
  CHECK(rehooked && b.at<uint32_t>(w::Wire::kDemandRing + w::Wire::kRecSeq) == 17);
}

void test_owed_delta() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t miss[] = {5};
  const float one[] = {1.0f};
  const SimRequest first = sim.post(0, miss, one, false, soon());
  CHECK(first.kinds[0] == int32_t(w::Wire::kKindMissGpu) && first.slots[0] == 0 && first.chain == 2);
  // The host has not published delta 2: the row's next post waits for it, then refuses at its deadline.
  CHECK_THROWS(sim.post(0, miss, one, true, monotonic_ns() + 1'000'000), "delta 2");
  CHECK(b.at<uint32_t>(w::Wire::kDemandHead) == 1);  // the refused post wrote nothing
  // Delta 2, as the host publishes it: expert 5 in slot 0, the victim slot 3 now staging[0].
  b.delta(0, 2, {3, 1, 2, -1, -1, -1, -1, -1}, {{5, 0}});
  const SimRequest hit = sim.post(0, miss, one, true, soon());
  CHECK(hit.kinds[0] == int32_t(w::Wire::kKindHitCpu) && hit.slots[0] == 0 && hit.chain == 0 && sim.map_chain(0) == 2);
  CHECK(sim.staging(0)[0] == 3);
  // The next miss takes the new staging[0] and the next chain number.
  const int32_t other[] = {6};
  const SimRequest second = sim.post(0, other, one, true, soon());
  CHECK(second.kinds[0] == int32_t(w::Wire::kKindMissGpu) && second.slots[0] == 3 && second.chain == 3);
}

void test_typing() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}, {2, 5}});  // experts 0 and 2: one home node, so one split table decides
  b.armed(true);
  b.split({0, 0, 0, 0, 0, 0, 0, 0, 0});
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t two[] = {0, 2};
  const float halves[] = {0.5f, 0.5f};
  SimRequest r = sim.post(0, two, halves, true, soon());  // split[2] = 0: no CPU lane
  CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitSm) && !DeviceSim::needs_copy_wait(r));
  b.split({0, 1, 1, 1, 1, 1, 1, 1, 1});  // split[2] = 1: the last eligible lane only
  r = sim.post(0, two, halves, true, soon());
  CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitCpu) && DeviceSim::needs_copy_wait(r));
  b.split(kAllToCpu);
  r = sim.post(0, two, halves, false, soon());  // uncaptured: no host lanes
  CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitSm));
  b.armed(false);
  r = sim.post(0, two, halves, true, soon());  // the copy engine is not armed: no host lanes
  CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitSm));
  Blocks c(1);
  c.delta(0, 1, kStaging012, {{0, 4}});
  c.split(kAllToCpu);
  c.armed(true);
  DeviceSim cold(c.page.get(), c.lease.get(), 1, 8);  // no set_row_cpu: the row's layer is not registered
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  CHECK(cold.post(0, e, wt, true, soon()).kinds[0] == int32_t(w::Wire::kKindHitSm));
  CHECK_THROWS(cold.post(0, two, wt, true, soon()), "weights");
  const int32_t dup[] = {0, 0};
  CHECK_THROWS(cold.post(0, dup, halves, true, soon()), "twice");
}

void test_epoch_wrap() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}});
  b.split(kAllToCpu);
  b.armed(true);
  const uint32_t last = 0xFFFFFFFFu;
  std::memcpy(b.page.get() + w::Wire::kDemandHead, &last, 4);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8, /*epoch=*/7);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  const SimRequest r = sim.post(0, e, wt, true, soon());
  CHECK(r.seq == 1 && r.idx == 0 && sim.epoch() == 8 && r.gen == (uint64_t{8} << 32 | 1));
  CHECK(b.at<uint32_t>(w::Wire::kDemandRing + w::Wire::kRecEpoch) == 8 && b.at<uint32_t>(w::Wire::kDemandHead) == 1);
}

void test_copy_wait_gate() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}});
  b.split(kAllToCpu);
  b.armed(true);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  const SimRequest r = sim.post(0, e, wt, true, soon());
  // No CopyDone: the wait ends at its deadline with the gate closed for G (the watchdog's evidence).
  CHECK(!sim.copy_wait(r, monotonic_ns() + 2'000'000));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::Wire::kLeaseGateClosed));
  // CopyDone stored before the close (the host's open found nothing closed): CW opens the gate itself.
  __atomic_store_n(
      reinterpret_cast<uint64_t*>(b.lease.get() + w::Wire::kLeaseCopyDone + r.idx * w::Wire::kLeaseCopyDoneBytes),
      r.gen,
      __ATOMIC_RELEASE);
  CHECK(sim.copy_wait(r, soon()));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::Wire::kLeaseGateOpen));
  CHECK(sim.copy_done(r) == r.gen);
}

// ---- row images ----

void test_image_stamp(const std::filesystem::path& dir) {
  const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
  CHECK(layout.image_bytes == 3072 && layout.row_stride == 4096 && layout.name_offsets[5] == 2560);
  CHECK_THROWS(image_layout({512, 512, 512, 512, 512, 100}), "multiple of 512");
  const std::filesystem::path path = dir / "selftest-stamp.rows";
  int fills = 0;
  auto fill = [&](int64_t, uint8_t* image) {
    ++fills;
    image[0] = 1;
  };
  CHECK(write_row_image(path, layout, 2, fill, "fixture A"));
  CHECK(!write_row_image(path, layout, 2, fill, "fixture A"));  // same stamp, right size: kept
  CHECK(write_row_image(path, layout, 2, fill, "fixture B"));   // another fixture's images: rewritten
  CHECK(write_row_image(path, layout, 2, fill, ""));            // no stamp: always written
  CHECK(fills == 6 && std::filesystem::file_size(path) == 2 * 4096);
}

// ---- the real stack, synthetic rows, a fake kernel ----

constexpr int64_t kSelfRows = 2;
constexpr int64_t kSelfExperts = 8;
constexpr int64_t kSelfCapacity = 7;  // 3 staging slots, 4 mappable
constexpr int64_t kSelfHidden = 64;

// One call of the fake kernel, recorded for the test to inspect.
struct FakeCall {
  std::vector<int32_t> slots;
  std::vector<float> weights;
  int32_t threads;
  bool accumulate;
  int core;  // worker 0's core: the group's first
};

// The kernel's stand-in: out[h] = h + x[0] + sum_i weights[i] * (slots[i] + 1), x[0] read as an integer, so the output
// proves the row's x, the slots and the weights reached it.
class FakeKernel final : public ce::CpuExpertKernel {
 public:
  struct Layer final : ce::CpuExpertLayer {
    explicit Layer(const CpuExpertKernel& k) : CpuExpertLayer(k) {}
  };
  explicit FakeKernel(const char* name = "bench-fake") : name_(name) {}
  const char* name() const noexcept override {
    return name_;
  }
  std::unique_ptr<ce::CpuExpertLayer> make_layer(const ce::LayerSlabs&, std::span<const std::byte>) const override {
    return std::make_unique<Layer>(*this);
  }
  void forward(const ce::CpuExpertLayer& layer, const ce::ForwardCall& c) const override {
    if (&layer.kernel() != this) throw std::invalid_argument("bench fake: another kernel's layer");
    uint16_t x0;
    std::memcpy(&x0, c.x, 2);
    float sum = 0.0f;
    for (int32_t i = 0; i < c.k; ++i)
      sum += c.weights[i] * static_cast<float>(c.slots[i] + 1);
    for (int64_t h = 0; h < kSelfHidden; ++h)
      c.out[h] = (c.accumulate ? c.out[h] : 0.0f) + static_cast<float>(h) + static_cast<float>(x0) + sum;
    std::lock_guard<std::mutex> guard(mutex);
    calls.push_back(
        {std::vector<int32_t>(c.slots, c.slots + c.k),
         std::vector<float>(c.weights, c.weights + c.k),
         c.threads,
         c.accumulate,
         c.cores.empty() ? -1 : c.cores.front()});
  }
  void keep_warm(std::span<const int>, int32_t, const uint32_t*, uint32_t, int64_t) const override {}

  mutable std::mutex mutex;
  mutable std::vector<FakeCall> calls;

 private:
  const char* name_;
};

// The byte filling expert `expert`'s `name` slab row in row `row`'s image, so a landed slot identifies its source.
uint8_t pattern(int64_t row, int64_t expert, int name) {
  return static_cast<uint8_t>(1 + row * 64 + expert * 6 + name);
}

// True when part 0 of an output row is the fake kernel's result for the given x and weight sum.
bool part0_is(const float* part0, float offset) {
  for (int64_t h = 0; h < kSelfHidden; ++h)
    if (part0[h] != static_cast<float>(h) + offset) return false;
  return true;
}

// Drives the real RamTier, RamThread, copy engine and CPU expert engine through DeviceSim: loading, SM hits, CPU hits,
// split 0, and a hit with a miss in one post; checks the fake kernel's inputs and outputs and the staging slots.
void test_stack(const Placement& placement, const std::filesystem::path& dir) {
  if constexpr (w::Wire::kNodes != 1) return;  // one group's stack; test_two_groups builds the multi-group one
  const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
  std::vector<std::array<AlignedBuffer, kNames>> slabs(kSelfRows);
  RowSet set;
  set.layout = layout;
  set.experts = kSelfExperts;
  set.capacity = kSelfCapacity;
  for (int64_t row = 0; row < kSelfRows; ++row) {
    std::array<uint8_t*, kNames> bases{};
    for (int n = 0; n < kNames; ++n) {
      slabs[row][n] = aligned_zeroed(kSelfCapacity * 512);
      bases[n] = slabs[row][n].get();
    }
    set.slabs.push_back(bases);
    const auto path = dir / ("selftest-layer-" + std::to_string(row) + ".rows");
    write_row_image(
        path,
        layout,
        kSelfExperts,
        [&](int64_t e, uint8_t* image) {
          for (int n = 0; n < kNames; ++n)
            std::memset(image + layout.name_offsets[n], pattern(row, e, n), 512);
        },
        "");
    set.paths.push_back(path.string());
  }
  AlignedBuffer x = aligned_zeroed(kSelfRows * 2 * kSelfHidden);
  AlignedBuffer out = aligned_zeroed(kSelfRows * 2 * kSelfHidden * 4);
  auto* part0 = reinterpret_cast<float*>(out.get());  // row 0, part 0
  FakeKernel fake;
  StackConfig config;
  config.rows = set;
  config.staging = 3;
  config.kernel = &fake;
  config.x_base = x.get();
  config.x_stride = 2 * kSelfHidden;
  config.out_base = out.get();
  config.out_stride = 2 * kSelfHidden * 4;
  config.hidden = kSelfHidden;
  config.copy_cpu = placement.copy;
  StackConfig::Group group;
  group.service_cpu = placement.groups[0].service;
  group.cores.assign(placement.groups[0].workers.begin(), placement.groups[0].workers.end());
  group.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  config.groups.push_back(group);
  config.trace_capacity = 256;
  const int64_t timeout = 1'000'000'000;
  {
    PinScope writer(placement.writer);
    Stack<BenchBuild> stack(std::move(config));
    DeviceSim sim(stack.page(), stack.lease(), kSelfRows, kSelfExperts);

    // Load three experts: one uncaptured post of three misses, into staging slots 0, 1, 2; the victims 3, 4, 5 (the
    // lowest free slots) become the staging slots.
    const int32_t three[] = {0, 1, 2};
    const uint32_t loaded = load_experts(sim, 0, three, 3, timeout);
    CHECK(sim.ram_slot(0, 0) == 0 && sim.ram_slot(0, 1) == 1 && sim.ram_slot(0, 2) == 2);
    CHECK(sim.staging(0)[0] == 3 && sim.staging(0)[1] == 4 && sim.staging(0)[2] == 5 && sim.staging(0)[3] == -1);
    CHECK(sim.map_chain(0) == 2);
    CHECK(stack.wait_handled(loaded, monotonic_ns() + timeout));
    CHECK(stack.mirror(0, 0) == 0 && stack.mirror(0, 2) == 2 && stack.mirror(0, 3) == -1);
    bool landed = true;  // the real reader read each expert's image into its slot
    for (int64_t e = 0; e < 3; ++e)
      for (int n = 0; n < kNames; ++n)
        landed = landed && slabs[0][n][e * 512] == pattern(0, e, n) && slabs[0][n][e * 512 + 511] == pattern(0, e, n);
    CHECK(landed);

    // Row 0's layer is not registered yet: a captured post types SM hits, which nothing waits for.
    const int32_t two[] = {0, 1};
    const float halves[] = {0.5f, 0.5f};
    SimRequest r = sim.post(0, two, halves, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    {
      std::lock_guard<std::mutex> guard(fake.mutex);
      CHECK(fake.calls.empty());
    }

    // Registered: split[3] = 3, every lane is the CPU's; one CPU job computes slots 2, 0, 1 into part 0.
    stack.set_cpu_layer(0, fake.make_layer({}, {}));
    sim.set_row_cpu(0);
    uint16_t mark = 100;
    std::memcpy(x.get(), &mark, 2);  // the post kernel's x store, before the record
    const int32_t order[] = {2, 0, 1};
    const float weights[] = {0.5f, 0.25f, 0.125f};
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    r = sim.post(0, order, weights, true, monotonic_ns() + timeout);
    CHECK(
        r.kinds[0] == int32_t(w::Wire::kKindHitCpu) && r.kinds[1] == int32_t(w::Wire::kKindHitCpu) &&
        r.kinds[2] == int32_t(w::Wire::kKindHitCpu));
    CHECK(sim.copy_wait(r, monotonic_ns() + timeout));
    {
      std::lock_guard<std::mutex> guard(fake.mutex);
      CHECK(fake.calls.size() == 1);
      if (fake.calls.size() == 1) {
        const FakeCall& call = fake.calls[0];
        CHECK(call.slots == std::vector<int32_t>({2, 0, 1}));
        CHECK(call.weights == std::vector<float>({0.5f, 0.25f, 0.125f}));
        CHECK(call.threads == static_cast<int32_t>(placement.groups[0].workers.size()) && !call.accumulate);
        CHECK(call.core == placement.groups[0].workers.front());
      }
    }
    CHECK(part0_is(part0, 100.0f + 2.0f));  // 0.5 * 3 + 0.25 * 1 + 0.125 * 2
    auto cpu = stack.cpu_stats();
    CHECK(cpu[0] == 1 && cpu[1] == 3 && cpu[2] > 0);
    CHECK(sim.copy_gate() == gate_word(r.seq, w::Wire::kLeaseGateOpen));
    if constexpr (BenchBuild::kMetrics) {
      auto stage = std::make_unique<es::StageRecord>();
      CHECK(stack.drain_stage(*stage, r.seq, monotonic_ns() + timeout));
      CHECK(stage->observed > 0 && stage->done >= stage->observed && stage->lanes == 3);
    }

    // split[n] = 0: no CPU lane, nothing for the CPU.
    stack.set_split({0, 0, 0, 0, 0, 0, 0, 0, 0});
    r = sim.post(0, two, halves, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitSm) && r.kinds[1] == int32_t(w::Wire::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    CHECK(stack.cpu_stats()[0] == 1);
    stack.set_split({0, 1, 2, 3, 4, 5, 6, 7, 8});

    // A hit and a miss: the hit is the CPU's; the miss is read into staging[0] = 3 for the device (cpu_misses off).
    const int32_t mixed[] = {0, 6};
    const float mixed_weights[] = {0.75f, 0.5f};
    mark = 101;
    std::memcpy(x.get(), &mark, 2);
    r = sim.post(0, mixed, mixed_weights, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::Wire::kKindHitCpu) && r.kinds[1] == int32_t(w::Wire::kKindMissGpu));
    CHECK(r.slots[1] == 3 && r.chain == 3);
    CHECK(sim.copy_wait(r, monotonic_ns() + timeout));
    CHECK(sim.wait_pieces(r, 1, monotonic_ns() + timeout));
    CHECK(part0_is(part0, 101.0f + 0.75f));
    landed = true;
    for (int n = 0; n < kNames; ++n)
      landed = landed && slabs[0][n][3 * 512] == pattern(0, 6, n);
    CHECK(landed);
    cpu = stack.cpu_stats();
    CHECK(cpu[0] == 2 && cpu[1] == 4);
    sim.sync_row(0, monotonic_ns() + timeout);  // delta 3: expert 6 in slot 3; the victim, slot 6, now staging[0]
    CHECK(sim.ram_slot(0, 6) == 3 && sim.staging(0)[0] == 6);
    const float* row1 = part0 + 2 * kSelfHidden;
    bool untouched = true;
    for (int64_t i = 0; i < 2 * kSelfHidden; ++i)
      untouched = untouched && row1[i] == 0.0f;
    CHECK(untouched);
  }  // teardown: open the gate, stop the service, settle, stop the copy and CPU threads
  CHECK(fake.calls.size() == 2);
}

// Two NUMA groups' synthetic rows, row images and pinned buffers (test_two_groups, test_groups_must_share_one_kernel):
// per group 3 staging slots and 4 mappable; config() sends every eligible lane to the CPU.
struct TwoGroupRig {
  static constexpr int64_t kGroup = 7;
  std::vector<std::array<AlignedBuffer, kNames>> slabs;
  RowSet set;
  AlignedBuffer x;
  AlignedBuffer out;

  TwoGroupRig(const std::filesystem::path& dir, const std::string& name)
      : slabs(kSelfRows),
        x(aligned_zeroed(kSelfRows * 2 * kSelfHidden)),
        out(aligned_zeroed(kSelfRows * 4 * kSelfHidden * 4)) {  // two parts per group
    const ImageLayout layout = image_layout({512, 512, 512, 512, 512, 512});
    set.layout = layout;
    set.experts = kSelfExperts;
    set.capacity = 2 * kGroup;
    for (int64_t row = 0; row < kSelfRows; ++row) {
      std::array<uint8_t*, kNames> bases{};
      for (int n = 0; n < kNames; ++n) {
        slabs[row][n] = aligned_zeroed(2 * kGroup * 512);
        bases[n] = slabs[row][n].get();
      }
      set.slabs.push_back(bases);
      const auto path = dir / (name + "-layer-" + std::to_string(row) + ".rows");
      write_row_image(
          path,
          layout,
          kSelfExperts,
          [&](int64_t e, uint8_t* image) {
            for (int n = 0; n < kNames; ++n)
              std::memset(image + layout.name_offsets[n], pattern(row, e, n), 512);
          },
          "");
      set.paths.push_back(path.string());
    }
  }

  StackConfig config(const Placement& placement, const ce::CpuExpertKernel& kernel) const {
    StackConfig c;
    c.rows = set;
    c.staging = 3;
    c.kernel = &kernel;
    c.x_base = x.get();
    c.x_stride = 2 * kSelfHidden;
    c.out_base = out.get();
    c.out_stride = 4 * kSelfHidden * 4;
    c.hidden = kSelfHidden;
    c.copy_cpu = placement.copy;
    c.ranges = {{0, kGroup}, {kGroup, 2 * kGroup}};
    for (int g = 0; g < 2; ++g) {
      StackConfig::Group group;
      group.service_cpu = placement.groups[g].service;
      group.cores.assign(placement.groups[g].workers.begin(), placement.groups[g].workers.end());
      group.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
      c.groups.push_back(group);
    }
    return c;
  }
};

// Two NUMA groups (Wire::kNodes == 2): each group's CPU lanes reach the kernel on its own cores with only its own
// slots, and each group's service thread runs on its own CPU.
void test_two_groups(const Placement& placement, const std::filesystem::path& dir) {
  if constexpr (w::Wire::kNodes != 2) {
    return;
  } else {
    constexpr int64_t kGroup = TwoGroupRig::kGroup;
    TwoGroupRig rig(dir, "selftest2");
    FakeKernel fake;
    StackConfig config = rig.config(placement, fake);
    PinScope writer(placement.writer);
    Stack<BenchBuild> stack(std::move(config));
    DeviceSim sim(stack.page(), stack.lease(), kSelfRows, kSelfExperts);
    const int32_t four[] = {0, 1, 2, 3};
    // One load per group, so no post mixes groups' misses: a mixed post leaves the tier's staging order unlike the
    // lowest-first order the slot checks below read (expert 3 landed in slot 10).
    const int32_t home0[] = {0, 2};
    const int32_t home1[] = {1, 3};
    load_experts(sim, 0, home0, 3, soon());
    load_experts(sim, 0, home1, 3, soon());
    stack.set_cpu_layer(0, fake.make_layer({}, {}));
    sim.set_row_cpu(0);
    const float ones[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const SimRequest r = sim.post(0, four, ones, /*captured=*/true, soon());
    CHECK(sim.copy_wait(r, soon()));
    {
      std::lock_guard<std::mutex> guard(fake.mutex);
      CHECK(fake.calls.size() == 2);  // one CPU-hit job per group
      std::vector<int64_t> groups;
      for (const FakeCall& call : fake.calls) {
        int64_t g = -1;
        for (int candidate = 0; candidate < 2; ++candidate)
          if (call.core == placement.groups[candidate].workers.front()) g = candidate;
        groups.push_back(g);
        std::vector<int32_t> slots = call.slots;
        std::sort(slots.begin(), slots.end());
        // Each group's staging slots are the lowest of its range: experts {0, 2} (home 0) land in 0, 1 and {1, 3}
        // (home 1) in kGroup, kGroup + 1.
        CHECK(g == 0 || g == 1);
        CHECK(slots == std::vector<int32_t>({int32_t(g * kGroup), int32_t(g * kGroup + 1)}));
      }
      std::sort(groups.begin(), groups.end());
      CHECK(groups == std::vector<int64_t>({0, 1}));
    }
    // Group g's part 0 sits at floats [2 g hidden, (2 g + 1) hidden) of the row: the fake forward wrote
    // h + x[0] + sum(weight * (slot + 1)) there, 1 * (0 + 1) + 1 * (1 + 1) for group 0 and 1 * 8 + 1 * 9 for group 1.
    const auto* row0 = reinterpret_cast<const float*>(rig.out.get());
    CHECK(part0_is(row0, 3.0f));
    CHECK(part0_is(row0 + 2 * kSelfHidden, 17.0f));
    for (int g = 0; g < 2; ++g)
      CHECK(stack.group_counters(g)[es::kSpinCpu] == placement.groups[g].service);
  }
}

// Review Focus 2: one tier runs one kernel. Group 1 naming a second kernel is refused while the stack is built, the
// message naming both kernels. Runs only in the two-node build: BENCH kiface-bench-n2 gates it, kiface-bench (one node)
// returns at once.
void test_groups_must_share_one_kernel(const Placement& placement, const std::filesystem::path& dir) {
  if constexpr (w::Wire::kNodes != 2) {
    return;
  } else {
    TwoGroupRig rig(dir, "selftest-kernels");
    FakeKernel first("fake-a"), second("fake-b");
    StackConfig config = rig.config(placement, first);
    config.groups[1].kernel = &second;
    PinScope writer(placement.writer);
    CHECK_THROWS(Stack<BenchBuild> stack(std::move(config)), "group 1 names fake-b, another group fake-a");
  }
}

}  // namespace

int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  test_placement();
  test_two_group_placement();
  test_record_bytes();
  test_full_width_record();
  test_seqlock_order();
  test_owed_delta();
  test_typing();
  test_epoch_wrap();
  test_copy_wait_gate();
  require_o_direct(image_dir);
  test_image_stamp(image_dir);
  test_stack(placement, image_dir);
  test_two_groups(placement, image_dir);
  test_groups_must_share_one_kernel(placement, image_dir);
  std::fprintf(
      stderr, "self-test (%s): %d checks, %d failed\n", std::string(BenchBuild::kName).c_str(), checks, failures);
  return failures;
}

}  // namespace fullstack
