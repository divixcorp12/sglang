#include "self_test.h"

#include <array>
#include <cstring>
#include <utility>

#include "aligned.h"
#include "device_sim.h"
#include "expert_stream/host/tier_protocol.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace fullstack {
namespace {

int checks = 0;
int failures = 0;
namespace w = ::sglang::expert_stream::wire;
namespace es = ::sglang::expert_stream;

void check(bool ok, const char* what, const char* file, int line) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
  }
}

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

// divix01: node 0 = 0-17,36-53; node 1 = 18-35,54-71; c and c + 36 are SMT siblings. Allowed: the partition.
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

Placement production_placement() {
  Placement p;
  p.writer = 16;
  p.service = 17;
  p.copy = 52;
  p.workers = parse_cpus("18-33");
  return p;
}

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
  twice.service = 16;
  CHECK_THROWS(validate_placement(twice, t, true), "two roles");

  Placement sibling = production_placement();
  sibling.copy = 53;  // 17's SMT sibling
  CHECK_THROWS(validate_placement(sibling, t, true), "physical core of the service CPU 17");

  Placement outside = production_placement();
  outside.copy = 34;
  CHECK_THROWS(validate_placement(outside, t, true), "outside the process's allowed CPUs");

  Placement writer_node = production_placement();
  writer_node.writer = 19;
  writer_node.workers = parse_cpus("18,20-33");
  CHECK_THROWS(validate_placement(writer_node, t, true), "writer CPU 19 is on NUMA node 1, not node 0");
  CHECK(passes(writer_node, t, false));  // the self-test's mode: no node rules

  Topology wider = fake_topology();
  CPU_SET(10, &wider.allowed);
  Placement worker_node = production_placement();
  worker_node.workers.back() = 10;
  CHECK_THROWS(validate_placement(worker_node, wider, true), "worker CPU 10 is on NUMA node 0, not node 1");

  std::vector<int> expected = expected_threads(production_placement());
  std::vector<int> want = {17};  // sorted: service, workers 18-33, then the copy CPU twice (copy thread, watchdog)
  for (int cpu = 18; cpu <= 33; ++cpu) want.push_back(cpu);
  want.push_back(52);
  want.push_back(52);
  CHECK(expected == want);
  CHECK(cpu_list({52, 16, 17}) == "16,17,52");
}

// ---- DeviceSim, on a standalone page and lease block (no service: the host's words are written by hand) ----

uint32_t gate_word(uint32_t seq, uint32_t low) {
  return ((seq & w::kLeaseGateSeqMask) << w::kLeaseGateSeqShift) | low;
}

int64_t soon() {
  return monotonic_ns() + 100'000'000;
}

constexpr std::array<int16_t, 8> kStaging012 = {0, 1, 2, -1, -1, -1, -1, -1};
constexpr std::array<int32_t, 9> kAllToCpu = {0, 1, 2, 3, 4, 5, 6, 7, 8};

struct Blocks {
  explicit Blocks(int64_t rows)
      : page(aligned_zeroed(w::kPageBytes)),
        lease_bytes(w::kLeaseBlockBytes + round_up(rows * w::kDeltaStride, 4096)),
        lease(aligned_zeroed(lease_bytes)) {}

  // The host's delta record for `row`: payload, then the tag with a release (RamTier::publish_delta_locked).
  void delta(int64_t row, uint64_t tag, std::array<int16_t, 8> staging, std::vector<std::pair<int16_t, int16_t>> entries) {
    uint8_t* d = lease.get() + w::kDeltaBase + row * w::kDeltaStride;
    const auto count = static_cast<uint32_t>(entries.size());
    std::memcpy(d + w::kDeltaCount, &count, 4);
    std::memcpy(d + w::kDeltaStaging, staging.data(), 16);
    for (size_t i = 0; i < entries.size(); ++i) {
      const int16_t entry[2] = {entries[i].first, entries[i].second};
      std::memcpy(d + w::kDeltaEntries + 4 * i, entry, 4);
    }
    __atomic_store_n(reinterpret_cast<uint64_t*>(d + w::kDeltaTag), tag, __ATOMIC_RELEASE);
  }
  void split(std::array<int32_t, 9> table) {
    std::memcpy(lease.get() + w::kSplit, table.data(), sizeof(table));
  }
  void armed(bool on) {
    const uint32_t value = on ? 1 : 0;
    std::memcpy(lease.get() + w::kCopyArmed, &value, 4);
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
  CHECK(r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindMissGpu));
  CHECK(r.slots[0] == 4 && r.slots[1] == 0);
  CHECK(r.seq == 1 && r.gen == 1 && r.idx == 0 && r.chain == 2 && sim.map_chain(0) == 2);
  const int64_t rec = w::kDemandRing;
  CHECK(b.at<uint32_t>(w::kDemandHead) == 1);
  CHECK(b.at<uint32_t>(rec + w::kRecSeq) == 1);
  CHECK(b.at<uint16_t>(rec + w::kRecRow) == 0);
  CHECK(b.at<uint8_t>(rec + w::kRecCounts) == (2 | 2 << 4));
  CHECK(b.at<uint8_t>(rec + w::kRecFlags) == w::kRecFlagCaptured);
  CHECK(b.at<uint64_t>(rec + w::kRecChain) == 2);
  CHECK(b.at<uint32_t>(rec + w::kRecEpoch) == 0);
  CHECK(b.at<uint32_t>(rec + w::kRecKinds) == (3u | 4u << 4));
  const int16_t ids[8] = {3, 5, -1, -1, -1, -1, -1, -1};
  const int16_t slots[8] = {4, 0, -1, -1, -1, -1, -1, -1};
  const int16_t dst[8] = {0, 1, -1, -1, -1, -1, -1, -1};
  const float lane_weights[8] = {0.5f, 0.25f, 0, 0, 0, 0, 0, 0};
  CHECK(std::memcmp(b.page.get() + rec + w::kRecProtect, ids, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneExpert, ids, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneSlot, slots, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneDst, dst, 16) == 0);
  CHECK(std::memcmp(b.page.get() + rec + w::kRecLaneWeight, lane_weights, 32) == 0);
  // The production parser reads it back as the device meant it.
  es::Request req;
  CHECK(es::read_record(b.page.get() + rec, 1, &req) == es::RecordRead::kOk);
  CHECK(req.gen == 1 && req.row == 0 && req.captured && req.chain == 2);
  CHECK(req.lanes.size() == 2 && req.protect.size() == 2);
  CHECK(req.lanes[0].expert == 3 && req.lanes[0].slot == 4 && req.lanes[0].dst == 0 && req.lanes[0].weight == 0.5f &&
        req.lanes[0].kind == w::kKindHitCpu);
  CHECK(req.lanes[1].expert == 5 && req.lanes[1].slot == 0 && req.lanes[1].kind == w::kKindMissGpu);
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
    std::memcpy(&seq, record + w::kRecSeq, 4);
    std::memcpy(&head, b.page.get() + w::kDemandHead, 4);
    std::memcpy(&kinds, record + w::kRecKinds, 4);
    std::memcpy(&weight, record + w::kRecLaneWeight, 4);
    CHECK(seq == 0);   // the seqlock word is 0 while the payload is in place
    CHECK(head == 0);  // demand_head moves only after the seq
    CHECK(kinds == w::kKindHitCpu && weight == 0.75f);
  });
  CHECK(hooked);
  CHECK(b.at<uint32_t>(w::kDemandRing + w::kRecSeq) == 1 && b.at<uint32_t>(w::kDemandHead) == 1);
  // A rewrite of a ring slot that held an earlier record also zeroes its seq first: seq 17 reuses slot 0.
  for (int i = 0; i < 15; ++i) sim.post(0, e, wt, true, soon());
  bool rehooked = false;
  sim.post(0, e, wt, true, soon(), [&](const uint8_t* record) {
    rehooked = true;
    uint32_t seq;
    std::memcpy(&seq, record + w::kRecSeq, 4);
    CHECK(record == b.page.get() + w::kDemandRing);
    CHECK(seq == 0);
  });
  CHECK(rehooked && b.at<uint32_t>(w::kDemandRing + w::kRecSeq) == 17);
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
  CHECK(first.kinds[0] == int32_t(w::kKindMissGpu) && first.slots[0] == 0 && first.chain == 2);
  // The host has not published delta 2: the row's next post waits for it, then refuses at its deadline.
  CHECK_THROWS(sim.post(0, miss, one, true, monotonic_ns() + 1'000'000), "delta 2");
  CHECK(b.at<uint32_t>(w::kDemandHead) == 1);  // the refused post wrote nothing
  // Delta 2, as the host publishes it: expert 5 in slot 0, the victim slot 3 now staging[0].
  b.delta(0, 2, {3, 1, 2, -1, -1, -1, -1, -1}, {{5, 0}});
  const SimRequest hit = sim.post(0, miss, one, true, soon());
  CHECK(hit.kinds[0] == int32_t(w::kKindHitCpu) && hit.slots[0] == 0 && hit.chain == 0 && sim.map_chain(0) == 2);
  CHECK(sim.staging(0)[0] == 3);
  // The next miss takes the new staging[0] and the next chain number.
  const int32_t other[] = {6};
  const SimRequest second = sim.post(0, other, one, true, soon());
  CHECK(second.kinds[0] == int32_t(w::kKindMissGpu) && second.slots[0] == 3 && second.chain == 3);
}

void test_typing() {
  Blocks b(1);
  b.delta(0, 1, kStaging012, {{0, 4}, {1, 5}});
  b.armed(true);
  b.split({0, 0, 0, 0, 0, 0, 0, 0, 0});
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8);
  sim.set_row_cpu(0);
  const int32_t two[] = {0, 1};
  const float halves[] = {0.5f, 0.5f};
  SimRequest r = sim.post(0, two, halves, true, soon());  // split[2] = 0: no CPU lane
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm) && !DeviceSim::needs_copy_wait(r));
  b.split({0, 1, 1, 1, 1, 1, 1, 1, 1});  // split[2] = 1: the last eligible lane only
  r = sim.post(0, two, halves, true, soon());
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitCpu) && DeviceSim::needs_copy_wait(r));
  b.split(kAllToCpu);
  r = sim.post(0, two, halves, false, soon());  // uncaptured: no host lanes
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
  b.armed(false);
  r = sim.post(0, two, halves, true, soon());  // the copy engine is not armed: no host lanes
  CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
  Blocks c(1);
  c.delta(0, 1, kStaging012, {{0, 4}});
  c.split(kAllToCpu);
  c.armed(true);
  DeviceSim cold(c.page.get(), c.lease.get(), 1, 8);  // no set_row_cpu: the row's layer is not registered
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  CHECK(cold.post(0, e, wt, true, soon()).kinds[0] == int32_t(w::kKindHitSm));
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
  std::memcpy(b.page.get() + w::kDemandHead, &last, 4);
  DeviceSim sim(b.page.get(), b.lease.get(), 1, 8, /*epoch=*/7);
  sim.set_row_cpu(0);
  const int32_t e[] = {0};
  const float wt[] = {1.0f};
  const SimRequest r = sim.post(0, e, wt, true, soon());
  CHECK(r.seq == 1 && r.idx == 0 && sim.epoch() == 8 && r.gen == (uint64_t{8} << 32 | 1));
  CHECK(b.at<uint32_t>(w::kDemandRing + w::kRecEpoch) == 8 && b.at<uint32_t>(w::kDemandHead) == 1);
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
  CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateClosed));
  // CopyDone stored before the close (the host's open found nothing closed): CW opens the gate itself.
  __atomic_store_n(reinterpret_cast<uint64_t*>(b.lease.get() + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes),
                   r.gen, __ATOMIC_RELEASE);
  CHECK(sim.copy_wait(r, soon()));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateOpen));
  CHECK(sim.copy_done(r) == r.gen);
}

}  // namespace

int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  (void)placement;
  (void)image_dir;
  test_placement();
  test_record_bytes();
  test_seqlock_order();
  test_owed_delta();
  test_typing();
  test_epoch_wrap();
  test_copy_wait_gate();
  std::fprintf(stderr, "self-test: %d checks, %d failed\n", checks, failures);
  return failures;
}

}  // namespace fullstack
