// The bench's --self-test: checks the harness itself, so a bench result is never the harness's fault.
//
// Needs no fixture file and no GPU. In order:
//   test_placement          the placement rules, against a fake topology of the reference machine
//   test_record_bytes ...   DeviceSim's records, seqlock order, delta handling, lane typing, epoch wrap and copy-wait
//                           gate, on a standalone page and lease block with the host's words written by hand
//   test_image_stamp        row-image layout and stamp reuse
//   test_stack              the real stack (this binary's build) on synthetic rows with a fake forward
// Each failed check prints "FAIL file:line" and counts toward run_self_test's return value.
#include "self_test.h"

#include "aligned.h"
#include "device_sim.h"
#include "expert_stream/host/tier_protocol.h"
#include "row_images.h"
#include "stack.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace fullstack {
namespace {

int checks = 0;
int failures = 0;
namespace w = ::sglang::expert_stream::wire;
namespace es = ::sglang::expert_stream;

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
  p.service = 17;
  p.copy = 52;
  p.workers = parse_cpus("18-33");
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
  for (int cpu = 18; cpu <= 33; ++cpu)
    want.push_back(cpu);
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

// A standalone request page and lease block, with helpers that write the host's words (deltas, the split table, the
// armed flag) the way the host does, and read the device's words back.
struct Blocks {
  explicit Blocks(int64_t rows)
      : page(aligned_zeroed(w::kPageBytes)),
        lease_bytes(w::kLeaseBlockBytes + round_up(rows * w::kDeltaStride, 4096)),
        lease(aligned_zeroed(lease_bytes)) {}

  // The host's delta record for `row`: payload, then the tag with a release (RamTier::publish_delta_locked).
  void
  delta(int64_t row, uint64_t tag, std::array<int16_t, 8> staging, std::vector<std::pair<int16_t, int16_t>> entries) {
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
  CHECK(
      req.lanes[0].expert == 3 && req.lanes[0].slot == 4 && req.lanes[0].dst == 0 && req.lanes[0].weight == 0.5f &&
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
  for (int i = 0; i < 15; ++i)
    sim.post(0, e, wt, true, soon());
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
  __atomic_store_n(
      reinterpret_cast<uint64_t*>(b.lease.get() + w::kLeaseCopyDone + r.idx * w::kLeaseCopyDoneBytes),
      r.gen,
      __ATOMIC_RELEASE);
  CHECK(sim.copy_wait(r, soon()));
  CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateOpen));
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

// ---- the real stack, synthetic rows, a fake forward ----

constexpr int64_t kSelfRows = 2;
constexpr int64_t kSelfExperts = 8;
constexpr int64_t kSelfCapacity = 7;  // 3 staging slots, 4 mappable
constexpr int64_t kSelfHidden = 64;
constexpr int64_t kFakeHandle = 7;

// One call of the fake forward, recorded for the test to inspect.
struct FakeCall {
  int64_t layer;
  std::vector<int32_t> slots;
  std::vector<float> weights;
  int32_t threads;
  int32_t accumulate;
};
std::mutex fake_mutex;
std::vector<FakeCall> fake_calls;

// The kernel's stand-in: out[h] = h + x[0] + sum_i weights[i] * (slots[i] + 1), x[0] read as an integer, so the output
// proves the row's x, the slots and the weights reached it.
int fake_forward(
    int64_t layer,
    const void* x,
    const int32_t* slots,
    const float* weights,
    int32_t k,
    float* out,
    int32_t threads,
    int32_t accumulate) {
  uint16_t x0;
  std::memcpy(&x0, x, 2);
  float sum = 0.0f;
  for (int32_t i = 0; i < k; ++i)
    sum += weights[i] * static_cast<float>(slots[i] + 1);
  for (int64_t h = 0; h < kSelfHidden; ++h)
    out[h] = (accumulate != 0 ? out[h] : 0.0f) + static_cast<float>(h) + static_cast<float>(x0) + sum;
  std::lock_guard<std::mutex> guard(fake_mutex);
  fake_calls.push_back(
      {layer, std::vector<int32_t>(slots, slots + k), std::vector<float>(weights, weights + k), threads, accumulate});
  return 0;
}

// The byte filling expert `expert`'s `name` slab row in row `row`'s image, so a landed slot identifies its source.
uint8_t pattern(int64_t row, int64_t expert, int name) {
  return static_cast<uint8_t>(1 + row * 64 + expert * 6 + name);
}

// True when part 0 of an output row is fake_forward's result for the given x and weight sum.
bool part0_is(const float* part0, float offset) {
  for (int64_t h = 0; h < kSelfHidden; ++h)
    if (part0[h] != static_cast<float>(h) + offset) return false;
  return true;
}

// Drives the real RamTier, RamThread, copy engine and CPU expert engine through DeviceSim: loading, SM hits, CPU hits,
// split 0, and a hit with a miss in one post; checks the fake forward's inputs and outputs and the staging slots.
void test_stack(const Placement& placement, const std::filesystem::path& dir) {
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
  StackConfig config;
  config.rows = set;
  config.staging = 3;
  config.forward = &fake_forward;
  config.threads = static_cast<int>(placement.workers.size());
  config.cores.assign(placement.workers.begin(), placement.workers.end());
  config.x_base = x.get();
  config.x_stride = 2 * kSelfHidden;
  config.out_base = out.get();
  config.out_stride = 2 * kSelfHidden * 4;
  config.hidden = kSelfHidden;
  config.service_cpu = placement.service;
  config.copy_cpu = placement.copy;
  config.split = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  config.trace_capacity = 256;
  fake_calls.clear();
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
    CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    CHECK(fake_calls.empty());

    // Registered: split[3] = 3, every lane is the CPU's; one CPU job computes slots 2, 0, 1 into part 0.
    stack.set_cpu_layer(0, kFakeHandle);
    sim.set_row_cpu(0);
    uint16_t mark = 100;
    std::memcpy(x.get(), &mark, 2);  // the post kernel's x store, before the record
    const int32_t order[] = {2, 0, 1};
    const float weights[] = {0.5f, 0.25f, 0.125f};
    if constexpr (BenchBuild::kMetrics) stack.drain_all();
    r = sim.post(0, order, weights, true, monotonic_ns() + timeout);
    CHECK(
        r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindHitCpu) &&
        r.kinds[2] == int32_t(w::kKindHitCpu));
    CHECK(sim.copy_wait(r, monotonic_ns() + timeout));
    {
      std::lock_guard<std::mutex> guard(fake_mutex);
      CHECK(fake_calls.size() == 1);
      if (fake_calls.size() == 1) {
        const FakeCall& call = fake_calls[0];
        CHECK(call.layer == kFakeHandle && call.slots == std::vector<int32_t>({2, 0, 1}));
        CHECK(call.weights == std::vector<float>({0.5f, 0.25f, 0.125f}));
        CHECK(call.threads == static_cast<int32_t>(placement.workers.size()) && call.accumulate == 0);
      }
    }
    CHECK(part0_is(part0, 100.0f + 2.0f));  // 0.5 * 3 + 0.25 * 1 + 0.125 * 2
    auto cpu = stack.cpu_stats();
    CHECK(cpu[0] == 1 && cpu[1] == 3 && cpu[2] > 0);
    CHECK(sim.copy_gate() == gate_word(r.seq, w::kLeaseGateOpen));
    if constexpr (BenchBuild::kMetrics) {
      auto stage = std::make_unique<es::StageRecord>();
      CHECK(stack.drain_stage(*stage, r.seq, monotonic_ns() + timeout));
      CHECK(stage->observed > 0 && stage->done >= stage->observed && stage->lanes == 3);
    }

    // split[n] = 0: no CPU lane, nothing for the CPU.
    stack.set_split({0, 0, 0, 0, 0, 0, 0, 0, 0});
    r = sim.post(0, two, halves, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitSm) && r.kinds[1] == int32_t(w::kKindHitSm));
    CHECK(stack.wait_handled(r.seq, monotonic_ns() + timeout));
    CHECK(stack.cpu_stats()[0] == 1);
    stack.set_split({0, 1, 2, 3, 4, 5, 6, 7, 8});

    // A hit and a miss: the hit is the CPU's; the miss is read into staging[0] = 3 for the device (cpu_misses off).
    const int32_t mixed[] = {0, 6};
    const float mixed_weights[] = {0.75f, 0.5f};
    mark = 101;
    std::memcpy(x.get(), &mark, 2);
    r = sim.post(0, mixed, mixed_weights, true, monotonic_ns() + timeout);
    CHECK(r.kinds[0] == int32_t(w::kKindHitCpu) && r.kinds[1] == int32_t(w::kKindMissGpu));
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
  CHECK(fake_calls.size() == 2);
}

}  // namespace

int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  test_placement();
  test_record_bytes();
  test_seqlock_order();
  test_owed_delta();
  test_typing();
  test_epoch_wrap();
  test_copy_wait_gate();
  require_o_direct(image_dir);
  test_image_stamp(image_dir);
  test_stack(placement, image_dir);
  std::fprintf(
      stderr, "self-test (%s): %d checks, %d failed\n", std::string(BenchBuild::kName).c_str(), checks, failures);
  return failures;
}

}  // namespace fullstack
