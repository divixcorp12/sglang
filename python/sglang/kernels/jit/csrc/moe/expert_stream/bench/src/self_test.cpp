#include "self_test.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace fullstack {
namespace {

int checks = 0;
int failures = 0;

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

}  // namespace

int run_self_test(const Placement& placement, const std::filesystem::path& image_dir) {
  (void)placement;
  (void)image_dir;
  test_placement();
  std::fprintf(stderr, "self-test: %d checks, %d failed\n", checks, failures);
  return failures;
}

}  // namespace fullstack
