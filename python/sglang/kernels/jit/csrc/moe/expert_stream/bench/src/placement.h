// Where the bench's threads run, the refusals before setup, and the affinity check after it.
//
// Each thread of the stack gets one CPU of its own. The busy-polling service never yields, so no other role may share
// its physical core, and the CPU expert workers must sit on a different NUMA node from the host-side threads. Setup
// is refused when a rule is broken, and afterwards every created thread's affinity is checked against the placement.
// See python/sglang/kernels/jit/csrc/moe/expert_stream/bench/README.txt, "Full-stack bench".
#pragma once

#include <cstdint>
#include <functional>
#include <sched.h>
#include <set>
#include <string>
#include <vector>

namespace fullstack {

// Parses a non-negative decimal integer; throws on anything else.
int number(const std::string& text);

// Parses a CPU list such as "16-17,52": ranges and singles. Throws on a duplicate, an empty list or a bad range.
std::vector<int32_t> parse_cpus(const std::string& text);

// The CPU of each role of the stack, and the NUMA node each group must be on.
struct Placement {
  int writer = -1;   // the GPU stand-in: posts records, spins on CopyDone
  int service = -1;  // RamThread, busy-polling: a physical core of its own
  int copy = -1;     // the copy engine thread and RamThread's watchdog (both inherit the enabling thread's affinity)
  std::vector<int32_t> workers;  // CPU experts: worker 0 (the CPU expert thread, the kernel's caller), then helpers
  int host_node = 0;             // writer, service and copy
  int worker_node = 1;           // every worker
};

// The machine facts validate_placement consults, injectable so the self-test can check the rules without sysfs.
struct Topology {
  std::function<int(int)> node_of;                   // a CPU's NUMA node, -1 when unknown
  std::function<std::vector<int>(int)> siblings_of;  // a CPU's SMT siblings, itself included
  cpu_set_t allowed;                                 // the process's CPUs
};

// The real topology, from sysfs and sched_getaffinity.
Topology system_topology();

// Throws, naming the CPU and the rule: duplicate roles, a CPU outside `allowed`, a role on the service CPU's SMT
// sibling, and (check_nodes) a role off its node.
void validate_placement(const Placement& placement, const Topology& topology, bool check_nodes);

// Pins the calling thread, and only it, to `cpu`. Throws on failure.
void pin_self(int cpu);

// Pins the calling thread to `cpu` and restores its previous affinity on destruction. Threads created inside the
// scope inherit `cpu`.
class PinScope {
 public:
  explicit PinScope(int cpu);
  ~PinScope();
  PinScope(const PinScope&) = delete;
  PinScope& operator=(const PinScope&) = delete;

 private:
  cpu_set_t saved_;
};

// The ids of this process's threads, from /proc/self/task.
std::set<int> task_ids();

// The CPUs the threads created during setup must be pinned to, one each: the service, the copy thread and the
// watchdog (both on `copy`), the CPU expert thread (workers[0]) and the kernel's helpers (workers[1..]).
std::vector<int> expected_threads(const Placement& placement);
// Every thread not in `before`, except io_uring's kernel workers ("iou-*"), must be pinned to exactly one CPU, and
// those CPUs, sorted, must equal `expected` sorted. Retried for up to 2 s (exiting threads); throws otherwise.
void verify_threads(const std::set<int>& before, std::vector<int> expected);
// Formats sorted CPUs as "16,17,52".
std::string cpu_list(std::vector<int> cpus);

}  // namespace fullstack
