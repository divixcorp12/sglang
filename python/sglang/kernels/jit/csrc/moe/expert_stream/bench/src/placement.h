// Where the bench's threads run (spec, "Placement"), the refusals before setup, and the affinity check after it.
#pragma once

#include <sched.h>

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace fullstack {

int number(const std::string& text);
std::vector<int32_t> parse_cpus(const std::string& text);  // "16-17,52": ranges and singles, no duplicates

struct Placement {
  int writer = -1;   // the GPU stand-in: posts records, spins on CopyDone
  int service = -1;  // RamThread, busy-polling: a physical core of its own
  int copy = -1;     // the copy engine thread and RamThread's watchdog (both inherit the enabling thread's affinity)
  std::vector<int32_t> workers;  // CPU experts: worker 0 (the CPU expert thread, the kernel's caller), then helpers
  int host_node = 0;    // writer, service and copy
  int worker_node = 1;  // every worker
};

struct Topology {
  std::function<int(int)> node_of;                  // a CPU's NUMA node, -1 when unknown
  std::function<std::vector<int>(int)> siblings_of; // a CPU's SMT siblings, itself included
  cpu_set_t allowed;                                // the process's CPUs
};

Topology system_topology();  // sysfs and sched_getaffinity

// Throws, naming the CPU and the rule: duplicate roles, a CPU outside `allowed`, a role on the service CPU's SMT
// sibling, and (check_nodes) a role off its node.
void validate_placement(const Placement& placement, const Topology& topology, bool check_nodes);

void pin_self(int cpu);  // the calling thread only

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

std::set<int> task_ids();
// The CPUs the threads created during setup must be pinned to, one each: the service, the copy thread and the
// watchdog (both on `copy`), the CPU expert thread (workers[0]) and the kernel's helpers (workers[1..]).
std::vector<int> expected_threads(const Placement& placement);
// Every thread not in `before`, except io_uring's kernel workers ("iou-*"), must be pinned to exactly one CPU, and
// those CPUs, sorted, must equal `expected` sorted. Throws otherwise.
void verify_threads(const std::set<int>& before, std::vector<int> expected);
std::string cpu_list(std::vector<int> cpus);  // "16,17,52"

}  // namespace fullstack
