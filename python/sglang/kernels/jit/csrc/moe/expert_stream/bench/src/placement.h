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

// One NUMA group's CPUs: its service thread and its CPU experts. One per NUMA group of the build; at one group the
// workers sit on `node` and the service on host_node, the measured cross-socket layout; above one group a group's
// service and workers sit on its own node.
struct GroupPlacement {
  int service = -1;  // RamThread's group thread, busy-polling: a physical core of its own
  std::vector<int32_t> workers;  // CPU experts: worker 0 (the CPU expert thread, the kernel's caller), then helpers
  int node = 1;                  // the group's NUMA node
};

// The CPU of each role of the stack, and the NUMA node each role must be on.
struct Placement {
  int writer = -1;  // the GPU stand-in: posts records, spins on CopyDone
  int copy = -1;    // the copy engine thread and RamThread's watchdog (both inherit the enabling thread's affinity)
  int host_node = 0;  // writer and copy, and at one group the service
  std::vector<GroupPlacement> groups;
};

// The machine facts validate_placement consults, injectable so the self-test can check the rules without sysfs.
struct Topology {
  std::function<int(int)> node_of;                   // a CPU's NUMA node, -1 when unknown
  std::function<std::vector<int>(int)> siblings_of;  // a CPU's SMT siblings, itself included
  cpu_set_t allowed;                                 // the process's CPUs
};

// The real topology, from sysfs and sched_getaffinity.
Topology system_topology();

// Throws, naming the CPU and the rule: duplicate roles, a CPU outside `allowed`, a role on a service CPU's SMT
// sibling, and (check_nodes) a role off its node.
void validate_placement(const Placement& placement, const Topology& topology, bool check_nodes);

// Binds the whole pages of [address, address + bytes) to NUMA node `node`, moving any already touched
// (mbind MPOL_BIND | MPOL_MF_MOVE). The pages that hold a range's ends are left where they are. Throws on failure.
void bind_pages(void* address, size_t bytes, int node);

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

// The CPUs the threads created during setup must be pinned to, one each: every group's service, the copy thread and
// the watchdog (both on `copy`), and every group's CPU expert thread (workers[0]) and kernel helpers (workers[1..]).
std::vector<int> expected_threads(const Placement& placement);
// Every thread not in `before`, except io_uring's kernel workers ("iou-*"), must be pinned to exactly one CPU, and
// those CPUs, sorted, must equal `expected` sorted. Retried for up to 2 s (exiting threads); throws otherwise.
void verify_threads(const std::set<int>& before, std::vector<int> expected);
// Formats sorted CPUs as "16,17,52".
std::string cpu_list(std::vector<int> cpus);

}  // namespace fullstack
