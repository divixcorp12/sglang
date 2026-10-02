#include "placement.h"

#include "expert_stream/host/core_topology.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <pthread.h>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace fullstack {
namespace fs = std::filesystem;

int number(const std::string& text) {
  size_t end = 0;
  int n = 0;
  try {
    n = std::stoi(text, &end);
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid integer: " + text);
  }
  if (end != text.size() || n < 0) throw std::runtime_error("Invalid integer: " + text);
  return n;
}

std::vector<int32_t> parse_cpus(const std::string& text) {
  std::vector<int32_t> cores;
  std::stringstream list(text);
  std::string item;
  while (std::getline(list, item, ',')) {
    const auto dash = item.find('-');
    const int first = number(item.substr(0, dash));
    const int last = dash == std::string::npos ? first : number(item.substr(dash + 1));
    if (first > last || last >= CPU_SETSIZE) throw std::runtime_error("Invalid CPU range: " + item);
    for (int cpu = first; cpu <= last; ++cpu) {
      if (std::find(cores.begin(), cores.end(), cpu) != cores.end())
        throw std::runtime_error("Duplicate CPU: " + std::to_string(cpu));
      cores.push_back(cpu);
    }
  }
  if (cores.empty()) throw std::runtime_error("Empty CPU list");
  return cores;
}

Topology system_topology() {
  Topology t;
  t.node_of = [](int cpu) {
    for (int node = 0; node < 64; ++node) {
      if (fs::exists("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(node))) return node;
    }
    return -1;
  };
  t.siblings_of = [](int cpu) { return sglang::expert_stream::core_siblings(cpu); };
  CPU_ZERO(&t.allowed);
  if (sched_getaffinity(0, sizeof(t.allowed), &t.allowed) != 0)
    throw std::runtime_error("Cannot read the process's CPU affinity");
  return t;
}

void validate_placement(const Placement& p, const Topology& t, bool check_nodes) {
  if (p.workers.empty()) throw std::runtime_error("placement: no CPU expert cores");
  struct Role {
    const char* name;
    int cpu;
  };
  std::vector<Role> roles = {{"writer", p.writer}, {"service", p.service}, {"copy", p.copy}};
  for (int32_t cpu : p.workers)
    roles.push_back({"worker", cpu});
  std::set<int> seen;
  for (const Role& role : roles) {
    if (role.cpu < 0 || role.cpu >= CPU_SETSIZE)
      throw std::runtime_error(
          std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) + " is out of range");
    if (!seen.insert(role.cpu).second)
      throw std::runtime_error("placement: CPU " + std::to_string(role.cpu) + " has two roles");
    if (!CPU_ISSET(role.cpu, &t.allowed))
      throw std::runtime_error(
          std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) +
          " is outside the process's allowed CPUs");
  }
  // A busy-polling service never yields: no other role may share its physical core.
  for (int sibling : t.siblings_of(p.service)) {
    if (sibling != p.service && seen.contains(sibling))
      throw std::runtime_error(
          "placement: CPU " + std::to_string(sibling) + " shares the physical core of the service CPU " +
          std::to_string(p.service));
  }
  if (!check_nodes) return;
  for (const Role& role : roles) {
    const int want = std::string(role.name) == "worker" ? p.worker_node : p.host_node;
    const int node = t.node_of(role.cpu);
    if (node != want)
      throw std::runtime_error(
          std::string("placement: the ") + role.name + " CPU " + std::to_string(role.cpu) + " is on NUMA node " +
          std::to_string(node) + ", not node " + std::to_string(want));
  }
}

void pin_self(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0)
    throw std::runtime_error("Cannot pin the calling thread to CPU " + std::to_string(cpu));
}

PinScope::PinScope(int cpu) {
  CPU_ZERO(&saved_);
  if (sched_getaffinity(0, sizeof(saved_), &saved_) != 0) throw std::runtime_error("Cannot read the thread's affinity");
  pin_self(cpu);
}

PinScope::~PinScope() {
  sched_setaffinity(0, sizeof(saved_), &saved_);
}

std::set<int> task_ids() {
  std::set<int> result;
  for (const auto& entry : fs::directory_iterator("/proc/self/task"))
    result.insert(number(entry.path().filename().string()));
  return result;
}

std::vector<int> expected_threads(const Placement& p) {
  std::vector<int> cpus = {p.service, p.copy, p.copy};
  cpus.insert(cpus.end(), p.workers.begin(), p.workers.end());
  std::sort(cpus.begin(), cpus.end());
  return cpus;
}

std::string cpu_list(std::vector<int> cpus) {
  std::sort(cpus.begin(), cpus.end());
  std::string text;
  for (int cpu : cpus)
    text += (text.empty() ? "" : ",") + std::to_string(cpu);
  return text;
}

namespace {

// Throws unless every thread created since `before` is pinned to exactly one CPU and those CPUs equal `expected`.
void census(const std::set<int>& before, const std::vector<int>& expected) {
  std::vector<int> pinned;
  for (int tid : task_ids()) {
    if (before.contains(tid)) continue;
    std::ifstream comm_file("/proc/self/task/" + std::to_string(tid) + "/comm");
    std::string comm;
    std::getline(comm_file, comm);
    if (comm.starts_with("iou-")) continue;  // io_uring's kernel workers
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(tid, sizeof(mask), &mask) != 0 || CPU_COUNT(&mask) != 1)
      throw std::runtime_error("thread " + std::to_string(tid) + " (" + comm + ") is not pinned to one CPU");
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &mask)) pinned.push_back(cpu);
  }
  std::sort(pinned.begin(), pinned.end());
  if (pinned != expected)
    throw std::runtime_error("threads are pinned to {" + cpu_list(pinned) + "}, expected {" + cpu_list(expected) + "}");
}

}  // namespace

void verify_threads(const std::set<int>& before, std::vector<int> expected) {
  std::sort(expected.begin(), expected.end());
  // A released OpenMP team's helpers exit asynchronously: give them up to 2 s to leave /proc/self/task.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  for (;;) {
    try {
      census(before, expected);
      return;
    } catch (const std::exception&) {
      if (std::chrono::steady_clock::now() > deadline) throw;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace fullstack
