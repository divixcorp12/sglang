// The physical-core check a busy-polling service thread needs (RamThread, busy_poll).
#pragma once

#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixed_vec.h"

namespace sglang::expert_stream {

// The SMT siblings of `core`, the core included, from sysfs ("17,53" or "16-17").
inline std::vector<int> core_siblings(int core) {
  std::ifstream in("/sys/devices/system/cpu/cpu" + std::to_string(core) + "/topology/thread_siblings_list");
  std::string list;
  if (!std::getline(in, list)) throw std::runtime_error("cannot read the SMT siblings of core " + std::to_string(core));
  std::vector<int> cores;
  std::stringstream items(list);
  for (std::string item; std::getline(items, item, ',');) {
    const size_t dash = item.find('-');
    const int first = std::stoi(item.substr(0, dash));
    const int last = dash == std::string::npos ? first : std::stoi(item.substr(dash + 1));
    for (int c = first; c <= last; ++c)
      cores.push_back(c);
  }
  return cores;
}

// A busy-polling service never yields its core, so it needs the physical core to itself: no SMT sibling of `core`
// (the core included) may be in the caller's affinity, which the process's other threads inherit, or among the CPU
// experts' cores, which their threads pin themselves to.
inline void check_dedicated_core(int core, const std::vector<int>& cpu_expert_cores, const std::string& prefix) {
  if (core < 0) {
    throw std::runtime_error(prefix + "busy_poll needs cpu_core: a busy-polling service is pinned to a core of its own");
  }
  cpu_set_t caller;
  CPU_ZERO(&caller);
  if (pthread_getaffinity_np(pthread_self(), sizeof(caller), &caller) != 0)
    throw std::runtime_error(prefix + "busy_poll: cannot read the caller's affinity");
  for (const int sibling : core_siblings(core)) {
    if (CPU_ISSET(sibling, &caller)) {
      throw std::runtime_error(
          prefix + "busy_poll: core " + std::to_string(sibling) + " shares the physical core of cpu_core " +
          std::to_string(core) + " and is in the caller's affinity, which the process's other threads inherit");
    }
    if (listed(cpu_expert_cores, sibling)) {
      throw std::runtime_error(
          prefix + "busy_poll: core " + std::to_string(sibling) + " shares the physical core of cpu_core " +
          std::to_string(core) + " and is a CPU expert core (SGLANG_DSV41_CPU_EXPERTS_CORES)");
    }
  }
}

}  // namespace sglang::expert_stream
