// Core lists and pinning for the CPU experts' threads.
#pragma once
#include <pthread.h>
#include <sched.h>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace sglang::cpu_experts {

// Throws std::invalid_argument unless every core is in [0, CPU_SETSIZE) and none repeats. Not checked against the
// caller's affinity: the owner may run under a mask that excludes the expert cores, and the workers pin themselves
// outside it; a core that cannot be pinned fails the Team's construction (std::runtime_error).
inline void check_cores(std::span<const int> cores)
{
    for (size_t i = 0; i < cores.size(); ++i) {
        if (cores[i] < 0 || cores[i] >= CPU_SETSIZE)
            throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " is outside [0, CPU_SETSIZE)");
        for (size_t j = 0; j < i; ++j)
            if (cores[j] == cores[i])
                throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " repeats");
    }
}

// Pins the calling thread to `core`; false when it cannot.
inline bool pin_thread(int core)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

}  // namespace sglang::cpu_experts
