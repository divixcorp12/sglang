#pragma once
#include <chrono>
#include <cstdint>

namespace sglang::cpu_experts {

// CLOCK_MONOTONIC in ns, the clock of every deadline here (libstdc++'s steady_clock reads it).
inline int64_t team_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace sglang::cpu_experts
