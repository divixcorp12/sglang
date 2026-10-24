// StageGate::wait, declared in kernel.hpp; defined here so that header stays free of <immintrin.h>.
#pragma once
#include "clock.hpp"
#include "kernel.hpp"
#include <immintrin.h>

namespace sglang::cpu_experts {

inline void StageGate::wait() noexcept
{
    ++calls;
    if (timed) called_ns = team_now_ns();
    if (word->load(std::memory_order_acquire) == open) return;
    spun = true;
    while (word->load(std::memory_order_acquire) != open && !(stop && stop->load(std::memory_order_relaxed)))
        _mm_pause();
    if (timed) waited_ns = team_now_ns() - called_ns;
}

}  // namespace sglang::cpu_experts
