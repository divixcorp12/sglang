// Register-only work at a forward's vector width (the GEMV inner loop's vpmaddwd/vpaddd chains), so a core holds the
// license the forward runs at between calls. Polls the word every 16 iterations and the clock every 1024; the deadline
// is the engine's CLOCK_MONOTONIC, which libstdc++'s steady_clock reads.
#pragma once
#include "isa.hpp"
#include "team.hpp"
#include <immintrin.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace sglang::cpu_experts {
namespace keep_warm_detail {

inline bool done(const uint32_t* word, uint32_t seen, int64_t deadline_ns, uint32_t tick)
{
    if (__atomic_load_n(word, __ATOMIC_ACQUIRE) != seen) return true;
    return (tick & 63) == 0
           && std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                      .count()
                  >= deadline_ns;
}

SGLANG_TARGET_BW __attribute__((noinline)) inline int32_t bw(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    __m512i a = _mm512_set1_epi16(3), b = _mm512_set1_epi16(5), c0 = _mm512_setzero_si512(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !done(word, seen, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm512_add_epi32(c0, _mm512_madd_epi16(a, b));
            c1 = _mm512_add_epi32(c1, _mm512_madd_epi16(b, a));
            c2 = _mm512_add_epi32(c2, _mm512_madd_epi16(a, a));
            c3 = _mm512_add_epi32(c3, _mm512_madd_epi16(b, b));
            a = _mm512_xor_si512(a, c3);
        }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, a)));
}

SGLANG_TARGET_AVX2 __attribute__((noinline)) inline int32_t avx2(const uint32_t* word, uint32_t seen,
                                                                  int64_t deadline_ns)
{
    __m256i a = _mm256_set1_epi16(3), b = _mm256_set1_epi16(5), c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !done(word, seen, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a, b));
            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(b, a));
            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a, a));
            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(b, b));
            a = _mm256_xor_si256(a, c3);
        }
    const __m256i t = _mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, a));
    // Unsigned: the lanes wrap, and a signed sum of two would overflow.
    return int32_t(uint32_t(_mm256_extract_epi32(t, 0)) + uint32_t(_mm256_extract_epi32(t, 7)));
}

inline int32_t scalar(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    for (uint32_t tick = 0; !done(word, seen, deadline_ns, tick); ++tick) _mm_pause();
    return 0;
}

// Keeps the loops' results observable so the compiler cannot drop the register work.
inline std::atomic<int32_t> sink{0};

}  // namespace keep_warm_detail

// Holds `threads` workers (the caller as worker 0, each pinned as the forward pins it) in register-only work of tier
// `isa` until *word != seen or CLOCK_MONOTONIC reaches deadline_ns. Returns 0, 1 on a kernel error (a failed pin), 2
// on invalid arguments or more threads than the configured cores.
inline int keep_warm(Isa isa, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept
{
    if (threads < 1 || word == nullptr) return 2;
    try {
        Cores::freeze();
        if (!Cores::frozen().empty() && size_t(threads) > Cores::frozen().size()) return 2;
        std::atomic<int> pin_error{0};
        #pragma omp parallel num_threads(threads) shared(pin_error)
        {
            Cores::pin(omp_get_thread_num(), pin_error);
            const int32_t r = isa == Isa::Scalar ? keep_warm_detail::scalar(word, seen, deadline_ns)
                              : isa == Isa::Avx2 ? keep_warm_detail::avx2(word, seen, deadline_ns)
                                                 : keep_warm_detail::bw(word, seen, deadline_ns);
            keep_warm_detail::sink.fetch_add(r, std::memory_order_relaxed);
        }
        return pin_error.load(std::memory_order_relaxed) ? 1 : 0;
    } catch (...) {
        return 1;
    }
}

}  // namespace sglang::cpu_experts
