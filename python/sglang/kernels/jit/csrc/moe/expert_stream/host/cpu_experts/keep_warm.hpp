// Holds a forward's team between calls: register-only work at the forward's vector width (the GEMV inner loop's
// vpmaddwd/vpaddd chains), so a core keeps the license the forward runs at, then PAUSE once the warm window ends, until
// the release time. Polls the word every 16 iterations and the clock every 1024; the window and the release are the
// engine's CLOCK_MONOTONIC, which libstdc++'s steady_clock reads.
#pragma once
#include "isa.hpp"
#include "team.hpp"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {
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

inline bool done_either(const uint32_t* a, uint32_t seen_a, const uint32_t* b, uint32_t seen_b, int64_t deadline_ns,
                        uint32_t tick)
{
    if (__atomic_load_n(a, __ATOMIC_ACQUIRE) != seen_a || __atomic_load_n(b, __ATOMIC_ACQUIRE) != seen_b) return true;
    return (tick & 63) == 0
           && std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                      .count()
                  >= deadline_ns;
}

SGLANG_TARGET_BW __attribute__((noinline)) inline int32_t bw_either(const uint32_t* a, uint32_t seen_a, const uint32_t* b,
                                                                     uint32_t seen_b, int64_t deadline_ns)
{
    __m512i x = _mm512_set1_epi16(3), y = _mm512_set1_epi16(5), c0 = _mm512_setzero_si512(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !done_either(a, seen_a, b, seen_b, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm512_add_epi32(c0, _mm512_madd_epi16(x, y));
            c1 = _mm512_add_epi32(c1, _mm512_madd_epi16(y, x));
            c2 = _mm512_add_epi32(c2, _mm512_madd_epi16(x, x));
            c3 = _mm512_add_epi32(c3, _mm512_madd_epi16(y, y));
            x = _mm512_xor_si512(x, c3);
        }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, x)));
}

SGLANG_TARGET_AVX2 __attribute__((noinline)) inline int32_t avx2_either(const uint32_t* a, uint32_t seen_a,
                                                                         const uint32_t* b, uint32_t seen_b,
                                                                         int64_t deadline_ns)
{
    __m256i x = _mm256_set1_epi16(3), y = _mm256_set1_epi16(5), c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !done_either(a, seen_a, b, seen_b, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(x, y));
            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(y, x));
            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(x, x));
            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(y, y));
            x = _mm256_xor_si256(x, c3);
        }
    const __m256i t = _mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, x));
    return int32_t(uint32_t(_mm256_extract_epi32(t, 0)) + uint32_t(_mm256_extract_epi32(t, 7)));
}

inline int32_t scalar_either(const uint32_t* a, uint32_t seen_a, const uint32_t* b, uint32_t seen_b, int64_t deadline_ns)
{
    for (uint32_t tick = 0; !done_either(a, seen_a, b, seen_b, deadline_ns, tick); ++tick) _mm_pause();
    return 0;
}

// Keeps the loops' results observable so the compiler cannot drop the register work.
inline std::atomic<int32_t> sink{0};

}  // namespace keep_warm_detail

// The loop for tier `isa`, clamped to Top: a quant compiles no vector code above its top tier, so a Scalar quant's
// portable build carries no AVX instructions from here.
template <Isa Top>
int32_t keep_warm_loop(Isa isa, const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    if constexpr (Top >= Isa::Bw)
        if (isa >= Isa::Bw) return keep_warm_detail::bw(word, seen, deadline_ns);
    if constexpr (Top >= Isa::Avx2)
        if (isa >= Isa::Avx2) return keep_warm_detail::avx2(word, seen, deadline_ns);
    return keep_warm_detail::scalar(word, seen, deadline_ns);
}

// Holds `threads` workers (the caller as worker 0, each pinned to `cores` as the forward pins them; empty: unpinned)
// until *word != seen or CLOCK_MONOTONIC reaches release_ns: register-only work of tier min(isa, Top) until
// warm_until_ns, then PAUSE. Throws std::invalid_argument for no worker, no word or more threads than `cores`,
// std::runtime_error for a failed pin.
template <Isa Top>
void keep_warm(Isa isa, std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
               int64_t warm_until_ns, int64_t release_ns)
{
    if (threads < 1 || word == nullptr || (!cores.empty() && size_t(threads) > cores.size()))
        throw std::invalid_argument("CPU expert keep-warm needs a worker, a word and no more workers than its cores");
    std::atomic<int> pin_error{0};
    #pragma omp parallel num_threads(threads) shared(cores, pin_error)
    {
        pin(omp_get_thread_num(), cores, pin_error);
        const int32_t warm = keep_warm_loop<Top>(isa, word, seen, std::min(warm_until_ns, release_ns));
        keep_warm_detail::sink.fetch_add(warm, std::memory_order_relaxed);
        keep_warm_detail::scalar(word, seen, release_ns);
    }
    if (pin_error.load(std::memory_order_relaxed)) throw std::runtime_error("cannot pin CPU expert worker to its core");
}

// keep_warm on two words: the same hold, ended by either word moving.
template <Isa Top>
void keep_warm_either(Isa isa, std::span<const int> cores, int32_t threads, const uint32_t* word_a, uint32_t seen_a,
                      const uint32_t* word_b, uint32_t seen_b, int64_t warm_until_ns, int64_t release_ns)
{
    if (threads < 1 || word_a == nullptr || word_b == nullptr || (!cores.empty() && size_t(threads) > cores.size()))
        throw std::invalid_argument("CPU expert keep-warm needs a worker, two words and no more workers than its cores");
    std::atomic<int> pin_error{0};
    #pragma omp parallel num_threads(threads) shared(cores, pin_error)
    {
        pin(omp_get_thread_num(), cores, pin_error);
        const int64_t warm_deadline = std::min(warm_until_ns, release_ns);
        int32_t warm = 0;
        if constexpr (Top >= Isa::Bw) {
            if (isa >= Isa::Bw) warm = keep_warm_detail::bw_either(word_a, seen_a, word_b, seen_b, warm_deadline);
            else if (isa >= Isa::Avx2) warm = keep_warm_detail::avx2_either(word_a, seen_a, word_b, seen_b, warm_deadline);
            else warm = keep_warm_detail::scalar_either(word_a, seen_a, word_b, seen_b, warm_deadline);
        } else if constexpr (Top >= Isa::Avx2) {
            if (isa >= Isa::Avx2) warm = keep_warm_detail::avx2_either(word_a, seen_a, word_b, seen_b, warm_deadline);
            else warm = keep_warm_detail::scalar_either(word_a, seen_a, word_b, seen_b, warm_deadline);
        } else {
            warm = keep_warm_detail::scalar_either(word_a, seen_a, word_b, seen_b, warm_deadline);
        }
        keep_warm_detail::sink.fetch_add(warm, std::memory_order_relaxed);
        keep_warm_detail::scalar_either(word_a, seen_a, word_b, seen_b, release_ns);
    }
    if (pin_error.load(std::memory_order_relaxed)) throw std::runtime_error("cannot pin CPU expert worker to its core");
}

}  // namespace
}  // namespace sglang::cpu_experts
