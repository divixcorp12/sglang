// The AVX2 tier's GEMV tiles (Tiles<Isa::Avx2>), present in builds whose -march has AVX2, FMA and F16C.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#if defined(SGLANG_CPU_EXPERTS_AVX2)
#include <immintrin.h>
#include <array>
#include <cstddef>
#include <cstdint>

namespace sglang::exl3_cpu {
namespace {

// State decode: AVX2 has no cross-lane permute wider than one 256-bit (8-lane) register, unlike AVX-512's vpermt2var.
// A row's 8-column half can draw from any of the `bits` registers a k-tile occupies, so each half permutes every
// register that contributes a lane to it and blends them, all selected at compile time per (bits, row, half). The row
// loop must be compile-time for the blend masks to be immediates; a gather measured only 1.7x over scalar decode.
template <>
struct Tiles<Isa::Avx2>
{
    using Acc = __m256i[GENERIC_MAX_M][2];

    static void run(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
    {
        with_count<8>(mat.bits, [&]<int bits>() { tiles<bits>(mat, in, tout, m, tn0, tn1); });
    }

    // Which lanes of half `half` of row `row`'s gather come from register Reg.
    template <int bits, int row, bool second_word, int half, int Reg>
    static constexpr uint8_t reg_mask()
    {
        constexpr auto idx16 = make_row_indices<bits, row, second_word>();
        uint8_t mask = 0;
        for (int i = 0; i < 8; ++i)
            if (idx16[half * 8 + i] / 8 == Reg) mask |= uint8_t(1) << i;
        return mask;
    }

    template <int bits, int row, bool second_word, int half>
    static constexpr std::array<int32_t, 8> lane_idx()
    {
        constexpr auto idx16 = make_row_indices<bits, row, second_word>();
        std::array<int32_t, 8> out{};
        for (int i = 0; i < 8; ++i) out[i] = idx16[half * 8 + i] % 8;
        return out;
    }

    // The words of one 8-column half: the last register's permute, then each lower register that contributes a lane
    // blended over it (a register with no lane costs nothing; low bit widths collapse to one permute).
    template <int bits, int row, bool second_word, int half>
    EXL3_ALWAYS_INLINE static __m256i gather_half(const __m256i (&preg)[bits])
    {
        constexpr auto li = lane_idx<bits, row, second_word, half>();
        const __m256i lane_idx_v = _mm256_setr_epi32(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7]);
        __m256i out = _mm256_permutevar8x32_epi32(preg[bits - 1], lane_idx_v);
        unroll<bits - 1>([&]<int I>() EXL3_ALWAYS_INLINE {
            constexpr int Reg = bits - 2 - I;
            constexpr uint8_t mask = reg_mask<bits, row, second_word, half, Reg>();
            if constexpr (mask != 0) out = _mm256_blend_epi32(out, _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v), mask);
        });
        return out;
    }

    // Shift-merge one half's codes from its two gathered word vectors, `delta` bits further down (row pairing).
    template <int s>
    EXL3_ALWAYS_INLINE static __m256i merge(__m256i a, __m256i b)
    {
        return _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi32(b, s), _mm256_slli_epi32(a, 32 - s)),
                                _mm256_set1_epi32(0xffff));
    }

    // Bytesum-first accumulate of one k-row: maddubs(prod, 0x01010101) sums each product-byte pair into an i16 lane
    // (<= 510), so vpmaddubsw cannot saturate at full +-127 activations, and one vpmaddwd per token row against
    // splat_dup folds both pairs into one i32. The token loop is compile-time: with a run-time bound GCC spills the
    // pair sums (~1.4x on Zen 3).
    EXL3_ALWAYS_INLINE static void accumulate_row(__m256i codes_lo, __m256i codes_hi, const int32_t* splat_dup, int k,
                                                  int m, Acc& acc, int row)
    {
        const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(kMul1Mult));
        const __m256i ones32 = _mm256_set1_epi32(0x01010101);
        const __m256i p_lo = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_lo, mult), ones32);
        const __m256i p_hi = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_hi, mult), ones32);
        unroll<GENERIC_MAX_M>([&]<int I>() EXL3_ALWAYS_INLINE {
            if (I < m) {
                const __m256i xs = _mm256_set1_epi32(splat_dup[size_t(I) * k + row]);
                acc[I][0] = _mm256_add_epi32(acc[I][0], _mm256_madd_epi16(p_lo, xs));
                acc[I][1] = _mm256_add_epi32(acc[I][1], _mm256_madd_epi16(p_hi, xs));
            }
        });
    }

    // The k-tile's 16 rows. Row pairing (word_pair_ok) only at bits == 8: measured +11% there, neutral-to-negative at
    // every other K (K3 -5% 1T / -14% 24T-cold on the 7960X), as the 4 shared word registers spill on 16 ymm.
    template <int bits>
    EXL3_ALWAYS_INLINE static void accumulate_tile(const __m256i (&preg)[bits], const int32_t* splat_dup, int k, int m,
                                                   Acc& acc)
    {
        if constexpr (bits == 8) {
            unroll<8>([&]<int P>() EXL3_ALWAYS_INLINE {
                constexpr int row = 2 * P;
                static_assert(word_pair_ok<bits, row>(), "K8 pairs are fully eligible by layout");
                const __m256i a_lo = gather_half<bits, row, false, 0>(preg);
                const __m256i b_lo = gather_half<bits, row, true, 0>(preg);
                const __m256i a_hi = gather_half<bits, row, false, 1>(preg);
                const __m256i b_hi = gather_half<bits, row, true, 1>(preg);
                constexpr int s0 = row_shift<bits, row>(0);
                constexpr int s1 = row_shift<bits, row>(8);
                accumulate_row(merge<s0>(a_lo, b_lo), merge<s1>(a_hi, b_hi), splat_dup, k, m, acc, row);
                // The odd row: the same words, shifted by an extra `bits` (a shift by >= 32 is a well-defined zero).
                accumulate_row(merge<s0 - bits>(a_lo, b_lo), merge<s1 - bits>(a_hi, b_hi), splat_dup, k, m, acc, row + 1);
            });
        } else {
            unroll<16>([&]<int row>() EXL3_ALWAYS_INLINE {
                const __m256i a_lo = gather_half<bits, row, false, 0>(preg);
                const __m256i b_lo = gather_half<bits, row, true, 0>(preg);
                const __m256i a_hi = gather_half<bits, row, false, 1>(preg);
                const __m256i b_hi = gather_half<bits, row, true, 1>(preg);
                constexpr int s0 = row_shift<bits, row>(0);
                constexpr int s1 = row_shift<bits, row>(8);
                accumulate_row(merge<s0>(a_lo, b_lo), merge<s1>(a_hi, b_hi), splat_dup, k, m, acc, row);
            });
        }
    }

    template <int bits>
    static void tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
    {
        const int tiles_k = mat.k / 16;
        const int tiles_n = mat.n / 16;
        constexpr int packed_size = 16 * bits;
        // The k-major stream strides row_stride (>= 8 KB) per step, beyond what the HW prefetcher tracks. Cold-stack
        // microbench: +20..70% at K>=4, warm-neutral; distance 4 measured best cold. bits == 6 (96 B rows) collapses at
        // distance 4 when cold (~2x slower on the 7960X and Zen 5), so it uses 2. Prefetching past the end is safe.
        constexpr int pf_lines = (32 * bits + 63) / 64;
        constexpr int pf_dist = (bits == 6) ? 2 : 4;

        for (int tile_n = tn0; tile_n < tn1; ++tile_n) {
            Acc acc;
            for (int i = 0; i < m; ++i) {
                acc[i][0] = _mm256_setzero_si256();
                acc[i][1] = _mm256_setzero_si256();
            }
            const uint16_t* packed = mat.trellis + size_t(tile_n) * packed_size;
            const size_t row_stride = size_t(tiles_n) * packed_size;
            for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed += row_stride) {
                const uint16_t* pf = packed + row_stride * pf_dist;
                for (int l = 0; l < pf_lines; ++l) _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);
                // One register per bit: the k-tile's 16 * bits u16 states are bits * 8 u32 words.
                __m256i preg[bits];
                for (int i = 0; i < bits; ++i) preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
                accumulate_tile<bits>(preg, in.splat_dup + tile_k * 16, mat.k, m, acc);
            }
            for (int i = 0; i < m; ++i) {
                const float scale = mul1_k_inv() * in.q[i];
                const __m256 corr = _mm256_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
                float* out = tout + size_t(i) * mat.n + tile_n * 16;
                _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][0]), _mm256_set1_ps(scale), corr));
                _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][1]), _mm256_set1_ps(scale), corr));
            }
        }
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
#endif  // SGLANG_CPU_EXPERTS_AVX2
