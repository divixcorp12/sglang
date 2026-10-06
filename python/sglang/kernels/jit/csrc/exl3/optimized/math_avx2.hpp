// The AVX2 tier's GEMV tiles (avx2_tiles), compiled for AVX2/FMA/F16C by attribute (M1_TARGET_AVX2).
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace sglang::exl3_cpu {
namespace {

// -------------------------------------------------------------------------------------------
//   AVX2
// -------------------------------------------------------------------------------------------

// maddubs saturates its i16 pair sums (2 * 255 * 127 > 32767), so the product bytes are split
// even/odd; each masked pair then holds a single u8 x s8 product

// State decode: AVX2 has no cross-lane permute wider than one 256-bit (8-lane) register, unlike
// AVX-512's vpermt2var (16-wide, spanning 2 registers = 32 lanes). A row's 8-column half can draw
// from any of the `bits` registers a k-tile occupies (packed_size = 16*bits u16 = bits registers
// of 8 u32 each), so instead of the VNNI path's fixed two-register-pair span, this walks every
// candidate register and blends in only the ones that actually contribute for a given (bits,
// row, half) -- resolved entirely at compile time via row/half/word-selector being template
// parameters, exactly mirroring how VNNI's hi/lo masks are compile-time per row. Requires the
// row loop itself to be compile-time-unrolled (below), not the runtime loop the gather-based
// first cut of this used: a runtime row made the blend masks runtime values too, which needs a
// variable blend (or a gather) instead of a free compile-time-immediate blend, and benchmarking
// showed AVX2 gather is not a win on this hardware (a modest 1.7x over the scalar-decode
// baseline, vs the several-x this register-permute version gets).

template <int bits, int row, bool second_word, int half, int Reg>
constexpr uint8_t avx2_reg_mask()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    uint8_t mask = 0;
    for (int i = 0; i < 8; ++i)
        if (idx16[half * 8 + i] / 8 == Reg) mask |= uint8_t(1) << i;
    return mask;
}

template <int bits, int row, bool second_word, int half>
constexpr std::array<int32_t, 8> avx2_lane_idx()
{
    constexpr auto idx16 = make_row_indices<bits, row, second_word>();
    std::array<int32_t, 8> out{};
    for (int i = 0; i < 8; ++i) out[i] = idx16[half * 8 + i] % 8;
    return out;
}

// Permutes+blends together only the registers that actually contribute a lane to this half, in
// increasing Reg order (skipped candidates cost nothing -- if constexpr eliminates them, so low
// bitrates collapse to a single unconditional permute, same as VNNI's cheapest case)
template <int bits, int row, bool second_word, int half, int Reg = 0>
M1_TARGET_AVX2
inline __m256i avx2_gather_half(const __m256i (&preg)[bits])
{
    // All-compile-time-constant arguments: the compiler folds this to a single constant load,
    // same as a hand-written lookup table. No lambda (this function is target-attributed, and
    // GCC does not propagate the target to a lambda's closure -- see file header note).
    constexpr auto li = avx2_lane_idx<bits, row, second_word, half>();
    const __m256i lane_idx_v = _mm256_setr_epi32(li[0], li[1], li[2], li[3], li[4], li[5], li[6], li[7]);
    if constexpr (Reg + 1 >= bits)
    {
        // last candidate: every column not already claimed must come from here
        return _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
    }
    else
    {
        constexpr uint8_t mask = avx2_reg_mask<bits, row, second_word, half, Reg>();
        if constexpr (mask == 0)
        {
            return avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
        }
        else
        {
            const __m256i cur = _mm256_permutevar8x32_epi32(preg[Reg], lane_idx_v);
            const __m256i rest = avx2_gather_half<bits, row, second_word, half, Reg + 1>(preg);
            return _mm256_blend_epi32(rest, cur, mask);
        }
    }
}

// Decodes row `row`'s 16 states into two 8-wide registers (cols 0-7, cols 8-15) from the k-tile's
// pre-loaded registers; no scalar decode_state_scalar calls. b0/b1 and the funnel shift follow
// the same bit layout as decode_state_scalar; the shift is shared across each 8-column half (by
// construction of the tile's tensor-core permutation, the same invariant the VNNI path relies on
// for its single per-half s0/s1).
template <int bits, int row>
M1_TARGET_AVX2
inline void avx2_row_codes(const __m256i (&preg)[bits], __m256i& codes_lo, __m256i& codes_hi)
{
    const __m256i a_lo = avx2_gather_half<bits, row, false, 0>(preg);
    const __m256i b_lo = avx2_gather_half<bits, row, true, 0>(preg);
    const __m256i a_hi = avx2_gather_half<bits, row, false, 1>(preg);
    const __m256i b_hi = avx2_gather_half<bits, row, true, 1>(preg);
    constexpr int s0 = row_shift<bits, row>(0);
    constexpr int s1 = row_shift<bits, row>(8);
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    codes_lo = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
    codes_hi = _mm256_and_si256(_mm256_or_si256(
        _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
}

M1_TARGET_AVX2
inline void avx2_accum_row(__m256i codes_lo, __m256i codes_hi, const int32_t* splat_dup, int k,
    int m, __m256i (&acc)[MAX_M][2], const __m256i& mult, const __m256i& ones32, int row)
{
    // Bytesum-first accumulate: maddubs(prod, 0x01010101) sums each product-byte pair into an
    // i16 lane ((b0+b1), (b2+b3), <= 510). x is OUTSIDE the pair so vpmaddubsw cannot saturate
    // at full +-127 activations; one vpmaddwd per token row against splat_dup (x8 in both 16-bit
    // slots) then folds (b0+b1)*x+(b2+b3)*x into a single i32. 4 shared + 4 per-row ops,
    // bit-exact vs the 16-op masked accumulate it replaces (verified K1-K8 x m1-4 against an
    // exact scalar reference). The token loop is unrolled by hand: with a runtime-bounded loop
    // GCC spills the pair sums and pays per-iteration overhead (~1.4x on Zen 3).
    const __m256i p_lo = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_lo, mult), ones32);
    const __m256i p_hi = _mm256_maddubs_epi16(_mm256_mullo_epi32(codes_hi, mult), ones32);
    // Rows 0..7 cover the largest MAX_M (8); rows >= MAX_M are constant-false and index row 0 so the dead code is
    // still in bounds.
    #define ACC_ROW(i) \
        if ((i) < MAX_M && (i) < m) { \
            constexpr int r = (i) < MAX_M ? (i) : 0; \
            const __m256i xs = _mm256_set1_epi32(splat_dup[static_cast<size_t>(i) * k + row]); \
            acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(p_lo, xs)); \
            acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(p_hi, xs)); \
        }
    ACC_ROW(0) ACC_ROW(1) ACC_ROW(2) ACC_ROW(3) ACC_ROW(4) ACC_ROW(5) ACC_ROW(6) ACC_ROW(7)
    #undef ACC_ROW
}

// Word-level row pairing on AVX2 is gated to bits == 8 ONLY: measured +11% there (each gather
// walks all 8 candidate registers, so halving gathers wins even though the 4 shared word
// registers staying live across the even row's accumulate spills -- AVX2 has 16 architectural
// ymm registers). At every other K the same restructure measured neutral-to-negative
// (K3 -5% 1T / -14% 24T-cold on the 7960X); do not widen the gate without re-measuring.
template <int bits, int row = 0>
M1_TARGET_AVX2
inline void avx2_rows_accum(
    const __m256i (&preg)[bits], const int32_t* splat_dup, int k, int m, __m256i (&acc)[MAX_M][2],
    const __m256i& mult, const __m256i& ones32)
{
    if constexpr (bits == 8)
    {
        if constexpr (row < 16)
        {
            static_assert(word_pair_ok<bits, row>(), "K8 pairs are fully eligible by layout");
            const __m256i a_lo = avx2_gather_half<bits, row, false, 0>(preg);
            const __m256i b_lo = avx2_gather_half<bits, row, true, 0>(preg);
            const __m256i a_hi = avx2_gather_half<bits, row, false, 1>(preg);
            const __m256i b_hi = avx2_gather_half<bits, row, true, 1>(preg);
            constexpr int s0 = row_shift<bits, row>(0);
            constexpr int s1 = row_shift<bits, row>(8);
            const __m256i mask16 = _mm256_set1_epi32(0xffff);
            __m256i codes_lo = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_lo, s0), _mm256_slli_epi32(a_lo, 32 - s0)), mask16);
            __m256i codes_hi = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_hi, s1), _mm256_slli_epi32(a_hi, 32 - s1)), mask16);
            avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row);
            // Odd row: same gathered words, shifted by an extra `bits` (>= 32 shifts are
            // well-defined zero, so the slli term drops out cleanly when s - bits == 0)
            codes_lo = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_lo, s0 - bits), _mm256_slli_epi32(a_lo, 32 - (s0 - bits))), mask16);
            codes_hi = _mm256_and_si256(_mm256_or_si256(
                _mm256_srli_epi32(b_hi, s1 - bits), _mm256_slli_epi32(a_hi, 32 - (s1 - bits))), mask16);
            avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row + 1);
            avx2_rows_accum<bits, row + 2>(preg, splat_dup, k, m, acc, mult, ones32);
        }
    }
    else if constexpr (row < 16)
    {
        __m256i codes_lo, codes_hi;
        avx2_row_codes<bits, row>(preg, codes_lo, codes_hi);
        avx2_accum_row(codes_lo, codes_hi, splat_dup, k, m, acc, mult, ones32, row);
        avx2_rows_accum<bits, row + 1>(preg, splat_dup, k, m, acc, mult, ones32);
    }
}

template <int bits>
M1_TARGET_AVX2
void avx2_tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(MUL1_MULT));
    const __m256i ones32 = _mm256_set1_epi32(0x01010101);
    const int32_t* splat_dup = in.splat_dup;

    // The k-major stream strides row_stride (>= 8 KB) per step, beyond what the HW prefetcher
    // tracks (and the swizzled layout the VNNI path uses is not applied for AVX2). Cold-stack
    // (offloaded expert) microbench: +20..70% at K>=4, largest at K8, warm-neutral, so always
    // on. Distance 4 measured best cold (>= 2 everywhere within noise, 4 adds another +10..35%
    // at K5-K8 cold); prefetching past the allocation end is architecturally safe.
    constexpr int pf_lines = (32 * bits + 63) / 64;   // cache lines per tile row
    // bits==6 (96B rows) collapses at distance 4 when cold (reproducibly ~2x slower on both
    // the 7960X and this Zen5 box; the 3-line window from 4 rows out interacts badly with the
    // 96B stride). Distance 2 measures >= everywhere else for K6 while costing <2% warm.
    constexpr int pf_dist = (bits == 6) ? 2 : 4;

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        __m256i acc[MAX_M][2];
        for (int i = 0; i < m; ++i)
        {
            acc[i][0] = _mm256_setzero_si256();
            acc[i][1] = _mm256_setzero_si256();
        }

        const uint16_t* packed = mat.trellis + static_cast<size_t>(tile_n) * packed_size;
        const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed += row_stride)
        {
            const uint16_t* pf = packed + row_stride * pf_dist;
            #pragma unroll
            for (int l = 0; l < pf_lines; ++l)
                _mm_prefetch(reinterpret_cast<const char*>(pf) + l * 64, _MM_HINT_T0);

            const int32_t* splat_k = splat_dup + tile_k * 16;
            // One 256-bit (8xu32) register per bits: covers packed_size = 16*bits u16 = bits*8
            // u32 words exactly, the whole k-tile's row of packed states
            __m256i preg[bits];
            for (int i = 0; i < bits; ++i)
                preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
            avx2_rows_accum<bits>(preg, splat_k, mat.k, m, acc, mult, ones32);
        }

        for (int i = 0; i < m; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m256 corr = _mm256_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            float* out = tout + static_cast<size_t>(i) * mat.n + tile_n * 16;
            _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][0]), _mm256_set1_ps(scale), corr));
            _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][1]), _mm256_set1_ps(scale), corr));
        }
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
