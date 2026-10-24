// The 3-bit register kernels (register_tiles<M>): the DSV4.1 plan's GEMVs (compact int16 activations, chunks of
// 1..CHUNK_M tokens, all M tokens sharing each decoded tile) and, on the Bw tier, the generic plan's one-token 3-bit
// chunks. They read the original 96-byte tiles two at a time (a tile pair: 32 output columns), transposing each column
// pair's three trellis words into a, b, c and decoding the 16 states per lane in registers (register_state).
//
// Every output keeps the generic path's rounding: the int32 sum of each (128-input block, quantized row), one FMA scale
// each, fp32 sums in increasing block order, then token + residual once.
//
// The integer accumulate is the build's: vpdpwssd on a VNNI tier, else vpmaddwd + vpaddd (both wrap mod 2^32, so the
// tiers are bit-identical); on VNNI, one-token compact chunks dot the product bytes with vpdpbusd and skip the byte sum.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math_avx512.hpp"
#if defined(SGLANG_CPU_EXPERTS_AVX512)
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>

namespace sglang::exl3_cpu {
namespace {

#if defined(SGLANG_CPU_EXPERTS_VNNI)
constexpr bool kVnniAccumulate = true;
#else
constexpr bool kVnniAccumulate = false;
#endif

// acc + sum . pair: the pairwise 16-bit dot of a byte-sum vector with a broadcast activation pair.
EXL3_ALWAYS_INLINE inline __m512i dot_pairs(__m512i acc, __m512i sum, __m512i pair)
{
#if defined(SGLANG_CPU_EXPERTS_VNNI)
    return _mm512_dpwssd_epi32(acc, sum, pair);
#else
    return _mm512_add_epi32(acc, _mm512_madd_epi16(sum, pair));
#endif
}

// The mul1 byte sum of 32 16-bit states: the four bytes of state * kMul1Mult, from its low and high 16-bit halves,
// summed into [0, 1020].
EXL3_ALWAYS_INLINE inline __m512i register_bytesum(__m512i state)
{
    const __m512i ml = _mm512_set1_epi16(int16_t(kMul1Mult & 0xffff));
    const __m512i mh = _mm512_set1_epi16(int16_t(kMul1Mult >> 16));
    const __m512i lo = _mm512_mullo_epi16(state, ml);
    const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml), _mm512_mullo_epi16(state, mh));
    const __m512i ones = _mm512_set1_epi8(1);
    return _mm512_add_epi16(_mm512_maddubs_epi16(lo, ones), _mm512_maddubs_epi16(hi, ones));
}

// -------------------------------------------------------------------------------------------
//   Register budget
// -------------------------------------------------------------------------------------------

// The register budget, one entry per tokens-per-chunk M = 1..CHUNK_M (register_tiles<M>): pairs, the tile pairs one
// register_band call holds; traversal, whether whole 128-output groups take the grouped traversal (IntegerAccum holds
// one token's rows); halves, whether each 128-input block runs once per output half (register_band_halves), holding half
// the accumulators. Raising EXL3_MOE_CPU_MAX_M past twice the entries fails to compile below until one is added.
struct RegisterBudget
{
    int pairs;
    bool traversal;
    bool halves = false;
};
constexpr RegisterBudget kRegisterBudget[] = {
    {4, true},         // M = 1 (register_band<1..4>, traversal_kblock's four bands)
    {2, false},        // M = 2
    {1, false},        // M = 3
    {1, false},        // M = 4
    {1, false, true},  // M = 5..8: 4 * M accumulators per tile pair exceed the registers; 2 * M per half fit
    {1, false, true},
    {1, false, true},
    {1, false, true},
};
// zmm registers register_rows keeps live besides the integer accumulators: prev, a, b, c, the state, its byte sum, two
// products, the multiplier halves, the ones vector and the broadcast activation pair.
constexpr int kRegisterDecodeZmm = 12;

// Integer accumulators one register_band call holds for M tokens: two output halves of 2 * M rows per tile pair, or
// one half at a time.
constexpr int register_accumulators(int m, const RegisterBudget& b) { return (b.halves ? 2 : 4) * m * b.pairs; }

constexpr bool register_budget_fits()
{
    for (int m = 1; m <= int(std::size(kRegisterBudget)); ++m)
        if (kRegisterBudget[m - 1].pairs < 1
            || register_accumulators(m, kRegisterBudget[m - 1]) + kRegisterDecodeZmm > 32)
            return false;
    return true;
}
static_assert(register_budget_fits(), "kRegisterBudget: an entry's integer accumulators (register_accumulators) plus "
                                      "kRegisterDecodeZmm exceed the 32 zmm registers");
static_assert(CHUNK_M <= int(std::size(kRegisterBudget)),
              "kRegisterBudget has no entry for this CHUNK_M: add one before raising EXL3_MOE_CPU_MAX_M");

// -------------------------------------------------------------------------------------------
//   Tile pairs
// -------------------------------------------------------------------------------------------

// Each column pair's 32 three-bit codes occupy three consecutive uint32 words of the two tiles: transpose those
// triplets across sixteen lanes (eight per output tile). Word Q of lane l is word 3l + Q.
template <int Q>
constexpr std::array<int32_t, 16> register_word_indices()
{
    std::array<int32_t, 16> indices{};
    for (int lane = 0; lane < 16; ++lane) indices[lane] = lane * 3 + Q;
    return indices;
}

template <int Q>
EXL3_ALWAYS_INLINE inline __m512i register_word(__m512i p0, __m512i p1, __m512i p2)
{
    alignas(64) static constexpr auto idx = register_word_indices<Q>();
    constexpr __mmask16 mask = Q == 2 ? 0xfc00 : 0xf800;
    const __m512i index = _mm512_load_si512(idx.data());
    const __m512i low = _mm512_permutex2var_epi32(p0, index, p1);
    // vpermd uses only the low four index bits: indices 32..47 address p2.
    return _mm512_mask_permutexvar_epi32(low, mask, index, p2);
}

// A tile pair's transposed words. prev is the preceding lane's last word, which supplies the circular trellis's
// initial state.
struct TilePair
{
    __m512i prev, a, b, c;
};

// Prefetches the tile pair two k-tiles ahead (step: the bytes between k-tiles), then loads and transposes this one.
EXL3_ALWAYS_INLINE inline TilePair load_tile_pair(const uint32_t* packed, size_t step, bool prefetch = true)
{
    alignas(64) static constexpr int previous[16] = {7, 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14};
    if (prefetch) {
        // The prefetch may point past the final tile: form its address as an integer, not an out-of-object pointer.
        const uintptr_t future = reinterpret_cast<uintptr_t>(packed) + step * 2;
#pragma GCC unroll 3
        for (int line = 0; line < 3; ++line) _mm_prefetch(reinterpret_cast<const char*>(future + line * 64), _MM_HINT_T0);
    }
    const __m512i p0 = _mm512_loadu_si512(packed), p1 = _mm512_loadu_si512(packed + 16), p2 = _mm512_loadu_si512(packed + 32);
    TilePair t;
    t.a = register_word<0>(p0, p1, p2);
    t.b = register_word<1>(p0, p1, p2);
    t.c = register_word<2>(p0, p1, p2);
    t.prev = _mm512_permutexvar_epi32(_mm512_load_si512(previous), t.c);
    return t;
}

// Tile (tile_k, nt) of a 3-bit matrix: 48 u16 per tile, swizzled or not.
EXL3_ALWAYS_INLINE inline const uint32_t* tile_words(const Exl3Projection& mat, bool swz, int tiles_k, int tiles_n,
                                                     int tile_k, int nt)
{
    const size_t tile = swz ? size_t(nt / 8) * tiles_k * 8 + size_t(tile_k) * 8 + nt % 8 : size_t(tile_k) * tiles_n + nt;
    return reinterpret_cast<const uint32_t*>(mat.trellis + tile * 48);
}

// Decoded state J (0..15) of a tile pair: 32 16-bit trellis states.
template <int J>
EXL3_ALWAYS_INLINE inline __m512i register_state(const TilePair& t)
{
    constexpr int offset = 19 + 6 * J, q = offset / 32, s = offset % 32;
    __m512i lo, hi;
    if constexpr (q == 0) { lo = t.prev; hi = t.a; }
    else if constexpr (q == 1) { lo = t.a; hi = t.b; }
    else if constexpr (q == 2) { lo = t.b; hi = t.c; }
    else { lo = t.c; hi = _mm512_setzero_si512(); }
    if constexpr (s <= 13) {
        // Only v[31:13] survives the halfword selection. hi contributes to v[s-1:0], so it is irrelevant here.
        return _mm512_mask_blend_epi16(0xaaaaaaaaU, _mm512_srli_epi32(lo, 16 - s), _mm512_slli_epi32(lo, s + 3));
    } else if constexpr (s == 15) {
        return _mm512_mask_blend_epi16(0xaaaaaaaaU, _mm512_srli_epi32(lo, 1),
                                       _mm512_or_si512(_mm512_slli_epi32(lo, 18), _mm512_srli_epi32(hi, 14)));
    } else if constexpr (s == 29) {
        return _mm512_mask_blend_epi16(0xaaaaaaaaU, _mm512_or_si512(_mm512_slli_epi32(lo, 13), _mm512_srli_epi32(hi, 19)),
                                       hi);
    } else if constexpr (s == 31) {
        return _mm512_mask_blend_epi16(0xaaaaaaaaU, _mm512_or_si512(_mm512_slli_epi32(lo, 15), _mm512_srli_epi32(hi, 17)),
                                       _mm512_slli_epi32(hi, 2));
    } else {
        const __m512i v = _mm512_or_si512(_mm512_slli_epi32(lo, s), _mm512_srli_epi32(hi, 32 - s));
        return _mm512_mask_blend_epi16(0xaaaaaaaaU, _mm512_srli_epi32(v, 16), _mm512_slli_epi32(v, 3));
    }
}

// State J's k-row within the tile and its output half (columns 0-15 or 16-31 of the pair).
template <int J>
constexpr int state_row = (J / 4) * 2 + (J % 2) * 8;
template <int J>
constexpr int state_half = (J / 2) % 2;

// -------------------------------------------------------------------------------------------
//   register_band: tile pairs [n0, n0 + 2 * Pairs) of M tokens
// -------------------------------------------------------------------------------------------

// The byte path's activations: each compact int16 x (|x| <= 127) as the byte pair (x, x), so a broadcast activation
// pair reads as the bytes (xE, xE, xO, xO). rows quantized rows of one 128-input block, 128 apart, as in compact.
EXL3_ALWAYS_INLINE inline void byte_dup_rows(const int16_t* src, int16_t* dst, int rows)
{
    const __m512i low = _mm512_set1_epi16(0x00ff);
    for (int i = 0; i < rows * 128; i += 32) {
        const __m512i x = _mm512_and_si512(_mm512_loadu_si512(src + i), low);
        _mm512_store_si512(dst + i, _mm512_or_si512(x, _mm512_slli_epi16(x, 8)));
    }
}

// One-token compact chunks on VNNI: the product bytes dotted with (xE, xE, xO, xO) by vpdpbusd (byte_dup_rows).
template <bool Compact, int M>
constexpr bool kBytePath = kVnniAccumulate && Compact && M == 1;

// A tile pair's 16 states into acc[band][half][0 .. 2M): 2 * M quantized rows, 128 apart (M token rows, then their
// residual rows). splat is the compact rows (int16, the activation pair at row) or splat_dup (int32, its middle bytes).
template <int Pairs, bool Compact, int M>
EXL3_ALWAYS_INLINE inline void register_rows(const TilePair& t, const std::conditional_t<Compact, int16_t, int32_t>* splat,
                                             __m512i (&acc)[Pairs][2][2 * M], int band)
{
    Unroll<16>{}([&](auto J) EXL3_ALWAYS_INLINE {
        const __m512i state = register_state<J>(t);
        constexpr int row = state_row<J>, half = state_half<J>;
        if constexpr (kBytePath<Compact, M>) {
            // register_bytesum's 32-bit product halves, each dotted bytewise with (xE, xE, xO, xO): the sum of all four
            // product bytes times x per state, the integer the byte sum gives.
            const __m512i ml = _mm512_set1_epi16(int16_t(kMul1Mult & 0xffff));
            const __m512i mh = _mm512_set1_epi16(int16_t(kMul1Mult >> 16));
            const __m512i lo = _mm512_mullo_epi16(state, ml);
            const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml), _mm512_mullo_epi16(state, mh));
            for (int i = 0; i < 2; ++i) {
                uint32_t pair;
                std::memcpy(&pair, splat + i * 128 + row, 4);
                const __m512i x = _mm512_set1_epi32(int32_t(pair));
                acc[band][half][i] = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc[band][half][i], lo, x), hi, x);
            }
        } else {
            const __m512i sum = register_bytesum(state);
            for (int i = 0; i < 2 * M; ++i) {
                // An int32 splat_dup entry holds its activation twice: the middle four bytes of two adjacent entries
                // are the pair.
                uint32_t pair;
                std::memcpy(&pair, reinterpret_cast<const char*>(splat + i * 128 + row) + (Compact ? 0 : 2), 4);
                acc[band][half][i] = dot_pairs(acc[band][half][i], sum, _mm512_set1_epi32(int32_t(pair)));
            }
        }
    });
}

// The fp32 scale of each quantized row's block-kb sums: k_inv * q, and the -510 * sum correction.
template <int R>
EXL3_ALWAYS_INLINE inline void block_scales(const PreparedIn& in, int kb, __m512 (&scales)[R], __m512 (&corrections)[R])
{
    for (int i = 0; i < R; ++i) {
        const float scale = kMul1KInv * in.bq[kb * MAX_M + i];
        scales[i] = _mm512_set1_ps(scale);
        corrections[i] = _mm512_set1_ps(-510.0f * float(in.bsum[kb * MAX_M + i]) * scale);
    }
}

// Token row t at tout + t * n gets token + residual; its residual row stays at tout + (M + t) * n, as the generic path
// leaves it. sums[b][h][i]: tile pair b, output half h, quantized row i.
template <int Pairs, int M>
EXL3_ALWAYS_INLINE inline void store_pairs(const __m512 (&sums)[Pairs][2][2 * M], float* tout, int n, int n0)
{
    for (int b = 0; b < Pairs; ++b)
        for (int t = 0; t < M; ++t) {
            float* row = tout + size_t(t) * n;
            float* res = tout + size_t(M + t) * n;
            const __m512 lo = _mm512_add_ps(sums[b][0][t], sums[b][0][M + t]);
            const __m512 hi = _mm512_add_ps(sums[b][1][t], sums[b][1][M + t]);
            _mm512_storeu_ps(row + (n0 + 2 * b) * 16, _mm512_shuffle_f32x4(lo, hi, 0x44));
            _mm512_storeu_ps(row + (n0 + 2 * b + 1) * 16, _mm512_shuffle_f32x4(lo, hi, 0xee));
            _mm512_storeu_ps(res + (n0 + 2 * b) * 16, _mm512_shuffle_f32x4(sums[b][0][M + t], sums[b][1][M + t], 0x44));
            _mm512_storeu_ps(res + (n0 + 2 * b + 1) * 16, _mm512_shuffle_f32x4(sums[b][0][M + t], sums[b][1][M + t], 0xee));
        }
}

template <int Pairs, bool Compact = false, int M = 1>
void register_band(const Exl3Projection& mat, const PreparedIn& in, float* tout, int n0)
{
    constexpr int R = 2 * M;  // quantized rows: M token rows, then their M residual rows
    const int k = mat.k, n = mat.n;
    const bool swz = mat.swz;
    const int tiles_k = k / 16, tiles_n = n / 16;
    const size_t step = size_t(swz ? 8 : tiles_n) * 96;
    __m512 sums[Pairs][2][R];
    constexpr bool bytes = kBytePath<Compact, M>;
    [[maybe_unused]] alignas(64) int16_t dup[bytes ? R * 128 : 1];
    for (int kb = 0; kb < k / 128; ++kb) {
        __m512i acc[Pairs][2][R];
        for (int b = 0; b < Pairs; ++b)
            for (int h = 0; h < 2; ++h)
                for (int i = 0; i < R; ++i) acc[b][h][i] = _mm512_setzero_si512();
        if constexpr (bytes) byte_dup_rows(in.compact + size_t(kb) * R * 128, dup, R);
        for (int kt = 0; kt < 8; ++kt) {
            const auto* splat = [&]() {
                if constexpr (bytes) return static_cast<const int16_t*>(dup) + kt * 16;
                else if constexpr (Compact) return in.compact + size_t(kb) * R * 128 + kt * 16;
                else return in.splat_dup + size_t(kb) * R * 128 + kt * 16;
            }();
            for (int band = 0; band < Pairs; ++band) {
                const TilePair t = load_tile_pair(tile_words(mat, swz, tiles_k, tiles_n, kb * 8 + kt, n0 + 2 * band), step);
                register_rows<Pairs, Compact, M>(t, splat, acc, band);
            }
        }
        __m512 scales[R], corrections[R];
        block_scales(in, kb, scales, corrections);
        for (int b = 0; b < Pairs; ++b)
            for (int h = 0; h < 2; ++h)
                for (int i = 0; i < R; ++i) {
                    const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]), scales[i], corrections[i]);
                    if (kb == 0) sums[b][h][i] = v;
                    else sums[b][h][i] = _mm512_add_ps(sums[b][h][i], v);
                }
    }
    store_pairs<Pairs, M>(sums, tout, n, n0);
}

// register_rows for one output half: the states J with state_half<J> == Half, each decoded once, into that half's 2 * M
// rows (register_band_halves).
template <int Pairs, int M, int Half>
EXL3_ALWAYS_INLINE inline void register_rows_half(const TilePair& t, const int16_t* splat, __m512i (&acc)[Pairs][2 * M],
                                                  int band)
{
    Unroll<16>{}([&](auto J) EXL3_ALWAYS_INLINE {
        if constexpr (state_half<J> == Half) {
            const __m512i sum = register_bytesum(register_state<J>(t));
            constexpr int row = state_row<J>;
            for (int i = 0; i < 2 * M; ++i) {
                uint32_t pair;
                std::memcpy(&pair, splat + i * 128 + row, 4);
                acc[band][i] = dot_pairs(acc[band][i], sum, _mm512_set1_epi32(int32_t(pair)));
            }
        }
    });
}

// register_band<Pairs, true, M> for chunks whose 4 * M accumulators per tile pair do not fit the registers
// (kRegisterBudget's halves entries). Each 128-input block runs its eight k-tiles once per output half, holding only
// that half's 2 * M accumulators; the second pass reloads the tiles from L1 and decodes the other half's states, so
// every state is still decoded once. Each output's integer sums, per-block scale, fp32 block order and token + residual
// are register_band's.
template <int Pairs, int M>
void register_band_halves(const Exl3Projection& mat, const PreparedIn& in, float* tout, int n0)
{
    constexpr int R = 2 * M;
    const int k = mat.k, n = mat.n, tiles_n = n / 16;
    const size_t step = size_t(tiles_n) * 96;
    __m512 sums[Pairs][2][R];
    for (int kb = 0; kb < k / 128; ++kb) {
        __m512 scales[R], corrections[R];
        block_scales(in, kb, scales, corrections);
        auto half = [&]<int Half>() {
            __m512i acc[Pairs][R];
            for (int b = 0; b < Pairs; ++b)
                for (int i = 0; i < R; ++i) acc[b][i] = _mm512_setzero_si512();
            for (int kt = 0; kt < 8; ++kt) {
                const int16_t* splat = in.compact + size_t(kb) * R * 128 + kt * 16;
                for (int band = 0; band < Pairs; ++band) {
                    const TilePair t = load_tile_pair(tile_words(mat, false, 0, tiles_n, kb * 8 + kt, n0 + 2 * band),
                                                      step, Half == 0);
                    register_rows_half<Pairs, M, Half>(t, splat, acc, band);
                }
            }
            for (int b = 0; b < Pairs; ++b)
                for (int i = 0; i < R; ++i) {
                    const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), scales[i], corrections[i]);
                    if (kb == 0) sums[b][Half][i] = v;
                    else sums[b][Half][i] = _mm512_add_ps(sums[b][Half][i], v);
                }
        };
        half.template operator()<0>();
        half.template operator()<1>();
    }
    store_pairs<Pairs, M>(sums, tout, n, n0);
}

// -------------------------------------------------------------------------------------------
//   One-token traversal (whole 128-output groups)
// -------------------------------------------------------------------------------------------

// Visit neighboring 128-output groups before advancing to the next 128-input block. Original packed weights and each
// output's increasing-kb fp32 addition order stay unchanged. One integer group is live at a time; the fp32 partials
// reside in private aligned scratch.
struct IntegerAccum
{
    __m512i lanes[4][2][2];
};

// One 128-input block of one 128-output group (its four tile pairs), added into sums.
__attribute__((noinline)) void traversal_kblock(const Exl3Projection& mat, const PreparedIn& in,
                                                __m512 (&sums)[4][2][2], int n0, int kb)
{
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const bool swz = mat.swz;
    const size_t step = size_t(swz ? 8 : tiles_n) * 96;
    IntegerAccum first;
    for (int b = 0; b < 4; ++b)
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 2; ++i) first.lanes[b][h][i] = _mm512_setzero_si512();
    auto& acc = first.lanes;
    constexpr bool bytes = kBytePath<true, 1>;
    [[maybe_unused]] alignas(64) int16_t dup[bytes ? 256 : 1];
    if constexpr (bytes) byte_dup_rows(in.compact + size_t(kb) * 256, dup, 2);
    for (int kt = 0; kt < 8; ++kt) {
        const int16_t* splat = (bytes ? static_cast<const int16_t*>(dup) : in.compact + size_t(kb) * 256) + kt * 16;
#pragma omp unroll full
        for (int band = 0; band < 4; ++band) {
            const TilePair t = load_tile_pair(tile_words(mat, swz, tiles_k, tiles_n, kb * 8 + kt, n0 + 2 * band), step);
            register_rows<4, true, 1>(t, splat, acc, band);
        }
    }
    __m512 scales[2], corrections[2];
    block_scales(in, kb, scales, corrections);
    for (int b = 0; b < 4; ++b)
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < 2; ++i) {
                const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][h][i]), scales[i], corrections[i]);
                if (kb == 0) sums[b][h][i] = v;
                else sums[b][h][i] = _mm512_add_ps(sums[b][h][i], v);
            }
}

template <int Groups>
void traversal_group(const Exl3Projection& mat, const PreparedIn& in, float* tout, int n0)
{
    static_assert(Groups >= 1 && Groups <= 4);
    // Each group has 16 zmm partial sums: 1024 bytes per 128 outputs.
    alignas(64) __m512 sums[Groups][4][2][2];
    for (int kb = 0; kb < mat.k / 128; ++kb)
        for (int group = 0; group < Groups; ++group) traversal_kblock(mat, in, sums[group], n0 + group * 8, kb);
    for (int group = 0; group < Groups; ++group) store_pairs<4, 1>(sums[group], tout, mat.n, n0 + group * 8);
}

// Whole groups [t0, t1). Range-aware: three-group runs for longer ranges (9 = 3+3+3, 10 = 3+3+4), else up to four.
void traversal_tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int t0, int t1, bool grouped)
{
    for (int t = t0; t < t1;) {
        const int remaining = (t1 - t) / 8;
        const int cap = grouped ? (remaining > 4 ? 3 : 4) : 1;
        const int groups = std::min(cap, remaining);
        with_count<4>(groups, [&]<int G>() { traversal_group<G>(mat, in, tout, t); });
        t += groups * 8;
    }
}

// Compact tile pairs [t0, t1) inside one 128-output group, t1 - t0 in {0, 2, 4, 6}: the same per-output arithmetic as
// traversal_kblock, which takes whole groups.
void compact_pairs(const Exl3Projection& mat, const PreparedIn& in, float* tout, int t0, int t1)
{
    with_count<3>((t1 - t0) / 2, [&]<int P>() { register_band<P, true>(mat, in, tout, t0); });
}

// -------------------------------------------------------------------------------------------
//   One-token, one-tile kernel on the int32 activations (the generic plan's Bw tier)
// -------------------------------------------------------------------------------------------

// Decodes two adjacent tile rows at once into 32 interleaved 16-bit lanes. For a 16-bit state s and
// M = (M_hi << 16) + M_lo, the product's low and high halves are mullo(s, M_lo) and mulhi_unsigned(s, M_lo) +
// mullo(s, M_hi) mod 2^16; summing their bytes gives the dword kernel's mul1 byte sum, and vpmaddwd combines both rows
// without saturation. Each lane takes the two halfwords enclosing its state in the (previous : current dword) window.
template <int row, bool high>
constexpr std::array<uint16_t, 32> bw3_word_indices()
{
    std::array<uint16_t, 32> idx{};
    constexpr auto inv = make_tc_perm_inv();
    for (int col = 0; col < 16; ++col)
        for (int r = 0; r < 2; ++r) {
            const int t = inv[(row + r) * 16 + col];
            const int b0 = t * 3 + 3 - 16 + 768;
            const int b1 = b0 + 16;
            const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
            const int half = shift / 16 + int(high);
            idx[col * 2 + r] = half < 2 ? (((b1 - 1) / 32) % 24) * 2 + half : ((b0 / 32) % 24) * 2 + half - 2;
        }
    return idx;
}

template <int row, bool left>
constexpr std::array<uint16_t, 32> bw3_word_shifts()
{
    std::array<uint16_t, 32> shifts{};
    for (int col = 0; col < 16; ++col) {
        const int s0 = row_shift<3, row>(col) % 16;
        const int s1 = row_shift<3, row + 1>(col) % 16;
        shifts[col * 2] = left ? 16 - s0 : s0;
        shifts[col * 2 + 1] = left ? 16 - s1 : s1;
    }
    return shifts;
}

// One 3-bit tile (96 bytes: both halfword gathers fit in p0/p1) of token row 0 and its residual row 1.
EXL3_ALWAYS_INLINE inline void bw3_rows(__m512i p0, __m512i p1, const int32_t* splat_dup, int k, __m512i (&acc)[2],
                                        const char* future_weight)
{
    // Prefetch a future tile while decoding this one's register data.
    _mm_prefetch(future_weight, _MM_HINT_T0);
    _mm_prefetch(future_weight + 64, _MM_HINT_T0);
    Unroll<8>{}([&](auto P) EXL3_ALWAYS_INLINE {
        constexpr int R = 2 * P;
        alignas(64) static constexpr auto il = bw3_word_indices<R, false>();
        alignas(64) static constexpr auto ih = bw3_word_indices<R, true>();
        alignas(64) static constexpr auto sr = bw3_word_shifts<R, false>();
        alignas(64) static constexpr auto sl = bw3_word_shifts<R, true>();
        const __m512i lo = _mm512_permutex2var_epi16(p0, _mm512_load_si512(il.data()), p1);
        const __m512i hi = _mm512_permutex2var_epi16(p0, _mm512_load_si512(ih.data()), p1);
        const __m512i state = _mm512_or_si512(_mm512_srlv_epi16(lo, _mm512_load_si512(sr.data())),
                                              _mm512_sllv_epi16(hi, _mm512_load_si512(sl.data())));
        const __m512i sum = register_bytesum(state);
        for (int i = 0; i < 2; ++i) {
            // Each i32 holds two copies of a signed i16 activation: the middle four bytes of two adjacent entries
            // hold the pair.
            uint32_t pair;
            std::memcpy(&pair, reinterpret_cast<const char*>(splat_dup + size_t(i) * k + R) + 2, sizeof(pair));
            acc[i] = _mm512_add_epi32(acc[i], _mm512_madd_epi16(sum, _mm512_set1_epi32(static_cast<int32_t>(pair))));
        }
    });
}

// Output tile n0 of one token: each 128-input block's int32 sums scaled with their own FMA, the base and residual rows
// summed separately in fp32, the residual added once (the generic path's rounding).
void bw3_blocked_tile(const Exl3Projection& mat, const PreparedIn& in, float* tout, int n0)
{
    constexpr int B = 128, packed_size = 48;
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t step = size_t(mat.swz ? 8 : tiles_n) * packed_size;
    __m512 sums[2];
    for (int kb = 0; kb < mat.k / B; ++kb) {
        __m512i acc[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
        for (int kt = 0; kt < B / 16; ++kt) {
            const int tile_k = kb * (B / 16) + kt;
            const int32_t* splat = in.splat_dup + size_t(kb) * 2 * B + kt * 16;
            const size_t tile = mat.swz ? size_t(n0 / 8) * tiles_k * 8 + size_t(tile_k) * 8 + (n0 % 8)
                                        : size_t(tile_k) * tiles_n + n0;
            const uint16_t* packed = mat.trellis + tile * packed_size;
            // Two k-tiles ahead (step is in uint16_t), formed as an integer: it may point past the final tile.
            const char* pf = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(packed) + 4 * step);
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_loadu_si512(pw);
            // The tile tail is exactly 32 bytes: a 256-bit load avoids a masked 512-bit load and zeroes the upper lanes.
            const __m512i p1 = _mm512_zextsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pw + 16)));
            bw3_rows(p0, p1, splat, B, acc, pf);
        }
        for (int i = 0; i < 2; ++i) {
            const float scale = kMul1KInv * in.bq[kb * MAX_M + i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.bsum[kb * MAX_M + i]) * scale);
            const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[i]), _mm512_set1_ps(scale), corr);
            if (kb == 0) sums[i] = v;
            else sums[i] = _mm512_add_ps(sums[i], v);
        }
    }
    _mm512_storeu_ps(tout + n0 * 16, _mm512_add_ps(sums[0], sums[1]));
    _mm512_storeu_ps(tout + mat.n + n0 * 16, sums[1]);
}

// -------------------------------------------------------------------------------------------
//   register_tiles<M>: output tiles [t0, t1) of one chunk of M tokens
// -------------------------------------------------------------------------------------------

// M = 1 is the one-token kernel either plan calls (compact or not, swizzled or not; the grouped traversal:
// kRegisterBudget[0]). M > 1 is the DSV4.1 plan's compact, unswizzled input, in runs of up to
// kRegisterBudget[M - 1].pairs tile pairs; M = 5..8 (halves entries) through register_band_halves.
template <int M>
void register_tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int t0, int t1,
                    [[maybe_unused]] bool grouped)
{
    [[maybe_unused]] constexpr RegisterBudget budget = kRegisterBudget[M - 1];
    if constexpr (M == 1) {
        if (in.compact) {
            if (mat.swz) {
                // The swizzled layout stores a 128-output group's eight tiles together.
                EXL3_CHECK(t0 % 8 == 0 && t1 % 8 == 0, "compact swizzled input requires whole output blocks");
                for (int t = t0; t < t1; t += 8) register_band<4, true>(mat, in, tout, t);
                return;
            }
            // Unswizzled: whole groups through the traversal, a partial group at either end as tile pairs.
            EXL3_CHECK(t0 % 2 == 0 && t1 % 2 == 0, "compact input requires whole tile pairs");
            const int a0 = std::min(t1, (t0 + 7) / 8 * 8), a1 = std::max(a0, t1 / 8 * 8);
            compact_pairs(mat, in, tout, t0, a0);
            if (a1 > a0) traversal_tiles(mat, in, tout, a0, a1, grouped);
            compact_pairs(mat, in, tout, a1, t1);
            return;
        }
        for (int n0 = t0; n0 < t1;) {
            if ((n0 % 2) || t1 - n0 < 2) {
                bw3_blocked_tile(mat, in, tout, n0++);
                continue;
            }
            const int pairs = std::min({4, (t1 - n0) / 2, (8 - n0 % 8) / 2});
            with_count<4>(pairs, [&]<int P>() { register_band<P>(mat, in, tout, n0); });
            n0 += pairs * 2;
        }
    } else {
        static_assert(!budget.traversal, "the grouped traversal holds one token's rows (IntegerAccum)");
        EXL3_CHECK(in.compact && !mat.swz && t0 % 2 == 0 && t1 % 2 == 0,
                   "a chunk of several tokens needs compact, unswizzled tile pairs");
        if constexpr (budget.halves) {
            static_assert(budget.pairs == 1, "register_band_halves runs one tile pair per call here");
            for (int n0 = t0; n0 < t1; n0 += 2) register_band_halves<1, M>(mat, in, tout, n0);
            return;
        }
        for (int n0 = t0; n0 < t1;) {
            const int pairs = std::min(budget.pairs, (t1 - n0) / 2);
            with_count<budget.pairs>(pairs, [&]<int P>() { register_band<P, true, M>(mat, in, tout, n0); });
            n0 += pairs * 2;
        }
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
#endif  // SGLANG_CPU_EXPERTS_AVX512
