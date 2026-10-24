// The AVX-512 tiers' GEMV tiles (Tiles<Bw | Vnni | Vbmi>): bands of output tiles held in zmm accumulators while the
// k-tiles stream through. One band driver (band_tiles) serves the three tiers; BandRows<Isa> is each tier's decode and
// integer accumulate of one k-tile's 16 rows. The DSV4.1 plan's register kernels are register_kernels.hpp.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#if defined(SGLANG_CPU_EXPERTS_AVX512)
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace sglang::exl3_cpu {
namespace {

// -------------------------------------------------------------------------------------------
//   Dword state extraction (the Bw and Vnni tiers)
// -------------------------------------------------------------------------------------------

// Gathers the two 32-bit word vectors covering row `row`'s 16-bit states (the permute stage of the extraction, split
// out so a row pair can share it).
template <int bits, int row>
inline void dword_gather(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i& a, __m512i& b)
{
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    a = _mm512_permutex2var_epi32(p0, i0, p1);
    b = _mm512_permutex2var_epi32(p0, i1, p1);
    if constexpr (bits > 4) {
        // Up to 64 packed words: indices >= 32 select from the second register pair. vpermt2var uses index bits [4:0],
        // so the same index vectors address both pairs; constexpr masks choose per lane.
        constexpr __mmask16 hm0 = make_row_himask<bits, row, false>();
        constexpr __mmask16 hm1 = make_row_himask<bits, row, true>();
        if constexpr (hm0 != 0) a = _mm512_mask_blend_epi32(hm0, a, _mm512_permutex2var_epi32(p2, i0, p3));
        if constexpr (hm1 != 0) b = _mm512_mask_blend_epi32(hm1, b, _mm512_permutex2var_epi32(p2, i1, p3));
    }
}

// Per-lane shift counts for the funnel merge: cols 0-7 use s0, cols 8-15 use s1.
template <int s0, int s1>
constexpr std::array<int32_t, 16> make_lane_shifts()
{
    std::array<int32_t, 16> v{};
    for (int i = 0; i < 16; ++i) v[i] = i < 8 ? s0 : s1;
    return v;
}

// Shift-merges the codes of `row` out of its gathered word vectors; delta = bits extracts row + 1 from row's own gather
// (valid when word_pair_ok). Vector shifts by >= 32 are a well-defined zero. The two half-rows generally need different
// shifts: one per-lane variable funnel shift merges both in 4 uops, where two immediate shifts plus a blend took 8.
template <int bits, int row, int delta>
inline __m512i dword_codes(__m512i a, __m512i b)
{
    constexpr int s0 = row_shift<bits, row>(0) - delta;
    constexpr int s1 = row_shift<bits, row>(8) - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds shift headroom");
    if constexpr (s0 == s1) {
        const __m512i c = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    } else {
        alignas(64) static constexpr auto sh = make_lane_shifts<s0, s1>();
        alignas(64) static constexpr auto shc = make_lane_shifts<32 - s0, 32 - s1>();
        const __m512i c = _mm512_or_si512(_mm512_srlv_epi32(b, _mm512_load_si512(sh.data())),
                                          _mm512_sllv_epi32(a, _mm512_load_si512(shc.data())));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
}

template <int bits, int row>
inline __m512i extract_row(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    __m512i a, b;
    dword_gather<bits, row>(p0, p1, p2, p3, a, b);
    return dword_codes<bits, row, 0>(a, b);
}

// The codes of rows R and R + 1, sharing one gather where the layout allows (word_pair_ok).
template <int bits, int R>
EXL3_ALWAYS_INLINE inline void dword_row_pair(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i& code0,
                                              __m512i& code1)
{
    __m512i a, wb;
    dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
    code0 = dword_codes<bits, R, 0>(a, wb);
    if constexpr (word_pair_ok<bits, R>()) {
        code1 = dword_codes<bits, R, bits>(a, wb);
    } else {
        dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
        code1 = dword_codes<bits, R + 1, 0>(a, wb);
    }
}

// The swizzled layout needs bands that divide 8 (whole or partial groups); the dword scheme's live temporaries leave
// room for 8 accumulators at one row. Native: up to 12 accumulators. 16 at one row measured no better than 8 for
// decode-shape jobs (7960X: the prefetcher covers 8-tile bursts and the extra accumulators cost scheduling registers).
template <int rows>
constexpr int dword_max_band(bool swz)
{
    return swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2) : (rows == 1 ? 8 : std::min(12 / rows, 8));
}

// -------------------------------------------------------------------------------------------
//   BandRows<Isa>: one k-tile's 16 rows into a band column's accumulators
// -------------------------------------------------------------------------------------------

// accumulate<bits, rows, band>(p0..p3, b, x, k, acc): decodes the k-tile in p0..p3 (16 * bits words) and adds each of
// its rows' 16 product sums times activation row i's x[i * k + row] into acc[b][i], for i < rows. kDup: x is
// PreparedIn::splat_dup, else splat32. max_band<bits, rows>(swz): the widest band of tiles a call holds.
template <Isa I>
struct BandRows;

// AVX-512F/BW/VL without VNNI (Skylake-SP/X): the dword extraction with the AVX2 tier's accumulate, vpmaddubsw of the
// product bytes against 0x01 pairs giving (b0+b1), (b2+b3) as i16 lanes (<= 510), then one vpmaddwd per token row.
// Force-inlined: GCC otherwise outlines the 16-row chain behind a call on every tile step, and with every zmm caller-
// saved the band loop then reloads all of its live constants after each call (+8-9% inlined, Skylake-SP).
template <>
struct BandRows<Isa::Bw>
{
    static constexpr bool kDup = true;

    template <int bits, int rows>
    static constexpr int max_band(bool swz) { return dword_max_band<rows>(swz); }

    template <int bits, int rows, int band>
    EXL3_ALWAYS_INLINE static void accumulate(__m512i p0, __m512i p1, __m512i p2, __m512i p3, int b,
                                              const int32_t* splat_dup, int k, __m512i (&acc)[band][GENERIC_MAX_M])
    {
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(kMul1Mult));
        const __m512i ones = _mm512_set1_epi32(0x01010101);
        if constexpr (dword_pair_wins<bits>()) {
            unroll<8>([&]<int P>() EXL3_ALWAYS_INLINE {
                constexpr int R = 2 * P;
                __m512i code0, code1;
                dword_row_pair<bits, R>(p0, p1, p2, p3, code0, code1);
                const __m512i ps0 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code0, mult), ones);
                const __m512i ps1 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code1, mult), ones);
                for (int i = 0; i < rows; ++i) {
                    const __m512i x0 = _mm512_set1_epi32(splat_dup[size_t(i) * k + R]);
                    const __m512i x1 = _mm512_set1_epi32(splat_dup[size_t(i) * k + R + 1]);
                    acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps0, x0));
                    acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps1, x1));
                }
            });
        } else {
            unroll<16>([&]<int R>() EXL3_ALWAYS_INLINE {
                const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
                const __m512i ps = _mm512_maddubs_epi16(_mm512_mullo_epi32(code, mult), ones);
                for (int i = 0; i < rows; ++i)
                    acc[b][i] = _mm512_add_epi32(
                        acc[b][i], _mm512_madd_epi16(ps, _mm512_set1_epi32(splat_dup[size_t(i) * k + R])));
            });
        }
    }
};

#if defined(SGLANG_CPU_EXPERTS_VNNI)
// AVX-512 VNNI: the product bytes straight into vpdpbusd against the activation replicated x4.
template <>
struct BandRows<Isa::Vnni>
{
    static constexpr bool kDup = false;

    template <int bits, int rows>
    static constexpr int max_band(bool swz) { return dword_max_band<rows>(swz); }

    template <int bits, int rows, int band>
    EXL3_ALWAYS_INLINE static void accumulate(__m512i p0, __m512i p1, __m512i p2, __m512i p3, int b,
                                              const int32_t* splat, int k, __m512i (&acc)[band][GENERIC_MAX_M])
    {
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(kMul1Mult));
        if constexpr (dword_pair_wins<bits>()) {
            unroll<8>([&]<int P>() EXL3_ALWAYS_INLINE {
                constexpr int R = 2 * P;
                __m512i code0, code1;
                dword_row_pair<bits, R>(p0, p1, p2, p3, code0, code1);
                const __m512i prod0 = _mm512_mullo_epi32(code0, mult);
                const __m512i prod1 = _mm512_mullo_epi32(code1, mult);
                for (int i = 0; i < rows; ++i) {
                    acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0, _mm512_set1_epi32(splat[size_t(i) * k + R]));
                    acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1, _mm512_set1_epi32(splat[size_t(i) * k + R + 1]));
                }
            });
        } else {
            unroll<16>([&]<int R>() EXL3_ALWAYS_INLINE {
                const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
                const __m512i prod = _mm512_mullo_epi32(code, mult);
                for (int i = 0; i < rows; ++i)
                    acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod, _mm512_set1_epi32(splat[size_t(i) * k + R]));
            });
        }
    }
};
#endif  // SGLANG_CPU_EXPERTS_VNNI

#if defined(SGLANG_CPU_EXPERTS_VBMI)
// AVX-512 VBMI: the dword extraction's two 32-bit cross permutes and shift-merge become one byte permute (vpermb /
// vpermt2b) gathering the 3 bytes that cover each column's 16-bit state into its dword lane, then one (even K) or two
// blended (odd K) sub-byte shifts and a mask, bit-exactly the same codes. Layout facts it relies on (make_tc_perm;
// validated bit-exact against the dword scheme for K1-8 x m1-4):
//   - within a half-row (cols 0-7 / 8-15) bit offsets step by 32 * bits per column, so shift % 8 is uniform per half;
//   - the two half-rows differ by 4 * bits bits: the same shift % 8 for even K, +-4 for odd K;
//   - rows (2p, 2p+1) differ by exactly `bits` bits, so when shift % 8 >= bits for every column of the even row, one
//     gather serves both rows ("byte pairing": K1/K2/K4 all 8 pairs, K3/K6 rows 8-15 only, K5/K7/K8 never).
// At 8 bits byte pairing is impossible (shift % 8 == 0) and the windows straddle the register pairs, measured slower
// than the dword scheme: the Vbmi tier runs BandRows<Vnni> there (Tiles::tiles).
template <>
struct BandRows<Isa::Vbmi>
{
    static constexpr bool kDup = false;

    // Byte gathers need few temporaries, so wider swizzled bands (up to 16 accumulators: rows 2 x band 8, rows 3-4 x
    // band 4) fit and keep the swizzled stream at full duty; narrow divisor bands measured below the native layout at
    // m > 1. K2 rows 4 prefers band 2 (216 vs 197 Gw/s at band 4); rows > 4 (a raised MAX_M) take two bands.
    template <int bits, int rows>
    static constexpr int max_band(bool swz)
    {
        if (!swz) return dword_max_band<rows>(false);
        return rows <= 2 ? 8 : rows > 4 ? 2 : (rows == 4 && bits == 2 ? 2 : 4);
    }

    // For each column, the indices (into the tile's 32 * bits bytes) of the 3 bytes covering bits [shift, shift + 16)
    // of its (w0:w1) word window; the 4th lane byte is unused (shift % 8 + 16 <= 23 keeps the value in the low 3).
    template <int bits, int row>
    static constexpr std::array<uint8_t, 64> byte_indices()
    {
        std::array<uint8_t, 64> idx{};
        const auto inv = make_tc_perm_inv();
        constexpr int words32 = bits * 256 / 32;
        for (int col = 0; col < 16; ++col) {
            const int t = inv[row * 16 + col];
            const int b0 = t * bits + bits - 16 + 256 * bits;
            const int b1 = b0 + 16;
            const int w0 = (b0 / 32) % words32;        // high (earlier) word
            const int w1 = ((b1 - 1) / 32) % words32;  // low (later) word
            const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
            const int fb = shift / 8;
            for (int byte = 0; byte < 3; ++byte) {
                const int mb = fb + byte;
                idx[col * 4 + byte] = static_cast<uint8_t>(mb < 4 ? w1 * 4 + mb : w0 * 4 + (mb - 4));
            }
            idx[col * 4 + 3] = 0;
        }
        return idx;
    }

    // For bits > 4 (a tile spans 4 zmm): the gathered bytes that come from the (p2, p3) pair. vpermt2b reads index
    // bits [6:0], so raw indices >= 128 address the high pair directly.
    template <int bits, int row>
    static constexpr uint64_t byte_himask()
    {
        const auto idx = byte_indices<bits, row>();
        uint64_t m = 0;
        for (int i = 0; i < 64; ++i)
            if (idx[i] >= 128) m |= uint64_t(1) << i;
        return m;
    }

    // Byte pairing is valid iff the odd row's value stays inside the even row's gathered byte window for every column:
    // shift % 8 >= bits everywhere (stricter than word_pair_ok, which needs only the full shift's headroom).
    template <int bits, int row>
    static constexpr bool byte_pair_ok()
    {
        for (int col = 0; col < 16; ++col)
            if (row_shift<bits, row>(col) % 8 < bits) return false;
        return true;
    }

    template <int bits, int row>
    static __m512i gather_bytes(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
    {
        alignas(64) static constexpr auto bidx = byte_indices<bits, row>();
        const __m512i idx = _mm512_load_si512(bidx.data());
        if constexpr (bits <= 2) {
            return _mm512_permutexvar_epi8(idx, p0);
        } else if constexpr (bits <= 4) {
            return _mm512_permutex2var_epi8(p0, idx, p1);
        } else {
            constexpr uint64_t hm = byte_himask<bits, row>();
            if constexpr (hm == 0)
                return _mm512_permutex2var_epi8(p0, idx, p1);
            else if constexpr (hm == ~uint64_t(0))
                return _mm512_permutex2var_epi8(p2, idx, p3);
            else
                return _mm512_mask_blend_epi8(static_cast<__mmask64>(hm), _mm512_permutex2var_epi8(p0, idx, p1),
                                              _mm512_permutex2var_epi8(p2, idx, p3));
        }
    }

    // delta = 0 extracts `row` itself; delta = bits extracts row + 1 from row's gathered bytes.
    template <int bits, int row, int delta>
    static __m512i shift_mask(__m512i g)
    {
        constexpr int s0 = row_shift<bits, row>(0) % 8 - delta;
        constexpr int s1 = row_shift<bits, row>(8) % 8 - delta;
        static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds sub-byte shift headroom");
        if constexpr (s0 == s1)
            return _mm512_and_si512(_mm512_srli_epi32(g, s0), _mm512_set1_epi32(0xffff));
        else
            return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00, _mm512_srli_epi32(g, s0), _mm512_srli_epi32(g, s1)),
                                    _mm512_set1_epi32(0xffff));
    }

    template <int bits, int rows, int band>
    EXL3_ALWAYS_INLINE static void accumulate(__m512i p0, __m512i p1, __m512i p2, __m512i p3, int b,
                                              const int32_t* splat, int k, __m512i (&acc)[band][GENERIC_MAX_M])
    {
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(kMul1Mult));
        unroll<8>([&]<int P>() EXL3_ALWAYS_INLINE {
            constexpr int R = P * 2;
            __m512i c0, c1;
            if constexpr (byte_pair_ok<bits, R>()) {
                const __m512i g = gather_bytes<bits, R>(p0, p1, p2, p3);
                c0 = shift_mask<bits, R, 0>(g);
                c1 = shift_mask<bits, R, bits>(g);
            } else {
                c0 = shift_mask<bits, R, 0>(gather_bytes<bits, R>(p0, p1, p2, p3));
                c1 = shift_mask<bits, R + 1, 0>(gather_bytes<bits, R + 1>(p0, p1, p2, p3));
            }
            const __m512i prod0 = _mm512_mullo_epi32(c0, mult);
            const __m512i prod1 = _mm512_mullo_epi32(c1, mult);
            for (int i = 0; i < rows; ++i) {
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0, _mm512_set1_epi32(splat[size_t(i) * k + R]));
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1, _mm512_set1_epi32(splat[size_t(i) * k + R + 1]));
            }
        });
    }
};
#endif  // SGLANG_CPU_EXPERTS_VBMI

// -------------------------------------------------------------------------------------------
//   The band driver
// -------------------------------------------------------------------------------------------

constexpr __mmask16 word_load_mask(int n)
{
    return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
}

// Output tiles [n0, n0 + band) of `rows` quantized rows: each k-tile's band of tiles is loaded and decoded into the
// band's accumulators (Rows::accumulate), then each tile is scaled once.
template <class Rows, int bits, int rows, int band>
void band_tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    constexpr __mmask16 mask0 = word_load_mask(words32 - 0);
    constexpr __mmask16 mask1 = word_load_mask(words32 - 16);
    constexpr __mmask16 mask2 = word_load_mask(words32 - 32);
    constexpr __mmask16 mask3 = word_load_mask(words32 - 48);

    __m512i acc[band][GENERIC_MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i) acc[b][i] = _mm512_setzero_si512();

    // Swizzled (band-contiguous) layout: tile (kt, nt) lives at group nt/8, then kt, then member nt%8, so a band's
    // k-stream is (near-)sequential instead of packed_size runs strided by the full row. The bands are divisors of 8
    // within one group (max_band and the group-aligned splits), and a k-block view keeps the whole matrix's group
    // stride (Exl3Projection::group_tiles_k).
    const size_t row_stride = size_t(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? size_t(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + size_t(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride) {
        const int32_t* x = (Rows::kDup ? in.splat_dup : in.splat32) + tile_k * 16;
        for (int b = 0; b < band; ++b) {
            const uint16_t* packed =
                mat.swz ? mat.trellis
                              + (size_t(n0 / 8) * mat.group_tiles_k * 8 + size_t(tile_k) * 8 + (n0 % 8) + b) * packed_size
                        : packed_row + b * packed_size;
            if (mat.swz && band == 8) {
                // A whole-group band on the swizzled layout reads sequentially: one line one step ahead is enough
                // and the HW prefetcher follows the run (wider or farther measured neutral on K4, 7960X).
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            } else {
                // A strided stream (the native layout, or a partial group on the swizzled one, which reads half a group
                // then skips half) outruns the HW prefetcher: touch every line of the tile row a few steps ahead.
                // +11% decode on K8 (7960X); bits == 6 keeps the shorter distance its 96-byte rows need (Tiles<Avx2>).
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
#pragma GCC unroll 4
                for (int l = 0; l < pf_lines; ++l) _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            Rows::template accumulate<bits, rows, band>(p0, p1, p2, p3, b, x, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i) {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + size_t(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <Isa I>
    requires(kAvx512<I>)
struct Tiles<I>
{
    static void run(const Exl3Projection& mat, const PreparedIn& in, float* tout, int rows, int tn0, int tn1)
    {
        with_count<8>(mat.bits, [&]<int bits>() {
            with_count<GENERIC_MAX_M>(rows, [&]<int R>() { tiles<bits, R>(mat, in, tout, tn0, tn1); });
        });
    }

    template <int bits, int rows>
    static void tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
    {
        using Rows = BandRows<(I == Isa::Vbmi && bits == 8) ? Isa::Vnni : I>;
        const int max_band = Rows::template max_band<bits, rows>(mat.swz != 0);
        for (int n0 = tn0; n0 < tn1;) {
            const int band = std::min(tn1 - n0, max_band);
            with_count<8>(band, [&]<int B>() { band_tiles<Rows, bits, rows, B>(mat, in, tout, n0); });
            n0 += band;
        }
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
#endif  // SGLANG_CPU_EXPERTS_AVX512
