// EXL3's arithmetic every tier shares: the build constants, the format tables and state decode, the Hadamard
// transform and the activation quantization. Each tier's GEMV tiles are math_scalar.hpp, math_avx2.hpp and
// math_avx512.hpp (Tiles<Isa>); the DSV4.1 plan's register kernels are register_kernels.hpp.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
//
// The mul1 codebook is affine in a byte-sum: w(s) = (bytesum(s * 0x83DCD12D) - 510) * k_inv with
// k_inv = fp16(0x1eee). With int8 activations x8 and one scale q per input block:
//   sum_k w_kn x_k = k_inv * q * ( sum_k bytesum(s_kn * M) * x8_k  -  510 * sum_k x8_k )
// and bytesum(s*M) * x8 is exactly one AVX-512 VNNI vpdpbusd per 16 weights (unsigned operand = product bytes, signed
// operand = the int8 activation replicated x4; u8*s8 word products stay below 2^15, which makes the operand order
// load-bearing). i32 accumulators are safe for k up to ~8192.
//
// Without VNNI (AVX2, AVX-512BW) the byte-sum is emulated with vpmaddubsw, whose i16 pair sums saturate once an
// activation is inside the pair. The accumulate therefore keeps the pair x-free: sum the product bytes once per k-row
// into i16 pair-sums <= 510, then multiply by x with one vpmaddwd per token row (bit-exact, ~2x the masked accumulate).
//
// The GEMV streams k-major with a contiguous band of output tiles held in register accumulators per worker, so cold
// expert weights are read near-sequentially from DRAM (measured 3.4x over per-output-column traversal on cold stacks).
//
// The build compiles one tier, its -march's (kIsa); code holding a tier's instructions sits under that tier's
// SGLANG_CPU_EXPERTS_* flag (isa.hpp), so a lower build never sees it.
#pragma once
#include "quant.hpp"
#if defined(SGLANG_CPU_EXPERTS_AVX2)
#include <immintrin.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace sglang::exl3_cpu {
namespace {

// The tier every plan runs at: the build's (Exl3Quant::kTopIsa is the highest tier).
constexpr Isa kIsa = kBuildIsa;
static_assert(Exl3Quant::kTopIsa == Isa::Vbmi);

#ifndef EXL3_MOE_CPU_MAX_M
#define EXL3_MOE_CPU_MAX_M 4
#endif
// Quantized activation rows per chunk: CHUNK_M tokens times kActRows. Every per-chunk array, stride and tile table
// derives from it. A build raises it with -DEXL3_MOE_CPU_MAX_M (SGLANG_EXL3_CPU_MAX_M; the bench's EXL3_MAX_M).
constexpr int MAX_M = EXL3_MOE_CPU_MAX_M;
static_assert(MAX_M >= 2 && MAX_M <= 16 && MAX_M % 2 == 0, "EXL3_MOE_CPU_MAX_M must be even, in [2, 16]");

// Activation quantization: each token row is quantized to int8 in blocks of kActBlock inputs (one scale per block), and
// its residual, the row minus its dequantized self, is quantized again as a second row (the tin row r + m, beside the
// token rows). Every layer's hidden and intermediate are multiples of kActBlock (Exl3Quant::validate).
constexpr int kActRows = 2;
constexpr int kActBlock = 128;
constexpr int CHUNK_M = MAX_M / kActRows;
// The generic plan's quantized rows per chunk: MAX_M up to 8, whatever the build's MAX_M. Its tile kernels are
// instantiated per row count, so a larger bound only lengthens the build; the DSV4.1 plan alone takes chunks past 4
// tokens (register_tiles<M>). Exl3Quant::dispatch splits a generic call at GENERIC_CHUNK_M.
constexpr int GENERIC_MAX_M = MAX_M < 8 ? MAX_M : 8;
constexpr int GENERIC_CHUNK_M = GENERIC_MAX_M / kActRows;

constexpr uint32_t kMul1Mult = 0x83DCD12Du;
constexpr float kHadScale = 0.088388347648f;
// 1/K of the mul1 decode: the fp16 0x1eee.
constexpr float kMul1KInv = 0x1.bb8p-8f;

#define EXL3_ALWAYS_INLINE __attribute__((always_inline))

// Unroll<N>{}(f, args...) calls f(std::integral_constant<int, I>{}, args...) for I = 0 .. N - 1 in order (nothing for
// N == 0): a loop the compiler sees fully unrolled.
template <int N>
struct Unroll {
    template <class F, class... Args>
    EXL3_ALWAYS_INLINE void operator()(F&& f, Args&&... args) const
    {
        [&]<int... I>(std::integer_sequence<int, I...>) EXL3_ALWAYS_INLINE {
            (f(std::integral_constant<int, I>{}, args...), ...);
        }(std::make_integer_sequence<int, N>{});
    }
};

// f.template operator()<n>() for a run-time n in [1, Max] (nothing for any other n): one direct call per value.
template <int Max, class F>
EXL3_ALWAYS_INLINE inline void with_count(int n, F&& f)
{
    [&]<int... I>(std::integer_sequence<int, I...>) EXL3_ALWAYS_INLINE {
        (void)((n == I + 1 && (f.template operator()<I + 1>(), true)) || ...);
    }(std::make_integer_sequence<int, Max>{});
}

// A violated precondition of the tile kernels (the plans only call them with whole tiles); out of line and cold.
[[noreturn, gnu::cold, gnu::noinline]] inline void check_failed(const char* message)
{
    throw std::invalid_argument(message);
}
#define EXL3_CHECK(cond, message) \
    do { if (!(cond)) [[unlikely]] check_failed(message); } while (0)

// The GEMV tiles of one tier (math_scalar.hpp, math_avx2.hpp, math_avx512.hpp): run(mat, in, tout, rows, tn0, tn1)
// writes output tiles [tn0, tn1) of `rows` quantized rows (the scalar tier: token rows of in.tin).
template <Isa I>
struct Tiles;

// -------------------------------------------------------------------------------------------
//   Format tables
// -------------------------------------------------------------------------------------------

// The tensor-core permutation baked into the EXL3 tile storage format.
constexpr std::array<uint16_t, 256> make_tc_perm()
{
    std::array<uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t) {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = r0 * 16 + c0;
        p[t * 8 + 1] = r1 * 16 + c0;
        p[t * 8 + 2] = r2 * 16 + c0;
        p[t * 8 + 3] = r3 * 16 + c0;
        p[t * 8 + 4] = r0 * 16 + c1;
        p[t * 8 + 5] = r1 * 16 + c1;
        p[t * 8 + 6] = r2 * 16 + c1;
        p[t * 8 + 7] = r3 * 16 + c1;
    }
    return p;
}

constexpr std::array<uint16_t, 256> make_tc_perm_inv()
{
    std::array<uint16_t, 256> inv{};
    const auto perm = make_tc_perm();
    for (int i = 0; i < 256; ++i) inv[perm[i]] = i;
    return inv;
}

// For each column of tile row `row`: the 32-bit word holding its state's first (second_word: last) bits.
template <int bits, int row, bool second_word>
constexpr std::array<int32_t, 16> make_row_indices()
{
    std::array<int32_t, 16> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col) {
        const int t = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        idx[col] = (second_word ? (b1 - 1) / 32 : b0 / 32) % words32;
    }
    return idx;
}

template <int bits, int row, bool second_word>
constexpr uint16_t make_row_himask()
{
    uint16_t mask = 0;
    const auto idx = make_row_indices<bits, row, second_word>();
    for (int col = 0; col < 16; ++col)
        if (idx[col] >= 32) mask |= uint16_t(1) << col;
    return mask;
}

// The funnel shift that brings column `col`'s state of tile row `row` to the low 16 bits.
template <int bits, int row>
constexpr int row_shift(int col)
{
    const auto inv = make_tc_perm_inv();
    const int t = inv[row * 16 + col];
    const int b1 = t * bits + bits + 256 * bits;
    return ((b1 - 1) / 32 + 1) * 32 - b1;
}

// Word-level row pairing: rows 2p/2p+1 differ by exactly `bits` in bit position, so when both half-row shifts of the
// even row have >= bits of headroom, one word gather serves both rows (the odd row is the same gather shifted by an
// extra `bits`). It pays only where the saved permutes beat the unchanged shift-merge epilogue: measured +7% K1/K2, +6%
// K5, -2..-5% K4/K6/K7 (7960X); even the gated-off pair-step body costs ~2% on K4, so other K keep the row-by-row body.
template <int bits, int row>
constexpr bool word_pair_ok()
{
    return row_shift<bits, row>(0) >= bits && row_shift<bits, row>(8) >= bits;
}

template <int bits>
constexpr bool dword_pair_wins()
{
    return bits == 1 || bits == 2 || bits == 5;
}

inline uint32_t load_u32(const uint16_t* ptr, int index)
{
    uint32_t v;
    std::memcpy(&v, ptr + index * 2, sizeof(v));
    return v;
}

template <int bits>
inline uint16_t decode_state_scalar(const uint16_t* packed, int t_offset)
{
    constexpr int words32 = bits * 256 / 32;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const uint64_t merged = (static_cast<uint64_t>(load_u32(packed, (b0 / 32) % words32)) << 32)
                            | load_u32(packed, ((b1 - 1) / 32) % words32);
    return static_cast<uint16_t>(merged >> shift);
}

inline float decode_mul1_scalar(uint16_t state)
{
    const uint32_t x = static_cast<uint32_t>(state) * kMul1Mult;
    const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    return (static_cast<float>(sum) - 510.0f) * kMul1KInv;
}

// -------------------------------------------------------------------------------------------
//   Hadamard transform (128 inputs)
// -------------------------------------------------------------------------------------------

void hadamard_128_scalar(float* v)
{
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i) {
                const float a = v[base + i];
                const float b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

#if defined(SGLANG_CPU_EXPERTS_AVX512)
template <int Width>
EXL3_ALWAYS_INLINE inline void butterfly_registers(__m512 (&r)[8])
{
#pragma GCC unroll 8
    for (int b = 0; b < 8; b += 2 * Width) {
#pragma GCC unroll 8
        for (int i = 0; i < Width; ++i) {
            const __m512 a = r[b + i], z = r[b + Width + i];
            r[b + i] = _mm512_add_ps(a, z);
            r[b + Width + i] = _mm512_sub_ps(a, z);
        }
    }
}

__attribute__((noipa, optimize("no-associative-math"))) void hadamard_128(float* v)
{
    __m512 r[8];
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) r[i] = _mm512_loadu_ps(v + 16 * i);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        __m512 t = _mm512_permute_ps(r[i], 0xb1);
        r[i] = _mm512_mask_sub_ps(_mm512_add_ps(r[i], t), 0xaaaa, t, r[i]);
        t = _mm512_permute_ps(r[i], 0x4e);
        r[i] = _mm512_mask_sub_ps(_mm512_add_ps(r[i], t), 0xcccc, t, r[i]);
        t = _mm512_shuffle_f32x4(r[i], r[i], 0xb1);
        r[i] = _mm512_mask_sub_ps(_mm512_add_ps(r[i], t), 0xf0f0, t, r[i]);
        t = _mm512_shuffle_f32x4(r[i], r[i], 0x4e);
        r[i] = _mm512_mask_sub_ps(_mm512_add_ps(r[i], t), 0xff00, t, r[i]);
    }
    butterfly_registers<1>(r);
    butterfly_registers<2>(r);
    butterfly_registers<4>(r);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) _mm512_storeu_ps(v + 16 * i, r[i]);
}
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
void hadamard_128(float* v)
{
    __m256 r[16];
    for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + i * 8);
    // Widths 1, 2 and 4: butterflies within a register (adjacent pairs, 64-bit pairs, 128-bit halves).
    for (int i = 0; i < 16; ++i) {
        const __m256 t = _mm256_permute_ps(r[i], 0b10110001);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b10101010);
    }
    for (int i = 0; i < 16; ++i) {
        const __m256 t = _mm256_permute_ps(r[i], 0b01001110);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11001100);
    }
    for (int i = 0; i < 16; ++i) {
        const __m256 t = _mm256_permute2f128_ps(r[i], r[i], 0x01);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11110000);
    }
    // Widths 8..64: whole-register butterflies.
    for (int width = 1; width < 16; width *= 2)
        for (int base = 0; base < 16; base += 2 * width)
            for (int i = 0; i < width; ++i) {
                const __m256 a = r[base + i];
                const __m256 b = r[base + width + i];
                r[base + i] = _mm256_add_ps(a, b);
                r[base + width + i] = _mm256_sub_ps(a, b);
            }
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + i * 8, r[i]);
}
#else
inline void hadamard_128(float* v) { hadamard_128_scalar(v); }
#endif

// -------------------------------------------------------------------------------------------
//   Prepared input (per GEMV, per chunk)
// -------------------------------------------------------------------------------------------

// One chunk's input to a projection's GEMV: m token rows, transformed (suh, Hadamard), then quantized (vector tiers).
// Quantized rows are laid out per block: block b of quantized row i (token rows 0..m-1, then their residual rows
// m..2m-1) at (b * 2m + i) * kActBlock, its scale and sum at bq / bsum[b * MAX_M + i].
struct PreparedIn
{
    float* tin;          // [2m][k] transformed fp32: token rows, then residual rows; the scalar tier reads the first m
    int32_t* splat32;    // the int8 activation replicated x4 (vpdpbusd's signed operand: the Vnni and Vbmi tiles)
    // x8 (two's-complement, read as i16) in both 16-bit halves of the dword: one vpmaddwd per token row against the
    // pair-summed product bytes (the Avx2 and Bw tiles). Null where no kernel reads it.
    int32_t* splat_dup;
    int16_t* compact;    // the DSV4.1 plan's int16 activations in place of the splats (register_kernels.hpp); or null
    float* bq;
    int32_t* bsum;
    // One block's scales and sums, as run_tiles hands a block's view to the tile kernels.
    float q[MAX_M];
    int32_t sum_x8[MAX_M];
};

// The tiers whose tile kernels read PreparedIn::splat_dup.
constexpr bool kSplatDup = kIsa == Isa::Avx2 || kIsa == Isa::Bw;

#if defined(SGLANG_CPU_EXPERTS_AVX2)
// Quantizes k inputs of src to int8 at one scale q (amax / 127): splat gets each x4, splat_dup (when non-null) each in
// both 16-bit halves; s_out is the activations' sum.
void quantize_row(const float* src, int32_t* splat, int32_t* splat_dup, int k, float& q_out, int32_t& s_out)
{
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k; i += 8) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(src + i)));
    alignas(32) float mx[8];
    _mm256_store_ps(mx, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 8; ++i) amax = std::max(amax, mx[i]);
    const float q = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m256 rq = _mm256_set1_ps(1.0f / q);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    const __m256i rep = _mm256_set1_epi32(0x01010101);
    const __m256i mask8 = _mm256_set1_epi32(0xff);
    __m256i vsum = _mm256_setzero_si256();
    for (int i = 0; i < k; i += 8) {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(src + i), rq));
        v = _mm256_min_epi32(hi, _mm256_max_epi32(lo, v));
        vsum = _mm256_add_epi32(vsum, v);
        const __m256i b = _mm256_and_si256(v, mask8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(splat + i), _mm256_mullo_epi32(b, rep));
        // The low 16 bits of v are already the two's-complement i16 activation; replicate them into the high half.
        if (splat_dup) {
            const __m256i low16 = _mm256_and_si256(v, _mm256_set1_epi32(0xffff));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(splat_dup + i),
                                _mm256_or_si256(low16, _mm256_slli_epi32(low16, 16)));
        }
    }
    alignas(32) int32_t sm[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sm), vsum);
    s_out = sm[0] + sm[1] + sm[2] + sm[3] + sm[4] + sm[5] + sm[6] + sm[7];
    q_out = q;
}

#if defined(SGLANG_CPU_EXPERTS_AVX512)
// The DSV4.1 plan's compact form of quantize_row over one block, at 256 bits (AVX-512VL's vpmovdw): int16 activations,
// and the residual (src minus its dequantized self) into `residual` when non-null.
void compact_quantize_row(const float* src, int16_t* dst, float* residual, float& q_out, int32_t& sum_out)
{
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < 128; i += 8) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(src + i)));
    alignas(32) float mx[8];
    _mm256_store_ps(mx, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 8; ++i) amax = std::max(amax, mx[i]);
    const float q = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m256 rq = _mm256_set1_ps(1.0f / q), qv = _mm256_set1_ps(q);
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    __m256i vsum = _mm256_setzero_si256();
    for (int i = 0; i < 128; i += 8) {
        const __m256 x = _mm256_loadu_ps(src + i);
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(x, rq));
        v = _mm256_min_epi32(hi, _mm256_max_epi32(lo, v));
        vsum = _mm256_add_epi32(vsum, v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm256_cvtepi32_epi16(v));
        if (residual) _mm256_storeu_ps(residual + i, _mm256_fnmadd_ps(qv, _mm256_cvtepi32_ps(v), x));
    }
    alignas(32) int32_t sm[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(sm), vsum);
    sum_out = sm[0] + sm[1] + sm[2] + sm[3] + sm[4] + sm[5] + sm[6] + sm[7];
    q_out = q;
}

// compact_quantize_row at 512 bits: the DSV4.1 plan's one-token, one-expert forward (PlanTraits::kWideSingleExpert).
void compact_quantize_row512(const float* src, int16_t* dst, float* residual, float& q_out, int32_t& sum_out)
{
    __m512 vmax = _mm512_setzero_ps();
    const __m512 sign = _mm512_set1_ps(-0.0f);
    for (int i = 0; i < 128; i += 16) vmax = _mm512_max_ps(vmax, _mm512_andnot_ps(sign, _mm512_loadu_ps(src + i)));
    const float amax = _mm512_reduce_max_ps(vmax);
    const float q = amax > 0.0f ? amax / 127.0f : 1.0f;
    const __m512 rq = _mm512_set1_ps(1.0f / q), qv = _mm512_set1_ps(q);
    const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
    __m512i vsum = _mm512_setzero_si512();
    for (int i = 0; i < 128; i += 16) {
        const __m512 x = _mm512_loadu_ps(src + i);
        __m512i v = _mm512_cvtps_epi32(_mm512_mul_ps(x, rq));
        v = _mm512_min_epi32(hi, _mm512_max_epi32(lo, v));
        vsum = _mm512_add_epi32(vsum, v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm512_cvtepi32_epi16(v));
        if (residual) _mm512_storeu_ps(residual + i, _mm512_fnmadd_ps(qv, _mm512_cvtepi32_ps(v), x));
    }
    sum_out = _mm512_reduce_add_epi32(vsum);
    q_out = q;
}
#endif

// Quantizes block `block` of token row r (of the chunk's m) from src, the row's transformed block: the token row, then
// its residual, which is staged in tin row r + m. Wide: the compact rows at 512 bits.
template <bool Wide>
void quantize_block(PreparedIn& p, int r, int m, int k, int block, const float* src)
{
    float* residual = p.tin + size_t(r + m) * k + block * kActBlock;
    for (int pass = 0; pass < kActRows; ++pass) {
        const int row = r + pass * m;
        const size_t off = size_t(block) * kActRows * m * kActBlock + size_t(row) * kActBlock;
#if defined(SGLANG_CPU_EXPERTS_AVX512)
        if (p.compact) {
            float* res = pass ? nullptr : residual;
            if constexpr (Wide)
                compact_quantize_row512(pass ? residual : src, p.compact + off, res, p.bq[block * MAX_M + row],
                                        p.bsum[block * MAX_M + row]);
            else
                compact_quantize_row(pass ? residual : src, p.compact + off, res, p.bq[block * MAX_M + row],
                                     p.bsum[block * MAX_M + row]);
            continue;
        }
#endif
        const float* v = pass ? residual : src;
        int32_t* splat = p.splat32 + off;
        float q;
        int32_t sum;
        quantize_row(v, splat, p.splat_dup ? p.splat_dup + off : nullptr, kActBlock, q, sum);
        p.bq[block * MAX_M + row] = q;
        p.bsum[block * MAX_M + row] = sum;
        if (pass + 1 < kActRows)
            for (int i = 0; i < kActBlock; ++i)
                residual[i] = v[i] - q * float(static_cast<int8_t>(splat[i] & 0xff));
    }
}

// One 128-input block of an activation row, transformed: dst = Hadamard(src * suh) * kHadScale. src is fp16 or fp32.
void prepare_block(const void* srcv, bool f16, const Fp16* suh, float* dst)
{
    const __m256 hs = _mm256_set1_ps(kHadScale);
    for (int i = 0; i < 128; i += 8) {
        __m256 x;
        if (f16)
            x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const uint16_t*>(srcv) + i)));
        else
            x = _mm256_loadu_ps(static_cast<const float*>(srcv) + i);
        const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(suh + i)));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, s));
    }
    hadamard_128(dst);
    for (int i = 0; i < 128; i += 8) _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), hs));
}
#endif  // SGLANG_CPU_EXPERTS_AVX2

// The scalar tier's input: m rows gathered by token index from src (fp16 or fp32, one of them non-null), transformed
// into p.tin. Its tiles read the fp32 rows themselves, so nothing is quantized.
[[maybe_unused]] void prepare_rows_scalar(const Exl3Projection& mat, const Fp16* src_f16, const float* src_f32, int src_stride,
                         const int* token_idx, int m, PreparedIn& p)
{
    const int k = mat.k;
    for (int r = 0; r < m; ++r) {
        float* dst = p.tin + size_t(r) * k;
        const size_t src_off = size_t(token_idx[r]) * src_stride;
        for (int block = 0; block < k; block += 128) {
            float vals[128];
            for (int i = 0; i < 128; ++i) {
                const float xv = src_f16 ? fp16_to_float(src_f16[src_off + block + i]) : src_f32[src_off + block + i];
                vals[i] = xv * fp16_to_float(mat.suh[block + i]);
            }
            hadamard_128_scalar(vals);
            for (int i = 0; i < 128; ++i) dst[block + i] = vals[i] * kHadScale;
        }
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
