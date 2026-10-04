// EXL3's arithmetic every tier shares: the format tables and state decode, the Hadamard transforms and the
// activation quantization. Each tier's GEMV tiles are math_scalar.hpp, math_avx2.hpp and math_avx512.hpp.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
//
// CPU MoE expert GEMM for mul1 EXL3 tensors.
//
// The mul1 codebook is affine in a byte-sum: w(s) = (bytesum(s * 0x83DCD12D) - 510) * k_inv with
// k_inv = fp16(0x1eee). With int8 activations x8 and one scale q per input row:
//   sum_k w_kn x_k = k_inv * q * ( sum_k bytesum(s_kn * M) * x8_k  -  510 * sum_k x8_k )
// and bytesum(s*M) * x8 is exactly one AVX-512 VNNI vpdpbusd per 16 weights (unsigned operand =
// product bytes, signed operand = the int8 activation replicated x4; u8*s8 word products stay
// below 2^15, which makes the operand order load-bearing). Accuracy matches the GPU int8-GEMV
// mode-2 class (~0.9% per-call output RMS). i32 accumulators are safe for k up to ~8192.
//
// On the AVX2 and AVX-512BW tiers (no VNNI) the byte-sum is emulated with vpmaddubsw, whose i16 pair sums
// saturate once an activation is inside the pair. The accumulate therefore keeps the pair x-free:
// sum the product bytes once per k-row into i16 pair-sums <= 510, then multiply by x with one
// vpmaddwd per token row. +-127 activations, bit-exact vs the masked accumulate it replaces,
// ~2x its throughput (measured on Zen 3 + Zen 5).
//
// State extraction uses compile-time (bits, row) index tables for vpermt2var plus immediate
// funnel shifts, following benchmarks/exl3_cpu_gemm. The GEMV streams k-major with a contiguous
// band of output tiles held in register accumulators per worker, so cold expert weights are read
// near-sequentially from DRAM (measured 3.4x over per-output-column traversal on cold stacks).
//
// No generic lambdas inside target-attributed functions (GCC does not let lambdas inherit the
// target), hence the recursive-template row unrolling.

#pragma once
#include "quant.hpp"
#include <c10/util/Half.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sglang::exl3_cpu {
namespace {

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr float HAD_SCALE = 0.088388347648f;
constexpr int MAX_M = 4;

#ifndef EXL3_MOE_CPU_ACT_RESIDUAL
#define EXL3_MOE_CPU_ACT_RESIDUAL 0
#endif
#ifndef EXL3_MOE_CPU_ACT_BLOCK
#define EXL3_MOE_CPU_ACT_BLOCK 0
#endif
static_assert(EXL3_MOE_CPU_ACT_BLOCK >= 0 && EXL3_MOE_CPU_ACT_BLOCK % 16 == 0,
              "EXL3_MOE_CPU_ACT_BLOCK must be 0 or a multiple of 16");
// Quantized activation rows per token row, and so the most tokens a chunk can hold
constexpr int ACT_ROWS = EXL3_MOE_CPU_ACT_RESIDUAL ? 2 : 1;
constexpr int CHUNK_M = MAX_M / ACT_ROWS;

inline bool act_blocked(int k)
{
    return EXL3_MOE_CPU_ACT_BLOCK != 0 && k % EXL3_MOE_CPU_ACT_BLOCK == 0 && k > EXL3_MOE_CPU_ACT_BLOCK;
}

// k-tiles of the full matrix for the swizzled address of a k-block sub-view (0: the view is the
// whole matrix). Set by run_tiles around its per-block calls, read once per band call.
thread_local int tl_swz_tiles_k = 0;

#if defined(__GNUC__) && defined(__linux__)
#define M1_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define M1_TARGET_BW __attribute__((target("avx512f,avx512bw,avx512vl,fma,f16c")))
#define M1_TARGET_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,fma,f16c")))
#define M1_TARGET_VBMI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#define M1_ALWAYS_INLINE __attribute__((always_inline)) inline
#else
#define M1_TARGET_AVX2
#define M1_TARGET_BW
#define M1_TARGET_VNNI
#define M1_TARGET_VBMI
#define M1_ALWAYS_INLINE __forceinline
#endif


inline float half_to_float(at::Half h) { return static_cast<float>(h); }

inline float mul1_k_inv()
{
    // fp16 0x1eee
    static const float v = half_to_float(c10::Half(uint16_t(0x1eee), c10::Half::from_bits()));
    return v;
}

// -------------------------------------------------------------------------------------------
//   Format tables
// -------------------------------------------------------------------------------------------

// Matches tensor-core permutation baked into by EXL3 tile storage format
constexpr std::array<uint16_t, 256> make_tc_perm()
{
    std::array<uint16_t, 256> p{};
    #pragma unroll
    for (int t = 0; t < 32; ++t)
    {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = r0 * 16 + c0; p[t * 8 + 1] = r1 * 16 + c0;
        p[t * 8 + 2] = r2 * 16 + c0; p[t * 8 + 3] = r3 * 16 + c0;
        p[t * 8 + 4] = r0 * 16 + c1; p[t * 8 + 5] = r1 * 16 + c1;
        p[t * 8 + 6] = r2 * 16 + c1; p[t * 8 + 7] = r3 * 16 + c1;
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

template <int bits, int row>
constexpr int row_shift(int col)
{
    const auto inv = make_tc_perm_inv();
    const int t = inv[row * 16 + col];
    const int b1 = t * bits + bits + 256 * bits;
    return ((b1 - 1) / 32 + 1) * 32 - b1;
}

inline uint32_t load_u32_(const uint16_t* ptr, int index)
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
    const uint64_t merged = (static_cast<uint64_t>(load_u32_(packed, (b0 / 32) % words32)) << 32) |
                            load_u32_(packed, ((b1 - 1) / 32) % words32);
    return static_cast<uint16_t>(merged >> shift);
}

inline float decode_mul1_scalar(uint16_t state)
{
    const uint32_t x = static_cast<uint32_t>(state) * MUL1_MULT;
    const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    return (static_cast<float>(sum) - 510.0f) * mul1_k_inv();
}

// Word-level row pairing: rows 2p/2p+1 differ by exactly `bits` in bit position, so when both
// half-row shifts of the even row have >= bits of headroom, one word gather serves both rows
// (the odd row is the same gather shifted by an extra `bits`). AVX-512's 32 registers absorb
// the extra live values; the identical restructure measured a net LOSS on 16-register AVX2
// (spills), and on the dword path it only pays where the saved permutes beat the unchanged
// shift-merge epilogue: measured +7% K1/K2, +6% K5, -2..-5% K4/K6/K7 (7960X). Even the
// gated-off pair-step body costs ~2% on K4 (deferred dpbusd changes the dependency chains), so
// non-winning K keep the original row-by-row body verbatim.
template <int bits, int row>
constexpr bool word_pair_ok()
{
    return row_shift<bits, row>(0) >= bits && row_shift<bits, row>(8) >= bits;
}

template <int bits>
constexpr bool dword_pair_wins() { return bits == 1 || bits == 2 || bits == 5; }

// -------------------------------------------------------------------------------------------
//   ISA dispatch
// -------------------------------------------------------------------------------------------

// Isa and kAvx512 are cpu_experts_common/isa.hpp's. Vbmi = Vnni + AVX512-VBMI (Zen4+, Ice
// Lake+); kept as a separate tier because Cascade/Cooper Lake have VNNI without VBMI. Bw =
// AVX-512F/BW/VL without VNNI (Skylake-SP/X): the dword kernel with the AVX2-style accumulate.
// The forward is instantiated per tier; ExpertForward's detected tier picks the instantiation once per call
// (Exl3Quant::dispatch).

// -------------------------------------------------------------------------------------------
//   Transforms
// -------------------------------------------------------------------------------------------

void hadamard_128_scalar(float* v)
{
    #pragma unroll
    for (int width = 1; width < 128; width *= 2)
        #pragma unroll
        for (int base = 0; base < 128; base += 2 * width)
            #pragma unroll
            for (int i = 0; i < width; ++i) {
                const float a = v[base + i];
                const float b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

M1_TARGET_AVX2
void hadamard_128_current(float* v)
{
    __m256 r[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = _mm256_loadu_ps(v + i * 8);

    // width 1: butterfly within adjacent pairs
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b10110001);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b10101010);
    }

    // width 2: butterfly between 64-bit pairs
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute_ps(r[i], 0b01001110);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11001100);
    }

    // width 4: butterfly between 128-bit halves
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        const __m256 t = _mm256_permute2f128_ps(r[i], r[i], 0x01);
        r[i] = _mm256_blend_ps(_mm256_add_ps(r[i], t), _mm256_sub_ps(t, r[i]), 0b11110000);
    }

    // widths 8..64: whole-register butterflies
    #pragma unroll
    for (int width = 1; width < 16; width *= 2)
        #pragma unroll
        for (int base = 0; base < 16; base += 2 * width)
            #pragma unroll
            for (int i = 0; i < width; ++i)
            {
                const __m256 a = r[base + i];
                const __m256 b = r[base + width + i];
                r[base + i] = _mm256_add_ps(a, b);
                r[base + width + i] = _mm256_sub_ps(a, b);
            }

    #pragma unroll
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(v + i * 8, r[i]);
}
template<int Width>
__attribute__((always_inline)) inline void butterfly_registers(__m512 (&r)[8]) {
    #pragma GCC unroll 8
    for (int b=0; b<8; b+=2*Width) {
        #pragma GCC unroll 8
        for (int i=0; i<Width; ++i) {
            const __m512 a=r[b+i], z=r[b+Width+i];
            r[b+i]=_mm512_add_ps(a,z);
            r[b+Width+i]=_mm512_sub_ps(a,z);
        }
    }
}

M1_TARGET_BW __attribute__((noipa,optimize("no-associative-math"))) void hadamard_512(float* v) {
    __m512 r[8];
    #pragma GCC unroll 8
    for (int i=0; i<8; ++i) r[i]=_mm512_loadu_ps(v+16*i);
    #pragma GCC unroll 8
    for (int i=0; i<8; ++i) {
        __m512 t=_mm512_permute_ps(r[i],0xb1);
        r[i]=_mm512_mask_sub_ps(_mm512_add_ps(r[i],t),0xaaaa,t,r[i]);
        t=_mm512_permute_ps(r[i],0x4e);
        r[i]=_mm512_mask_sub_ps(_mm512_add_ps(r[i],t),0xcccc,t,r[i]);
        t=_mm512_shuffle_f32x4(r[i],r[i],0xb1);
        r[i]=_mm512_mask_sub_ps(_mm512_add_ps(r[i],t),0xf0f0,t,r[i]);
        t=_mm512_shuffle_f32x4(r[i],r[i],0x4e);
        r[i]=_mm512_mask_sub_ps(_mm512_add_ps(r[i],t),0xff00,t,r[i]);
    }
    butterfly_registers<1>(r);
    butterfly_registers<2>(r);
    butterfly_registers<4>(r);
    #pragma GCC unroll 8
    for (int i=0; i<8; ++i) _mm512_storeu_ps(v+16*i,r[i]);
}

template <Isa I>
M1_TARGET_AVX2 void hadamard_128_avx2(float* v) {
    if constexpr (kAvx512<I>) hadamard_512(v);
    else hadamard_128_current(v);
}


template <Isa I>
inline void hadamard_128(float* v)
{
    if constexpr (I != Isa::Scalar) hadamard_128_avx2<I>(v);
    else                            hadamard_128_scalar(v);
}

// -------------------------------------------------------------------------------------------
//   Prepared input (per GEMV, per chunk)
// -------------------------------------------------------------------------------------------

struct PreparedIn
{
    float* tin;         // m x k, transformed fp32 (scalar kernel)
    int32_t* splat32;   // m x k, int8 activation replicated x4
    // m x k, x8 (two's-complement, read as i16) in BOTH 16-bit slots of the dword. AVX2
    // bytesum-first accumulate: the product-byte pair sums (b0+b1, b2+b3, each <= 510) are
    // formed once per k-row with x OUTSIDE the pair (so no saturation at full +-127), then a
    // single vpmaddwd per token row against this pattern returns bytesum*x as one i32.
    // Bit-exact reassociation of the same integer sum: 4 ops/row + 4 shared, vs 16 (masked)
    // or 12 (activation-split). VNNI/VBMI ignore.
    int32_t* splat_dup;
    float q[MAX_M];
    int32_t sum_x8[MAX_M];
    // EXL3_MOE_CPU_ACT_BLOCK: per-block scales and sums, [k / B][MAX_M]
    float* bq = nullptr;
    int32_t* bsum = nullptr;
    int16_t* compact = nullptr;
};

M1_TARGET_AVX2
void quantize_row_avx2(const float* dst, int32_t* splat, int32_t* splat_dup,
    int k, float& q_out, int32_t& s_out)
{
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(dst + i)));
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
    #pragma unroll
    for (int i = 0; i < k; i += 8)
    {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(dst + i), rq));
        v = _mm256_min_epi32(hi, _mm256_max_epi32(lo, v));
        vsum = _mm256_add_epi32(vsum, v);
        const __m256i b = _mm256_and_si256(v, mask8);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(splat + i), _mm256_mullo_epi32(b, rep));
        // x in both 16-bit slots (see PreparedIn::splat_dup): the low 16 bits of v are already
        // the two's-complement i16 activation; replicate into the high slot
        if (splat_dup)
        {
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

// Quantize token row r of m (quantized tiers) under the accuracy options. Layout:
//   rows: ACT_ROWS per token row; the remainder row of token r is row r + m;
//   one scale per row: row i at i * k, scales in q[i] / sum_x8[i];
//   per block (act_blocked): block b of row i at b * rows * B + i * B, scales at bq / bsum[b * MAX_M + i].
// The remainder is staged in tin row r + m, which the quantized tiers never read.
void quantize_act(PreparedIn& p, int r, int m, int k, const float* src, bool dup)
{
    const int rows = ACT_ROWS * m;
    const int B = act_blocked(k) ? EXL3_MOE_CPU_ACT_BLOCK : k;
    float* res = p.tin + static_cast<size_t>(r + m) * k;
    for (int pass = 0; pass < ACT_ROWS; ++pass)
    {
        const int row = r + pass * m;
        const float* v = pass ? res : src;
        for (int b = 0; b < k / B; ++b)
        {
            const size_t off = B == k ? static_cast<size_t>(row) * k
                                      : static_cast<size_t>(b) * rows * B + static_cast<size_t>(row) * B;
            int32_t* splat = p.splat32 + off;
            float q;
            int32_t s;
            quantize_row_avx2(v + b * B, splat, dup ? p.splat_dup + off : nullptr, B, q, s);
            if (B == k) { p.q[row] = q; p.sum_x8[row] = s; }
            else { p.bq[b * MAX_M + row] = q; p.bsum[b * MAX_M + row] = s; }
            if (pass + 1 < ACT_ROWS)
                for (int i = 0; i < B; ++i)
                    res[b * B + i] = v[b * B + i] - q * static_cast<float>(static_cast<int8_t>(splat[i] & 0xff));
        }
    }
}

template <Isa I>
M1_TARGET_AVX2
void prepare_block_avx2(const void* srcv, bool f16, const at::Half* suh, float* dst, int k)
{
    const __m256 hs = _mm256_set1_ps(HAD_SCALE);
    for (int block = 0; block < k; block += 128)
    {
        #pragma unroll
        for (int i = 0; i < 128; i += 8)
        {
            __m256 x;
            if (f16)
                x = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    static_cast<const uint16_t*>(srcv) + block + i)));
            else
                x = _mm256_loadu_ps(static_cast<const float*>(srcv) + block + i);
            const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                reinterpret_cast<const uint16_t*>(suh) + block + i)));
            _mm256_storeu_ps(dst + block + i, _mm256_mul_ps(x, s));
        }
        hadamard_128_avx2<I>(dst + block);

        #pragma unroll
        for (int i = 0; i < 128; i += 8)
            _mm256_storeu_ps(dst + block + i,
                             _mm256_mul_ps(_mm256_loadu_ps(dst + block + i), hs));
    }
}



M1_TARGET_BW
void compact_quantize_row_reference(const float* src,int16_t* dst,float* residual,float& q_out,int32_t& sum_out) {
    __m256 vmax=_mm256_setzero_ps();
    const __m256 sign=_mm256_set1_ps(-0.0f);
    for(int i=0;i<128;i+=8)
        vmax=_mm256_max_ps(vmax,_mm256_andnot_ps(sign,_mm256_loadu_ps(src+i)));
    alignas(32) float mx[8];_mm256_store_ps(mx,vmax);
    float amax=0.0f;for(int i=0;i<8;++i)amax=std::max(amax,mx[i]);
    const float q=amax>0.0f?amax/127.0f:1.0f;
    const __m256 rq=_mm256_set1_ps(1.0f/q),qv=_mm256_set1_ps(q);
    const __m256i lo=_mm256_set1_epi32(-127),hi=_mm256_set1_epi32(127);
    __m256i vsum=_mm256_setzero_si256();
    for(int i=0;i<128;i+=8) {
        const __m256 x=_mm256_loadu_ps(src+i);
        __m256i v=_mm256_cvtps_epi32(_mm256_mul_ps(x,rq));
        v=_mm256_min_epi32(hi,_mm256_max_epi32(lo,v));
        vsum=_mm256_add_epi32(vsum,v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+i),_mm256_cvtepi32_epi16(v));
        if(residual) _mm256_storeu_ps(residual+i,_mm256_fnmadd_ps(qv,_mm256_cvtepi32_ps(v),x));
    }
    alignas(32) int32_t sm[8];_mm256_store_si256(reinterpret_cast<__m256i*>(sm),vsum);
    sum_out=sm[0]+sm[1]+sm[2]+sm[3]+sm[4]+sm[5]+sm[6]+sm[7];q_out=q;
}

M1_TARGET_BW
void compact_quantize_row512_reference(const float* src,int16_t* dst,float* residual,float& q_out,int32_t& sum_out) {
    __m512 vmax=_mm512_setzero_ps();
    const __m512 sign=_mm512_set1_ps(-0.0f);
    for(int i=0;i<128;i+=16)
        vmax=_mm512_max_ps(vmax,_mm512_andnot_ps(sign,_mm512_loadu_ps(src+i)));
    const float amax=_mm512_reduce_max_ps(vmax);
    const float q=amax>0.0f?amax/127.0f:1.0f;
    const __m512 rq=_mm512_set1_ps(1.0f/q),qv=_mm512_set1_ps(q);
    const __m512i lo=_mm512_set1_epi32(-127),hi=_mm512_set1_epi32(127);
    __m512i vsum=_mm512_setzero_si512();
    for(int i=0;i<128;i+=16) {
        const __m512 x=_mm512_loadu_ps(src+i);
        __m512i v=_mm512_cvtps_epi32(_mm512_mul_ps(x,rq));
        v=_mm512_min_epi32(hi,_mm512_max_epi32(lo,v));
        vsum=_mm512_add_epi32(vsum,v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+i),_mm512_cvtepi32_epi16(v));
        if(residual)_mm512_storeu_ps(residual+i,_mm512_fnmadd_ps(qv,_mm512_cvtepi32_ps(v),x));
    }
    sum_out=_mm512_reduce_add_epi32(vsum);q_out=q;
}

M1_TARGET_BW
void compact_quantize_row(const float* src,int16_t* dst,float* residual,float& q_out,int32_t& sum_out) {
    __m256 vmax=_mm256_setzero_ps();
    const __m256 sign=_mm256_set1_ps(-0.0f);
    for(int i=0;i<128;i+=8)
        vmax=_mm256_max_ps(vmax,_mm256_andnot_ps(sign,_mm256_loadu_ps(src+i)));
    alignas(32) float mx[8];_mm256_store_ps(mx,vmax);
    float amax=0.0f;for(int i=0;i<8;++i)amax=std::max(amax,mx[i]);
    const float q=amax>0.0f?amax/127.0f:1.0f;
    const __m256 rq=_mm256_set1_ps(1.0f/q),qv=_mm256_set1_ps(q);
    const __m256i lo=_mm256_set1_epi32(-127),hi=_mm256_set1_epi32(127);
    __m256i vsum=_mm256_setzero_si256();
    for(int i=0;i<128;i+=8) {
        const __m256 x=_mm256_loadu_ps(src+i);
        __m256i v=_mm256_cvtps_epi32(_mm256_mul_ps(x,rq));
        v=_mm256_min_epi32(hi,_mm256_max_epi32(lo,v));
        vsum=_mm256_add_epi32(vsum,v);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst+i),_mm256_cvtepi32_epi16(v));
        if(residual) _mm256_storeu_ps(residual+i,_mm256_fnmadd_ps(qv,_mm256_cvtepi32_ps(v),x));
    }
    alignas(32) int32_t sm[8];_mm256_store_si256(reinterpret_cast<__m256i*>(sm),vsum);
    sum_out=sm[0]+sm[1]+sm[2]+sm[3]+sm[4]+sm[5]+sm[6]+sm[7];q_out=q;
}

M1_TARGET_BW
void compact_quantize_row512(const float* src,int16_t* dst,float* residual,float& q_out,int32_t& sum_out) {
    __m512 vmax=_mm512_setzero_ps();
    const __m512 sign=_mm512_set1_ps(-0.0f);
    for(int i=0;i<128;i+=16)
        vmax=_mm512_max_ps(vmax,_mm512_andnot_ps(sign,_mm512_loadu_ps(src+i)));
    const float amax=_mm512_reduce_max_ps(vmax);
    const float q=amax>0.0f?amax/127.0f:1.0f;
    const __m512 rq=_mm512_set1_ps(1.0f/q),qv=_mm512_set1_ps(q);
    const __m512i lo=_mm512_set1_epi32(-127),hi=_mm512_set1_epi32(127);
    __m512i vsum=_mm512_setzero_si512();
    for(int i=0;i<128;i+=16) {
        const __m512 x=_mm512_loadu_ps(src+i);
        __m512i v=_mm512_cvtps_epi32(_mm512_mul_ps(x,rq));
        v=_mm512_min_epi32(hi,_mm512_max_epi32(lo,v));
        vsum=_mm512_add_epi32(vsum,v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+i),_mm512_cvtepi32_epi16(v));
        if(residual)_mm512_storeu_ps(residual+i,_mm512_fnmadd_ps(qv,_mm512_cvtepi32_ps(v),x));
    }
    sum_out=_mm512_reduce_add_epi32(vsum);q_out=q;
}

template<bool Wide = false>
void compact_quantize_block(PreparedIn& p,int r,int m,int k,int block,const float* src) {
    float* residual=p.tin+size_t(r+m)*k+block*128;
    for(int pass=0;pass<ACT_ROWS;++pass) {
        const int row=r+pass*m;
        const size_t off=size_t(block)*ACT_ROWS*m*128+row*128;
        if constexpr (Wide)
            compact_quantize_row512(pass?residual:src,p.compact+off,pass?nullptr:residual,
                                   p.bq[block*MAX_M+row],p.bsum[block*MAX_M+row]);
        else
        compact_quantize_row(pass?residual:src,p.compact+off,pass?nullptr:residual,
                             p.bq[block*MAX_M+row],p.bsum[block*MAX_M+row]);
    }
}

// src_f16 / src_f32: one of them non-null; rows gathered by token index
template <Isa I>
void prepare_rows
(
    const MoeCpuMatrix& mat,
    const at::Half* src_f16, const float* src_f32, int src_stride,
    const int* token_idx, int m,
    PreparedIn& p
)
{
    const int k = mat.k;
    if(p.compact) {
        for(int r=0;r<m;++r)for(int b=0;b<k/128;++b) {
            const size_t off=size_t(token_idx[r])*src_stride+b*128;
            float* dst=p.tin+size_t(r)*k+b*128;
            prepare_block_avx2<I>(src_f16?static_cast<const void*>(src_f16+off):static_cast<const void*>(src_f32+off),
                                  src_f16!=nullptr,mat.suh+b*128,dst,128);
            compact_quantize_block(p,r,m,k,b,dst);
        }
        return;
    }
    for (int r = 0; r < m; ++r)
    {
        float* dst = p.tin + static_cast<size_t>(r) * k;
        const size_t src_off = static_cast<size_t>(token_idx[r]) * src_stride;
        if constexpr (I != Isa::Scalar)
        {
            prepare_block_avx2<I>(src_f16 ? reinterpret_cast<const void*>(src_f16 + src_off)
                                          : reinterpret_cast<const void*>(src_f32 + src_off),
                                  src_f16 != nullptr, mat.suh, dst, k);
        }
        else
        {
            for (int block = 0; block < k; block += 128)
            {
                float vals[128];
                for (int i = 0; i < 128; ++i)
                {
                    const float xv = src_f16 ? half_to_float(src_f16[src_off + block + i])
                                             : src_f32[src_off + block + i];
                    vals[i] = xv * half_to_float(mat.suh[block + i]);
                }
                hadamard_128<I>(vals);
                for (int i = 0; i < 128; ++i)
                    dst[block + i] = vals[i] * HAD_SCALE;
            }
        }

        // int8 quantization, one scale per row
        int32_t* splat = p.splat32 + static_cast<size_t>(r) * k;
        // dup is only read by the AVX2/BW maddubs kernels; skip the stores on the VNNI/VBMI tiers
        int32_t* splat_dup = (p.splat_dup && (I == Isa::Avx2 || I == Isa::Bw))
            ? p.splat_dup + static_cast<size_t>(r) * k : nullptr;
        float q;
        int32_t s;
        if constexpr (I != Isa::Scalar)
        {
            if (ACT_ROWS > 1 || act_blocked(k))
            {
                quantize_act(p, r, m, k, dst, splat_dup != nullptr);
                continue;
            }
            quantize_row_avx2(dst, splat, splat_dup, k, q, s);
        }
        else
        {
            float amax = 0.0f;
            for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(dst[i]));
            q = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float rq = 1.0f / q;
            s = 0;
            for (int i = 0; i < k; ++i)
            {
                int v = static_cast<int>(std::lround(dst[i] * rq));
                v = std::clamp(v, -127, 127);
                s += v;
                splat[i] = static_cast<int32_t>(static_cast<uint8_t>(static_cast<int8_t>(v))) * 0x01010101;
            }
        }
        p.q[r] = q;
        p.sum_x8[r] = s;
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
