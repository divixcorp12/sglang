// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
#include <atomic>
#include <type_traits>
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include <c10/util/Half.h>
#include <ATen/ATen.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#else
// min/max macro suppression is handled globally (-DNOMINMAX in setup.py): this TU uses
// std::min/max/clamp throughout
#include <intrin.h>
#include <windows.h>
#endif

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



// Kept for upstream's bindings. Phase timing is compile-time here (ForwardPlan's Profile, forward_plan.hpp).
void exl3_moe_cpu_set_prof(bool) {}

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

// -------------------------------------------------------------------------------------------
//   ISA dispatch
// -------------------------------------------------------------------------------------------

// Declared early: the transforms below select on it. Vbmi = Vnni + AVX512-VBMI (Zen4+, Ice
// Lake+); kept as a separate tier because Cascade/Cooper Lake have VNNI without VBMI. Bw =
// AVX-512F/BW/VL without VNNI (Skylake-SP/X): the dword kernel with the AVX2-style accumulate.
enum class Isa { Scalar, Avx2, Bw, Vnni, Vbmi };
extern const Isa g_isa;
// The forward is instantiated per tier; g_isa picks the instantiation once per call (forward_raw).
template <Isa I> constexpr bool kAvx512 = I == Isa::Bw || I == Isa::Vnni || I == Isa::Vbmi;

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

// -------------------------------------------------------------------------------------------
//   AVX-512 VNNI banded kernel
// -------------------------------------------------------------------------------------------

// Gather the two 32-bit word vectors covering row `row`'s 16-bit states (the permute stage of
// the extraction, split out so a row pair can share it -- see vnni_band_rows)
template <int bits, int row>
M1_TARGET_BW
inline void dword_gather(__m512i p0, __m512i p1, __m512i p2, __m512i p3, __m512i& a, __m512i& b)
{
    alignas(64) static constexpr auto i0d = make_row_indices<bits, row, false>();
    alignas(64) static constexpr auto i1d = make_row_indices<bits, row, true>();
    const __m512i i0 = _mm512_load_si512(i0d.data());
    const __m512i i1 = _mm512_load_si512(i1d.data());
    a = _mm512_permutex2var_epi32(p0, i0, p1);
    b = _mm512_permutex2var_epi32(p0, i1, p1);
    if constexpr (bits > 4)
    {
        // Up to 64 packed words: indices >= 32 select from the second register pair. vpermt2var
        // uses index bits [4:0], so the same index vectors address both pairs; constexpr masks
        // choose per lane
        constexpr __mmask16 hm0 = make_row_himask<bits, row, false>();
        constexpr __mmask16 hm1 = make_row_himask<bits, row, true>();
        if constexpr (hm0 != 0)
            a = _mm512_mask_blend_epi32(hm0, a, _mm512_permutex2var_epi32(p2, i0, p3));
        if constexpr (hm1 != 0)
            b = _mm512_mask_blend_epi32(hm1, b, _mm512_permutex2var_epi32(p2, i1, p3));
    }
}

// Per-lane shift counts for the funnel merge: cols 0-7 use s0, cols 8-15 use s1
template <int s0, int s1>
constexpr std::array<int32_t, 16> make_lane_shifts()
{
    std::array<int32_t, 16> v{};
    for (int i = 0; i < 16; ++i) v[i] = i < 8 ? s0 : s1;
    return v;
}

// Shift-merge codes for `row` out of its gathered word vectors; delta = bits extracts row+1
// from row's own gather (valid when word_pair_ok). Vector shifts by >= 32 are well-defined
// zero, so the s' == 0 case needs no special path. The two half-rows generally need different
// shifts: one per-lane variable funnel shift (vpsrlvd/vpsllvd against compile-time count
// vectors) merges both in 4 uops (GCC fuses the or+and into vpternlogd), where two immediate
// funnel shifts plus a lane blend took 8.
template <int bits, int row, int delta>
M1_TARGET_BW
inline __m512i dword_codes(__m512i a, __m512i b)
{
    constexpr int s0 = row_shift<bits, row>(0) - delta;
    constexpr int s1 = row_shift<bits, row>(8) - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds shift headroom");
    if constexpr (s0 == s1)
    {
        const __m512i c = _mm512_or_si512(_mm512_srli_epi32(b, s0), _mm512_slli_epi32(a, 32 - s0));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
    else
    {
        alignas(64) static constexpr auto sh = make_lane_shifts<s0, s1>();
        alignas(64) static constexpr auto shc = make_lane_shifts<32 - s0, 32 - s1>();
        const __m512i c = _mm512_or_si512(_mm512_srlv_epi32(b, _mm512_load_si512(sh.data())),
                                          _mm512_sllv_epi32(a, _mm512_load_si512(shc.data())));
        return _mm512_and_si512(c, _mm512_set1_epi32(0xffff));
    }
}

template <int bits, int row>
M1_TARGET_BW
inline __m512i extract_row(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    __m512i a, b;
    dword_gather<bits, row>(p0, p1, p2, p3, a, b);
    return dword_codes<bits, row, 0>(a, b);
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

template <int bits, int rows, int band, int R>
M1_TARGET_VNNI
inline void vnni_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (dword_pair_wins<bits>())
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            __m512i a, wb;
            dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
            const __m512i code0 = dword_codes<bits, R, 0>(a, wb);
            __m512i code1;
            if constexpr (word_pair_ok<bits, R>())
            {
                code1 = dword_codes<bits, R, bits>(a, wb);
            }
            else
            {
                dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
                code1 = dword_codes<bits, R + 1, 0>(a, wb);
            }
            const __m512i prod0 = _mm512_mullo_epi32(code0, mult);
            const __m512i prod1 = _mm512_mullo_epi32(code1, mult);
            for (int i = 0; i < rows; ++i)
            {
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
            }
            vnni_band_rows<bits, rows, band, R + 2>(p0, p1, p2, p3, b, splat, k, acc);
        }
    }
    else
    {
        if constexpr (R < 16) {
            const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
            const __m512i prod = _mm512_mullo_epi32(code, _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT)));
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod,
                    _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
            vnni_band_rows<bits, rows, band, R + 1>(p0, p1, p2, p3, b, splat, k, acc);
        }
    }
}

template <int bits, int rows, int band>
M1_TARGET_VNNI
void vnni_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // The remaining-word count is computed at the call sites: MSVC rejects reading even a
    // constexpr local inside a capture-less lambda (C3493), unlike GCC/clang
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    // Swizzled (band-contiguous) trellis layout: tile (kt, nt) lives at group nt/8, then kt,
    // then member nt%8, so a band's k-stream is (near-)sequential instead of packed_size-sized
    // runs strided by the full row. Requires band-aligned tile ranges (divisors of 8 within one
    // group; enforced by the band tables in *_tiles and the group-aligned splits in
    // forward_phase). Prefetch step differs accordingly.
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vnni_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_VNNI
void vnni_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // m = 1 supports band widths up to 16 (16 zmm accumulators). Measured on the 7960X: 16 is
    // not better than 8 for decode-shape jobs (medians 1.01 vs 0.98 ms, interleaved A/B) -- the
    // prefetcher already covers 8-tile bursts and the extra accumulators cost load-scheduling
    // registers -- so 8 is fixed (was a runtime switch via EXL3_MOE_CPU_BAND during that
    // investigation; no case remained for deviating from 8, so removed ahead of further
    // microoptimization work that wants less runtime branching in this path)
    constexpr int band_cap = 8;

    // Swizzled layout requires bands that are divisors of 8 (whole or partial groups); the
    // VBMI tier widens these (see vbmi_tiles) but the dword scheme's extra live temporaries
    // don't leave the register headroom for that here
    const int max_band = mat.swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2)
                                 : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vnni_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vnni_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vnni_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vnni_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vnni_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vnni_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vnni_band<bits, rows, 7>(mat, in, tout, n0); break;
            case 8: vnni_band<bits, rows, 8>(mat, in, tout, n0); break;
            default:
                if constexpr (rows == 1)
                {
                    switch (band)
                    {
                        case 9: vnni_band<bits, 1, 9>(mat, in, tout, n0); break;
                        case 10: vnni_band<bits, 1, 10>(mat, in, tout, n0); break;
                        case 11: vnni_band<bits, 1, 11>(mat, in, tout, n0); break;
                        case 12: vnni_band<bits, 1, 12>(mat, in, tout, n0); break;
                        case 13: vnni_band<bits, 1, 13>(mat, in, tout, n0); break;
                        case 14: vnni_band<bits, 1, 14>(mat, in, tout, n0); break;
                        case 15: vnni_band<bits, 1, 15>(mat, in, tout, n0); break;
                        default: vnni_band<bits, 1, 16>(mat, in, tout, n0); break;
                    }
                }
                break;
        }
        n0 += band;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX-512 BW banded kernel
//
//   AVX-512F/BW/VL without VNNI (Skylake-SP/X), which otherwise fell through to the AVX2 tier.
//   The VNNI kernel's dword extraction and k-major band structure (pure AVX-512F) with the AVX2
//   tier's accumulate in place of vpdpbusd: vpmaddubsw of the product bytes against 0x01 pairs
//   gives (b0+b1), (b2+b3) as i16 lanes (<= 510, cannot saturate), then one vpmaddwd per token
//   row against splat_dup (x8 in both 16-bit slots). Bit-exact with the AVX2 and VNNI tiers.
//   Separate functions rather than a template flag on the VNNI kernel: a function carries one
//   target attribute, and compiling this accumulate under the VNNI target would permit
//   contracting vpmaddwd+vpaddd into vpdpwssd.
// -------------------------------------------------------------------------------------------

// Force-inlined: GCC otherwise outlines the 16-row chain behind a call on every tile step, and
// with every zmm register caller-saved the band loop then reloads all of its live constants
// (index tables, multiplier, shift vectors) after each call (+8-9% inlined, Skylake-SP)
template <int bits, int rows, int band, int R>
M1_TARGET_BW
M1_ALWAYS_INLINE void bw_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat_dup, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (dword_pair_wins<bits>())
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            const __m512i ones = _mm512_set1_epi32(0x01010101);
            __m512i a, wb;
            dword_gather<bits, R>(p0, p1, p2, p3, a, wb);
            const __m512i code0 = dword_codes<bits, R, 0>(a, wb);
            __m512i code1;
            if constexpr (word_pair_ok<bits, R>())
            {
                code1 = dword_codes<bits, R, bits>(a, wb);
            }
            else
            {
                dword_gather<bits, R + 1>(p0, p1, p2, p3, a, wb);
                code1 = dword_codes<bits, R + 1, 0>(a, wb);
            }
            const __m512i ps0 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code0, mult), ones);
            const __m512i ps1 = _mm512_maddubs_epi16(_mm512_mullo_epi32(code1, mult), ones);
            for (int i = 0; i < rows; ++i)
            {
                const __m512i x0 = _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R]);
                const __m512i x1 = _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R + 1]);
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps0, x0));
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps1, x1));
            }
            bw_band_rows<bits, rows, band, R + 2>(p0, p1, p2, p3, b, splat_dup, k, acc);
        }
    }
    else
    {
        if constexpr (R < 16) {
            const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
            const __m512i ones = _mm512_set1_epi32(0x01010101);
            const __m512i code = extract_row<bits, R>(p0, p1, p2, p3);
            const __m512i ps = _mm512_maddubs_epi16(_mm512_mullo_epi32(code, mult), ones);
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_add_epi32(acc[b][i], _mm512_madd_epi16(ps,
                    _mm512_set1_epi32(splat_dup[static_cast<size_t>(i) * k + R])));
            bw_band_rows<bits, rows, band, R + 1>(p0, p1, p2, p3, b, splat_dup, k, acc);
        }
    }
}

// Three-bit batched BW kernel. Decode two adjacent input rows at once into 32
// interleaved 16-bit lanes. For a 16-bit state s and M = (M_hi << 16) + M_lo,
// the low/high product halves are mullo(s, M_lo) and mulhi_unsigned(s, M_lo)
// + mullo(s, M_hi), modulo 2^16. Summing their bytes produces the same mul1
// byte-sum as the dword kernel; vpmaddwd combines both rows without saturation.
// Three-bit tiles occupy 96 bytes, so both halfword gathers fit in p0/p1.
// Used below by the block-128 residual path: one token has two activation rows.
//
// Extract two adjacent rows into interleaved 16-bit lanes. Each lane needs the two
// halfwords enclosing its state in the format's (previous dword : current dword) window.
template <int row, bool high>
constexpr std::array<uint16_t, 32> bw3_word_indices()
{
    std::array<uint16_t, 32> idx{};
    constexpr auto inv = make_tc_perm_inv();
    for (int col = 0; col < 16; ++col)
        for (int r = 0; r < 2; ++r)
        {
            const int t = inv[(row + r) * 16 + col];
            const int b0 = t * 3 + 3 - 16 + 768;
            const int b1 = b0 + 16;
            const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
            const int half = shift / 16 + int(high);
            idx[col * 2 + r] = half < 2 ? (((b1 - 1) / 32) % 24) * 2 + half
                                                  : ((b0 / 32) % 24) * 2 + half - 2;
        }
    return idx;
}

template <int row, bool left>
constexpr std::array<uint16_t, 32> bw3_word_shifts()
{
    std::array<uint16_t, 32> shifts{};
    for (int col = 0; col < 16; ++col)
    {
        const int s0 = row_shift<3, row>(col) % 16;
        const int s1 = row_shift<3, row + 1>(col) % 16;
        shifts[col * 2] = left ? 16 - s0 : s0;
        shifts[col * 2 + 1] = left ? 16 - s1 : s1;
    }
    return shifts;
}

template <int rows, int band, int P>
M1_TARGET_BW
M1_ALWAYS_INLINE void bw3_band_rows(__m512i p0, __m512i p1, int b,
    const int32_t* splat_dup, int k, __m512i (&acc)[band][MAX_M], const char* future_weight)
{
    if constexpr (P < 8)
    {
        // Prefetch a future tile while decoding the current tile's register data.
        // P is compile-time: issue these once per tile, not once per row pair.
        if constexpr (P == 0)
        {
            _mm_prefetch(future_weight, _MM_HINT_T0);
            _mm_prefetch(future_weight + 64, _MM_HINT_T0);
        }
        constexpr int R = 2 * P;
        alignas(64) static constexpr auto il = bw3_word_indices<R, false>();
        alignas(64) static constexpr auto ih = bw3_word_indices<R, true>();
        alignas(64) static constexpr auto sr = bw3_word_shifts<R, false>();
        alignas(64) static constexpr auto sl = bw3_word_shifts<R, true>();
        const __m512i lo = _mm512_permutex2var_epi16(p0, _mm512_load_si512(il.data()), p1);
        const __m512i hi = _mm512_permutex2var_epi16(p0, _mm512_load_si512(ih.data()), p1);
        const __m512i state = _mm512_or_si512(
            _mm512_srlv_epi16(lo, _mm512_load_si512(sr.data())),
            _mm512_sllv_epi16(hi, _mm512_load_si512(sl.data())));
        const __m512i ml = _mm512_set1_epi16(static_cast<int16_t>(MUL1_MULT & 0xffff));
        const __m512i mh = _mm512_set1_epi16(static_cast<int16_t>(MUL1_MULT >> 16));
        const __m512i prod_lo = _mm512_mullo_epi16(state, ml);
        const __m512i prod_hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml),
                                                _mm512_mullo_epi16(state, mh));
        const __m512i ones = _mm512_set1_epi8(1);
        // Sum each product's four unsigned bytes. The result is in [0, 1020],
        // safely representable as signed i16 for the adjacent-row dot product.
        const __m512i sum = _mm512_add_epi16(_mm512_maddubs_epi16(prod_lo, ones),
                                            _mm512_maddubs_epi16(prod_hi, ones));
        for (int i = 0; i < rows; ++i)
        {
            const size_t off = static_cast<size_t>(i) * k + R;
            // Each i32 holds two copies of a signed i16 activation. On x86,
            // the middle four bytes of two adjacent entries hold the needed pair.
            // memcpy permits the unaligned load without violating aliasing rules.
            uint32_t pair;
            std::memcpy(&pair, reinterpret_cast<const char*>(splat_dup + off) + 2, sizeof(pair));
            acc[b][i] = _mm512_add_epi32(acc[b][i],
                _mm512_madd_epi16(sum, _mm512_set1_epi32(static_cast<int32_t>(pair))));
        }
        bw3_band_rows<rows, band, P + 1>(p0, p1, b, splat_dup, k, acc, future_weight);
    }
}

template <int bits, int rows, int band>
M1_TARGET_BW
void bw_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    // Layout and prefetch handling as in vnni_band (see the comments there); the host hands
    // this tier the swizzled layout too (moe_cpu_host gates it on has_avx512_bw)
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat_dup = in.splat_dup + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                // GCC uses a different spelling; at most four cache lines per tile.
                #if defined(__GNUC__) && !defined(__clang__)
                #pragma GCC unroll 4
                #else
                #pragma unroll
                #endif
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            bw_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat_dup, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_BW
void bw_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // Same band widths as vnni_tiles (one more live temporary per band step, same budget)
    constexpr int band_cap = 8;
    const int max_band = mat.swz ? (rows == 1 ? 8 : rows <= 3 ? 4 : 2)
                                 : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: bw_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: bw_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: bw_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: bw_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: bw_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: bw_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: bw_band<bits, rows, 7>(mat, in, tout, n0); break;
            default: bw_band<bits, rows, 8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}

// -------------------------------------------------------------------------------------------
//   AVX-512 VBMI banded kernel
//
//   Replaces the dword extraction's two 32-bit cross permutes + shift-merge with a single
//   byte-level permute (vpermb / vpermt2b): the 3 bytes covering each column's 16-bit state
//   are gathered directly into the dword lane, then one (even K) or two blended (odd K)
//   sub-byte right-shifts + mask produce the same codes bit-exactly.
//
//   Layout facts this relies on (from make_tc_perm; validated bit-exact against the dword
//   scheme for K1-8 x m1-4 in benchmarks/moe_mul1_bench):
//   - within a half-row (cols 0-7 / 8-15) the inverse-permutation index steps by 32 per
//     column, so bit offsets step by 32*bits and shift % 8 is uniform per half-row at every K
//   - the two half-rows differ by 4*bits bits: same shift % 8 for even K, +/-4 for odd K
//   - rows (2p, 2p+1) differ by exactly `bits` bits, so when shift % 8 >= bits for every
//     column of the even row, one gather serves both rows ("byte pairing": K1/K2/K4 all 8
//     pairs, K3/K6 rows 8-15 only, K5/K7/K8 never)
// -------------------------------------------------------------------------------------------

// For each column, byte indices (into the tile's 32*bits packed bytes) of the 3 bytes covering
// bits [shift, shift+16) of the (w0:w1) combined word window; 4th lane byte unused (index 0,
// never observed: shift%8 + 16 <= 23 keeps the value inside the low 3 bytes)
template <int bits, int row>
constexpr std::array<uint8_t, 64> make_row_byte_indices()
{
    std::array<uint8_t, 64> idx{};
    const auto inv = make_tc_perm_inv();
    constexpr int words32 = bits * 256 / 32;
    for (int col = 0; col < 16; ++col)
    {
        const int t = inv[row * 16 + col];
        const int b0 = t * bits + bits - 16 + 256 * bits;
        const int b1 = b0 + 16;
        const int w0 = (b0 / 32) % words32;          // high (earlier) word
        const int w1 = ((b1 - 1) / 32) % words32;    // low (later) word
        const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
        const int fb = shift / 8;
        for (int byte = 0; byte < 3; ++byte)
        {
            const int mb = fb + byte;
            const int src = mb < 4 ? w1 * 4 + mb : w0 * 4 + (mb - 4);
            idx[col * 4 + byte] = static_cast<uint8_t>(src);
        }
        idx[col * 4 + 3] = 0;
    }
    return idx;
}

// For bits > 4 (tile spans 4 zmms): which gathered bytes come from the (p2,p3) pair.
// vpermt2b consumes idx bits [6:0], so raw indices >= 128 address the high pair directly.
template <int bits, int row>
constexpr uint64_t make_row_byte_himask()
{
    const auto idx = make_row_byte_indices<bits, row>();
    uint64_t m = 0;
    for (int i = 0; i < 64; ++i)
        if (idx[i] >= 128) m |= uint64_t(1) << i;
    return m;
}

// Byte-level pairing is valid iff the odd row's value stays inside the even row's gathered
// byte window for every column, i.e. shift % 8 >= bits everywhere (stricter than the dword
// path's word_pair_ok, which only needs the full shift's headroom)
template <int bits, int row>
constexpr bool byte_pair_ok()
{
    for (int col = 0; col < 16; ++col)
        if (row_shift<bits, row>(col) % 8 < bits) return false;
    return true;
}

template <int bits, int row>
M1_TARGET_VBMI
inline __m512i gather_row_bytes(__m512i p0, __m512i p1, __m512i p2, __m512i p3)
{
    alignas(64) static constexpr auto bidx = make_row_byte_indices<bits, row>();
    const __m512i idx = _mm512_load_si512(bidx.data());
    if constexpr (bits <= 2)
    {
        (void) p1; (void) p2; (void) p3;
        return _mm512_permutexvar_epi8(idx, p0);
    }
    else if constexpr (bits <= 4)
    {
        (void) p2; (void) p3;
        return _mm512_permutex2var_epi8(p0, idx, p1);
    }
    else
    {
        constexpr uint64_t hm = make_row_byte_himask<bits, row>();
        if constexpr (hm == 0)
            return _mm512_permutex2var_epi8(p0, idx, p1);
        else if constexpr (hm == ~uint64_t(0))
            return _mm512_permutex2var_epi8(p2, idx, p3);
        else
            return _mm512_mask_blend_epi8(static_cast<__mmask64>(hm),
                _mm512_permutex2var_epi8(p0, idx, p1),
                _mm512_permutex2var_epi8(p2, idx, p3));
    }
}

// delta = 0 extracts `row` itself; delta = bits extracts row+1 from row's gathered bytes
template <int bits, int row, int delta>
M1_TARGET_VBMI
inline __m512i shift_mask_row(__m512i g)
{
    constexpr int s0 = row_shift<bits, row>(0) % 8 - delta;
    constexpr int s1 = row_shift<bits, row>(8) % 8 - delta;
    static_assert(s0 >= 0 && s1 >= 0, "pairing delta exceeds sub-byte shift headroom");
    if constexpr (s0 == s1)
        return _mm512_and_si512(_mm512_srli_epi32(g, s0), _mm512_set1_epi32(0xffff));
    else
        return _mm512_and_si512(_mm512_mask_blend_epi32(0xff00,
            _mm512_srli_epi32(g, s0), _mm512_srli_epi32(g, s1)), _mm512_set1_epi32(0xffff));
}

template <int bits, int rows, int band, int P>
M1_TARGET_VBMI
inline void vbmi_band_rows
(
    __m512i p0, __m512i p1, __m512i p2, __m512i p3, int b, const int32_t* splat, int k,
    __m512i (&acc)[band][MAX_M]
)
{
    if constexpr (P < 8)
    {
        constexpr int R = P * 2;
        const __m512i mult = _mm512_set1_epi32(static_cast<int32_t>(MUL1_MULT));
        __m512i c0, c1;
        if constexpr (byte_pair_ok<bits, R>())
        {
            const __m512i g = gather_row_bytes<bits, R>(p0, p1, p2, p3);
            c0 = shift_mask_row<bits, R, 0>(g);
            c1 = shift_mask_row<bits, R, bits>(g);
        }
        else
        {
            c0 = shift_mask_row<bits, R, 0>(gather_row_bytes<bits, R>(p0, p1, p2, p3));
            c1 = shift_mask_row<bits, R + 1, 0>(gather_row_bytes<bits, R + 1>(p0, p1, p2, p3));
        }
        const __m512i prod0 = _mm512_mullo_epi32(c0, mult);
        const __m512i prod1 = _mm512_mullo_epi32(c1, mult);
        for (int i = 0; i < rows; ++i)
        {
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod0,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R]));
            acc[b][i] = _mm512_dpbusd_epi32(acc[b][i], prod1,
                _mm512_set1_epi32(splat[static_cast<size_t>(i) * k + R + 1]));
        }
        vbmi_band_rows<bits, rows, band, P + 1>(p0, p1, p2, p3, b, splat, k, acc);
    }
}

template <int bits, int rows, int band>
M1_TARGET_VBMI
void vbmi_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr int words32 = bits * 256 / 32;
    // Same MSVC-safe form as vnni_band (C3493: no constexpr locals read inside the lambda)
    constexpr auto ld_mask = [](int n) -> __mmask16
    {
        return n >= 16 ? 0xffffu : (n <= 0 ? 0x0000u : static_cast<__mmask16>((1u << n) - 1u));
    };
    constexpr __mmask16 mask0 = ld_mask(words32 - 0);
    constexpr __mmask16 mask1 = ld_mask(words32 - 16);
    constexpr __mmask16 mask2 = ld_mask(words32 - 32);
    constexpr __mmask16 mask3 = ld_mask(words32 - 48);

    __m512i acc[band][MAX_M];
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
            acc[b][i] = _mm512_setzero_si512();

    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const size_t pf_step = mat.swz ? static_cast<size_t>(8) * packed_size : row_stride;
    const uint16_t* packed_row = mat.trellis + static_cast<size_t>(n0) * packed_size;
    for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed_row += row_stride)
    {
        const int32_t* splat = in.splat32 + tile_k * 16;
        for (int b = 0; b < band; ++b)
        {
            const uint16_t* packed = mat.swz
                ? mat.trellis + (static_cast<size_t>(n0 / 8) * (tl_swz_tiles_k ? tl_swz_tiles_k : tiles_k) * 8
                                 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b) * packed_size
                : packed_row + b * packed_size;
            if (mat.swz && band == 8)
            {
                // Whole-group band on the swizzled layout: the k-stream is sequential, one line
                // one step ahead is enough and the HW prefetcher follows the run (wider/farther
                // measured neutral on K4, 7960X)
                _mm_prefetch(reinterpret_cast<const char*>(packed + pf_step), _MM_HINT_T1);
            }
            else
            {
                // Strided stream: the native layout (K8, or VNNI-only CPUs where nothing is
                // swizzled) strides row_stride per step, and a partial-group band on the
                // swizzled layout (rows 3-4 at band 4) reads half a group then skips half; both
                // outrun the HW prefetcher, so touch every line of the tile row a few steps
                // ahead, as the AVX2 tier does (PR #331). +11% decode on K8 (7960X, four
                // order-alternated pairs, 60.9 -> 67.9 tok/s), prefill unchanged; bits == 6 keeps
                // the shorter distance that tier found necessary for its 96-byte rows
                constexpr int pf_lines = (packed_size * 2 + 63) / 64;
                constexpr int pf_dist = (bits == 6) ? 2 : 4;
                const char* pf = reinterpret_cast<const char*>(packed + pf_step * pf_dist);
                for (int l = 0; l < pf_lines; ++l)
                    _mm_prefetch(pf + l * 64, _MM_HINT_T0);
            }
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
            const __m512i p0 = _mm512_maskz_loadu_epi32(mask0, pw);
            const __m512i p1 = _mm512_maskz_loadu_epi32(mask1, pw + 16);
            const __m512i p2 = mask2 ? _mm512_maskz_loadu_epi32(mask2, pw + 32) : _mm512_setzero_si512();
            const __m512i p3 = mask3 ? _mm512_maskz_loadu_epi32(mask3, pw + 48) : _mm512_setzero_si512();
            vbmi_band_rows<bits, rows, band, 0>(p0, p1, p2, p3, b, splat, mat.k, acc);
        }
    }
    for (int b = 0; b < band; ++b)
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            const __m512 out = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
            _mm512_storeu_ps(tout + static_cast<size_t>(i) * mat.n + (n0 + b) * 16, out);
        }
}

template <int bits, int rows>
M1_TARGET_VBMI
void vbmi_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    constexpr int band_cap = 8;

    // Swizzled layout: bands must be divisors of 8 (whole or partial groups). Unlike the
    // dword scheme, byte-gather extraction needs few temporaries, so wider bands (up to 16
    // zmm accumulators: rows2 x band8, rows3/4 x band4) fit the register budget and keep the
    // swizzled stream at full duty. Narrow divisor bands (read-N-skip-N) measured BELOW
    // native layout at m>1. K2 rows4 prefers band 2 (measured 216 vs 197 Gw/s at band 4).
    const int max_band = mat.swz
        ? (rows <= 2 ? 8 : (rows == 4 && bits == 2 ? 2 : 4))
        : (rows == 1 ? band_cap : (12 / rows < 8 ? 12 / rows : 8));
    int n0 = tn0;
    while (n0 < tn1)
    {
        const int band = std::min(tn1 - n0, max_band);
        switch (band)
        {
            case 1: vbmi_band<bits, rows, 1>(mat, in, tout, n0); break;
            case 2: vbmi_band<bits, rows, 2>(mat, in, tout, n0); break;
            case 3: vbmi_band<bits, rows, 3>(mat, in, tout, n0); break;
            case 4: vbmi_band<bits, rows, 4>(mat, in, tout, n0); break;
            case 5: vbmi_band<bits, rows, 5>(mat, in, tout, n0); break;
            case 6: vbmi_band<bits, rows, 6>(mat, in, tout, n0); break;
            case 7: vbmi_band<bits, rows, 7>(mat, in, tout, n0); break;
            default: vbmi_band<bits, rows, 8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}

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
    #define ACC_ROW(i) \
        if ((i) < m) { \
            const __m256i xs = _mm256_set1_epi32(splat_dup[static_cast<size_t>(i) * k + row]); \
            acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(p_lo, xs)); \
            acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(p_hi, xs)); \
        }
    ACC_ROW(0) ACC_ROW(1) ACC_ROW(2) ACC_ROW(3)
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
void avx2_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
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

// -------------------------------------------------------------------------------------------
//   Scalar fallback
// -------------------------------------------------------------------------------------------

template <int bits>
void scalar_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr auto perm = make_tc_perm();

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        float acc[MAX_M][16] = {};
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k)
        {
            const uint16_t* packed = mat.trellis + (static_cast<size_t>(tile_k) * tiles_n + tile_n) * packed_size;
            float tile[256];

            for (int t = 0; t < 256; ++t)
                tile[perm[t]] = decode_mul1_scalar(decode_state_scalar<bits>(packed, t));

            for (int i = 0; i < m; ++i)
            {
                const float* x = in.tin + static_cast<size_t>(i) * mat.k + tile_k * 16;
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 16; ++c)
                        acc[i][c] += x[r] * tile[r * 16 + c];
            }
        }
        for (int i = 0; i < m; ++i)
            std::memcpy(tout + static_cast<size_t>(i) * mat.n + tile_n * 16, acc[i], 16 * sizeof(float));
    }
}

// -------------------------------------------------------------------------------------------
//   Dispatch
// -------------------------------------------------------------------------------------------

Isa detect_isa()
{
    Isa hw;
#if defined(__GNUC__) && defined(__linux__)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("fma"))
    {
        if (__builtin_cpu_supports("avx512vnni"))
            hw = __builtin_cpu_supports("avx512vbmi") ? Isa::Vbmi : Isa::Vnni;
        else
            hw = Isa::Bw;
    }
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        hw = Isa::Avx2;
    else
        hw = Isa::Scalar;
#else
    // __builtin_cpu_supports checks OS state-saving internally; this branch must do it by hand:
    // CPUID feature bits report hardware capability only, so without OSXSAVE + XCR0 checks a
    // hypervisor/OS that doesn't context-switch YMM/ZMM state would pass detection and fault at
    // the first vector instruction. FMA (leaf 1) mirrors the Linux branch's avx2+fma gate.
    int l0[4];
    __cpuid(l0, 0);
    if (l0[0] < 7)
    {
        hw = Isa::Scalar;
    }
    else
    {
        int l1[4];
        __cpuid(l1, 1);
        const bool osxsave = (l1[2] & (1u << 27)) != 0;
        const bool fma = (l1[2] & (1u << 12)) != 0;
        const uint64_t xcr0 = osxsave ? _xgetbv(0) : 0;
        const bool ymm_os = (xcr0 & 0x06) == 0x06;          // XMM + YMM state
        const bool zmm_os = (xcr0 & 0xe6) == 0xe6;          // + opmask, ZMM_Hi256, Hi16_ZMM
        int info[4];
        __cpuidex(info, 7, 0);
        const bool avx512 = (info[1] & (1u << 16)) && (info[1] & (1u << 30)) && (info[1] & (1u << 31));
        const bool vnni = (info[2] & (1u << 11)) != 0;
        const bool vbmi = (info[2] & (1u << 1)) != 0;
        const bool avx2 = (info[1] & (1u << 5)) != 0;
        hw = (avx512 && fma && zmm_os) ? (vnni ? (vbmi ? Isa::Vbmi : Isa::Vnni) : Isa::Bw)
           : (avx2 && fma && ymm_os)    ? Isa::Avx2
           :                              Isa::Scalar;
    }
#endif

    // EXL3_MOE_CPU_MAX_ISA=scalar|avx2|bw|vnni|vbmi: cap detection at a lower tier for testing
    // (e.g. exercising the AVX2 path on AVX512-VNNI hardware, or the dword scheme on VBMI
    // hardware). Never upgrades past what the CPU actually supports; an unrecognized value is
    // ignored.
    if (const char* e = std::getenv("EXL3_MOE_CPU_MAX_ISA"))
    {
        std::string s(e);
        for (char& c : s) c = (char) std::tolower((unsigned char) c);
        Isa cap;
        if (s == "scalar") cap = Isa::Scalar;
        else if (s == "avx2") cap = Isa::Avx2;
        else if (s == "bw" || s == "avx512bw") cap = Isa::Bw;
        else if (s == "vnni" || s == "avx512") cap = Isa::Vnni;
        else if (s == "vbmi") cap = Isa::Vbmi;
        else return hw;
        if (cap < hw) hw = cap;
    }
    return hw;
}

const Isa g_isa = []{ return detect_isa(); }();

template <Isa I>
void run_tiles_raw(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Vbmi)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: vbmi_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: vbmi_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: vbmi_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: vbmi_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: vbmi_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: vbmi_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: vbmi_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: vbmi_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: vbmi_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: vbmi_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: vbmi_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: vbmi_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: vbmi_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: vbmi_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: vbmi_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: vbmi_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: vbmi_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: vbmi_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: vbmi_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: vbmi_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: vbmi_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: vbmi_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: vbmi_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: vbmi_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: vbmi_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: vbmi_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: vbmi_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: vbmi_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            // K8: byte pairing impossible (shift % 8 == 0) and the byte windows straddle
            // the register pairs -- measured slower than the dword scheme, so route there
            case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Vnni)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: vnni_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: vnni_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: vnni_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: vnni_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: vnni_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: vnni_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: vnni_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: vnni_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: vnni_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: vnni_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: vnni_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: vnni_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: vnni_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: vnni_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: vnni_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: vnni_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: vnni_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: vnni_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: vnni_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: vnni_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: vnni_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: vnni_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: vnni_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: vnni_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: vnni_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: vnni_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: vnni_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: vnni_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Bw)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: bw_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: bw_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: bw_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: bw_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: bw_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: bw_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: bw_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: bw_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: bw_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: bw_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: bw_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: bw_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: bw_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: bw_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: bw_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: bw_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: bw_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: bw_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: bw_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: bw_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: bw_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: bw_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: bw_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: bw_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: bw_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: bw_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: bw_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: bw_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 0: bw_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: bw_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: bw_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: bw_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Avx2)
    {
        switch (mat.bits)
        {
            case 1: avx2_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: avx2_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: avx2_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: avx2_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: avx2_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: avx2_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: avx2_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: avx2_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
    else
    {
        switch (mat.bits)
        {
            case 1: scalar_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: scalar_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: scalar_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: scalar_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: scalar_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: scalar_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: scalar_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: scalar_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
}

// K3/BW block-128 residual decode. Keep each output band's fp32 sums across input
// blocks, avoiding repeated dispatch/setup and the full-size intermediate output.
// Walking the input blocks within a band also follows the swizzled weight layout.
// Preserve the generic path's rounding: scale each int32 block with its own FMA,
// sum blocks separately for base/residual rows, then add the residual exactly once.
template <int band>
M1_TARGET_BW
void bw3_blocked_band(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int n0)
{
    constexpr int B = 128, rows = 2, packed_size = 48;
    const int tiles_k = mat.k / 16, tiles_n = mat.n / 16;
    const size_t step = static_cast<size_t>(mat.swz ? 8 : tiles_n) * packed_size;
    __m512 sums[band][rows];
    for (int kb = 0; kb < mat.k / B; ++kb)
    {
        __m512i acc[band][MAX_M];
        for (int b = 0; b < band; ++b)
            for (int i = 0; i < rows; ++i)
                acc[b][i] = _mm512_setzero_si512();
        for (int kt = 0; kt < B / 16; ++kt)
        {
            const int tile_k = kb * (B / 16) + kt;
            const int32_t* splat = in.splat_dup + static_cast<size_t>(kb) * rows * B + kt * 16;
            for (int b = 0; b < band; ++b)
            {
                const size_t tile = mat.swz
                    ? static_cast<size_t>(n0 / 8) * tiles_k * 8 + static_cast<size_t>(tile_k) * 8 + (n0 % 8) + b
                    : static_cast<size_t>(tile_k) * tiles_n + n0 + b;
                const uint16_t* packed = mat.trellis + tile * packed_size;
                // Lead by two K tiles (step is in uint16_t elements). The prefetch
                // may point beyond the final tile; form its address
                // as an integer instead of an out-of-object C++ array pointer.
                const char* pf = reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(packed) + 4 * step);
                const uint32_t* pw = reinterpret_cast<const uint32_t*>(packed);
                const __m512i p0 = _mm512_loadu_si512(pw);
                // The tile tail is exactly 32 bytes. A 256-bit load avoids a
                // masked 512-bit memory operation and keeps the upper lanes zero.
                const __m512i p1 = _mm512_zextsi256_si512(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pw + 16)));
                bw3_band_rows<rows, band, 0>(p0, p1, b, splat, B, acc, pf);
            }
        }
        for (int b = 0; b < band; ++b)
            for (int i = 0; i < rows; ++i)
            {
                const float scale = mul1_k_inv() * in.bq[kb * MAX_M + i];
                const __m512 corr = _mm512_set1_ps(-510.0f * static_cast<float>(in.bsum[kb * MAX_M + i]) * scale);
                const __m512 v = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[b][i]), _mm512_set1_ps(scale), corr);
                if (kb == 0) sums[b][i] = v;
                else sums[b][i] = _mm512_add_ps(sums[b][i], v);
            }
    }
    for (int b = 0; b < band; ++b)
    {
        _mm512_storeu_ps(tout + (n0 + b) * 16, _mm512_add_ps(sums[b][0], sums[b][1]));
        _mm512_storeu_ps(tout + mat.n + (n0 + b) * 16, sums[b][1]);
    }
}

M1_TARGET_BW
void bw3_blocked_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    // Match the eight output tiles stored together in the swizzled layout.
    // Keep each output's fp32 block accumulation order unchanged.
    constexpr int cap = 8;
    for (int n0 = tn0; n0 < tn1;)
    {
        const int band = std::min({cap, tn1 - n0, 8 - n0 % 8});
        switch (band)
        {
            case 1: bw3_blocked_band<1>(mat, in, tout, n0); break;
            case 2: bw3_blocked_band<2>(mat, in, tout, n0); break;
            case 3: bw3_blocked_band<3>(mat, in, tout, n0); break;
            case 4: bw3_blocked_band<4>(mat, in, tout, n0); break;
            case 5: bw3_blocked_band<5>(mat, in, tout, n0); break;
            case 6: bw3_blocked_band<6>(mat, in, tout, n0); break;
            case 7: bw3_blocked_band<7>(mat, in, tout, n0); break;
            case 8: bw3_blocked_band<8>(mat, in, tout, n0); break;
        }
        n0 += band;
    }
}


M1_TARGET_BW M1_ALWAYS_INLINE __m512i register_bytesum(__m512i state) {
    const __m512i ml = _mm512_set1_epi16(int16_t(MUL1_MULT & 0xffff));
    const __m512i mh = _mm512_set1_epi16(int16_t(MUL1_MULT >> 16));
    const __m512i lo = _mm512_mullo_epi16(state, ml);
    const __m512i hi = _mm512_add_epi16(_mm512_mulhi_epu16(state, ml), _mm512_mullo_epi16(state, mh));
    const __m512i ones = _mm512_set1_epi8(1);
    return _mm512_add_epi16(_mm512_maddubs_epi16(lo, ones), _mm512_maddubs_epi16(hi, ones));
}
#include "register.hpp"

// m token rows through the quantized kernels under the accuracy options (quantize_act's layout): per
// k-block sub-views summed in fp32, then each remainder row added onto its token row. Only this
// worker's columns [tn0, tn1) are touched, so the sums need no synchronization.
template <Isa I>
void run_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1, bool grouped = false)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)
    {
        if (m == 1 && mat.bits == 3 && act_blocked(mat.k))
        {
            register_tiles(mat,in,tout,tn0,tn1,grouped);
            return;
        }
    }
    if (I == Isa::Scalar || (ACT_ROWS == 1 && !act_blocked(mat.k)))
    {
        run_tiles_raw<I>(mat, in, tout, m, tn0, tn1);
        return;
    }
    const int rows = ACT_ROWS * m;
    const int n = mat.n;
    const int c0 = tn0 * 16, c1 = tn1 * 16;
    if (!act_blocked(mat.k))
    {
        run_tiles_raw<I>(mat, in, tout, rows, tn0, tn1);
    }
    else
    {
        const int B = EXL3_MOE_CPU_ACT_BLOCK;
        const int block_tiles = B / 16;
        const size_t packed_size = static_cast<size_t>(16) * mat.bits;
        thread_local std::vector<float> part;
        if (part.size() < static_cast<size_t>(rows) * n) part.resize(static_cast<size_t>(rows) * n);
        tl_swz_tiles_k = mat.k / 16;
        for (int b = 0; b < mat.k / B; ++b)
        {
            MoeCpuMatrix sub = mat;
            sub.k = B;
            // Native: k-tile rows are tiles_n tiles apart. Swizzled: consecutive k-tiles of an 8-tile
            // group are 8 tiles apart, with the group stride from tl_swz_tiles_k
            sub.trellis = mat.trellis + static_cast<size_t>(b) * block_tiles
                                        * (mat.swz ? 8 : n / 16) * packed_size;
            PreparedIn sub_in = in;
            sub_in.splat32 = in.splat32 + static_cast<size_t>(b) * rows * B;
            sub_in.splat_dup = in.splat_dup ? in.splat_dup + static_cast<size_t>(b) * rows * B : nullptr;
            for (int i = 0; i < rows; ++i)
            {
                sub_in.q[i] = in.bq[b * MAX_M + i];
                sub_in.sum_x8[i] = in.bsum[b * MAX_M + i];
            }
            float* dst = b ? part.data() : tout;
            run_tiles_raw<I>(sub, sub_in, dst, rows, tn0, tn1);
            if (b)
                for (int i = 0; i < rows; ++i)
                {
                    float* t = tout + static_cast<size_t>(i) * n;
                    const float* q = part.data() + static_cast<size_t>(i) * n;
                    for (int c = c0; c < c1; ++c) t[c] += q[c];
                }
        }
        tl_swz_tiles_k = 0;
    }
    for (int r = m; r < rows; ++r)
    {
        float* t = tout + static_cast<size_t>(r - m) * n;
        const float* q = tout + static_cast<size_t>(r) * n;
        for (int c = c0; c < c1; ++c) t[c] += q[c];
    }
}

// Worker cores set by sglang_exl3_cpu_experts_set_cores; copied into g_compute_cores at the first forward.
std::mutex g_cores_mutex;
std::vector<int> g_configured_cores;
std::atomic<bool> g_compute_started{false};
std::vector<int> g_compute_cores; // Immutable after release publication at first forward.

// Freezes the configured cores into g_compute_cores once, at the first forward or keep-warm. Steady-state calls
// acquire no pool mutex.
inline void freeze_compute_cores()
{
    if (g_compute_started.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_cores_mutex);
    if (!g_compute_started.load(std::memory_order_relaxed)) {
        g_compute_cores = g_configured_cores;
        g_compute_started.store(true, std::memory_order_release);
    }
}

// Inside a parallel region: pins OpenMP worker `worker` to its compute core (none configured: no-op), setting
// pin_error if it cannot.
inline void pin_compute_worker(int worker, std::atomic<int>& pin_error)
{
    if (g_compute_cores.empty()) return;
    const int core = g_compute_cores[worker];
    static thread_local int pinned_core = -1;
    if (pinned_core != core || sched_getcpu() != core) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
            pin_error.store(1, std::memory_order_relaxed);
        else pinned_core = core;
    }
}


// -------------------------------------------------------------------------------------------
//   Forward driver
// -------------------------------------------------------------------------------------------

struct Chunk
{
    int expert;
    int m;
    int token[MAX_M];
    float weight[MAX_M];
};

#include "experts.hpp"
#include "shapes.hpp"

// -------------------------------------------------------------------------------------------
//   MoE Layer registry
// -------------------------------------------------------------------------------------------

// A registered layer: make_layer's per-expert tables, or the slab registration's view (table == nullptr).
struct RegisteredLayer
{
    LayerInfo info;
    std::unique_ptr<MoeCpuLayer> table;
    StridedExperts<GenericShape> strided{};
};

std::vector<std::unique_ptr<RegisteredLayer>> g_layers;
std::mutex g_layers_mutex;

struct ForwardCtx
{
    LayerInfo info;
    const at::Half* x;
    float* out;
    int m_total;
    std::vector<Chunk> chunks;

    // workspace, per chunk (pointers into the persistent per-thread arena below: fresh
    // allocations per call cost more in first-touch page faults than the small phases do work)
    float* tout_g;       // chunks x m x I (quant space, then transformed in place)
    float* tout_u;
    float* tout_d;       // chunks x m x H
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    // Down tiles finished per (chunk, 128-output block), when a plan splits the down GEMVs finer than a block: the
    // worker that finishes a block's eighth tile applies its output transform.
    int* down_tiles_done;
};

struct ForwardArena
{
    std::vector<float> tin_g, tin_u, tin_d;
    std::vector<int32_t> splat_g, splat_u, splat_d;
    std::vector<int32_t> splat_dup_g, splat_dup_u, splat_dup_d;
    std::vector<int16_t> compact_g, compact_u, compact_d;
    std::vector<float> tout_g, tout_u, tout_d;
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    std::vector<float> bq_g, bq_u, bq_d;
    std::vector<int32_t> bsum_g, bsum_u, bsum_d;
    std::vector<int> down_tiles_done;
    // Moved into the call's ForwardCtx and back, so a forward allocates nothing once warm
    std::vector<std::vector<std::pair<int, float>>> per_expert;
    std::vector<Chunk> chunks;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

template <Isa I>
M1_TARGET_AVX2
void transform_out_avx2(const MoeCpuMatrix& mat, float* tout, int m)
{
    const __m256 hs = _mm256_set1_ps(HAD_SCALE);
    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128)
        {
            float* v = tout + static_cast<size_t>(r) * mat.n + block;
            hadamard_128_avx2<I>(v);
            for (int i = 0; i < 128; i += 8)
            {
                const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    reinterpret_cast<const uint16_t*>(mat.svh) + block + i)));
                __m256 x = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), hs), s);
                if (mat.bias)
                    x = _mm256_add_ps(x, _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(
                            reinterpret_cast<const uint16_t*>(mat.bias) + block + i))));
                _mm256_storeu_ps(v + i, x);
            }
        }
}

template <Isa I>
__attribute__((noipa)) void transform_out(const MoeCpuMatrix& mat, float* tout, int m)
{
    if constexpr (I != Isa::Scalar) { transform_out_avx2<I>(mat, tout, m); return; }
    else
    {
        for (int r = 0; r < m; ++r)
            for (int block = 0; block < mat.n; block += 128)
            {
                float* v = tout + static_cast<size_t>(r) * mat.n + block;
                hadamard_128<I>(v);
                if (mat.bias)
                    for (int i = 0; i < 128; ++i)
                        v[i] = v[i] * HAD_SCALE * half_to_float(mat.svh[block + i])
                               + half_to_float(mat.bias[block + i]);
                else
                    for (int i = 0; i < 128; ++i)
                        v[i] *= HAD_SCALE * half_to_float(mat.svh[block + i]);
            }
    }
}


// Assign this worker its share of a phase's `total` GEMVs (all of one width, tiles_n tiles),
// calling gemv(j, t0, t1) per tile range. Few GEMVs (decode, small batches; each expert read
// once): the GEMVs' tiles form one flat range, split evenly across workers in 8-tile groups so
// no piece crosses a group of its GEMV (the swizzled band kernels' invariant; n % 128 == 0
// makes every tiles_n a multiple of 8). Whole-GEMV assignment left a 2:1 imbalance whenever
// 2 * cold experts fell between multiples of the worker count (16 gate/up GEMVs on 20 workers:
// twelve single-worker GEMVs set the phase time while eight workers idled half of it). Many
// GEMVs (prefill): whole GEMVs strided across workers, so all workers stream the same expert's
// chunks together and L3 serves the repeats; the imbalance there is at most one GEMV in four.
constexpr int FLAT_MAX_GEMVS_PER_WORKER = 4;

// Unit is the flat split's granularity in tiles: 8 (one 128-output group, which the swizzled kernels need) or, for
// a plan whose kernels take any tile pair, 2 (finer balance: 36 gate/up groups over 16 workers leave the slowest
// worker 3 groups against a mean of 2.25, where 144 pairs give every worker 9).
template <int Unit = 8, typename Gemv>
inline void assign_gemvs(int worker, int num_workers, int total, int tiles_n, Gemv gemv)
{
    static_assert(Unit == 2 || Unit == 8);
    if (total > FLAT_MAX_GEMVS_PER_WORKER * num_workers)
    {
        for (int j = worker; j < total; j += num_workers) gemv(j, 0, tiles_n);
        return;
    }
    const int64_t groups = static_cast<int64_t>(total) * (tiles_n / Unit);
    const int f0 = static_cast<int>(groups * worker / num_workers) * Unit;
    const int f1 = static_cast<int>(groups * (worker + 1) / num_workers) * Unit;
    for (int j = f0 / tiles_n; j * tiles_n < f1; ++j)
        gemv(j, std::max(f0 - j * tiles_n, 0), std::min(f1 - j * tiles_n, tiles_n));
}

// Same scalar/vector operations and block scale ordering as the original phase 2.
// Blocks are independent: only their own gate/up outputs and prepared input slices are written.
template<class Shape, Isa I, bool Wide = false, class Experts>
void middle_blocks(ForwardCtx& c,const Experts& E,int worker,int num_workers) {
    const int I_=Shape::intermediate(c.info), nb=I_/128, nc=int(c.chunks.size());
    const bool gated=Shape::gated(c.info);
    const int first=nc*nb*worker/num_workers,last=nc*nb*(worker+1)/num_workers;
    for (int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb,block=b*128;
        const auto& ch=c.chunks[j];
        for (int r=0;r<ch.m;++r) {
            float* g=c.tout_g+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            float* u=c.tout_u+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            if (gated) {
                auto mat=E.gate(ch.expert);mat.n=128;mat.svh+=block;if(mat.bias)mat.bias+=block;
                transform_out<I>(mat,g,1);
            }
            auto up=E.up(ch.expert);up.n=128;up.svh+=block;if(up.bias)up.bias+=block;
            transform_out<I>(up,u,1);
            const size_t count=128;
            float* a=gated?g:u;
            const float lim=Shape::act_limit(c.info)!=0.0f?Shape::act_limit(c.info):std::numeric_limits<float>::infinity();
                switch (Shape::activation(c.info)) {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            const float av = std::min(gv * cdf, lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 3: {
                        // gpt-oss clamped swiglu: g = min(g, limit); a = (clamp(u, -l, l) + 1) * g *
                        // sigmoid(1.702 * g)
                        const float lim = Shape::act_limit(c.info);
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = std::min(g[i], lim);
                            const float uv = std::clamp(u[i], -lim, lim);
                            g[i] = (uv + 1.0f) * gv / (1.0f + std::exp(-1.702f * gv));
                        }
                        break;
                    }
                    default:
                        for (size_t i = 0; i < count; ++i) {
                            const float uv = u[i] > 0.0f ? u[i] : 0.0f;
                            u[i] = uv * uv;
                        }
                        break;
                }

            auto& p=c.prep_d[j];
            float* dst=p.tin+size_t(r)*I_+block;
            prepare_block_avx2<I>(a,false,E.down(ch.expert).suh+block,dst,128);
            if(p.compact){compact_quantize_block<Wide>(p,r,ch.m,I_,b,dst);continue;}
            float* residual=p.tin+size_t(r+ch.m)*I_+block;
            for (int pass=0;pass<ACT_ROWS;++pass) {
                const int row=r+pass*ch.m;
                const size_t offset=size_t(b)*ACT_ROWS*ch.m*128+row*128;
                int32_t* splat=p.splat32+offset;
                float q;int32_t sum;
                const float* v=pass?residual:dst;
                quantize_row_avx2(v,splat,p.splat_dup+offset,128,q,sum);
                p.bq[b*MAX_M+row]=q;p.bsum[b*MAX_M+row]=sum;
                if(pass+1<ACT_ROWS) for(int i=0;i<128;++i)
                    residual[i]=v[i]-q*float(static_cast<int8_t>(splat[i]&0xff));
            }
        }
    }
}


template<class Shape, Isa I, bool Wide = false, class Experts>
void prepare_gu_blocks(ForwardCtx& c,const Experts& E,int worker,int num_workers) {
    const int K=Shape::hidden(c.info),nb=K/128,nc=int(c.chunks.size()),gu=!Shape::gated(c.info)?1:2;
    const int first=nc*gu*nb*worker/num_workers,last=nc*gu*nb*(worker+1)/num_workers;
    for(int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb;
        const auto& ch=c.chunks[j/gu];
        const bool up=gu==1 || j%gu;
        const MoeCpuMatrix& mat=up?E.up(ch.expert):E.gate(ch.expert);
        auto& p=(up?c.prep_u:c.prep_g)[j/gu];
        for(int r=0;r<ch.m;++r) {
            float* dst=p.tin+size_t(r)*K+b*128;
            prepare_block_avx2<I>(c.x+size_t(ch.token[r])*K+b*128,true,mat.suh+b*128,dst,128);
            if(p.compact){compact_quantize_block<Wide>(p,r,ch.m,K,b,dst);continue;}
            float* residual=p.tin+size_t(r+ch.m)*K+b*128;
            for(int pass=0;pass<ACT_ROWS;++pass) {
                const int row=r+pass*ch.m;
                const size_t off=size_t(b)*ACT_ROWS*ch.m*128+row*128;
                float q;int32_t sum;const float* v=pass?residual:dst;
                int32_t* splat=p.splat32+off;
                quantize_row_avx2(v,splat,p.splat_dup+off,128,q,sum);
                p.bq[b*MAX_M+row]=q;p.bsum[b*MAX_M+row]=sum;
                if(pass+1<ACT_ROWS) for(int i=0;i<128;++i)
                    residual[i]=v[i]-q*float(static_cast<int8_t>(splat[i]&0xff));
            }
        }
    }
}


template <Isa I>
void transform_owned_blocks(const MoeCpuMatrix& mat,float* out,int m,int t0,int t1) {
    for(int r=0;r<m;++r) for(int block=t0*16;block<t1*16;block+=128) {
        auto sub=mat;sub.n=128;sub.svh+=block;if(sub.bias)sub.bias+=block;
        transform_out<I>(sub,out+size_t(r)*mat.n+block,1);
    }
}

// Coarse readiness is per expert: all gate/up outputs must exist before middle,
// and all middle blocks must be prepared before any down output band can run.
#include "forward_plan.hpp"

// Runs the call's plan. E reads the layer's experts under the generic plan; D reads the same experts under the
// DSV4.1 plan's assumptions (for a strided layer, the compile-time-shaped view of the same slabs).
template <class Experts, class Dsv41Experts>
void run_plan(ForwardCtx& ctx, const Experts& E, const Dsv41Experts& D, ForwardArena& ar, int threads)
{
    if (g_isa == Isa::Bw && Dsv41Shape::accepts(ctx.info, E, ctx.chunks)) {
        ForwardPlan<Dsv41Shape, Isa::Bw>::run(ctx, D, ar, threads);
        return;
    }
    switch (g_isa) {
        case Isa::Scalar: ForwardPlan<GenericShape, Isa::Scalar>::run(ctx, E, ar, threads); return;
        case Isa::Avx2:   ForwardPlan<GenericShape, Isa::Avx2>::run(ctx, E, ar, threads); return;
        case Isa::Bw:     ForwardPlan<GenericShape, Isa::Bw>::run(ctx, E, ar, threads); return;
        case Isa::Vnni:   ForwardPlan<GenericShape, Isa::Vnni>::run(ctx, E, ar, threads); return;
        case Isa::Vbmi:   ForwardPlan<GenericShape, Isa::Vbmi>::run(ctx, E, ar, threads); return;
    }
}



// The forward's OpenMP team for tier I: pins worker i to the configured core i, runs the phases with a barrier after
// each, and checks the team. Phase 4 (the whole-row down transform) is folded into phase 3's owned blocks.


} // namespace

// -------------------------------------------------------------------------------------------
//   Upstream link compatibility
// -------------------------------------------------------------------------------------------

// This file replaces upstream's cpu/moe_mul1.cpp inside the exllamav3 extension, whose
// bindings.cpp and cpu/moe_handoff.cu still reference these symbols. sglang never calls them:
// the staging copy belongs to upstream's handoff worker, and the pool they exercised was
// replaced by the OpenMP forward. They exist only so the extension links, and fail if called.

void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_stage_experts is not supported by the optimized CPU expert kernel");
}

int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_pool_stress is not supported by the optimized CPU expert kernel");
    return 0;
}

bool exl3_moe_cpu_has_avx2() { return g_isa != Isa::Scalar; }
bool exl3_moe_cpu_has_avx512_bw() { return g_isa >= Isa::Bw; }
bool exl3_moe_cpu_has_avx512_vnni() { return g_isa >= Isa::Vnni; }
bool exl3_moe_cpu_has_avx512_vbmi() { return g_isa == Isa::Vbmi; }

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && trellis.is_contiguous(), "trellis must be contiguous CPU");
    TORCH_CHECK(trellis.dim() == 3, "trellis must be [k/16, n/16, 16K]");
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
    m.suh = reinterpret_cast<const at::Half*>(suh.data_ptr());
    m.svh = reinterpret_cast<const at::Half*>(svh.data_ptr());
    m.bias = bias ? reinterpret_cast<const at::Half*>(bias->data_ptr()) : nullptr;
    m.k = static_cast<int>(trellis.size(0)) * 16;
    m.n = static_cast<int>(trellis.size(1)) * 16;
    m.bits = static_cast<int>(trellis.size(2)) / 16;
    // K8 tensors are exempt from swizzling (routed to the dword kernel, which would gain
    // nothing) -- the child loader applies the same bits != 8 rule when repacking, so the two
    // sides agree per tensor
    m.swz = swizzled && m.bits != 8 ? 1 : 0;
    TORCH_CHECK(m.bits >= 1 && m.bits <= 8, "CPU MoE requires K in [1, 8]");
    TORCH_CHECK(m.k % 128 == 0 && m.n % 128 == 0, "dims must be divisible by 128");
    TORCH_CHECK(m.k <= 8192, "k too large for i32 accumulation");
    return m;
}

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled
)
{
    auto table = std::unique_ptr<MoeCpuLayer>(new MoeCpuLayer);
    const bool swz = swizzled != 0;
    const size_t E = up_trellis.size();
    const bool gated = !gate_trellis.empty();
    TORCH_CHECK(down_trellis.size() == E && (!gated || gate_trellis.size() == E), "expert count mismatch");
    TORCH_CHECK(gated ? (activation == 0 || activation == 1 || activation == 3) : activation == 2, "gated experts take silu/gelu/swiglu_oai, gateless take relu2");
    TORCH_CHECK(gate_bias.empty() || gate_bias.size() == E, "gate bias count mismatch");
    TORCH_CHECK(up_bias.empty() || up_bias.size() == E, "up bias count mismatch");
    TORCH_CHECK(down_bias.empty() || down_bias.size() == E, "down bias count mismatch");
    table->num_experts = static_cast<int>(E);
    table->activation = static_cast<int>(activation);
    table->act_limit = static_cast<float>(act_limit);
    for (size_t e = 0; e < E; ++e) {
        if (gated) {
            table->gates.push_back(make_matrix(gate_trellis[e], gate_suh[e], gate_svh[e],
                                               gate_bias.empty() ? nullptr : &gate_bias[e], swz));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]})
                table->refs.push_back(t);
            if (!gate_bias.empty()) table->refs.push_back(gate_bias[e]);
        }
        table->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                         up_bias.empty() ? nullptr : &up_bias[e], swz));
        table->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                           down_bias.empty() ? nullptr : &down_bias[e], swz));
        for (auto& t : {up_trellis[e], up_suh[e], up_svh[e], down_trellis[e], down_suh[e], down_svh[e]})
            table->refs.push_back(t);
        if (!up_bias.empty()) table->refs.push_back(up_bias[e]);
        if (!down_bias.empty()) table->refs.push_back(down_bias[e]);
    }
    table->hidden_size = table->ups[0].k;
    table->interm_size = table->ups[0].n;
    TORCH_CHECK(table->downs[0].k == table->interm_size && table->downs[0].n == table->hidden_size,
                "expert shape mismatch");

    auto entry = std::make_unique<RegisteredLayer>();
    entry->info = {table->num_experts, table->hidden_size, table->interm_size, !table->gates.empty(),
                   table->activation, table->act_limit};
    entry->table = std::move(table);
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    g_layers.push_back(std::move(entry));
    return static_cast<int64_t>(g_layers.size() - 1);
}

void exl3_moe_cpu_free_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    if (handle >= 0 && handle < static_cast<int64_t>(g_layers.size()))
    {
        g_layers[handle].reset();
    }
}

static const RegisteredLayer& get_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    TORCH_CHECK(handle >= 0 && handle < static_cast<int64_t>(g_layers.size()) && g_layers[handle], "invalid CPU MoE layer handle");
    return *g_layers[handle];
}

// exl3_moe_cpu_forward_raw, adding into out when `accumulate`: the sglang C ABI's forward. Other units of the
// extension are built against upstream's header, so the public signature stays as upstream declares it.
static void forward_raw(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads,
    bool accumulate
)
{
    const RegisteredLayer& layer = get_layer(handle);
    const LayerInfo& info = layer.info;
    const int m_total = rows;
    const int top_k = topk;

    ForwardCtx ctx;
    ctx.info = info;
    ctx.x = x;
    ctx.out = out;
    ctx.m_total = m_total;
    if (!accumulate) std::memset(ctx.out, 0, static_cast<size_t>(m_total) * info.hidden * sizeof(float));

    ForwardArena& ar = ForwardArena::get();
    ctx.chunks = std::move(ar.chunks);
    ctx.chunks.clear();
    ctx.prep_g = std::move(ar.prep_g); ctx.prep_u = std::move(ar.prep_u); ctx.prep_d = std::move(ar.prep_d);
    const auto give_back = [&] {
        ar.chunks = std::move(ctx.chunks);
        ar.prep_g = std::move(ctx.prep_g); ar.prep_u = std::move(ctx.prep_u); ar.prep_d = std::move(ctx.prep_d);
    };

    // Group token assignments by expert, then split into chunks of CHUNK_M rows
    auto& per_expert = ar.per_expert;
    if (per_expert.size() < static_cast<size_t>(info.num_experts)) per_expert.resize(info.num_experts);
    for (int t = 0; t < m_total; ++t)
        for (int j = 0; j < top_k; ++j)
        {
            const int32_t e = sel[static_cast<size_t>(t) * top_k + j];
            if (e >= 0 && e < info.num_experts)
                per_expert[e].emplace_back(t, half_to_float(wts[static_cast<size_t>(t) * top_k + j]));
        }
    for (int e = 0; e < info.num_experts; ++e)
    {
        auto& lst = per_expert[e];
        for (size_t i = 0; i < lst.size(); i += CHUNK_M)
        {
            Chunk ch;
            ch.expert = e;
            ch.m = static_cast<int>(std::min<size_t>(CHUNK_M, lst.size() - i));
            for (int r = 0; r < ch.m; ++r)
            {
                ch.token[r] = lst[i + r].first;
                ch.weight[r] = lst[i + r].second;
            }
            ctx.chunks.push_back(ch);
        }
        lst.clear();
    }
    const int nc = static_cast<int>(ctx.chunks.size());
    if (!nc) { give_back(); return; }

    if (layer.table) {
        const TableExperts t{layer.table.get()};
        run_plan(ctx, t, t, ar, threads);
    } else {
        run_plan(ctx, layer.strided, layer.strided.as<Dsv41Shape>(), ar, threads);
    }
    give_back();
}

void exl3_moe_cpu_forward_raw(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    forward_raw(handle, x, sel, wts, out, rows, topk, threads, false);
}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(x.device().is_cpu() && selected.device().is_cpu() && weights.device().is_cpu() && out.device().is_cpu(), "CPU MoE tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kFloat, "dtype mismatch");

    const int m_total = static_cast<int>(x.size(0));
    const int top_k = static_cast<int>(selected.size(-1));

    // Raw path takes int32 selection
    std::vector<int32_t> sel32(static_cast<size_t>(m_total) * top_k);
    if (selected.scalar_type() == at::kLong)
    {
        const int64_t* s = selected.data_ptr<int64_t>();
        for (size_t i = 0; i < sel32.size(); ++i) sel32[i] = static_cast<int32_t>(s[i]);
    }
    else
    {
        TORCH_CHECK(selected.scalar_type() == at::kInt, "selected must be int32 or int64");
        std::memcpy(sel32.data(), selected.data_ptr<int32_t>(), sel32.size() * 4);
    }

    exl3_moe_cpu_forward_raw
    (
        handle,
        reinterpret_cast<const at::Half*>(x.data_ptr()),
        sel32.data(),
        reinterpret_cast<const at::Half*>(weights.data_ptr()),
        out.data_ptr<float>(),
        m_total, top_k,
        static_cast<int>(num_threads)
    );
}

// C ABI consumed by expert_stream/host/cpu_experts.h.
// expert_stream/host/cpu_experts.h calls these through addresses Exl3CpuQuantTrait resolves with dlsym, so the
// service module links nothing of this one. Neither throws across the boundary.

// CpuExpertForward: call->rows token rows (x fp16 [rows][hidden]) through each row's experts, into out
// (fp32 [rows][hidden]): overwritten, or added to when `accumulate` is nonzero. The calling thread is worker 0.
extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_forward(
    const SglangCpuExpertsForward* call) noexcept
{
    if (!call || call->abi_version != SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION) return 2;
    const SglangCpuExpertsForward& c = *call;
    if (c.rows < 1 || c.k < 0 || c.k > 32 || !c.x || !c.out || c.threads < 1 || (c.k && (!c.slots || !c.weights)))
        return 2;
    try
    {
        static thread_local std::vector<at::Half> wts;
        const size_t n = static_cast<size_t>(c.rows) * c.k;
        wts.resize(n);
        for (size_t i = 0; i < n; ++i) wts[i] = at::Half(c.weights[i]);
        forward_raw(c.layer, static_cast<const at::Half*>(c.x), c.slots, wts.data(), c.out, c.rows, c.k, c.threads,
                    c.accumulate != 0);
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

// Register-only work at the forward's vector width: the GEMV inner loop's vpmaddwd/vpaddd chains, so the core holds
// the license the forward runs at. Polls the word every 16 iterations and the clock every 1024 (the deadline is the
// engine's CLOCK_MONOTONIC, which libstdc++'s steady_clock reads).
namespace {
inline bool keep_warm_done(const uint32_t* word, uint32_t seen, int64_t deadline_ns, uint32_t tick)
{
    if (__atomic_load_n(word, __ATOMIC_ACQUIRE) != seen) return true;
    return (tick & 63) == 0 &&
           std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count() >= deadline_ns;
}
M1_TARGET_BW __attribute__((noinline)) int32_t keep_warm_bw(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    __m512i a = _mm512_set1_epi16(3), b = _mm512_set1_epi16(5), c0 = _mm512_setzero_si512(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !keep_warm_done(word, seen, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm512_add_epi32(c0, _mm512_madd_epi16(a, b));
            c1 = _mm512_add_epi32(c1, _mm512_madd_epi16(b, a));
            c2 = _mm512_add_epi32(c2, _mm512_madd_epi16(a, a));
            c3 = _mm512_add_epi32(c3, _mm512_madd_epi16(b, b));
            a = _mm512_xor_si512(a, c3);
        }
    return _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(c0, c1), _mm512_add_epi32(c2, a)));
}
M1_TARGET_AVX2 __attribute__((noinline)) int32_t keep_warm_avx2(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    __m256i a = _mm256_set1_epi16(3), b = _mm256_set1_epi16(5), c0 = _mm256_setzero_si256(), c1 = c0, c2 = c0, c3 = c0;
    for (uint32_t tick = 0; !keep_warm_done(word, seen, deadline_ns, tick); ++tick)
        for (int i = 0; i < 16; ++i) {
            c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(a, b));
            c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(b, a));
            c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(a, a));
            c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(b, b));
            a = _mm256_xor_si256(a, c3);
        }
    const __m256i t = _mm256_add_epi32(_mm256_add_epi32(c0, c1), _mm256_add_epi32(c2, a));
    return _mm256_extract_epi32(t, 0) + _mm256_extract_epi32(t, 7);
}
int32_t keep_warm_scalar(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    for (uint32_t tick = 0; !keep_warm_done(word, seen, deadline_ns, tick); ++tick) _mm_pause();
    return 0;
}
std::atomic<int32_t> g_keep_warm_sink{0};
}  // namespace

// CpuExpertKeepWarm: holds `threads` workers (the caller as worker 0, each pinned as the forward pins it) in
// register-only work of the forward's ISA until *word != seen or CLOCK_MONOTONIC reaches deadline_ns. Returns 0, 1 on
// a kernel error, 2 on invalid arguments.
extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_keep_warm(
    int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept
{
    if (threads < 1 || word == nullptr) return 2;
    try
    {
        freeze_compute_cores();
        if (!g_compute_cores.empty() && size_t(threads) > g_compute_cores.size()) return 2;
        std::atomic<int> pin_error{0};
        #pragma omp parallel num_threads(threads)
        {
            pin_compute_worker(omp_get_thread_num(), pin_error);
            const int32_t r = g_isa == Isa::Scalar ? keep_warm_scalar(word, seen, deadline_ns)
                              : g_isa == Isa::Avx2 ? keep_warm_avx2(word, seen, deadline_ns)
                                                   : keep_warm_bw(word, seen, deadline_ns);
            g_keep_warm_sink.fetch_add(r, std::memory_order_relaxed);
        }
        return pin_error.load(std::memory_order_relaxed) ? 1 : 0;
    }
    catch (...)
    {
        return 1;
    }
}

// Configure worker i on cores[i], with the calling thread as worker 0. Refuse
// reconfiguration after the first forward.
extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_set_cores(const int32_t* cores, int32_t n)
    noexcept
{
    if (n < 1 || n > CPU_SETSIZE || cores==nullptr) return 2;
    for(int i=0;i<n;++i) {
        if(cores[i]<0 || cores[i]>=CPU_SETSIZE)return 2;
        for(int j=0;j<i;++j)if(cores[i]==cores[j])return 2;
    }
    try {
        std::lock_guard<std::mutex> lock(g_cores_mutex);
        if (g_compute_started.load(std::memory_order_relaxed)) return 1;
        g_configured_cores.assign(cores, cores + n);
        return 0;
    } catch (...) {
        return 1;
    }
}

extern "C" __attribute__((visibility("default"))) int sglang_exl3_cpu_experts_register_slabs(
    const void* const* slabs, int32_t capacity, int32_t hidden, int32_t intermediate, int32_t bits, int32_t swizzled,
    float act_limit, int64_t* handle) noexcept
{
    if (!slabs || !handle || capacity < 1 || bits < 1 || bits > 8 || (swizzled != 0 && swizzled != 1)) return 2;
    // make_matrix's limits: 128-element blocks, and k (hidden for gate/up, intermediate for down) <= 8192 for the
    // int32 accumulators.
    if (hidden < 128 || intermediate < 128 || hidden % 128 || intermediate % 128 || hidden > 8192 || intermediate > 8192)
        return 2;
    if (!std::isfinite(act_limit) || act_limit < 0.0f) return 2;
    for (int i = 0; i < kSlabNames; ++i)
        if (!slabs[i]) return 2;
    try {
        auto entry = std::make_unique<RegisteredLayer>();
        entry->info = {capacity, hidden, intermediate, true, 0, act_limit};
        for (int i = 0; i < kSlabNames; ++i)
            entry->strided.base[i] = static_cast<const uint8_t*>(slabs[i]);
        entry->strided.hidden = hidden;
        entry->strided.intermediate = intermediate;
        entry->strided.bits = bits;
        entry->strided.swz = swizzled && bits != 8 ? 1 : 0;  // make_matrix's rule: K8 is never swizzled
        std::lock_guard<std::mutex> lock(g_layers_mutex);
        g_layers.push_back(std::move(entry));
        *handle = static_cast<int64_t>(g_layers.size() - 1);
        return 0;
    } catch (...) {
        return 1;
    }
}
