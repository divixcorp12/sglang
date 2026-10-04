// dot_rows<Isa::Avx2, M>: GGML's AVX2 dot product, compiled for AVX2/FMA by attribute (SGLANG_TARGET_AVX2) so a
// baseline x86-64 library carries it beside the scalar tier and runs it only where the host has AVX2.
// Derived from GGML x86 quants.c, llama.cpp 889edf43ddae0cfe9a4564a882764dc879759870.
// MIT, ggml authors; see ../upstream/LICENSE.llama.cpp.
// Changes: GPU row/scale access, adjacent-nibble interleave, return signature, several Q8_0 vectors per weight row;
// the helpers upstream defines under __AVX2__ are restated here under the target attribute.
#pragma once
#include "math.hpp"
#include <immintrin.h>

namespace sglang::nvfp4_cpu {
namespace {
namespace avx2_detail {

// GGML's MM256_SET_M128I(hi, lo).
SGLANG_TARGET_AVX2 inline __m256i set_m128i(__m128i hi, __m128i lo)
{
    return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

SGLANG_TARGET_AVX2 inline float hsum_float_8(const __m256 x)
{
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

SGLANG_TARGET_AVX2 inline __m256i mul_add_epi8(const __m256i x, const __m256i y)
{
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

}  // namespace avx2_detail

template <>
struct DotRows<Isa::Avx2>
{
    template <int M>
    SGLANG_TARGET_AVX2 static void rows(int n, const GpuRow& x, const block_q8_0* const* ys, float* out)
    {
        using namespace avx2_detail;
        assert(n % QK_NVFP4 == 0);
        const int nb = n / QK_NVFP4;

        const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
        const __m128i m4b  = _mm_set1_epi8(0x0f);
        const __m256i mone = _mm256_set1_epi16(1);

        __m256 accum[M];
        for (int t = 0; t < M; ++t) accum[t] = _mm256_setzero_ps();
        for (int ib = 0; ib < nb; ib++){

            const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 0));
            const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 16));

            const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
            const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
            const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
            const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

            //reordering
            const __m256i q4_01 = set_m128i(_mm_unpackhi_epi8(q4_01_lo,q4_01_hi), _mm_unpacklo_epi8(q4_01_lo,q4_01_hi));
            const __m256i q4_23 = set_m128i(_mm_unpackhi_epi8(q4_23_lo,q4_23_hi),_mm_unpacklo_epi8(q4_23_lo,q4_23_hi));

            const float w0 = x.scale(ib, 0), w1 = x.scale(ib, 1), w2 = x.scale(ib, 2), w3 = x.scale(ib, 3);

            for (int t = 0; t < M; ++t) {
                const block_q8_0* y = ys[t];
                const __m256i q8_01 = _mm256_loadu_si256((const __m256i *)y[2*ib + 0].qs);
                const __m256i q8_23 = _mm256_loadu_si256((const __m256i *)y[2*ib + 1].qs);

                const __m256i p01 = mul_add_epi8(q4_01,q8_01);
                const __m256i p_1 = _mm256_madd_epi16(p01, mone);

                const __m256i p23 = mul_add_epi8(q4_23,q8_23);
                const __m256i p_2 = _mm256_madd_epi16(p23, mone);

                const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
                const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

                const float s0 = w0 * dy0;
                const float s1 = w1 * dy0;
                const float s2 = w2 * dy1;
                const float s3 = w3 * dy1;

                const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
                const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

                accum[t] = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum[t]);
                accum[t] = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum[t]);
            }
        }
        for (int t = 0; t < M; ++t) out[t] = hsum_float_8(accum[t]);
    }
};

}  // namespace
}  // namespace sglang::nvfp4_cpu
