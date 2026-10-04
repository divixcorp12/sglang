// Derived from GGML x86 quants.c, llama.cpp 889edf43ddae0cfe9a4564a882764dc879759870.
// MIT, ggml authors; see ../upstream/LICENSE.llama.cpp.
// Changes: GPU row/scale access, adjacent-nibble interleave, return signature, several Q8_0 vectors per weight row.
#pragma once
// Included by quant.hpp inside its namespace, after rounded and sf_index.
#include <cstring>
#include "../upstream/kernels.h"

struct GpuRow {
    const uint8_t* w;
    const uint8_t* sf;
    int row, k;
    uint8_t tail[32]{};
    GpuRow(const uint8_t* weights, const uint8_t* scales, int r, int columns)
        : w(weights + size_t(r)*(columns/2)), sf(scales), row(r), k(columns) {
        if (k%64) std::memcpy(tail, w+(k/64)*32, (k%64)/2);
    }
    const uint8_t* bytes(int ib) const { return ib*64+64 <= k ? w+ib*32 : tail; }
    float scale(int ib, int group) const {
        int g=ib*4+group;
        if (g>=k/16) return 0;
        uint8_t b=sf[sf_index(row,g,k/16)];
        // GGML's LUT is doubled. Preserve the signed GPU E4M3 convention.
        float v=ggml_ue4m3_to_fp32(b & 127);
        return b & 128 ? -v : v;
    }
};

// x against M Q8_0 vectors ys[0..M) into out[0..M): each weight block is decoded once for all M. Each out[t] is
// computed by exactly the operations a one-vector call makes, so it is bitwise that call's result.
template <int M>
inline void dot_gpu_rows(int n, const GpuRow& x, const block_q8_0* const* ys, float* out) {
    assert(n % QK_NVFP4 == 0);
    const int nb = n / QK_NVFP4;
    int ib = 0;
    float sumf[M] = {};

#if defined(__AVX2__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum[M];
    for (int t = 0; t < M; ++t) accum[t] = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 16));

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        //reordering
        const __m256i q4_01 = MM256_SET_M128I(_mm_unpackhi_epi8(q4_01_lo,q4_01_hi), _mm_unpacklo_epi8(q4_01_lo,q4_01_hi));
        const __m256i q4_23 = MM256_SET_M128I(_mm_unpackhi_epi8(q4_23_lo,q4_23_hi),_mm_unpacklo_epi8(q4_23_lo,q4_23_hi));

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
    for (int t = 0; t < M; ++t) sumf[t] = hsum_float_8(accum[t]);

#elif defined(__AVX__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum[M];
    for (int t = 0; t < M; ++t) accum[t] = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x.bytes(ib) + 16));

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        const __m128i q4_0 = _mm_unpacklo_epi8(q4_01_lo, q4_01_hi);
        const __m128i q4_1 = _mm_unpackhi_epi8(q4_01_lo, q4_01_hi);
        const __m128i q4_2 = _mm_unpacklo_epi8(q4_23_lo, q4_23_hi);
        const __m128i q4_3 = _mm_unpackhi_epi8(q4_23_lo, q4_23_hi);

        const float w0 = x.scale(ib, 0), w1 = x.scale(ib, 1), w2 = x.scale(ib, 2), w3 = x.scale(ib, 3);

        for (int t = 0; t < M; ++t) {
            const block_q8_0* y = ys[t];
            const __m128i q8_0 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 0));
            const __m128i q8_1 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 16));
            const __m128i q8_2 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 0));
            const __m128i q8_3 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 16));

            const __m128i p0_i32 = mul_sum_i8_pairs(q4_0, q8_0);
            const __m128i p1_i32 = mul_sum_i8_pairs(q4_1, q8_1);
            const __m128i p2_i32 = mul_sum_i8_pairs(q4_2, q8_2);
            const __m128i p3_i32 = mul_sum_i8_pairs(q4_3, q8_3);

            const __m128 p0 = _mm_cvtepi32_ps(p0_i32);
            const __m128 p1 = _mm_cvtepi32_ps(p1_i32);
            const __m128 p2 = _mm_cvtepi32_ps(p2_i32);
            const __m128 p3 = _mm_cvtepi32_ps(p3_i32);

            const __m256 p01 = _mm256_set_m128(p1, p0);
            const __m256 p23 = _mm256_set_m128(p3, p2);

            const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
            const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

            const float s0 = w0 * dy0;
            const float s1 = w1 * dy0;
            const float s2 = w2 * dy1;
            const float s3 = w3 * dy1;

            const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
            const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

            accum[t] = _mm256_add_ps(accum[t], _mm256_mul_ps(p01, scales01));
            accum[t] = _mm256_add_ps(accum[t], _mm256_mul_ps(p23, scales23));
        }
    }
    for (int t = 0; t < M; ++t) sumf[t] = hsum_float_8(accum[t]);

#endif

    for (;ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = x.scale(ib, s_idx);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            for (int t = 0; t < M; ++t) {
                const block_q8_0* y = ys[t];
                const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

                int sumi_lo = 0, sumi_hi = 0;
                for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                    const uint8_t qv = x.bytes(ib)[s_idx*(QK_NVFP4_SUB/2) + j];
                    sumi_lo += y[2*ib + q8_block].qs[q8_off + 2*j] * kvalues_fp4[qv & 0xf];
                    sumi_hi += y[2*ib + q8_block].qs[q8_off + 2*j + 1] * kvalues_fp4[qv >>  4];
                }

                sumf[t] += dy * d * (sumi_lo + sumi_hi);
            }
        }
    }
    for (int t = 0; t < M; ++t) out[t] = sumf[t];
}

inline float dot_gpu(int n, const GpuRow& x, const block_q8_0* y) {
    float result;
    dot_gpu_rows<1>(n, x, &y, &result);
    return result;
}
