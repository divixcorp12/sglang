// Unmodified function bodies from llama.cpp 889edf43ddae0cfe9a4564a882764dc879759870.
#include "kernels.h"

void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;
    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m256i q8_01 = _mm256_loadu_si256((const __m256i *)y[2*ib + 0].qs);
        const __m256i q8_23 = _mm256_loadu_si256((const __m256i *)y[2*ib + 1].qs);

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        //reordering
        const __m256i q4_01 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_01_lo,q4_01_hi), _mm_unpacklo_epi64(q4_01_lo,q4_01_hi));
        const __m256i q4_23 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_23_lo,q4_23_hi),_mm_unpacklo_epi64(q4_23_lo,q4_23_hi));

        const __m256i p01 = mul_add_epi8(q4_01,q8_01);
        const __m256i p_1 = _mm256_madd_epi16(p01, mone);

        const __m256i p23 = mul_add_epi8(q4_23,q8_23);
        const __m256i p_2 = _mm256_madd_epi16(p23, mone);

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    sumf = hsum_float_8(accum);

#elif defined(__AVX__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m128i q8_0 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 0));
        const __m128i q8_1 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 16));
        const __m128i q8_2 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 0));
        const __m128i q8_3 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 16));

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        const __m128i q4_0 = _mm_unpacklo_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_1 = _mm_unpackhi_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_2 = _mm_unpacklo_epi64(q4_23_lo, q4_23_hi);
        const __m128i q4_3 = _mm_unpackhi_epi64(q4_23_lo, q4_23_hi);

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

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_add_ps(accum, _mm256_mul_ps(p01, scales01));
        accum = _mm256_add_ps(accum, _mm256_mul_ps(p23, scales23));
    }
    sumf = hsum_float_8(accum);

#endif

    for (;ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = GGML_CPU_UE4M3_TO_FP32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_fp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_fp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
