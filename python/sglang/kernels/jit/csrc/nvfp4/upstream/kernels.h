// Selected GGML CPU definitions; see README.md and LICENSE.llama.cpp.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <assert.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif
#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
#define QK8_0 32
typedef uint16_t ggml_fp16_t;
typedef struct { uint8_t d[4]; uint8_t qs[32]; } block_nvfp4;
typedef struct { ggml_fp16_t d; int8_t qs[32]; } block_q8_0;
static const int8_t kvalues_fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_CPU_UE4M3_TO_FP32(x) ggml_ue4m3_to_fp32(x)
#if defined(__AVX__)
#define MM256_SET_M128I(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(b),(a),1)
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}


#if defined(__AVX2__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
void ggml_vec_dot_nvfp4_q8_0(int, float*, size_t, const void*, size_t, const void*, size_t, int);
void quantize_row_q8_0_ref(const float*, block_q8_0*, int64_t);
#ifdef __cplusplus
}
#endif
