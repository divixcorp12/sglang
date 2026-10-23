// IEEE binary16 storage and its conversions, for the EXL3 CPU kernel (no torch types).
//
// With F16C in the build's -march the conversions are the hardware ones, as c10::Half's are; without it they are
// the exact software equivalents. Both round to nearest even, and half to float is exact, so the two agree on every
// finite input (NaN payloads aside, which the kernel never feeds them: weights are checked finite).
#pragma once
#include <cstdint>
#include <cstring>
#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace sglang::exl3_cpu {

/// \brief A binary16 value as stored: sign vectors and activations in the pinned tier are arrays of these.
struct Fp16 {
    uint16_t bits;
};

inline float fp16_to_float(Fp16 h)
{
#if defined(__F16C__)
    return _cvtsh_ss(h.bits);
#else
    const uint32_t sign = uint32_t(h.bits & 0x8000u) << 16;
    const uint32_t exponent = (h.bits >> 10) & 0x1fu;
    const uint32_t mantissa = h.bits & 0x3ffu;
    uint32_t out;
    if (exponent == 0) {
        // Zero or subnormal: mantissa * 2^-24, exact in binary32.
        const float magnitude = float(mantissa) * 0x1p-24f;
        std::memcpy(&out, &magnitude, sizeof(out));
        out |= sign;
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
#endif
}

inline Fp16 float_to_fp16(float f)
{
#if defined(__F16C__)
    return Fp16{_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT)};
#else
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t sign = uint16_t((x >> 16) & 0x8000u);
    const uint32_t a = x & 0x7fffffffu;
    if (a > 0x7f800000u) return Fp16{uint16_t(sign | 0x7e00u)};  // NaN
    if (a >= 0x477ff000u) return Fp16{uint16_t(sign | 0x7c00u)};  // >= 65520 rounds to infinity
    if (a < 0x38800000u) {                                         // below the smallest normal
        if (a <= 0x33000000u) return Fp16{sign};                   // at most half the smallest subnormal
        const uint32_t mantissa = (a & 0x7fffffu) | 0x800000u;
        const uint32_t shift = 126u - (a >> 23);
        uint32_t h = mantissa >> shift;
        const uint32_t rest = mantissa & ((1u << shift) - 1u), half = 1u << (shift - 1u);
        if (rest > half || (rest == half && (h & 1u))) ++h;
        return Fp16{uint16_t(sign | h)};
    }
    uint32_t h = (a - 0x38000000u) >> 13;
    const uint32_t rest = a & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return Fp16{uint16_t(sign | h)};
#endif
}

}  // namespace sglang::exl3_cpu
