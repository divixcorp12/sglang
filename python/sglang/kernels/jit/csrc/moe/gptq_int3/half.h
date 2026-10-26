// Explicit IEEE binary16 boundaries, including subnormals and ties-to-even. Bit tests survive -Ofast.
#pragma once
#include <bit>
#include <cstdint>

namespace sglang::gptq_int3 {
inline float from_half(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023;
  if (e == 0) {
    if (m == 0) return std::bit_cast<float>(sign);
    int exponent = -14;
    while ((m & 1024) == 0) {
      m <<= 1;
      --exponent;
    }
    return std::bit_cast<float>(sign | (uint32_t(exponent + 127) << 23) | ((m & 1023) << 13));
  }
  return std::bit_cast<float>(sign | ((e == 31 ? 255 : e + 112) << 23) | (m << 13));
}
inline uint16_t to_half(float x) {
  uint32_t bits = std::bit_cast<uint32_t>(x), sign = (bits >> 16) & 0x8000;
  uint32_t e = (bits >> 23) & 255, m = bits & 0x7fffff;
  if (e == 255) return uint16_t(sign | 0x7c00 | (m ? 0x200 : 0));
  int exponent = int(e) - 127 + 15;
  if (exponent >= 31) return uint16_t(sign | 0x7c00);
  if (exponent <= 0) {
    if (exponent < -10) return uint16_t(sign);
    m |= 0x800000;
    int shift = 14 - exponent;
    uint32_t value = m >> shift, remainder = m & ((1u << shift) - 1), tie = 1u << (shift - 1);
    return uint16_t(sign | (value + (remainder > tie || (remainder == tie && (value & 1)))));
  }
  uint32_t value = (uint32_t(exponent) << 10) | (m >> 13), remainder = m & 8191;
  return uint16_t(sign | (value + (remainder > 4096 || (remainder == 4096 && (value & 1)))));
}
inline float round_half(float x) {
  return from_half(to_half(x));
}
inline bool finite(float x) {
  return (std::bit_cast<uint32_t>(x) & 0x7f800000) != 0x7f800000;
}
}  // namespace sglang::gptq_int3
