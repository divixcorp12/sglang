// Copyright (c) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Adapted from intel/neural-speed bfd5d3c17dee18a20e2768f855f2d8fe132fc579:
// bestla/bestla/{kernel_ref.h,kernel_avx2.h,kernel_avx512f.h}, decompress_s3_s8 and unpack_{1,2}bits.
// Fixed 32/64-element blocks, byte pointers instead of bitfield types, memcpy for unaligned scalar loads,
// and compile-time ISA guards replace the upstream dispatcher, tail logic and library dependencies.
#pragma once
#include "../expert_stream/host/cpu_experts/isa.hpp"
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace sglang::gptq_int3::bestla {
inline void scalar(const uint8_t* lo, const uint8_t* hi, int8_t* dst, int count) {
  for (int i = 0; i < count; ++i)
    dst[i] = int8_t(((lo[i / 4] >> (2 * (i % 4))) & 3) | (((hi[i / 8] >> (i % 8)) & 1) << 2)) - 4;
}
#ifdef SGLANG_CPU_EXPERTS_AVX2
inline void avx2(const uint8_t* lo, const uint8_t* hi, int8_t* dst) {
  const auto shift = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
  const auto shuffle = _mm256_set_epi8(
      15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0, 15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
  auto raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(lo));
  auto ordered = _mm256_permutevar8x32_epi32(_mm256_broadcastq_epi64(raw), _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0));
  auto low = _mm256_shuffle_epi8(_mm256_and_si256(_mm256_srlv_epi32(ordered, shift), _mm256_set1_epi8(3)), shuffle);
  uint32_t bits;
  std::memcpy(&bits, hi, 4);
  auto high = _mm256_srlv_epi32(_mm256_set1_epi32(int32_t(bits)), _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0));
  high = _mm256_and_si256(high, _mm256_set1_epi32(15));
  high = _mm256_and_si256(
      _mm256_mullo_epi32(high, _mm256_set1_epi32((1 << 23) + (1 << 16) + (1 << 9) + (1 << 2))), _mm256_set1_epi8(4));
  _mm256_storeu_si256(
      reinterpret_cast<__m256i*>(dst), _mm256_sub_epi8(_mm256_or_si256(low, high), _mm256_set1_epi8(4)));
}
#endif
#ifdef SGLANG_CPU_EXPERTS_AVX512
inline void avx512(const uint8_t* lo, const uint8_t* hi, int8_t* dst) {
  auto raw = _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(lo)));
  auto ordered = _mm512_permutexvar_epi32(_mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0), raw);
  auto low = _mm512_and_si512(
      _mm512_srlv_epi32(ordered, _mm512_set_epi32(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0)),
      _mm512_set1_epi8(3));
  const auto shuffle = _mm512_broadcast_i32x4(_mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0));
  low = _mm512_shuffle_epi8(low, shuffle);
  uint64_t bits;
  std::memcpy(&bits, hi, 8);
  auto high = _mm512_maskz_mov_epi8(__mmask64(bits), _mm512_set1_epi8(4));
  _mm512_storeu_si512(dst, _mm512_sub_epi8(_mm512_or_si512(low, high), _mm512_set1_epi8(4)));
}
#endif
}  // namespace sglang::gptq_int3::bestla
