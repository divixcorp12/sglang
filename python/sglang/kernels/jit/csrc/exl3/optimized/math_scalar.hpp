// The scalar tier's GEMV tiles: each 16x16 tile decoded to fp32 and multiplied with the fp32 transformed rows (in.tin).
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sglang::exl3_cpu {
namespace {

template <>
struct Tiles<Isa::Scalar>
{
    template <bool Accumulate = false>
    static void run(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
    {
        with_count<8>(mat.bits, [&]<int bits>() { tiles<bits, Accumulate>(mat, in, tout, m, tn0, tn1); });
    }

    template <int bits, bool Accumulate>
    static void tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
    {
        const int tiles_k = mat.k / 16;
        const int tiles_n = mat.n / 16;
        constexpr int packed_size = 16 * bits;
        constexpr auto perm = make_tc_perm();

        for (int tile_n = tn0; tile_n < tn1; ++tile_n) {
            float acc[GENERIC_MAX_M][16] = {};
            for (int tile_k = 0; tile_k < tiles_k; ++tile_k) {
                const uint16_t* packed = mat.trellis + (size_t(tile_k) * tiles_n + tile_n) * packed_size;
                float tile[256];
                for (int t = 0; t < 256; ++t) tile[perm[t]] = decode_mul1_scalar(decode_state_scalar<bits>(packed, t));
                for (int i = 0; i < m; ++i) {
                    const float* x = in.tin + size_t(i) * mat.k + tile_k * 16;
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < 16; ++c) acc[i][c] += x[r] * tile[r * 16 + c];
                }
            }
            for (int i = 0; i < m; ++i) {
                float* out = tout + size_t(i) * mat.n + tile_n * 16;
                if constexpr (Accumulate) {
                    for (int c = 0; c < 16; ++c) out[c] += acc[i][c];
                } else {
                    std::memcpy(out, acc[i], 16 * sizeof(float));
                }
            }
        }
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
