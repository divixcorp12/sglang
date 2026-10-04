// The scalar tier's GEMV tiles (scalar_tiles).
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sglang::exl3_cpu {
namespace {

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

}  // namespace
}  // namespace sglang::exl3_cpu
