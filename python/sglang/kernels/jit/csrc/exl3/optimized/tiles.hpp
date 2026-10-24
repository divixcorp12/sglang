// run_tiles: one chunk's GEMV over output tiles [tn0, tn1), on the build's tier. The register kernels take the DSV4.1
// plan's compact input (and, on the Bw tier, the generic plan's one-token 3-bit chunks); every other call runs the
// tier's Tiles over each 128-input block's view (block 0 stores, later blocks add onto tout in their epilogue), then
// each residual row onto its token row.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
// The tier headers in this order: the definition order the kernels were validated in (bit-exact per tier).
#include "math_avx512.hpp"
#include "register_kernels.hpp"
#include "math_avx2.hpp"
#include "math_scalar.hpp"
#include <cstddef>

namespace sglang::exl3_cpu {
namespace {

// Only this worker's columns [tn0, tn1) of tout are touched, so the sums need no synchronization.
void run_tiles(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1,
               [[maybe_unused]] bool grouped)
{
    if (tn0 >= tn1) return;
#if defined(SGLANG_CPU_EXPERTS_AVX512)
    if (in.compact || (kIsa == Isa::Bw && m == 1 && mat.bits == 3 && mat.k > kActBlock)) {
        with_count<CHUNK_M>(m, [&]<int M>() { register_tiles<M>(mat, in, tout, tn0, tn1, grouped); });
        return;
    }
#endif
    if constexpr (kIsa == Isa::Scalar) {
        Tiles<kIsa>::run<false>(mat, in, tout, m, tn0, tn1);
        return;
    }
    const int rows = kActRows * m;
    const int n = mat.n;
    const int c0 = tn0 * 16, c1 = tn1 * 16;
    constexpr int block_tiles = kActBlock / 16;
    const size_t packed_size = size_t(16) * mat.bits;
    for (int b = 0; b < mat.k / kActBlock; ++b) {
        // The block's view: native k-tile rows are tiles_n tiles apart; consecutive k-tiles of a swizzled 8-tile group
        // are 8 apart, the group stride staying the whole matrix's (group_tiles_k).
        Exl3Projection sub = mat;
        sub.k = kActBlock;
        sub.trellis = mat.trellis + size_t(b) * block_tiles * (mat.swz ? 8 : n / 16) * packed_size;
        PreparedIn sub_in = in;
        sub_in.splat32 = in.splat32 + size_t(b) * rows * kActBlock;
        sub_in.splat_dup = in.splat_dup ? in.splat_dup + size_t(b) * rows * kActBlock : nullptr;
        for (int i = 0; i < rows; ++i) {
            sub_in.q[i] = in.bq[b * MAX_M + i];
            sub_in.sum_x8[i] = in.bsum[b * MAX_M + i];
        }
        if (b) Tiles<kIsa>::run<true>(sub, sub_in, tout, rows, tn0, tn1);
        else Tiles<kIsa>::run<false>(sub, sub_in, tout, rows, tn0, tn1);
    }
    for (int r = m; r < rows; ++r) {
        float* t = tout + size_t(r - m) * n;
        const float* q = tout + size_t(r) * n;
        for (int c = c0; c < c1; ++c) t[c] += q[c];
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
