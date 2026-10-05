// EXL3's forward: the tier dispatch of the GEMV tiles (run_tiles), the phase helpers, the plans and the plan entry
// (Exl3Quant::dispatch). Included by kernel.cpp.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is
// the ISA tier. The primary template is the generic plan. PlanTraits<Dsv41Shape, Isa::Bw> turns on the fast path that
// was measured and validated bit-exact on AVX-512BW (csrc/exl3/optimized/README.txt); every other (Shape, I) pair runs
// the generic plan. ForwardPlan<Shape, I, true> also times each phase and prints one line per forward.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
// The math headers in this order: the definition order the kernels were validated in (bit-exact per tier); another
// order changes what GCC inlines and clones.
#include "math_avx512.hpp"
#include "math_avx2.hpp"
#include "math_scalar.hpp"
#include "shapes.hpp"
#include <c10/util/Half.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ratio>
#include <utility>
#include <vector>

namespace sglang::exl3_cpu {
namespace {

// -------------------------------------------------------------------------------------------
//   Dispatch
// -------------------------------------------------------------------------------------------

template <Isa I>
void run_tiles_raw(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Vbmi)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: vbmi_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: vbmi_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: vbmi_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: vbmi_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: vbmi_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: vbmi_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: vbmi_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: vbmi_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: vbmi_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: vbmi_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: vbmi_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: vbmi_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: vbmi_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: vbmi_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: vbmi_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: vbmi_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: vbmi_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: vbmi_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: vbmi_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: vbmi_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: vbmi_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: vbmi_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: vbmi_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: vbmi_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: vbmi_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: vbmi_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: vbmi_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: vbmi_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            // K8: byte pairing impossible (shift % 8 == 0) and the byte windows straddle
            // the register pairs -- measured slower than the dword scheme, so route there
            case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Vnni)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: vnni_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: vnni_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: vnni_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: vnni_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: vnni_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: vnni_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: vnni_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: vnni_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: vnni_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: vnni_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: vnni_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: vnni_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: vnni_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: vnni_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: vnni_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: vnni_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: vnni_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: vnni_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: vnni_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: vnni_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: vnni_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: vnni_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: vnni_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: vnni_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: vnni_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: vnni_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: vnni_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: vnni_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 0: vnni_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: vnni_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: vnni_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: vnni_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Bw)
    {
        switch (mat.bits * 4 + m - 1)
        {
            case 1 * 4 + 0: bw_tiles<1, 1>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 1: bw_tiles<1, 2>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 2: bw_tiles<1, 3>(mat, in, tout, tn0, tn1); return;
            case 1 * 4 + 3: bw_tiles<1, 4>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 0: bw_tiles<2, 1>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 1: bw_tiles<2, 2>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 2: bw_tiles<2, 3>(mat, in, tout, tn0, tn1); return;
            case 2 * 4 + 3: bw_tiles<2, 4>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 0: bw_tiles<3, 1>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 1: bw_tiles<3, 2>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 2: bw_tiles<3, 3>(mat, in, tout, tn0, tn1); return;
            case 3 * 4 + 3: bw_tiles<3, 4>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 0: bw_tiles<4, 1>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 1: bw_tiles<4, 2>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 2: bw_tiles<4, 3>(mat, in, tout, tn0, tn1); return;
            case 4 * 4 + 3: bw_tiles<4, 4>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 0: bw_tiles<5, 1>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 1: bw_tiles<5, 2>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 2: bw_tiles<5, 3>(mat, in, tout, tn0, tn1); return;
            case 5 * 4 + 3: bw_tiles<5, 4>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 0: bw_tiles<6, 1>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 1: bw_tiles<6, 2>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 2: bw_tiles<6, 3>(mat, in, tout, tn0, tn1); return;
            case 6 * 4 + 3: bw_tiles<6, 4>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 0: bw_tiles<7, 1>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 1: bw_tiles<7, 2>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 2: bw_tiles<7, 3>(mat, in, tout, tn0, tn1); return;
            case 7 * 4 + 3: bw_tiles<7, 4>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 0: bw_tiles<8, 1>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 1: bw_tiles<8, 2>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 2: bw_tiles<8, 3>(mat, in, tout, tn0, tn1); return;
            case 8 * 4 + 3: bw_tiles<8, 4>(mat, in, tout, tn0, tn1); return;
        }
        return;
    }
    else if constexpr (I == Isa::Avx2)
    {
        switch (mat.bits)
        {
            case 1: avx2_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: avx2_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: avx2_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: avx2_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: avx2_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: avx2_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: avx2_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: avx2_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
    else
    {
        switch (mat.bits)
        {
            case 1: scalar_tiles<1>(mat, in, tout, m, tn0, tn1); return;
            case 2: scalar_tiles<2>(mat, in, tout, m, tn0, tn1); return;
            case 3: scalar_tiles<3>(mat, in, tout, m, tn0, tn1); return;
            case 4: scalar_tiles<4>(mat, in, tout, m, tn0, tn1); return;
            case 5: scalar_tiles<5>(mat, in, tout, m, tn0, tn1); return;
            case 6: scalar_tiles<6>(mat, in, tout, m, tn0, tn1); return;
            case 7: scalar_tiles<7>(mat, in, tout, m, tn0, tn1); return;
            default: scalar_tiles<8>(mat, in, tout, m, tn0, tn1); return;
        }
    }
}

// m token rows through the quantized kernels under the accuracy options (quantize_act's layout): per
// k-block sub-views summed in fp32, then each remainder row added onto its token row. Only this
// worker's columns [tn0, tn1) are touched, so the sums need no synchronization.
template <Isa I>
void run_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1, bool grouped = false)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)
    {
        if (m == 1 && mat.bits == 3 && act_blocked(mat.k))
        {
            register_tiles(mat,in,tout,tn0,tn1,grouped);
            return;
        }
    }
    if (I == Isa::Scalar || (ACT_ROWS == 1 && !act_blocked(mat.k)))
    {
        run_tiles_raw<I>(mat, in, tout, m, tn0, tn1);
        return;
    }
    const int rows = ACT_ROWS * m;
    const int n = mat.n;
    const int c0 = tn0 * 16, c1 = tn1 * 16;
    if (!act_blocked(mat.k))
    {
        run_tiles_raw<I>(mat, in, tout, rows, tn0, tn1);
    }
    else
    {
        const int B = EXL3_MOE_CPU_ACT_BLOCK;
        const int block_tiles = B / 16;
        const size_t packed_size = static_cast<size_t>(16) * mat.bits;
        thread_local std::vector<float> part;
        if (part.size() < static_cast<size_t>(rows) * n) part.resize(static_cast<size_t>(rows) * n);
        tl_swz_tiles_k = mat.k / 16;
        for (int b = 0; b < mat.k / B; ++b)
        {
            MoeCpuMatrix sub = mat;
            sub.k = B;
            // Native: k-tile rows are tiles_n tiles apart. Swizzled: consecutive k-tiles of an 8-tile
            // group are 8 tiles apart, with the group stride from tl_swz_tiles_k
            sub.trellis = mat.trellis + static_cast<size_t>(b) * block_tiles
                                        * (mat.swz ? 8 : n / 16) * packed_size;
            PreparedIn sub_in = in;
            sub_in.splat32 = in.splat32 + static_cast<size_t>(b) * rows * B;
            sub_in.splat_dup = in.splat_dup ? in.splat_dup + static_cast<size_t>(b) * rows * B : nullptr;
            for (int i = 0; i < rows; ++i)
            {
                sub_in.q[i] = in.bq[b * MAX_M + i];
                sub_in.sum_x8[i] = in.bsum[b * MAX_M + i];
            }
            float* dst = b ? part.data() : tout;
            run_tiles_raw<I>(sub, sub_in, dst, rows, tn0, tn1);
            if (b)
                for (int i = 0; i < rows; ++i)
                {
                    float* t = tout + static_cast<size_t>(i) * n;
                    const float* q = part.data() + static_cast<size_t>(i) * n;
                    for (int c = c0; c < c1; ++c) t[c] += q[c];
                }
        }
        tl_swz_tiles_k = 0;
    }
    for (int r = m; r < rows; ++r)
    {
        float* t = tout + static_cast<size_t>(r - m) * n;
        const float* q = tout + static_cast<size_t>(r) * n;
        for (int c = c0; c < c1; ++c) t[c] += q[c];
    }
}

// -------------------------------------------------------------------------------------------
//   Forward driver
// -------------------------------------------------------------------------------------------

struct ForwardCtx
{
    const Exl3Quant::Layer* layer;
    const at::Half* x;
    float* out;
    int m_total;
    bool zero_out;  // Overwrite: worker 0 zeroes out inside the team, so a team that never runs leaves out untouched.
    std::vector<Chunk> chunks;

    // workspace, per chunk (pointers into the persistent per-thread arena below: fresh
    // allocations per call cost more in first-touch page faults than the small phases do work)
    float* tout_g;       // chunks x m x I (quant space, then transformed in place)
    float* tout_u;
    float* tout_d;       // chunks x m x H
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    // Down tiles finished per (chunk, 128-output block), when a plan splits the down GEMVs finer than a block: the
    // worker that finishes a block's eighth tile applies its output transform.
    int* down_tiles_done;
};

template <Isa I>
M1_TARGET_AVX2
void transform_out_avx2(const MoeCpuMatrix& mat, float* tout, int m)
{
    const __m256 hs = _mm256_set1_ps(HAD_SCALE);
    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128)
        {
            float* v = tout + static_cast<size_t>(r) * mat.n + block;
            hadamard_128_avx2<I>(v);
            for (int i = 0; i < 128; i += 8)
            {
                const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    reinterpret_cast<const uint16_t*>(mat.svh) + block + i)));
                __m256 x = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), hs), s);
                if (mat.bias)
                    x = _mm256_add_ps(x, _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(
                            reinterpret_cast<const uint16_t*>(mat.bias) + block + i))));
                _mm256_storeu_ps(v + i, x);
            }
        }
}

template <Isa I>
__attribute__((noipa)) void transform_out(const MoeCpuMatrix& mat, float* tout, int m)
{
    if constexpr (I != Isa::Scalar) { transform_out_avx2<I>(mat, tout, m); return; }
    else
    {
        for (int r = 0; r < m; ++r)
            for (int block = 0; block < mat.n; block += 128)
            {
                float* v = tout + static_cast<size_t>(r) * mat.n + block;
                hadamard_128<I>(v);
                if (mat.bias)
                    for (int i = 0; i < 128; ++i)
                        v[i] = v[i] * HAD_SCALE * half_to_float(mat.svh[block + i])
                               + half_to_float(mat.bias[block + i]);
                else
                    for (int i = 0; i < 128; ++i)
                        v[i] *= HAD_SCALE * half_to_float(mat.svh[block + i]);
            }
    }
}


// Assign this worker its share of a phase's `total` GEMVs (all of one width, tiles_n tiles),
// calling gemv(j, t0, t1) per tile range. Few GEMVs (decode, small batches; each expert read
// once): the GEMVs' tiles form one flat range, split evenly across workers in 8-tile groups so
// no piece crosses a group of its GEMV (the swizzled band kernels' invariant; n % 128 == 0
// makes every tiles_n a multiple of 8). Whole-GEMV assignment left a 2:1 imbalance whenever
// 2 * cold experts fell between multiples of the worker count (16 gate/up GEMVs on 20 workers:
// twelve single-worker GEMVs set the phase time while eight workers idled half of it). Many
// GEMVs (prefill): whole GEMVs strided across workers, so all workers stream the same expert's
// chunks together and L3 serves the repeats; the imbalance there is at most one GEMV in four.
constexpr int FLAT_MAX_GEMVS_PER_WORKER = 4;

// Unit is the flat split's granularity in tiles: 8 (one 128-output group, which the swizzled kernels need) or, for
// a plan whose kernels take any tile pair, 2 (finer balance: 36 gate/up groups over 16 workers leave the slowest
// worker 3 groups against a mean of 2.25, where 144 pairs give every worker 9).
template <int Unit = 8, typename Gemv>
inline void assign_gemvs(int worker, int num_workers, int total, int tiles_n, Gemv gemv)
{
    static_assert(Unit == 2 || Unit == 8);
    if (total > FLAT_MAX_GEMVS_PER_WORKER * num_workers)
    {
        for (int j = worker; j < total; j += num_workers) gemv(j, 0, tiles_n);
        return;
    }
    const int64_t groups = static_cast<int64_t>(total) * (tiles_n / Unit);
    const int f0 = static_cast<int>(groups * worker / num_workers) * Unit;
    const int f1 = static_cast<int>(groups * (worker + 1) / num_workers) * Unit;
    for (int j = f0 / tiles_n; j * tiles_n < f1; ++j)
        gemv(j, std::max(f0 - j * tiles_n, 0), std::min(f1 - j * tiles_n, tiles_n));
}

// Same scalar/vector operations and block scale ordering as the original phase 2.
// Blocks are independent: only their own gate/up outputs and prepared input slices are written.
template<class Shape, Isa I, bool Wide = false, class Experts>
void middle_blocks(ForwardCtx& c,const Experts& E,int worker,int num_workers) {
    const int I_=Shape::intermediate(*c.layer), nb=I_/128, nc=int(c.chunks.size());
    const bool gated=Shape::gated(*c.layer);
    const int first=nc*nb*worker/num_workers,last=nc*nb*(worker+1)/num_workers;
    for (int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb,block=b*128;
        const auto& ch=c.chunks[j];
        for (int r=0;r<ch.m;++r) {
            float* g=c.tout_g+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            float* u=c.tout_u+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            if (gated) {
                auto mat=E.gate(ch.expert);mat.n=128;mat.svh+=block;if(mat.bias)mat.bias+=block;
                transform_out<I>(mat,g,1);
            }
            auto up=E.up(ch.expert);up.n=128;up.svh+=block;if(up.bias)up.bias+=block;
            transform_out<I>(up,u,1);
            const size_t count=128;
            float* a=gated?g:u;
            const float lim=Shape::act_limit(*c.layer)!=0.0f?Shape::act_limit(*c.layer):std::numeric_limits<float>::infinity();
                switch (Shape::activation(*c.layer)) {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            const float av = std::min(gv * cdf, lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 3: {
                        // gpt-oss clamped swiglu: g = min(g, limit); a = (clamp(u, -l, l) + 1) * g *
                        // sigmoid(1.702 * g)
                        const float lim = Shape::act_limit(*c.layer);
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = std::min(g[i], lim);
                            const float uv = std::clamp(u[i], -lim, lim);
                            g[i] = (uv + 1.0f) * gv / (1.0f + std::exp(-1.702f * gv));
                        }
                        break;
                    }
                    default:
                        for (size_t i = 0; i < count; ++i) {
                            const float uv = u[i] > 0.0f ? u[i] : 0.0f;
                            u[i] = uv * uv;
                        }
                        break;
                }

            auto& p=c.prep_d[j];
            float* dst=p.tin+size_t(r)*I_+block;
            prepare_block_avx2<I>(a,false,E.down(ch.expert).suh+block,dst,128);
            if(p.compact){compact_quantize_block<Wide>(p,r,ch.m,I_,b,dst);continue;}
            float* residual=p.tin+size_t(r+ch.m)*I_+block;
            for (int pass=0;pass<ACT_ROWS;++pass) {
                const int row=r+pass*ch.m;
                const size_t offset=size_t(b)*ACT_ROWS*ch.m*128+row*128;
                int32_t* splat=p.splat32+offset;
                float q;int32_t sum;
                const float* v=pass?residual:dst;
                quantize_row_avx2(v,splat,p.splat_dup+offset,128,q,sum);
                p.bq[b*MAX_M+row]=q;p.bsum[b*MAX_M+row]=sum;
                if(pass+1<ACT_ROWS) for(int i=0;i<128;++i)
                    residual[i]=v[i]-q*float(static_cast<int8_t>(splat[i]&0xff));
            }
        }
    }
}


template<class Shape, Isa I, bool Wide = false, class Experts>
void prepare_gu_blocks(ForwardCtx& c,const Experts& E,int worker,int num_workers) {
    const int K=Shape::hidden(*c.layer),nb=K/128,nc=int(c.chunks.size()),gu=!Shape::gated(*c.layer)?1:2;
    const int first=nc*gu*nb*worker/num_workers,last=nc*gu*nb*(worker+1)/num_workers;
    for(int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb;
        const auto& ch=c.chunks[j/gu];
        const bool up=gu==1 || j%gu;
        const MoeCpuMatrix& mat=up?E.up(ch.expert):E.gate(ch.expert);
        auto& p=(up?c.prep_u:c.prep_g)[j/gu];
        for(int r=0;r<ch.m;++r) {
            float* dst=p.tin+size_t(r)*K+b*128;
            prepare_block_avx2<I>(c.x+size_t(ch.token[r])*K+b*128,true,mat.suh+b*128,dst,128);
            if(p.compact){compact_quantize_block<Wide>(p,r,ch.m,K,b,dst);continue;}
            float* residual=p.tin+size_t(r+ch.m)*K+b*128;
            for(int pass=0;pass<ACT_ROWS;++pass) {
                const int row=r+pass*ch.m;
                const size_t off=size_t(b)*ACT_ROWS*ch.m*128+row*128;
                float q;int32_t sum;const float* v=pass?residual:dst;
                int32_t* splat=p.splat32+off;
                quantize_row_avx2(v,splat,p.splat_dup+off,128,q,sum);
                p.bq[b*MAX_M+row]=q;p.bsum[b*MAX_M+row]=sum;
                if(pass+1<ACT_ROWS) for(int i=0;i<128;++i)
                    residual[i]=v[i]-q*float(static_cast<int8_t>(splat[i]&0xff));
            }
        }
    }
}


template <Isa I>
void transform_owned_blocks(const MoeCpuMatrix& mat,float* out,int m,int t0,int t1) {
    for(int r=0;r<m;++r) for(int block=t0*16;block<t1*16;block+=128) {
        auto sub=mat;sub.n=128;sub.svh+=block;if(sub.bias)sub.bias+=block;
        transform_out<I>(sub,out+size_t(r)*mat.n+block,1);
    }
}

// Coarse readiness is per expert: all gate/up outputs must exist before middle,
// and all middle blocks must be prepared before any down output band can run.
// The forward's phases, in team order; a barrier separates each from the next. The values index the profiling line
// ("moe_cpu phases(us)"): 4, the old whole-row down transform, is gone (Down's owned blocks include it) and prints 0.
enum class Phase : int
{
    PrepareGateUp = 0,  // quantize gate/up inputs
    GateUp = 1,         // gate/up GEMVs
    Middle = 2,         // gate/up output transform, activation, down input
    Down = 3,           // down GEMVs with their output transform
    Accumulate = 5,     // routing-weighted sum into out
};

// The calling thread's scratch for every forward it runs: a plan sizes its share of it per call (prepare_scratch).
struct ForwardArena
{
    std::vector<float> tin_g, tin_u, tin_d;
    std::vector<int32_t> splat_g, splat_u, splat_d;
    std::vector<int32_t> splat_dup_g, splat_dup_u, splat_dup_d;
    std::vector<int16_t> compact_g, compact_u, compact_d;
    std::vector<float> tout_g, tout_u, tout_d;
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    std::vector<float> bq_g, bq_u, bq_d;
    std::vector<int32_t> bsum_g, bsum_u, bsum_d;
    std::vector<int> down_tiles_done;
    // Moved into the call's ForwardCtx and back, so a forward allocates nothing once warm
    std::vector<std::vector<std::pair<int, float>>> per_expert;
    std::vector<Chunk> chunks;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

// What a (Shape, ISA) plan turns on. The primary template is the generic plan.
template <class Shape, Isa I>
struct PlanTraits
{
    static constexpr bool kCompactScratch = false;    // int16 compact activations instead of the int32 splats
    static constexpr bool kGroupedTraversal = false;  // range-aware multi-band traversal when one expert is routed
    static constexpr bool kWideSingleExpert = false;  // 512-wide quantization for one token through one expert
    static constexpr int kSplitTiles = 8;             // GEMV split unit in tiles (assign_gemvs)
};

// DeepSeek V4.1 on AVX-512BW: the validated fast path.
template <>
struct PlanTraits<Dsv41Shape, Isa::Bw>
{
    static constexpr bool kCompactScratch = true;
    static constexpr bool kGroupedTraversal = true;
    static constexpr bool kWideSingleExpert = true;
    static constexpr int kSplitTiles = 2;  // compact unswizzled kernels take any tile pair (register_tiles)
};

// Profile: time each phase (its barrier included) on worker 0 and print "moe_cpu phases(us)" after the forward. Off,
// the plan holds no timing code at all.
template <class Shape, Isa I, bool Profile = false>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape, I>;

    // Sizes this call's scratch from the arena, runs the team, and returns. ctx.chunks must be non-empty.
    template <class Experts>
    static void run(ForwardCtx& ctx, const Experts& E, ForwardArena& ar, int threads)
    {
        prepare_scratch(ctx, ar);
        const int nc = static_cast<int>(ctx.chunks.size());
        const bool grouped = Traits::kGroupedTraversal && nc == 1;
        const bool wide = Traits::kWideSingleExpert && ctx.m_total == 1 && nc == 1;
        run_team(ctx, E, threads > 0 ? threads : 1, grouped, wide);
    }

private:
    static void prepare_scratch(ForwardCtx& ctx, ForwardArena& ar)
    {
        constexpr bool compact = Traits::kCompactScratch;
        const int nc = static_cast<int>(ctx.chunks.size());
        const int H = Shape::hidden(*ctx.layer);
        const int I_ = Shape::intermediate(*ctx.layer);
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(ar.tin_g, static_cast<size_t>(nc) * MAX_M * H);
        grow(ar.tin_u, static_cast<size_t>(nc) * MAX_M * H);
        grow(ar.tin_d, static_cast<size_t>(nc) * MAX_M * I_);

        if(!compact) {
            grow(ar.splat_g, static_cast<size_t>(nc) * MAX_M * H);
            grow(ar.splat_u, static_cast<size_t>(nc) * MAX_M * H);
            grow(ar.splat_d, static_cast<size_t>(nc) * MAX_M * I_);
            grow(ar.splat_dup_g, static_cast<size_t>(nc) * MAX_M * H);
            grow(ar.splat_dup_u, static_cast<size_t>(nc) * MAX_M * H);
            grow(ar.splat_dup_d, static_cast<size_t>(nc) * MAX_M * I_);
        } else {
            grow(ar.compact_g,size_t(nc)*ACT_ROWS*H);
            grow(ar.compact_u,size_t(nc)*ACT_ROWS*H);
            grow(ar.compact_d,size_t(nc)*ACT_ROWS*I_);
        }
        grow(ar.tout_g, static_cast<size_t>(nc) * MAX_M * I_);
        grow(ar.tout_u, static_cast<size_t>(nc) * MAX_M * I_);
        grow(ar.tout_d, static_cast<size_t>(nc) * MAX_M * H);
        grow(ctx.prep_g, nc); grow(ctx.prep_u, nc); grow(ctx.prep_d, nc);
        ctx.tout_g = ar.tout_g.data();
        ctx.tout_u = ar.tout_u.data();
        ctx.tout_d = ar.tout_d.data();
        for (int j = 0; j < nc; ++j)
        {
            ctx.prep_g[j] = { ar.tin_g.data() + static_cast<size_t>(j) * MAX_M * H,
                              compact?nullptr:(ar.splat_g.data() + static_cast<size_t>(j) * MAX_M * H),
                              compact?nullptr:(ar.splat_dup_g.data() + static_cast<size_t>(j) * MAX_M * H), {}, {} };
            ctx.prep_u[j] = { ar.tin_u.data() + static_cast<size_t>(j) * MAX_M * H,
                              compact?nullptr:(ar.splat_u.data() + static_cast<size_t>(j) * MAX_M * H),
                              compact?nullptr:(ar.splat_dup_u.data() + static_cast<size_t>(j) * MAX_M * H), {}, {} };
            ctx.prep_d[j] = { ar.tin_d.data() + static_cast<size_t>(j) * MAX_M * I_,
                              compact?nullptr:(ar.splat_d.data() + static_cast<size_t>(j) * MAX_M * I_),
                              compact?nullptr:(ar.splat_dup_d.data() + static_cast<size_t>(j) * MAX_M * I_), {}, {} };
            if(compact) {
                ctx.prep_g[j].compact=ar.compact_g.data()+size_t(j)*ACT_ROWS*H;
                ctx.prep_u[j].compact=ar.compact_u.data()+size_t(j)*ACT_ROWS*H;
                ctx.prep_d[j].compact=ar.compact_d.data()+size_t(j)*ACT_ROWS*I_;
            }
        }
        if constexpr (Traits::kSplitTiles < 8)
        {
            const size_t blocks = static_cast<size_t>(nc) * (H / 128);
            grow(ar.down_tiles_done, blocks);
            std::fill_n(ar.down_tiles_done.data(), blocks, 0);
            ctx.down_tiles_done = ar.down_tiles_done.data();
        }
        if (EXL3_MOE_CPU_ACT_BLOCK)
        {
            // Sized for one block per 16 inputs, the smallest B allows; only k / B entries are used
            const size_t sh = static_cast<size_t>(MAX_M) * (H / 16), si = static_cast<size_t>(MAX_M) * (I_ / 16);
            grow(ar.bq_g, nc * sh); grow(ar.bq_u, nc * sh); grow(ar.bq_d, nc * si);
            grow(ar.bsum_g, nc * sh); grow(ar.bsum_u, nc * sh); grow(ar.bsum_d, nc * si);
            for (int j = 0; j < nc; ++j)
            {
                ctx.prep_g[j].bq = ar.bq_g.data() + j * sh; ctx.prep_g[j].bsum = ar.bsum_g.data() + j * sh;
                ctx.prep_u[j].bq = ar.bq_u.data() + j * sh; ctx.prep_u[j].bsum = ar.bsum_u.data() + j * sh;
                ctx.prep_d[j].bq = ar.bq_d.data() + j * si; ctx.prep_d[j].bsum = ar.bsum_d.data() + j * si;
            }
        }
    }

    // The phases on the framework's pinned team (team.hpp's run_team), which throws when the team is short or a
    // worker cannot be pinned.
    template <class Experts>
    static void run_team(ForwardCtx& ctx, const Experts& E, int count, bool grouped, bool wide)
    {
        [[maybe_unused]] double phase_us[6]{};
        ::sglang::cpu_experts::run_team(count, [&](int worker, int n) {
            if (ctx.zero_out && worker == 0)
                std::memset(ctx.out, 0, static_cast<size_t>(ctx.m_total) * ctx.layer->slabs.hidden * sizeof(float));
            step<Phase::PrepareGateUp>(ctx, E, worker, n, grouped, wide, phase_us);
            step<Phase::GateUp>(ctx, E, worker, n, grouped, wide, phase_us);
            step<Phase::Middle>(ctx, E, worker, n, grouped, wide, phase_us);
            step<Phase::Down>(ctx, E, worker, n, grouped, wide, phase_us);
            step<Phase::Accumulate>(ctx, E, worker, n, grouped, wide, phase_us);
        });
        if constexpr (Profile)
            printf("moe_cpu phases(us): %.1f %.1f %.1f %.1f %.1f %.1f\n",
                   phase_us[0], phase_us[1], phase_us[2], phase_us[3], phase_us[4], phase_us[5]);
    }

    // One phase of the team's sequence: run it, then wait for the whole team unless it is the last. Called inside
    // run_team's parallel region (an orphaned barrier binds to that team). Profile: phase_us is indexed by the
    // phase's value; otherwise it is untouched.
    template <Phase P, class Experts>
    static void step(ForwardCtx& ctx, const Experts& E, int worker, int n, bool grouped, bool wide,
                     [[maybe_unused]] double* phase_us)
    {
        using Clock = std::chrono::steady_clock;
        [[maybe_unused]] Clock::time_point begin;
        if constexpr (Profile) begin = Clock::now();
        phase<P>(ctx, E, worker, n, grouped, wide);
        if constexpr (P != Phase::Accumulate) {
            #pragma omp barrier
        }
        if constexpr (Profile) {
            if (worker == 0)
                phase_us[static_cast<int>(P)] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        }
    }

    // One phase for this worker; P picks the phase at compile time, so each instantiation holds one phase's code.
    template <Phase P, class Experts>
    static void phase(ForwardCtx& c, const Experts& E, int worker, int num_workers, bool grouped, bool wide)
    {
        [[maybe_unused]] const int nc = static_cast<int>(c.chunks.size());
        [[maybe_unused]] const int H = Shape::hidden(*c.layer);
        [[maybe_unused]] const int I_ = Shape::intermediate(*c.layer);

        if constexpr (P == Phase::PrepareGateUp)
        {
            if (EXL3_MOE_CPU_ACT_BLOCK==128 && act_blocked(H) && I==Isa::Bw) {
                if (wide) prepare_gu_blocks<Shape, I, true>(c,E,worker,num_workers);
                else prepare_gu_blocks<Shape, I, false>(c,E,worker,num_workers);
                return;
            }
            // Prepare gate and up inputs, distributed over (chunk, gate/up)
            const int gu = !Shape::gated(*c.layer) ? 1 : 2;
            for (int j = worker; j < nc * gu; j += num_workers)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = gu == 1 || (j % gu);
                const MoeCpuMatrix& mat = up ? E.up(ch.expert) : E.gate(ch.expert);
                PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                prepare_rows<I>(mat, c.x, nullptr, H, ch.token, ch.m, p);
            }
        }
        else if constexpr (P == Phase::GateUp)
        {
            // Gate + up GEMVs (see assign_gemvs)
            const int gu = !Shape::gated(*c.layer) ? 1 : 2;
            assign_gemvs<Traits::kSplitTiles>(worker, num_workers, nc * gu, I_ / 16, [&](int j, int t0, int t1)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = gu == 1 || (j % gu);
                const MoeCpuMatrix& mat = up ? E.up(ch.expert) : E.gate(ch.expert);
                const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                float* tout = (up ? c.tout_u : c.tout_g) + static_cast<size_t>(j / gu) * MAX_M * I_;
                run_tiles<I>(mat, p, tout, ch.m, t0, t1, grouped);
            });
        }
        else if constexpr (P == Phase::Middle)
        {
            if (EXL3_MOE_CPU_ACT_BLOCK==128 && act_blocked(I_) && I!=Isa::Scalar) {
                if (wide) middle_blocks<Shape, I, true>(c,E,worker,num_workers);
                else middle_blocks<Shape, I, false>(c,E,worker,num_workers);
                return;
            }
            // Output transform for gate/up, activation, prepare down input; per chunk. Gated: act(g)
            // * u accumulated into g; gateless: relu2 applied to u in place
            const bool gated = Shape::gated(*c.layer);
            for (int j = worker; j < nc; j += num_workers) {
                const Chunk& ch = c.chunks[j];
                float* g = c.tout_g + static_cast<size_t>(j) * MAX_M * I_;
                float* u = c.tout_u + static_cast<size_t>(j) * MAX_M * I_;
                if (gated) transform_out<I>(E.gate(ch.expert), g, ch.m);
                transform_out<I>(E.up(ch.expert), u, ch.m);
                const size_t count = static_cast<size_t>(ch.m) * I_;
                float* a = gated ? g : u;
                // Nonzero act_limit clamps the up path symmetrically and the activated gate
                // from above, BEFORE the multiply (matching the GPU act_mul kernels). DS4
                // ships swiglu_limit = 10 with plain silu: hidden states deep into a long
                // context push |u| into the thousands, and skipping the clamp here made
                // offloaded experts diverge arbitrarily far from their GPU-resident twins
                const float lim = Shape::act_limit(*c.layer) != 0.0f
                    ? Shape::act_limit(*c.layer) : std::numeric_limits<float>::infinity();
                switch (Shape::activation(*c.layer)) {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            const float av = std::min(gv * cdf, lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 3: {
                        // gpt-oss clamped swiglu: g = min(g, limit); a = (clamp(u, -l, l) + 1) * g *
                        // sigmoid(1.702 * g)
                        const float lim = Shape::act_limit(*c.layer);
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = std::min(g[i], lim);
                            const float uv = std::clamp(u[i], -lim, lim);
                            g[i] = (uv + 1.0f) * gv / (1.0f + std::exp(-1.702f * gv));
                        }
                        break;
                    }
                    default:
                        for (size_t i = 0; i < count; ++i) {
                            const float uv = u[i] > 0.0f ? u[i] : 0.0f;
                            u[i] = uv * uv;
                        }
                        break;
                }
                static const int idx4[MAX_M] = {0, 1, 2, 3};
                prepare_rows<I>(E.down(ch.expert), nullptr, a, I_, idx4, ch.m, c.prep_d[j]);
            }
        }
        else if constexpr (P == Phase::Down)
        {
            // Down GEMVs
            assign_gemvs<Traits::kSplitTiles>(worker, num_workers, nc, H / 16, [&](int j, int t0, int t1)
            {
                const Chunk& ch = c.chunks[j];
                float* tout = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                run_tiles<I>(E.down(ch.expert), c.prep_d[j], tout, ch.m, t0, t1, grouped);
                if constexpr (Traits::kSplitTiles == 8) {
                    transform_owned_blocks<I>(E.down(ch.expert),tout,ch.m,t0,t1);
                } else {
                    // A block's tiles may come from several workers: the one that completes it transforms it. The
                    // acq_rel count makes the other workers' tile stores visible to that worker.
                    for (int b = t0 / 8; b * 8 < t1; ++b) {
                        const int done = std::min(t1, b * 8 + 8) - std::max(t0, b * 8);
                        std::atomic_ref<int> count(c.down_tiles_done[static_cast<size_t>(j) * (H / 128) + b]);
                        if (count.fetch_add(done, std::memory_order_acq_rel) + done == 8)
                            transform_owned_blocks<I>(E.down(ch.expert),tout,ch.m,b*8,b*8+8);
                    }
                }
            });
        }
        else
        {
            static_assert(P == Phase::Accumulate);
            const int c0 = ((H/16) * worker / num_workers)*16;
            const int c1 = ((H/16) * (worker+1) / num_workers)*16;
            for (int j = 0; j < nc; ++j) {
                const Chunk& ch = c.chunks[j];
                const float* d = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                for (int r = 0; r < ch.m; ++r) {
                    float* dst = c.out + static_cast<size_t>(ch.token[r]) * H;
                    const float* src = d + static_cast<size_t>(r) * H;
                    const float w = ch.weight[r];
                    for (int col = c0; col < c1; ++col)
                        dst[col] += w * src[col];
                }
            }
        }
    }
};

// Runs the call's plan at tier `isa`. E reads the layer's experts under the generic plan; D reads the same experts
// under the DSV4.1 plan's assumptions (for a strided layer, the compile-time-shaped view of the same slabs).
template <class Experts, class Dsv41Experts>
void run_plan(ForwardCtx& ctx, const Experts& E, const Dsv41Experts& D, ForwardArena& ar, int threads, Isa isa)
{
    if (isa == Isa::Bw && Dsv41Shape::accepts(*ctx.layer, E, ctx.chunks)) {
        ForwardPlan<Dsv41Shape, Isa::Bw>::run(ctx, D, ar, threads);
        return;
    }
    switch (isa) {
        case Isa::Scalar: ForwardPlan<GenericShape, Isa::Scalar>::run(ctx, E, ar, threads); return;
        case Isa::Avx2:   ForwardPlan<GenericShape, Isa::Avx2>::run(ctx, E, ar, threads); return;
        case Isa::Bw:     ForwardPlan<GenericShape, Isa::Bw>::run(ctx, E, ar, threads); return;
        case Isa::Vnni:   ForwardPlan<GenericShape, Isa::Vnni>::run(ctx, E, ar, threads); return;
        case Isa::Vbmi:   ForwardPlan<GenericShape, Isa::Vbmi>::run(ctx, E, ar, threads); return;
    }
}

// The plan entry: groups the routes by expert and splits them into chunks of CHUNK_M rows. This grouping, not the
// framework's RouteTable, fixes EXL3's accumulation order, so it reads the call's slots and FP16 weights as given.
void run_forward(const Exl3Quant::Layer& layer, Isa isa, const at::Half* x, const int32_t* sel, const at::Half* wts,
                 float* out, int rows, int topk, int threads, bool accumulate)
{
    const LayerSlabs& slabs = layer.slabs;
    const int m_total = rows;
    const int top_k = topk;

    ForwardCtx ctx;
    ctx.layer = &layer;
    ctx.x = x;
    ctx.out = out;
    ctx.m_total = m_total;
    ctx.zero_out = !accumulate;

    ForwardArena& ar = ForwardArena::get();
    ctx.chunks = std::move(ar.chunks);
    ctx.chunks.clear();
    ctx.prep_g = std::move(ar.prep_g); ctx.prep_u = std::move(ar.prep_u); ctx.prep_d = std::move(ar.prep_d);
    const auto give_back = [&] {
        ar.chunks = std::move(ctx.chunks);
        ar.prep_g = std::move(ctx.prep_g); ar.prep_u = std::move(ctx.prep_u); ar.prep_d = std::move(ctx.prep_d);
    };

    // Group token assignments by expert, then split into chunks of CHUNK_M rows
    auto& per_expert = ar.per_expert;
    if (per_expert.size() < static_cast<size_t>(slabs.capacity)) per_expert.resize(slabs.capacity);
    for (int t = 0; t < m_total; ++t)
        for (int j = 0; j < top_k; ++j)
        {
            const int32_t e = sel[static_cast<size_t>(t) * top_k + j];
            if (e >= 0 && e < slabs.capacity)
                per_expert[e].emplace_back(t, half_to_float(wts[static_cast<size_t>(t) * top_k + j]));
        }
    for (int e = 0; e < slabs.capacity; ++e)
    {
        auto& lst = per_expert[e];
        for (size_t i = 0; i < lst.size(); i += CHUNK_M)
        {
            Chunk ch;
            ch.expert = e;
            ch.m = static_cast<int>(std::min<size_t>(CHUNK_M, lst.size() - i));
            for (int r = 0; r < ch.m; ++r)
            {
                ch.token[r] = lst[i + r].first;
                ch.weight[r] = lst[i + r].second;
            }
            ctx.chunks.push_back(ch);
        }
        lst.clear();
    }
    const int nc = static_cast<int>(ctx.chunks.size());
    if (!nc) {
        if (ctx.zero_out) std::memset(ctx.out, 0, static_cast<size_t>(m_total) * slabs.hidden * sizeof(float));
        give_back();
        return;
    }

    if (layer.table) {
        const TableExperts t{layer.table.get()};
        run_plan(ctx, t, t, ar, threads, isa);
    } else {
        const StridedExperts<GenericShape> s{&layer};
        run_plan(ctx, s, s.as<Dsv41Shape>(), ar, threads, isa);
    }
    give_back();
}

// ExpertForward has validated the call (every slot in [-1, capacity), finite weights, rows and k in range); the
// weights are converted to FP16, the registered kernel's convention.
int Exl3Quant::dispatch(const Layer& l, const ForwardCall& c, const RouteTable&, Isa isa)
{
    static thread_local std::vector<at::Half> wts;
    const size_t n = static_cast<size_t>(c.rows) * c.k;
    wts.resize(n);
    for (size_t i = 0; i < n; ++i) wts[i] = at::Half(c.weights[i]);
    run_forward(l, isa, static_cast<const at::Half*>(c.x), c.slots, wts.data(), c.out, c.rows, c.k, c.threads,
                c.accumulate != 0);
    return 0;
}

}  // namespace
}  // namespace sglang::exl3_cpu
