// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
#include <atomic>
#include <type_traits>
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include "quant.hpp"
#include "../../cpu_experts_common/cabi.hpp"
#include <c10/util/Half.h>
#include <ATen/ATen.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <chrono>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#else
// min/max macro suppression is handled globally (-DNOMINMAX in setup.py): this TU uses
// std::min/max/clamp throughout
#include <intrin.h>
#include <windows.h>
#endif



// Kept for upstream's bindings. Phase timing is compile-time here (ForwardPlan's Profile, forward_plan.hpp).
void exl3_moe_cpu_set_prof(bool) {}

namespace sglang::exl3_cpu {
namespace {
using namespace ::sglang::cpu_experts;

#include "math.hpp"

#include "math_avx512.hpp"

#include "math_avx2.hpp"

#include "math_scalar.hpp"

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

struct Chunk
{
    int expert;
    int m;
    int token[MAX_M];
    float weight[MAX_M];
};

#include "shapes.hpp"

struct ForwardCtx
{
    LayerInfo info;
    const at::Half* x;
    float* out;
    int m_total;
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
    const int I_=Shape::intermediate(c.info), nb=I_/128, nc=int(c.chunks.size());
    const bool gated=Shape::gated(c.info);
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
            const float lim=Shape::act_limit(c.info)!=0.0f?Shape::act_limit(c.info):std::numeric_limits<float>::infinity();
                switch (Shape::activation(c.info)) {
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
                        const float lim = Shape::act_limit(c.info);
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
    const int K=Shape::hidden(c.info),nb=K/128,nc=int(c.chunks.size()),gu=!Shape::gated(c.info)?1:2;
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
#include "forward_plan.hpp"

// Runs the call's plan at tier `isa`. E reads the layer's experts under the generic plan; D reads the same experts
// under the DSV4.1 plan's assumptions (for a strided layer, the compile-time-shaped view of the same slabs).
template <class Experts, class Dsv41Experts>
void run_plan(ForwardCtx& ctx, const Experts& E, const Dsv41Experts& D, ForwardArena& ar, int threads, Isa isa)
{
    if (isa == Isa::Bw && Dsv41Shape::accepts(ctx.info, E, ctx.chunks)) {
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
    const LayerInfo& info = layer.info;
    const int m_total = rows;
    const int top_k = topk;

    ForwardCtx ctx;
    ctx.info = info;
    ctx.x = x;
    ctx.out = out;
    ctx.m_total = m_total;
    if (!accumulate) std::memset(ctx.out, 0, static_cast<size_t>(m_total) * info.hidden * sizeof(float));

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
    if (per_expert.size() < static_cast<size_t>(info.num_experts)) per_expert.resize(info.num_experts);
    for (int t = 0; t < m_total; ++t)
        for (int j = 0; j < top_k; ++j)
        {
            const int32_t e = sel[static_cast<size_t>(t) * top_k + j];
            if (e >= 0 && e < info.num_experts)
                per_expert[e].emplace_back(t, half_to_float(wts[static_cast<size_t>(t) * top_k + j]));
        }
    for (int e = 0; e < info.num_experts; ++e)
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
    if (!nc) { give_back(); return; }

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
int Exl3Quant::dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable&, Isa isa)
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

SGLANG_CPU_EXPERTS_DEFINE_CABI(exl3, ::sglang::exl3_cpu::Exl3Quant)

namespace {
using ::sglang::exl3_cpu::Exl3Quant;
using Exl3Forward = ::sglang::cpu_experts::ExpertForward<Exl3Quant>;

// A nonzero ExpertForward status as a torch error.
void check_status(int status, const char* what)
{
    TORCH_CHECK(status != 2, what, ": invalid arguments (status 2: an unknown or freed handle, a slot outside the "
                "layer, a non-finite weight, or rows/k/threads out of range)");
    TORCH_CHECK(status != 3, what, ": another forward or free is running (status 3)");
    TORCH_CHECK(status == 0, what, ": kernel error (status ", status, "): ", ::sglang::cpu_experts::last_error());
}
}  // namespace

// -------------------------------------------------------------------------------------------
//   Upstream link compatibility
// -------------------------------------------------------------------------------------------

// This file replaces upstream's cpu/moe_mul1.cpp inside the exllamav3 extension, whose
// bindings.cpp and cpu/moe_handoff.cu still reference these symbols. sglang never calls them:
// the staging copy belongs to upstream's handoff worker, and the pool they exercised was
// replaced by the OpenMP forward. They exist only so the extension links, and fail if called.

void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_stage_experts is not supported by the optimized CPU expert kernel");
}

int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_pool_stress is not supported by the optimized CPU expert kernel");
    return 0;
}

bool exl3_moe_cpu_has_avx2() { return Exl3Forward::isa() != ::sglang::cpu_experts::Isa::Scalar; }
bool exl3_moe_cpu_has_avx512_bw() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Bw; }
bool exl3_moe_cpu_has_avx512_vnni() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Vnni; }
bool exl3_moe_cpu_has_avx512_vbmi() { return Exl3Forward::isa() == ::sglang::cpu_experts::Isa::Vbmi; }

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && trellis.is_contiguous(), "trellis must be contiguous CPU");
    TORCH_CHECK(trellis.dim() == 3, "trellis must be [k/16, n/16, 16K]");
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
    m.suh = reinterpret_cast<const at::Half*>(suh.data_ptr());
    m.svh = reinterpret_cast<const at::Half*>(svh.data_ptr());
    m.bias = bias ? reinterpret_cast<const at::Half*>(bias->data_ptr()) : nullptr;
    m.k = static_cast<int>(trellis.size(0)) * 16;
    m.n = static_cast<int>(trellis.size(1)) * 16;
    m.bits = static_cast<int>(trellis.size(2)) / 16;
    // K8 tensors are exempt from swizzling (routed to the dword kernel, which would gain
    // nothing) -- the child loader applies the same bits != 8 rule when repacking, so the two
    // sides agree per tensor
    m.swz = swizzled && m.bits != 8 ? 1 : 0;
    TORCH_CHECK(m.bits >= 1 && m.bits <= 8, "CPU MoE requires K in [1, 8]");
    TORCH_CHECK(m.k % 128 == 0 && m.n % 128 == 0, "dims must be divisible by 128");
    TORCH_CHECK(m.k <= 8192, "k too large for i32 accumulation");
    return m;
}

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled
)
{
    auto table = std::unique_ptr<MoeCpuLayer>(new MoeCpuLayer);
    const bool swz = swizzled != 0;
    const size_t E = up_trellis.size();
    const bool gated = !gate_trellis.empty();
    TORCH_CHECK(down_trellis.size() == E && (!gated || gate_trellis.size() == E), "expert count mismatch");
    TORCH_CHECK(gated ? (activation == 0 || activation == 1 || activation == 3) : activation == 2, "gated experts take silu/gelu/swiglu_oai, gateless take relu2");
    TORCH_CHECK(gate_bias.empty() || gate_bias.size() == E, "gate bias count mismatch");
    TORCH_CHECK(up_bias.empty() || up_bias.size() == E, "up bias count mismatch");
    TORCH_CHECK(down_bias.empty() || down_bias.size() == E, "down bias count mismatch");
    table->num_experts = static_cast<int>(E);
    table->activation = static_cast<int>(activation);
    table->act_limit = static_cast<float>(act_limit);
    for (size_t e = 0; e < E; ++e) {
        if (gated) {
            table->gates.push_back(make_matrix(gate_trellis[e], gate_suh[e], gate_svh[e],
                                               gate_bias.empty() ? nullptr : &gate_bias[e], swz));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]})
                table->refs.push_back(t);
            if (!gate_bias.empty()) table->refs.push_back(gate_bias[e]);
        }
        table->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                         up_bias.empty() ? nullptr : &up_bias[e], swz));
        table->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                           down_bias.empty() ? nullptr : &down_bias[e], swz));
        for (auto& t : {up_trellis[e], up_suh[e], up_svh[e], down_trellis[e], down_suh[e], down_svh[e]})
            table->refs.push_back(t);
        if (!up_bias.empty()) table->refs.push_back(up_bias[e]);
        if (!down_bias.empty()) table->refs.push_back(down_bias[e]);
    }
    table->hidden_size = table->ups[0].k;
    table->interm_size = table->ups[0].n;
    TORCH_CHECK(table->downs[0].k == table->interm_size && table->downs[0].n == table->hidden_size,
                "expert shape mismatch");

    // A table layer: ExpertForward refuses a slot at or past its expert count, as it does a slab layer's capacity.
    const ::sglang::exl3_cpu::LayerInfo info{table->num_experts, table->hidden_size, table->interm_size,
                                             !table->gates.empty(), table->activation, table->act_limit};
    Exl3Quant::Layer layer{info, {}, 0, 0, std::move(table)};
    layer.rows.capacity = info.num_experts;
    auto entry = std::make_shared<const Exl3Quant::Layer>(std::move(layer));
    std::lock_guard<std::mutex> lock(Exl3Forward::registry_mutex);
    Exl3Forward::layers.push_back(std::move(entry));
    return static_cast<int64_t>(Exl3Forward::layers.size() - 1);
}

void exl3_moe_cpu_free_layer(int64_t handle)
{
    const int status = Exl3Forward::free_layer(handle);
    if (status == 2) return;  // an unknown or already freed handle: a no-op, as upstream's free is
    check_status(status, "exl3_moe_cpu_free_layer");
}

// The C ABI's forward with accumulate 0: the FP16 weights widen to FP32 exactly and the kernel narrows them back.
void exl3_moe_cpu_forward_raw(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    if (rows == 0) return;  // upstream's forward of no tokens is a no-op; the C ABI refuses rows < 1
    static thread_local std::vector<float> weights;
    const size_t n = static_cast<size_t>(rows) * topk;
    weights.resize(n);
    for (size_t i = 0; i < n; ++i) weights[i] = static_cast<float>(wts[i]);
    SglangCpuExpertsForward call{};
    call.abi_version = SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION;
    call.rows = rows;
    call.layer = handle;
    call.x = x;
    call.slots = sel;
    call.weights = weights.data();
    call.out = out;
    call.k = topk;
    call.threads = std::max(threads, 1);  // upstream's forward ran fewer than one thread as one
    call.accumulate = 0;
    check_status(Exl3Forward::forward(&call), "exl3_moe_cpu_forward");
}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(x.device().is_cpu() && selected.device().is_cpu() && weights.device().is_cpu() && out.device().is_cpu(), "CPU MoE tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kFloat, "dtype mismatch");

    const int m_total = static_cast<int>(x.size(0));
    const int top_k = static_cast<int>(selected.size(-1));

    // Raw path takes int32 selection
    std::vector<int32_t> sel32(static_cast<size_t>(m_total) * top_k);
    if (selected.scalar_type() == at::kLong)
    {
        const int64_t* s = selected.data_ptr<int64_t>();
        for (size_t i = 0; i < sel32.size(); ++i) sel32[i] = static_cast<int32_t>(s[i]);
    }
    else
    {
        TORCH_CHECK(selected.scalar_type() == at::kInt, "selected must be int32 or int64");
        std::memcpy(sel32.data(), selected.data_ptr<int32_t>(), sel32.size() * 4);
    }

    exl3_moe_cpu_forward_raw
    (
        handle,
        reinterpret_cast<const at::Half*>(x.data_ptr()),
        sel32.data(),
        reinterpret_cast<const at::Half*>(weights.data_ptr()),
        out.data_ptr<float>(),
        m_total, top_k,
        static_cast<int>(num_threads)
    );
}
