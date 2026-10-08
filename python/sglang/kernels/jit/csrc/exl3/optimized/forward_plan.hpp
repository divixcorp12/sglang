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
#include "worker_trace.hpp"
#include "tile_assignment.hpp"
#include <c10/util/Half.h>
#include <algorithm>
#include <array>
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

using TilesFn = void (*)(const Exl3Projection&, const PreparedIn&, float*, int, int);

// One AVX-512 tier's GEMV tiles for `bits` and `rows` quantized rows. VBMI at 8 bits runs the VNNI tiles: byte pairing
// is impossible there (shift % 8 == 0) and the byte windows straddle the register pairs, measured slower than the
// dword scheme.
template <Isa I, int bits, int rows>
void tiles_for(const Exl3Projection& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    if constexpr (I == Isa::Vbmi && bits != 8) vbmi_tiles<bits, rows>(mat, in, tout, tn0, tn1);
    else if constexpr (I == Isa::Vbmi || I == Isa::Vnni) vnni_tiles<bits, rows>(mat, in, tout, tn0, tn1);
    else bw_tiles<bits, rows>(mat, in, tout, tn0, tn1);
}

template <Isa I, int bits, int... R>
constexpr std::array<TilesFn, sizeof...(R)> tiles_rows(std::integer_sequence<int, R...>)
{
    return {&tiles_for<I, bits, R + 1>...};
}

template <Isa I, int... B>
constexpr std::array<std::array<TilesFn, MAX_M>, sizeof...(B)> tiles_table(std::integer_sequence<int, B...>)
{
    return {tiles_rows<I, B + 1>(std::make_integer_sequence<int, MAX_M>{})...};
}

// [bits - 1][rows - 1] for bits 1..8 and rows 1..MAX_M, generated from MAX_M: a larger MAX_M instantiates its rows
// here with no hand-written case.
template <Isa I>
constexpr auto kTiles = tiles_table<I>(std::make_integer_sequence<int, 8>{});

template <Isa I>
void run_tiles_raw(const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Vbmi || I == Isa::Vnni || I == Isa::Bw)
    {
        kTiles<I>[mat.bits - 1][m - 1](mat, in, tout, tn0, tn1);
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

// register_tiles<M> for M = 1..CHUNK_M, generated from CHUNK_M: one direct call per M.
template <int... Ms>
M1_ALWAYS_INLINE void register_tiles_for(std::integer_sequence<int, Ms...>, const Exl3Projection& mat,
                                         const PreparedIn& in, float* tout, int m, int tn0, int tn1, bool grouped)
{
    (void)((m == Ms + 1 && (register_tiles<Ms + 1>(mat, in, tout, tn0, tn1, grouped), true)) || ...);
}

// m token rows through the quantized kernels under the accuracy options (quantize_act's layout): per
// k-block sub-views summed in fp32, then each remainder row added onto its token row. Only this
// worker's columns [tn0, tn1) are touched, so the sums need no synchronization.
template <Isa I>
void run_tiles(
    const Exl3Projection& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1, bool grouped = false)
{
    if (tn0 >= tn1) return;
    if constexpr (I == Isa::Bw && ACT_ROWS == 2 && EXL3_MOE_CPU_ACT_BLOCK == 128)
    {
        // One token from either plan; several only as the DSV4.1 plan's compact input (PlanTraits::kCompactScratch).
        if ((m == 1 || in.compact) && mat.bits == 3 && act_blocked(mat.k))
        {
            register_tiles_for(std::make_integer_sequence<int, CHUNK_M>{}, mat, in, tout, m, tn0, tn1, grouped);
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
            Exl3Projection sub = mat;
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
    const ExpertLayer* layer;
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
void transform_out_avx2(const Exl3Projection& mat, float* tout, int m)
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
__attribute__((noipa)) void transform_out(const Exl3Projection& mat, float* tout, int m)
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


// The gated SiLU, into g: act(g) * u. A nonzero act_limit clamps the up path symmetrically and the activated gate from
// above, BEFORE the multiply (matching the GPU act_mul kernels). DS4 ships swiglu_limit = 10 with plain silu: hidden
// states deep into a long context push |u| into the thousands, and skipping the clamp here made offloaded experts
// diverge arbitrarily far from their GPU-resident twins.
inline void silu_mul(float* g, const float* u, size_t count, float act_limit)
{
    const float lim = act_limit != 0.0f ? act_limit : std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < count; ++i) {
        const float gv = g[i];
        const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
        g[i] = av * std::clamp(u[i], -lim, lim);
    }
}

// Same scalar/vector operations and block scale ordering as the original phase 2.
// Blocks are independent: only their own gate/up outputs and prepared input slices are written.
template<class Shape, Isa I, bool Wide = false>
void middle_blocks(ForwardCtx& c,const Experts<Shape>& E,int worker,int num_workers) {
    const int I_=Shape::intermediate(*c.layer), nb=I_/128, nc=int(c.chunks.size());
    const int first=nc*nb*worker/num_workers,last=nc*nb*(worker+1)/num_workers;
    for (int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb,block=b*128;
        const auto& ch=c.chunks[j];
        for (int r=0;r<ch.m;++r) {
            float* g=c.tout_g+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            float* u=c.tout_u+size_t(j)*MAX_M*I_+size_t(r)*I_+block;
            auto gate=E.gate(ch.expert);gate.n=128;gate.svh+=block;if(gate.bias)gate.bias+=block;
            transform_out<I>(gate,g,1);
            auto up=E.up(ch.expert);up.n=128;up.svh+=block;if(up.bias)up.bias+=block;
            transform_out<I>(up,u,1);
            silu_mul(g,u,128,Shape::act_limit(*c.layer));

            auto& p=c.prep_d[j];
            float* dst=p.tin+size_t(r)*I_+block;
            prepare_block_avx2<I>(g,false,E.down(ch.expert).suh+block,dst,128);
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


template<class Shape, Isa I, bool Wide = false>
void prepare_gu_blocks(ForwardCtx& c,const Experts<Shape>& E,int worker,int num_workers) {
    const int K=Shape::hidden(*c.layer),nb=K/128,nc=int(c.chunks.size()),gu=2;
    const int first=nc*gu*nb*worker/num_workers,last=nc*gu*nb*(worker+1)/num_workers;
    for(int task=first;task<last;++task) {
        const int j=task/nb,b=task%nb;
        const auto& ch=c.chunks[j/gu];
        const bool up=j%gu;
        const Exl3Projection& mat=up?E.up(ch.expert):E.gate(ch.expert);
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
void transform_owned_blocks(const Exl3Projection& mat,float* out,int m,int t0,int t1) {
    for(int r=0;r<m;++r) for(int block=t0*16;block<t1*16;block+=128) {
        auto sub=mat;sub.n=128;sub.svh+=block;if(sub.bias)sub.bias+=block;
        transform_out<I>(sub,out+size_t(r)*mat.n+block,1);
    }
}

// Coarse readiness is per expert: all gate/up outputs must exist before middle,
// and all middle blocks must be prepared before any down output band can run.
// The forward's phases, in team order; a barrier separates each from the next. The values index the profiling line
// ("moe_cpu phases(us)"): 4, the old whole-row down transform, is gone (Down's owned blocks include it) and prints 0.
// A chunk's down input row r is its gate output row r: the identity over MAX_M rows (prepare_rows' token_idx).
constexpr std::array<int, MAX_M> kRowIndex = [] {
    std::array<int, MAX_M> rows{};
    for (int i = 0; i < MAX_M; ++i) rows[i] = i;
    return rows;
}();

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
    std::vector<std::pair<int32_t, int32_t>> routes;  // the call's live routes, (slot, t * k + j)
    // Moved into the call's ForwardCtx and back, so a forward allocates nothing once warm
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
    static constexpr bool kGroupedTraversal = false;  // range-aware multi-band traversal when one expert is routed (one-token chunks: kRegisterBudget[0])
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
template <class Shape, Isa I, bool Profile = false, bool Trace = EXL3_MOE_CPU_WORKER_TRACE != 0>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape, I>;

    // Sizes this call's scratch from the arena, runs the team, and returns. ctx.chunks must be non-empty.
    static void run(ForwardCtx& ctx, const Experts<Shape>& E, ForwardArena& ar, Team& team)
    {
        worker_trace::Capture<Trace> trace(team.workers(), ctx.m_total, int(ctx.chunks.size()));
        prepare_scratch(ctx, ar, trace);
        const int nc = static_cast<int>(ctx.chunks.size());
        const bool grouped = Traits::kGroupedTraversal && nc == 1;
        const bool wide = Traits::kWideSingleExpert && ctx.m_total == 1 && nc == 1;
        trace.team_start();
        run_team(ctx, E, team, grouped, wide, trace);
        trace.finish();
    }

private:
    static void prepare_scratch(ForwardCtx& ctx, ForwardArena& ar, worker_trace::Capture<Trace>& trace)
    {
        constexpr bool compact = Traits::kCompactScratch;
        const int nc = static_cast<int>(ctx.chunks.size());
        const int H = Shape::hidden(*ctx.layer);
        const int I_ = Shape::intermediate(*ctx.layer);
        auto grow = [&](auto& v, size_t n, const char* name) { trace.grow(v, n, name); };
        grow(ar.tin_g, static_cast<size_t>(nc) * MAX_M * H, "ar.tin_g");
        grow(ar.tin_u, static_cast<size_t>(nc) * MAX_M * H, "ar.tin_u");
        grow(ar.tin_d, static_cast<size_t>(nc) * MAX_M * I_, "ar.tin_d");

        if(!compact) {
            grow(ar.splat_g, static_cast<size_t>(nc) * MAX_M * H, "ar.splat_g");
            grow(ar.splat_u, static_cast<size_t>(nc) * MAX_M * H, "ar.splat_u");
            grow(ar.splat_d, static_cast<size_t>(nc) * MAX_M * I_, "ar.splat_d");
            grow(ar.splat_dup_g, static_cast<size_t>(nc) * MAX_M * H, "ar.splat_dup_g");
            grow(ar.splat_dup_u, static_cast<size_t>(nc) * MAX_M * H, "ar.splat_dup_u");
            grow(ar.splat_dup_d, static_cast<size_t>(nc) * MAX_M * I_, "ar.splat_dup_d");
        } else {
            // Compact: each chunk's ACT_ROWS * m rows follow the previous chunk's (a call of one-token chunks keeps chunk
            // j at j * ACT_ROWS rows).
            size_t rows = 0;
            for (const Chunk& ch : ctx.chunks) rows += size_t(ACT_ROWS) * ch.m;
            grow(ar.compact_g, rows*H, "ar.compact_g");
            grow(ar.compact_u, rows*H, "ar.compact_u");
            grow(ar.compact_d, rows*I_, "ar.compact_d");
        }
        grow(ar.tout_g, static_cast<size_t>(nc) * MAX_M * I_, "ar.tout_g");
        grow(ar.tout_u, static_cast<size_t>(nc) * MAX_M * I_, "ar.tout_u");
        grow(ar.tout_d, static_cast<size_t>(nc) * MAX_M * H, "ar.tout_d");
        grow(ctx.prep_g, nc, "ctx.prep_g"); grow(ctx.prep_u, nc, "ctx.prep_u"); grow(ctx.prep_d, nc, "ctx.prep_d");
        ctx.tout_g = ar.tout_g.data();
        ctx.tout_u = ar.tout_u.data();
        ctx.tout_d = ar.tout_d.data();
        size_t row0 = 0;
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
                ctx.prep_g[j].compact=ar.compact_g.data()+row0*H;
                ctx.prep_u[j].compact=ar.compact_u.data()+row0*H;
                ctx.prep_d[j].compact=ar.compact_d.data()+row0*I_;
                row0+=size_t(ACT_ROWS)*ctx.chunks[j].m;
            }
        }
        if constexpr (Traits::kSplitTiles < 8)
        {
            const size_t blocks = static_cast<size_t>(nc) * (H / 128);
            grow(ar.down_tiles_done, blocks, "ar.down_tiles_done");
            std::fill_n(ar.down_tiles_done.data(), blocks, 0);
            ctx.down_tiles_done = ar.down_tiles_done.data();
        }
        if (EXL3_MOE_CPU_ACT_BLOCK)
        {
            // Sized for one block per 16 inputs, the smallest B allows; only k / B entries are used
            const size_t sh = static_cast<size_t>(MAX_M) * (H / 16), si = static_cast<size_t>(MAX_M) * (I_ / 16);
            grow(ar.bq_g, nc * sh, "ar.bq_g"); grow(ar.bq_u, nc * sh, "ar.bq_u"); grow(ar.bq_d, nc * si, "ar.bq_d");
            grow(ar.bsum_g, nc * sh, "ar.bsum_g"); grow(ar.bsum_u, nc * sh, "ar.bsum_u"); grow(ar.bsum_d, nc * si, "ar.bsum_d");
            for (int j = 0; j < nc; ++j)
            {
                ctx.prep_g[j].bq = ar.bq_g.data() + j * sh; ctx.prep_g[j].bsum = ar.bsum_g.data() + j * sh;
                ctx.prep_u[j].bq = ar.bq_u.data() + j * sh; ctx.prep_u[j].bsum = ar.bsum_u.data() + j * sh;
                ctx.prep_d[j].bq = ar.bq_d.data() + j * si; ctx.prep_d[j].bsum = ar.bsum_d.data() + j * si;
            }
        }
    }

    // The phases on the engine's team (team.hpp), every worker in step, the team's owner as worker 0.
    static void run_team(ForwardCtx& ctx, const Experts<Shape>& E, Team& team, bool grouped, bool wide,
                         worker_trace::Capture<Trace>& trace)
    {
        [[maybe_unused]] double phase_us[6]{};
        team.run([&](int worker, int n) {
            trace.worker_start(worker);
            if (ctx.zero_out && worker == 0)
                std::memset(ctx.out, 0, static_cast<size_t>(ctx.m_total) * ctx.layer->hidden * sizeof(float));
            step<Phase::PrepareGateUp>(ctx, E, team, worker, n, grouped, wide, phase_us, trace);
            step<Phase::GateUp>(ctx, E, team, worker, n, grouped, wide, phase_us, trace);
            step<Phase::Middle>(ctx, E, team, worker, n, grouped, wide, phase_us, trace);
            step<Phase::Down>(ctx, E, team, worker, n, grouped, wide, phase_us, trace);
            step<Phase::Accumulate>(ctx, E, team, worker, n, grouped, wide, phase_us, trace);
            trace.worker_end(worker);
        });
        if constexpr (Profile)
            printf("moe_cpu phases(us): %.1f %.1f %.1f %.1f %.1f %.1f\n",
                   phase_us[0], phase_us[1], phase_us[2], phase_us[3], phase_us[4], phase_us[5]);
    }

    // One phase of the team's sequence: run it, then wait for the whole team unless it is the last. Profile: phase_us
    // is indexed by the phase's value; otherwise it is untouched.
    template <Phase P>
    static void step(ForwardCtx& ctx, const Experts<Shape>& E, Team& team, int worker, int n, bool grouped, bool wide,
                     [[maybe_unused]] double* phase_us, worker_trace::Capture<Trace>& trace)
    {
        using Clock = std::chrono::steady_clock;
        [[maybe_unused]] Clock::time_point begin;
        if constexpr (Profile) begin = Clock::now();
        trace.begin(worker, static_cast<int>(P));
        phase<P>(ctx, E, worker, n, grouped, wide, trace);
        trace.work_end(worker, static_cast<int>(P));
        if constexpr (P != Phase::Accumulate) team.barrier();
        trace.end(worker, static_cast<int>(P));
        if constexpr (Profile) {
            if (worker == 0)
                phase_us[static_cast<int>(P)] = std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
        }
    }

    // One phase for this worker; P picks the phase at compile time, so each instantiation holds one phase's code.
    template <Phase P>
    static void phase(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers, bool grouped, bool wide,
                      worker_trace::Capture<Trace>& trace)
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
            const int gu = 2;
            for (int j = worker; j < nc * gu; j += num_workers)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = j % gu;
                const Exl3Projection& mat = up ? E.up(ch.expert) : E.gate(ch.expert);
                PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                prepare_rows<I>(mat, c.x, nullptr, H, ch.token, ch.m, p);
            }
        }
        else if constexpr (P == Phase::GateUp)
        {
            // Gate + up GEMVs (see assign_gemvs)
            const int gu = 2;
            assign_plan_gemvs<Traits::kSplitTiles>(worker, num_workers, nc * gu, I_ / 16,
                [&](int j) { return c.chunks[j / gu].m; }, [&](int j, int t0, int t1)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = j % gu;
                const Exl3Projection& mat = up ? E.up(ch.expert) : E.gate(ch.expert);
                const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                float* tout = (up ? c.tout_u : c.tout_g) + static_cast<size_t>(j / gu) * MAX_M * I_;
                trace.add_work(worker, static_cast<int>(P), int64_t(t1 - t0) * ch.m);
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
            // Output transform for gate/up, the gated SiLU into g, prepare down input; per chunk.
            for (int j = worker; j < nc; j += num_workers) {
                const Chunk& ch = c.chunks[j];
                float* g = c.tout_g + static_cast<size_t>(j) * MAX_M * I_;
                float* u = c.tout_u + static_cast<size_t>(j) * MAX_M * I_;
                transform_out<I>(E.gate(ch.expert), g, ch.m);
                transform_out<I>(E.up(ch.expert), u, ch.m);
                silu_mul(g, u, static_cast<size_t>(ch.m) * I_, Shape::act_limit(*c.layer));
                prepare_rows<I>(E.down(ch.expert), nullptr, g, I_, kRowIndex.data(), ch.m, c.prep_d[j]);
            }
        }
        else if constexpr (P == Phase::Down)
        {
            // Down GEMVs
            assign_plan_gemvs<Traits::kSplitTiles>(worker, num_workers, nc, H / 16,
                [&](int j) { return c.chunks[j].m; }, [&](int j, int t0, int t1)
            {
                const Chunk& ch = c.chunks[j];
                float* tout = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                trace.add_work(worker, static_cast<int>(P), int64_t(t1 - t0) * ch.m);
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

// Forwards by plan since load: [0] the DSV4.1 plan, [1] the generic plan. Relaxed: counts only, read through
// exl3_cpu_plan_calls (kernel.cpp, the one file that includes this header).
std::atomic<int64_t> g_plan_calls[2];

// Runs the call's plan at tier `isa`: the DSV4.1 plan when it accepts the call, else the generic plan at the tier.
void run_plan(ForwardCtx& ctx, const Exl3Quant::Params& p, ForwardArena& ar, Team& team, Isa isa)
{
    const ExpertLayer& l = *ctx.layer;
    if (isa == Isa::Bw && Dsv41Shape::accepts(l, p, ctx.chunks)) {
        g_plan_calls[0].fetch_add(1, std::memory_order_relaxed);
        ForwardPlan<Dsv41Shape, Isa::Bw>::run(ctx, Experts<Dsv41Shape>{&l, p}, ar, team);
        return;
    }
    g_plan_calls[1].fetch_add(1, std::memory_order_relaxed);
    const Experts<GenericShape> E{&l, p};
    switch (isa) {
        case Isa::Scalar: ForwardPlan<GenericShape, Isa::Scalar>::run(ctx, E, ar, team); return;
        case Isa::Avx2:   ForwardPlan<GenericShape, Isa::Avx2>::run(ctx, E, ar, team); return;
        case Isa::Bw:     ForwardPlan<GenericShape, Isa::Bw>::run(ctx, E, ar, team); return;
        case Isa::Vnni:   ForwardPlan<GenericShape, Isa::Vnni>::run(ctx, E, ar, team); return;
        case Isa::Vbmi:   ForwardPlan<GenericShape, Isa::Vbmi>::run(ctx, E, ar, team); return;
    }
}

// The plan entry: groups the routes by expert and splits them into chunks of CHUNK_M rows. This grouping, not the
// framework's RouteTable, fixes EXL3's accumulation order (expert ascending, then token, then route), so it reads the
// call's slots as given and runs zero-weight routes. Each weight is rounded to FP16, the registered kernel's
// convention. ExpertForward has validated the call (every slot in [-1, capacity), finite weights, rows and k in range).
int Exl3Quant::dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa isa, Team& team)
{
    ForwardCtx ctx;
    ctx.layer = &l;
    ctx.x = static_cast<const at::Half*>(c.x);
    ctx.out = c.out;
    ctx.m_total = c.rows;
    ctx.zero_out = !c.accumulate;

    ForwardArena& ar = ForwardArena::get();
    ctx.chunks = std::move(ar.chunks);
    ctx.chunks.clear();
    ctx.prep_g = std::move(ar.prep_g); ctx.prep_u = std::move(ar.prep_u); ctx.prep_d = std::move(ar.prep_d);
    const auto give_back = [&] {
        ar.chunks = std::move(ctx.chunks);
        ar.prep_g = std::move(ctx.prep_g); ar.prep_u = std::move(ctx.prep_u); ar.prep_d = std::move(ctx.prep_d);
    };

    // The live routes as (slot, route index t * k + j), sorted: each expert's routes in token, then route, order.
    auto& routes = ar.routes;
    routes.clear();
    const int n = c.rows * c.k;
    for (int i = 0; i < n; ++i)
        if (c.slots[i] >= 0) routes.emplace_back(c.slots[i], i);
    std::sort(routes.begin(), routes.end());
    for (size_t i = 0; i < routes.size();)
    {
        Chunk ch;
        ch.expert = routes[i].first;
        ch.m = 0;
        for (; i < routes.size() && routes[i].first == ch.expert && ch.m < CHUNK_M; ++i, ++ch.m)
        {
            ch.token[ch.m] = routes[i].second / c.k;
            ch.weight[ch.m] = half_to_float(at::Half(c.weights[routes[i].second]));
        }
        ctx.chunks.push_back(ch);
    }
    if (ctx.chunks.empty()) {
        if (ctx.zero_out) std::memset(ctx.out, 0, static_cast<size_t>(ctx.m_total) * l.hidden * sizeof(float));
    } else {
        run_plan(ctx, p, ar, team, isa);
    }
    give_back();
    return 0;
}

}  // namespace
}  // namespace sglang::exl3_cpu
