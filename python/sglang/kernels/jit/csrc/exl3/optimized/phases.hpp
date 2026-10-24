// The forward's state (ForwardCtx, the per-thread ForwardArena) and the work of its phases short of the GEMVs: input
// preparation, the output transform, the gated SiLU and the down input. forward_plan.hpp sequences them.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "shapes.hpp"
#include "tiles.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace sglang::exl3_cpu {
namespace {

// The forward's phases, in team order; a barrier separates each from the next. The values are the worker trace's phase
// ids (worker_trace.hpp): 4, an old whole-row down transform now fused into Down, is unused.
enum class Phase : int
{
    PrepareGateUp = 0,  // quantize gate/up inputs
    GateUp = 1,         // gate/up GEMVs
    Middle = 2,         // gate/up output transform, activation, down input
    Down = 3,           // down GEMVs with their output transform
    Accumulate = 5,     // routing-weighted sum into out
};

struct ForwardCtx
{
    const ExpertLayer* layer;
    const Fp16* x;
    float* out;
    int m_total;
    bool zero_out;  // Overwrite: worker 0 zeroes out inside the team, so a team that never runs leaves out untouched.
    // Overwrite by store (OutMode::kOverwriteByStore): no zeroing up front; Accumulate stores each token's first sum
    // (Chunk::first) and zeroes the `unrouted` tokens' rows, each worker its own columns.
    bool store_first;
    const int* unrouted;
    int n_unrouted;
    StageGate* stage_two;  // ForwardCall::stage_two: worker 0 waits on it between GateUp and Middle (null: never)
    std::vector<Chunk> chunks;

    // Per chunk, MAX_M rows each: pointers into the calling thread's ForwardArena (fresh allocations per call cost more
    // in first-touch page faults than the small phases do work).
    float* tout_g;  // [chunks][MAX_M][I]: quant space, then transformed in place
    float* tout_u;
    float* tout_d;  // [chunks][MAX_M][H]
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    // Down tiles finished per (chunk, 128-output block), when a plan splits the down GEMVs finer than a block: the
    // worker that finishes a block's eighth tile applies its output transform.
    int* down_tiles_done;
    // PlanTraits::kClaimMixedChunks on a call of differing chunk sizes: the next unclaimed group of gate/up ([0]) and
    // of down ([16], its own cache line); null: the static split.
    int* claims = nullptr;
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
    std::vector<int> claims;                          // ForwardCtx::claims
    std::vector<std::pair<int32_t, int32_t>> routes;  // the call's live routes, (slot, t * k + j)
    std::vector<uint8_t> written;                     // store_first: per token, whether a chunk row writes it
    std::vector<int> unrouted;                        // store_first: the tokens no chunk row writes
    // Moved into the call's ForwardCtx and back, so a forward allocates nothing once warm.
    std::vector<Chunk> chunks;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

// The output transform of m rows: Hadamard per 128 outputs, then * kHadScale * svh (+ bias).
__attribute__((noipa)) void transform_out(const Exl3Projection& mat, float* tout, int m)
{
    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128) {
            float* v = tout + size_t(r) * mat.n + block;
#if defined(SGLANG_CPU_EXPERTS_AVX2)
            hadamard_128(v);
            const __m256 hs = _mm256_set1_ps(kHadScale);
            for (int i = 0; i < 128; i += 8) {
                const __m256 s = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mat.svh + block + i)));
                __m256 x = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(v + i), hs), s);
                if (mat.bias)
                    x = _mm256_add_ps(
                        x, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(mat.bias + block + i))));
                _mm256_storeu_ps(v + i, x);
            }
#else
            hadamard_128_scalar(v);
            if (mat.bias)
                for (int i = 0; i < 128; ++i)
                    v[i] = v[i] * kHadScale * fp16_to_float(mat.svh[block + i]) + fp16_to_float(mat.bias[block + i]);
            else
                for (int i = 0; i < 128; ++i) v[i] *= kHadScale * fp16_to_float(mat.svh[block + i]);
#endif
        }
}

// The output transform of one 128-output block of a projection's m rows.
void transform_block(Exl3Projection mat, float* tout, int block)
{
    mat.n = 128;
    mat.svh += block;
    if (mat.bias) mat.bias += block;
    transform_out(mat, tout, 1);
}

// The output transform of tiles [t0, t1)'s 128-output blocks of m rows (row stride mat.n).
void transform_owned_blocks(const Exl3Projection& mat, float* out, int m, int t0, int t1)
{
    for (int r = 0; r < m; ++r)
        for (int block = t0 * 16; block < t1 * 16; block += 128) transform_block(mat, out + size_t(r) * mat.n + block, block);
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

// The worker's share [first, last) of `tasks` split evenly over the team.
inline std::pair<int, int> worker_share(int tasks, int worker, int num_workers)
{
    return {tasks * worker / num_workers, tasks * (worker + 1) / num_workers};
}

#if defined(SGLANG_CPU_EXPERTS_AVX2)
// PrepareGateUp on the vector tiers: tasks over (chunk, gate | up, 128-input block), each its rows' transform and
// quantization. A task writes only its own block of its own projection's input.
template <class Shape, bool Wide>
void prepare_gate_up_blocks(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers)
{
    const int K = Shape::hidden(*c.layer), nb = K / 128, nc = int(c.chunks.size());
    const auto [first, last] = worker_share(nc * 2 * nb, worker, num_workers);
    for (int task = first; task < last; ++task) {
        const int j = task / nb, b = task % nb;
        const Chunk& ch = c.chunks[j / 2];
        const bool up = j % 2;
        const Exl3Projection& mat = up ? E.up(ch.expert) : E.gate(ch.expert);
        PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / 2];
        for (int r = 0; r < ch.m; ++r) {
            float* dst = p.tin + size_t(r) * K + b * 128;
            prepare_block(c.x + size_t(ch.token[r]) * K + b * 128, true, mat.suh + b * 128, dst);
            quantize_block<Wide>(p, r, ch.m, K, b, dst);
        }
    }
}

// Middle on the vector tiers: tasks over (chunk, 128-intermediate block), each its rows' gate/up output transform,
// gated SiLU and down input (transform and quantization). A task writes only its own blocks.
template <class Shape, bool Wide>
void middle_blocks(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers)
{
    const int I_ = Shape::intermediate(*c.layer), nb = I_ / 128, nc = int(c.chunks.size());
    const auto [first, last] = worker_share(nc * nb, worker, num_workers);
    for (int task = first; task < last; ++task) {
        const int j = task / nb, b = task % nb, block = b * 128;
        const Chunk& ch = c.chunks[j];
        for (int r = 0; r < ch.m; ++r) {
            float* g = c.tout_g + size_t(j) * MAX_M * I_ + size_t(r) * I_ + block;
            float* u = c.tout_u + size_t(j) * MAX_M * I_ + size_t(r) * I_ + block;
            transform_block(E.gate(ch.expert), g, block);
            transform_block(E.up(ch.expert), u, block);
            silu_mul(g, u, 128, Shape::act_limit(*c.layer));
            PreparedIn& p = c.prep_d[j];
            float* dst = p.tin + size_t(r) * I_ + block;
            prepare_block(g, false, E.down(ch.expert).suh + block, dst);
            quantize_block<Wide>(p, r, ch.m, I_, b, dst);
        }
    }
}
#endif  // SGLANG_CPU_EXPERTS_AVX2

// A chunk's down input row r is its gate output row r: the identity over MAX_M rows (prepare_rows_scalar's token_idx).
constexpr std::array<int, MAX_M> kRowIndex = [] {
    std::array<int, MAX_M> rows{};
    for (int i = 0; i < MAX_M; ++i) rows[i] = i;
    return rows;
}();

// PrepareGateUp on the scalar tier: tasks over (chunk, gate | up).
template <class Shape>
void prepare_gate_up_scalar(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers)
{
    const int H = Shape::hidden(*c.layer), nc = int(c.chunks.size());
    for (int j = worker; j < nc * 2; j += num_workers) {
        const Chunk& ch = c.chunks[j / 2];
        const bool up = j % 2;
        prepare_rows_scalar(up ? E.up(ch.expert) : E.gate(ch.expert), c.x, nullptr, H, ch.token, ch.m,
                            (up ? c.prep_u : c.prep_g)[j / 2]);
    }
}

// Middle on the scalar tier: tasks over chunks.
template <class Shape>
void middle_scalar(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers)
{
    const int I_ = Shape::intermediate(*c.layer), nc = int(c.chunks.size());
    for (int j = worker; j < nc; j += num_workers) {
        const Chunk& ch = c.chunks[j];
        float* g = c.tout_g + size_t(j) * MAX_M * I_;
        float* u = c.tout_u + size_t(j) * MAX_M * I_;
        transform_out(E.gate(ch.expert), g, ch.m);
        transform_out(E.up(ch.expert), u, ch.m);
        silu_mul(g, u, size_t(ch.m) * I_, Shape::act_limit(*c.layer));
        prepare_rows_scalar(E.down(ch.expert), nullptr, g, I_, kRowIndex.data(), ch.m, c.prep_d[j]);
    }
}

}  // namespace
}  // namespace sglang::exl3_cpu
