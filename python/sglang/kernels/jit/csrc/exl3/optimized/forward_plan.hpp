// EXL3's job: the plans (ForwardPlan<Shape>) that lay a call out as phases of tasks, the job's preparation
// (prepare_job) and the plan registry. Included by kernel.cpp.
//
// One job = prepare_job at admission, then the five phases of ForwardPlan<Shape> at the build's tier (kIsa), each task
// run by whichever worker the scheduler (host/cpu_experts/job_window.hpp) gives it. Shape (shapes.hpp) fixes what the plan
// may assume about the layer; PlanTraits<Shape> what it turns on. The primary traits are the generic plan;
// PlanTraits<Dsv41Shape> is the fast path measured and validated bit-exact for DeepSeek V4.1 from AVX-512BW up
// (README.txt). A task's arithmetic does not depend on the worker that runs it.
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "phases.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace sglang::exl3_cpu {
namespace {

// What a plan turns on. The primary template is the generic plan.
template <class Shape>
struct PlanTraits
{
    static constexpr bool kCompactScratch = false;    // int16 compact activations instead of the int32 splats
    static constexpr bool kGroupedTraversal = false;  // range-aware multi-band traversal when one expert is routed
    static constexpr bool kWideSingleExpert = false;  // 512-wide quantization for one token through one expert
};

// DeepSeek V4.1 from AVX-512BW up: the validated fast path.
template <>
struct PlanTraits<Dsv41Shape>
{
    static constexpr bool kCompactScratch = true;
    static constexpr bool kGroupedTraversal = true;  // one-token chunks (kRegisterBudget[0])
    static constexpr bool kWideSingleExpert = true;
};

template <class Shape>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape>;

    // GateUp and DownTiles tasks are tile pairs claimed in runs of whole 8-tile groups: the generic plan's tile kernels
    // start and end a range on one (the compact register kernels take any pair).
    static constexpr int kTileAlign = 8;

    // Sizes the job's scratch from its arena and points the context at it. ctx.chunks is final.
    static void prepare_scratch(ForwardJob& job)
    {
        ForwardCtx& ctx = job.ctx;
        ForwardArena& ar = job.ar;
        constexpr bool compact = Traits::kCompactScratch;
        const int nc = int(ctx.chunks.size());
        const size_t H = Shape::hidden(*ctx.layer);
        const size_t I_ = Shape::intermediate(*ctx.layer);
        const auto grow = [](auto& v, size_t n) {
            if (v.size() < n) v.resize(n);
        };
        grow(ar.tin_g, nc * MAX_M * H);
        grow(ar.tin_u, nc * MAX_M * H);
        grow(ar.tin_d, nc * MAX_M * I_);
        grow(ar.tout_g, nc * MAX_M * I_);
        grow(ar.tout_u, nc * MAX_M * I_);
        grow(ar.tout_d, nc * MAX_M * H);
        grow(ctx.prep_g, nc);
        grow(ctx.prep_u, nc);
        grow(ctx.prep_d, nc);
        ctx.tout_g = ar.tout_g.data();
        ctx.tout_u = ar.tout_u.data();
        ctx.tout_d = ar.tout_d.data();
        if constexpr (compact) {
            // Each chunk's kActRows * m rows follow the previous chunk's (a call of one-token chunks keeps chunk j at
            // j * kActRows rows).
            size_t rows = 0;
            for (const Chunk& ch : ctx.chunks) rows += size_t(kActRows) * ch.m;
            grow(ar.compact_g, rows * H);
            grow(ar.compact_u, rows * H);
            grow(ar.compact_d, rows * I_);
        } else {
            grow(ar.splat_g, nc * MAX_M * H);
            grow(ar.splat_u, nc * MAX_M * H);
            grow(ar.splat_d, nc * MAX_M * I_);
            if constexpr (kSplatDup) {
                grow(ar.splat_dup_g, nc * MAX_M * H);
                grow(ar.splat_dup_u, nc * MAX_M * H);
                grow(ar.splat_dup_d, nc * MAX_M * I_);
            }
        }
        // One scale and sum per (128-input block, quantized row).
        const size_t sh = MAX_M * (H / kActBlock), si = MAX_M * (I_ / kActBlock);
        grow(ar.bq_g, nc * sh);
        grow(ar.bq_u, nc * sh);
        grow(ar.bq_d, nc * si);
        grow(ar.bsum_g, nc * sh);
        grow(ar.bsum_u, nc * sh);
        grow(ar.bsum_d, nc * si);
        size_t row0 = 0;
        for (int j = 0; j < nc; ++j) {
            const size_t g = j * MAX_M * H, d = j * MAX_M * I_;
            PreparedIn& pg = ctx.prep_g[j];
            PreparedIn& pu = ctx.prep_u[j];
            PreparedIn& pd = ctx.prep_d[j];
            pg = {ar.tin_g.data() + g, nullptr, nullptr, nullptr, ar.bq_g.data() + j * sh, ar.bsum_g.data() + j * sh, {}, {}};
            pu = {ar.tin_u.data() + g, nullptr, nullptr, nullptr, ar.bq_u.data() + j * sh, ar.bsum_u.data() + j * sh, {}, {}};
            pd = {ar.tin_d.data() + d, nullptr, nullptr, nullptr, ar.bq_d.data() + j * si, ar.bsum_d.data() + j * si, {}, {}};
            if constexpr (compact) {
                pg.compact = ar.compact_g.data() + row0 * H;
                pu.compact = ar.compact_u.data() + row0 * H;
                pd.compact = ar.compact_d.data() + row0 * I_;
                row0 += size_t(kActRows) * ctx.chunks[j].m;
            } else {
                pg.splat32 = ar.splat_g.data() + g;
                pu.splat32 = ar.splat_u.data() + g;
                pd.splat32 = ar.splat_d.data() + d;
                if constexpr (kSplatDup) {
                    pg.splat_dup = ar.splat_dup_g.data() + g;
                    pu.splat_dup = ar.splat_dup_u.data() + g;
                    pd.splat_dup = ar.splat_dup_d.data() + d;
                }
            }
        }
        ctx.grouped = Traits::kGroupedTraversal && nc == 1;
        ctx.wide = Traits::kWideSingleExpert && ctx.m_total == 1 && nc == 1;
        ctx.out_blocks = ctx.zero_first || ctx.store_first || nc > 0 ? int(H / 128) : 0;
    }

    // The phase's tasks:
    //   PrepareGateUp  (chunk, gate | up, 128-input block) on the vector tiers; (chunk, gate | up) on the scalar tier
    //   GateUp         (chunk, gate | up, tile pair), claimed in runs of kTileAlign
    //   Middle         (chunk, 128-intermediate block) on the vector tiers; a chunk on the scalar tier
    //   DownTiles      (chunk, tile pair), claimed in runs of kTileAlign
    //   DownOut        a 128-output block, over every chunk in order
    static Tasks tasks(const ForwardCtx& c, int phase)
    {
        const int nc = int(c.chunks.size());
        const int H = Shape::hidden(*c.layer);
        const int I_ = Shape::intermediate(*c.layer);
        switch (static_cast<Phase>(phase)) {
            case Phase::PrepareGateUp:
#if defined(SGLANG_CPU_EXPERTS_AVX2)
                return {nc * 2 * (H / 128), 1};
#else
                return {nc * 2, 1};
#endif
            case Phase::GateUp: return {nc * 2 * (I_ / 32), kTileAlign};
            case Phase::Middle:
#if defined(SGLANG_CPU_EXPERTS_AVX2)
                return {nc * (I_ / 128), 1};
#else
                return {nc, 1};
#endif
            case Phase::DownTiles: return {nc * (H / 32), kTileAlign};
            case Phase::DownOut: return {c.out_blocks, 1};
        }
        return {};
    }

    // Tasks [t0, t1) of the phase.
    static void run(ForwardCtx& c, int phase, int t0, int t1)
    {
        const Experts<Shape> E{c.layer, c.layer->params};
        switch (static_cast<Phase>(phase)) {
            case Phase::PrepareGateUp:
#if defined(SGLANG_CPU_EXPERTS_AVX2)
                if (c.wide) prepare_gate_up_blocks<Shape, true>(c, E, t0, t1);
                else prepare_gate_up_blocks<Shape, false>(c, E, t0, t1);
#else
                prepare_gate_up_scalar(c, E, t0, t1);
#endif
                break;
            case Phase::GateUp: gate_up(c, E, t0, t1); break;
            case Phase::Middle:
#if defined(SGLANG_CPU_EXPERTS_AVX2)
                if (c.wide) middle_blocks<Shape, true>(c, E, t0, t1);
                else middle_blocks<Shape, false>(c, E, t0, t1);
#else
                middle_scalar(c, E, t0, t1);
#endif
                break;
            case Phase::DownTiles: down_tiles(c, E, t0, t1); break;
            case Phase::DownOut: down_out(c, E, t0, t1); break;
        }
    }

private:
    // Gate and up GEMVs, GEMV j = 2 * chunk + (up ? 1 : 0), a run split at the GEMVs' ends.
    static void gate_up(ForwardCtx& c, const Experts<Shape>& E, int t0, int t1)
    {
        const int I_ = Shape::intermediate(*c.layer);
        const int pairs = I_ / 32;
        for (int t = t0; t < t1;) {
            const int j = t / pairs, end = std::min(t1, (j + 1) * pairs);
            const Chunk& ch = c.chunks[j / 2];
            const bool up = j % 2;
            const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / 2];
            float* tout = (up ? c.tout_u : c.tout_g) + size_t(j / 2) * MAX_M * I_;
            run_tiles(up ? E.up(ch.expert) : E.gate(ch.expert), p, tout, ch.m, (t - j * pairs) * 2,
                      (end - j * pairs) * 2, c.grouped);
            t = end;
        }
    }

    // Down GEMVs, GEMV j = chunk j, into tout_d (quant space; DownOut transforms it).
    static void down_tiles(ForwardCtx& c, const Experts<Shape>& E, int t0, int t1)
    {
        const int H = Shape::hidden(*c.layer);
        const int pairs = H / 32;
        for (int t = t0; t < t1;) {
            const int j = t / pairs, end = std::min(t1, (j + 1) * pairs);
            const Chunk& ch = c.chunks[j];
            run_tiles(E.down(ch.expert), c.prep_d[j], c.tout_d + size_t(j) * MAX_M * H, ch.m, (t - j * pairs) * 2,
                      (end - j * pairs) * 2, c.grouped);
            t = end;
        }
    }

    // out[token] (+)= weight * down row for the 128-output blocks [b0, b1), over every chunk in order after the block's
    // output transform.
    static void down_out(ForwardCtx& c, const Experts<Shape>& E, int b0, int b1)
    {
        const int H = Shape::hidden(*c.layer);
        for (int b = b0; b < b1; ++b) {
            const int col = b * 128;
            if (c.zero_first)
                for (int t = 0; t < c.m_total; ++t) std::memset(c.out + size_t(t) * H + col, 0, 128 * sizeof(float));
            for (size_t j = 0; j < c.chunks.size(); ++j) {
                const Chunk& ch = c.chunks[j];
                float* d = c.tout_d + j * MAX_M * H;
                transform_rows_block(E.down(ch.expert), d, ch.m, col);
                for (int r = 0; r < ch.m; ++r) {
                    float* dst = c.out + size_t(ch.token[r]) * H + col;
                    const float* src = d + size_t(r) * H + col;
                    const float w = ch.weight[r];
                    if (c.store_first && (ch.first >> r & 1u)) {
                        // The zeroed row's sum, 0 + w * src, as the line below computes it: a zero the compiler cannot
                        // see keeps the add (and its contraction) under -Ofast, so the bits match.
                        float zero = 0.0f;
                        asm("" : "+x"(zero));
                        for (int i = 0; i < 128; ++i) dst[i] = w * src[i] + zero;
                    } else {
                        for (int i = 0; i < 128; ++i) dst[i] += w * src[i];
                    }
                }
            }
            if (c.store_first)
                for (const int t : c.unrouted) std::fill_n(c.out + size_t(t) * H + col, 128, 0.0f);
        }
    }
};

// Calls f.template operator()<Shape>() for the layer's plan (p.plan, which registration checked against the layer and
// this build).
template <class F>
decltype(auto) with_plan([[maybe_unused]] const Exl3Quant::Params& p, F&& f)
{
#if defined(SGLANG_CPU_EXPERTS_AVX512)
    static_assert(Dsv41Shape::kMinIsa == Isa::Bw, "the DSV4.1 plan is compiled where SGLANG_CPU_EXPERTS_AVX512 is");
    if (p.plan == Dsv41Shape::kPlan) return f.template operator()<Dsv41Shape>();
#endif
    return f.template operator()<GenericShape>();
}

// The job's serial setup, at admission: groups the call's routes by expert and splits them into chunks of CHUNK_M tokens
// (GENERIC_CHUNK_M for the generic plan), then sizes the scratch. This grouping, not the framework's RouteTable, fixes
// EXL3's accumulation order (expert ascending, then token, then route), so it reads the call's slots as given and runs
// zero-weight routes. Each weight is rounded to FP16, the registered kernel's convention. ExpertForward has validated
// the call (every slot in [-1, capacity), finite weights, rows and k in range). Throws std::invalid_argument for a plan
// this build does not compile.
void prepare_job(ForwardJob& job, const ExpertLayer& l, const ForwardCall& c)
{
    const Exl3Quant::Params& p = l.params;
#if !defined(SGLANG_CPU_EXPERTS_AVX512)
    // A fixed plan reaching here means its branch was compiled out while plan_info listed it available.
    EXL3_CHECK(p.plan == GenericShape::kPlan, "the layer's plan is not compiled into this build");
#endif
    ForwardCtx& ctx = job.ctx;
    ForwardArena& ar = job.ar;
    ctx.layer = &l;
    ctx.x = static_cast<const Fp16*>(c.x);
    ctx.out = c.out;
    ctx.m_total = c.rows;
    ctx.zero_first = !c.accumulate() && !c.store_first();
    ctx.store_first = c.store_first();

    // The live routes as (slot, route index t * k + j), sorted: each expert's routes in token, then route, order.
    auto& routes = ar.routes;
    routes.clear();
    const int n = c.rows * c.k;
    for (int i = 0; i < n; ++i)
        if (c.slots[i] >= 0) routes.emplace_back(c.slots[i], i);
    std::sort(routes.begin(), routes.end());
    const int chunk_m = p.plan == Dsv41Shape::kPlan ? CHUNK_M : GENERIC_CHUNK_M;
    ctx.chunks.clear();
    for (size_t i = 0; i < routes.size();) {
        Chunk ch;
        ch.expert = routes[i].first;
        ch.m = 0;
        for (; i < routes.size() && routes[i].first == ch.expert && ch.m < chunk_m; ++i, ++ch.m) {
            ch.token[ch.m] = routes[i].second / c.k;
            ch.weight[ch.m] = fp16_to_float(float_to_fp16(c.weights[routes[i].second]));
        }
        ctx.chunks.push_back(ch);
    }
    // A token's first write is the first of its rows in DownOut's order (chunk, then row); the rest add to it.
    ctx.unrouted.clear();
    if (ctx.store_first) {
        auto& written = ar.written;
        written.assign(size_t(c.rows), 0);
        for (Chunk& ch : ctx.chunks)
            for (int r = 0; r < ch.m; ++r)
                if (!written[ch.token[r]]) {
                    written[ch.token[r]] = 1;
                    ch.first |= 1u << r;
                }
        for (int t = 0; t < c.rows; ++t)
            if (!written[t]) ctx.unrouted.push_back(t);
    }
    with_plan(p, [&]<class Shape>() { ForwardPlan<Shape>::prepare_scratch(job); });
}

Tasks job_tasks(const ForwardCtx& c, int phase)
{
    return with_plan(c.layer->params, [&]<class Shape>() { return ForwardPlan<Shape>::tasks(c, phase); });
}

void job_run(ForwardCtx& c, int phase, int t0, int t1)
{
    with_plan(c.layer->params, [&]<class Shape>() { ForwardPlan<Shape>::run(c, phase, t0, t1); });
}

// -------------------------------------------------------------------------------------------
//   Plan registry
// -------------------------------------------------------------------------------------------

template <class Shape>
SglangExl3CpuPlanInfo plan_info()
{
    SglangExl3CpuPlanInfo info{};
    info.id = Shape::kPlan;
    info.name = Shape::kName;
    info.fixed = Shape::kFixed;
    if constexpr (Shape::kFixed) {
        info.hidden = Shape::kHidden;
        info.intermediate = Shape::kIntermediate;
        info.bits_w13 = Shape::kBitsW13;
        info.bits_w2 = Shape::kBitsW2;
        info.swizzled = Shape::kSwizzled;
        info.act_limit = Shape::kActLimit;
    }
    info.min_isa = isa_name(Shape::kMinIsa);
    info.available = kIsa >= Shape::kMinIsa;
    return info;
}

// Every compiled plan, the generic plan first; the host's cpu_kernel_plans lists these.
std::vector<SglangExl3CpuPlanInfo> plan_infos() { return {plan_info<GenericShape>(), plan_info<Dsv41Shape>()}; }

const char* plan_rejects(const ExpertLayer& l, const SglangExl3CpuParams& p)
{
    if (p.plan == GenericShape::kPlan) return nullptr;
    if (p.plan != Dsv41Shape::kPlan) return "plan is not a compiled plan id (cpu_kernel_plans)";
    if (kIsa < Dsv41Shape::kMinIsa)
        return "plan dsv41 is not built: it needs AVX-512BW, above this build's -march (SGLANG_JIT_HOST_MARCH)";
    if (!Dsv41Shape::matches(l, p))
        return "plan dsv41 does not accept the layer: it runs hidden 5120, intermediate 2304, act_limit 10, 3-bit "
               "gate/up and down, unswizzled";
    return nullptr;
}

}  // namespace
}  // namespace sglang::exl3_cpu
