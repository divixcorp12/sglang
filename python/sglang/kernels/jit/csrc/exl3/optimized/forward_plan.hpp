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
#include <atomic>
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
    // GateUp and DownTiles claim tile pairs in runs of this many: the generic tile kernels start and end a range on an
    // 8-tile group.
    static constexpr int kTileAlign = 4;
};

// DeepSeek V4.1 from AVX-512BW up: the validated fast path.
template <>
struct PlanTraits<Dsv41Shape>
{
    static constexpr bool kCompactScratch = true;
    static constexpr bool kGroupedTraversal = true;  // one-token chunks (kRegisterBudget[0])
    static constexpr bool kWideSingleExpert = true;
    // The compact register kernels take any tile pair. A coarser run leaves workers idle at the end of the
    // memory-bound GEMVs: 144 pairs in runs of 8 put 32 tiles on the slowest of 16 workers, against 18 evenly.
    static constexpr int kTileAlign = 1;
};

template <class Shape>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape>;
    static constexpr int kTileAlign = Traits::kTileAlign;

    // Lays the job's scratch out for its chunks and points the context into it. ctx.chunks is final but for row0.
    static void prepare_scratch(ForwardJob& job)
    {
        ForwardCtx& ctx = job.ctx;
        constexpr bool compact = Traits::kCompactScratch;
        constexpr bool scalar = kIsa == Isa::Scalar;  // its tiles read the transformed fp32 rows (PreparedIn::tin)
        const size_t nc = ctx.chunks.size();
        const size_t H = Shape::hidden(*ctx.layer);
        const size_t I_ = Shape::intermediate(*ctx.layer);
        // Each chunk's kActRows * m compact rows follow the previous chunk's (a call of one-token chunks keeps chunk j
        // at j * kActRows rows).
        size_t compact_rows = 0;
        for (Chunk& ch : ctx.chunks) {
            ch.row0 = int(compact_rows);
            compact_rows += size_t(kActRows) * ch.m;
        }

        // The regions' offsets; an absent region's is unused.
        struct ActRegions { size_t tin, splat, splat_dup, compact, bq, bsum; };
        ::sglang::cpu_experts::ScratchLayout layout;
        const auto add_act = [&](size_t k) {
            ActRegions a{};
            if constexpr (scalar) a.tin = layout.add<float>(nc * MAX_M * k);
            if constexpr (compact) a.compact = layout.add<int16_t>(compact_rows * k);
            else if constexpr (!scalar) {
                a.splat = layout.add<int32_t>(nc * MAX_M * k);
                if constexpr (kSplatDup) a.splat_dup = layout.add<int32_t>(nc * MAX_M * k);
            }
            // One scale and sum per (128-input block, quantized row).
            a.bq = layout.add<float>(nc * MAX_M * (k / kActBlock));
            a.bsum = layout.add<int32_t>(nc * MAX_M * (k / kActBlock));
            return a;
        };
        const ActRegions rg = add_act(H), ru = add_act(H), rd = add_act(I_);
        const size_t tout_g = layout.add<float>(nc * MAX_M * I_);
        const size_t tout_u = layout.add<float>(nc * MAX_M * I_);
        const size_t tout_d = layout.add<float>(nc * MAX_M * H);
        const size_t tiles_done = layout.add<int>(nc * (H / 128));

        auto& s = job.scratch;
        s.fit(layout);
        const auto act = [&](const ActRegions& a, size_t k) {
            Activations v{};
            if constexpr (scalar) v.tin = s.at<float>(a.tin);
            if constexpr (compact) v.compact = s.at<int16_t>(a.compact);
            else if constexpr (!scalar) {
                v.splat32 = s.at<int32_t>(a.splat);
                if constexpr (kSplatDup) v.splat_dup = s.at<int32_t>(a.splat_dup);
            }
            v.bq = s.at<float>(a.bq);
            v.bsum = s.at<int32_t>(a.bsum);
            v.k = int(k);
            return v;
        };
        ctx.act_g = act(rg, H);
        ctx.act_u = act(ru, H);
        ctx.act_d = act(rd, I_);
        ctx.tout_g = s.at<float>(tout_g);
        ctx.tout_u = s.at<float>(tout_u);
        ctx.tout_d = s.at<float>(tout_d);
        ctx.down_tiles_done = s.at<int>(tiles_done);
        std::fill_n(ctx.down_tiles_done, nc * (H / 128), 0);
    }

    // The phase's tasks:
    //   PrepareGateUp  (chunk, gate | up, 128-input block) on the vector tiers; (chunk, gate | up) on the scalar tier
    //   GateUp         (chunk, gate | up, tile pair), claimed in runs of kTileAlign
    //   Middle         (chunk, 128-intermediate block) on the vector tiers; a chunk on the scalar tier
    //   DownTiles      (chunk, tile pair), claimed in runs of kTileAlign
    //   DownOut        a 16-output unit, over every chunk in order
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
            case Phase::DownOut: return {out_units(c), 1};
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
                if (wide(c)) prepare_gate_up_blocks<Shape, true>(c, E, t0, t1);
                else prepare_gate_up_blocks<Shape, false>(c, E, t0, t1);
#else
                prepare_gate_up_scalar(c, E, t0, t1);
#endif
                break;
            case Phase::GateUp: gate_up(c, E, t0, t1); break;
            case Phase::Middle:
#if defined(SGLANG_CPU_EXPERTS_AVX2)
                if (wide(c)) middle_blocks<Shape, true>(c, E, t0, t1);
                else middle_blocks<Shape, false>(c, E, t0, t1);
#else
                middle_scalar(c, E, t0, t1);
#endif
                break;
            case Phase::DownTiles: down_tiles(c, E, t0, t1); break;
            case Phase::DownOut: down_out(c, t0, t1); break;
        }
    }

private:
    // The grouped traversal: one expert routed (PlanTraits::kGroupedTraversal).
    static bool grouped(const ForwardCtx& c) { return Traits::kGroupedTraversal && c.chunks.size() == 1; }
    // The 512-wide quantization: one token through one expert (PlanTraits::kWideSingleExpert).
    static bool wide(const ForwardCtx& c) { return Traits::kWideSingleExpert && c.m_total == 1 && c.chunks.size() == 1; }
    // DownOut's tasks: H / 16 units, or none when the job writes nothing.
    static int out_units(const ForwardCtx& c)
    {
        return c.zero_first || c.store_first || !c.chunks.empty() ? Shape::hidden(*c.layer) / 16 : 0;
    }

    // Gate and up GEMVs, GEMV j = 2 * chunk + (up ? 1 : 0), a run split at the GEMVs' ends.
    static void gate_up(ForwardCtx& c, const Experts<Shape>& E, int t0, int t1)
    {
        const int I_ = Shape::intermediate(*c.layer);
        const int pairs = I_ / 32;
        for (int t = t0; t < t1;) {
            const int j = t / pairs, end = std::min(t1, (j + 1) * pairs);
            const Chunk& ch = c.chunks[j / 2];
            const bool up = j % 2;
            const PreparedIn p = (up ? c.act_u : c.act_g).chunk(j / 2, ch);
            float* tout = (up ? c.tout_u : c.tout_g) + size_t(j / 2) * MAX_M * I_;
            run_tiles(up ? E.up(ch.expert) : E.gate(ch.expert), p, tout, ch.m, (t - j * pairs) * 2,
                      (end - j * pairs) * 2, grouped(c));
            t = end;
        }
    }

    // Down GEMVs, GEMV j = chunk j, into tout_d, each 128-output block transformed in place by the worker that completes
    // its eighth tile. The acq_rel count makes the other workers' tile stores visible to that worker.
    static void down_tiles(ForwardCtx& c, const Experts<Shape>& E, int t0, int t1)
    {
        const int H = Shape::hidden(*c.layer);
        const int pairs = H / 32;
        for (int t = t0; t < t1;) {
            const int j = t / pairs, end = std::min(t1, (j + 1) * pairs);
            const Chunk& ch = c.chunks[j];
            const Exl3Projection down = E.down(ch.expert);
            float* tout = c.tout_d + size_t(j) * MAX_M * H;
            const int tile0 = (t - j * pairs) * 2, tile1 = (end - j * pairs) * 2;
            run_tiles(down, c.act_d.chunk(j, ch), tout, ch.m, tile0, tile1, grouped(c));
            for (int b = tile0 / 8; b * 8 < tile1; ++b) {
                const int done = std::min(tile1, b * 8 + 8) - std::max(tile0, b * 8);
                std::atomic_ref<int> count(c.down_tiles_done[size_t(j) * (H / 128) + b]);
                if (count.fetch_add(done, std::memory_order_acq_rel) + done == 8)
                    transform_rows_block(down, tout, ch.m, b * 128);
            }
            t = end;
        }
    }

    // out[token] (+)= weight * down row for the 16-output units [u0, u1), over every chunk in order.
    static void down_out(ForwardCtx& c, int u0, int u1)
    {
        const int H = Shape::hidden(*c.layer);
        // The columns as a run-time range, as the loops below were compiled in the barrier-phased forward: under -Ofast a
        // fixed 16-wide loop fuses its first element differently on the vector tiers.
        const int c0 = u0 * 16, c1 = u1 * 16;
        if (c.zero_first)
            for (int t = 0; t < c.m_total; ++t) std::memset(c.out + size_t(t) * H + c0, 0, size_t(c1 - c0) * sizeof(float));
        for (size_t j = 0; j < c.chunks.size(); ++j) {
            const Chunk& ch = c.chunks[j];
            const float* d = c.tout_d + j * MAX_M * H;
            for (int r = 0; r < ch.m; ++r) {
                float* dst = c.out + size_t(ch.token[r]) * H;
                const float* src = d + size_t(r) * H;
                const float w = ch.weight[r];
                if (c.store_first && (ch.first >> r & 1u)) {
                    // The zeroed row's sum, 0 + w * src, as the line below computes it: a zero the compiler cannot see
                    // keeps the add (and its contraction) under -Ofast, so the bits match.
                    float zero = 0.0f;
                    asm("" : "+x"(zero));
                    for (int col = c0; col < c1; ++col) dst[col] = w * src[col] + zero;
                } else {
                    for (int col = c0; col < c1; ++col) dst[col] += w * src[col];
                }
            }
        }
        if (c.store_first)
            for (const int t : c.unrouted) std::fill_n(c.out + size_t(t) * H + c0, c1 - c0, 0.0f);
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
    ctx.layer = &l;
    ctx.x = static_cast<const Fp16*>(c.x);
    ctx.out = c.out;
    ctx.m_total = c.rows;
    ctx.zero_first = !c.accumulate() && !c.store_first();
    ctx.store_first = c.store_first();

    // The live routes as (slot, route index t * k + j), sorted: each expert's routes in token, then route, order.
    std::vector<std::pair<int32_t, int32_t>> routes;
    const int n = c.rows * c.k;
    routes.reserve(size_t(n));
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
        std::vector<uint8_t> written(size_t(c.rows), 0);  // per token, whether a chunk row writes it
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
