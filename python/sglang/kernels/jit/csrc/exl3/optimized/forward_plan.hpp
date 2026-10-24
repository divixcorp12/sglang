// EXL3's forward: the plans (ForwardPlan<Shape>), the plan entry (Exl3Quant::dispatch) and the plan registry.
// Included by kernel.cpp.
//
// One forward = ForwardPlan<Shape>::run, at the build's tier (kIsa). Shape (shapes.hpp) fixes what the plan may assume
// about the layer; PlanTraits<Shape> what it turns on. The primary traits are the generic plan; PlanTraits<Dsv41Shape>
// is the fast path measured and validated bit-exact for DeepSeek V4.1 from AVX-512BW up (README.txt).
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "phases.hpp"
#include "tile_assignment.hpp"
#include "worker_trace.hpp"
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
    static constexpr int kSplitTiles = 8;             // GEMV split unit in tiles (assign_gemvs)
    static constexpr bool kClaimMixedChunks = false;  // chunks of differing token counts claim groups (claim_groups)
};

// DeepSeek V4.1 from AVX-512BW up: the validated fast path.
template <>
struct PlanTraits<Dsv41Shape>
{
    static constexpr bool kCompactScratch = true;
    static constexpr bool kGroupedTraversal = true;  // one-token chunks (kRegisterBudget[0])
    static constexpr bool kWideSingleExpert = true;
    static constexpr int kSplitTiles = 2;  // compact unswizzled kernels take any tile pair (register_tiles)
    // A call whose chunks differ in token count claims 8-tile groups instead of a static split, which neither evenly
    // nor by rows matches a chunk's cost (memory-bound at one token, compute-bound at several): rows 3/4/8 random
    // -14/-21/-26% at 16 workers. Uniform calls keep the static split, its grouped traversal included.
    static constexpr bool kClaimMixedChunks = true;
};

template <class Shape, bool Trace = EXL3_MOE_CPU_WORKER_TRACE != 0>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape>;
    using Capture = worker_trace::Capture<Trace>;

    // Sizes this call's scratch from the arena, runs the team, and returns. ctx.chunks must be non-empty.
    static void run(ForwardCtx& ctx, const Experts<Shape>& E, ForwardArena& ar, Team& team)
    {
        Capture trace(team.workers(), ctx.m_total, int(ctx.chunks.size()));
        prepare_scratch(ctx, ar, trace);
        const int nc = int(ctx.chunks.size());
        const bool grouped = Traits::kGroupedTraversal && nc == 1;
        const bool wide = Traits::kWideSingleExpert && ctx.m_total == 1 && nc == 1;
        trace.team_start();
        // The phases on the engine's team (team.hpp), every worker in step, the team's owner as worker 0.
        team.run([&](int worker, int n) {
            trace.worker_start(worker);
            if (ctx.zero_out && !ctx.store_first && worker == 0)
                std::memset(ctx.out, 0, size_t(ctx.m_total) * ctx.layer->hidden * sizeof(float));
            step<Phase::PrepareGateUp>(ctx, E, team, worker, n, grouped, wide, trace);
            step<Phase::GateUp>(ctx, E, team, worker, n, grouped, wide, trace);
            if (ctx.stage_two) {
                if (worker == 0) ctx.stage_two->wait();
                team.barrier();
            }
            step<Phase::Middle>(ctx, E, team, worker, n, grouped, wide, trace);
            step<Phase::Down>(ctx, E, team, worker, n, grouped, wide, trace);
            step<Phase::Accumulate>(ctx, E, team, worker, n, grouped, wide, trace);
            trace.worker_end(worker);
        });
        trace.finish();
    }

private:
    static void prepare_scratch(ForwardCtx& ctx, ForwardArena& ar, Capture& trace)
    {
        constexpr bool compact = Traits::kCompactScratch;
        const int nc = int(ctx.chunks.size());
        const size_t H = Shape::hidden(*ctx.layer);
        const size_t I_ = Shape::intermediate(*ctx.layer);
        auto grow = [&](auto& v, size_t n, const char* name) { trace.grow(v, n, name); };
        grow(ar.tin_g, nc * MAX_M * H, "ar.tin_g");
        grow(ar.tin_u, nc * MAX_M * H, "ar.tin_u");
        grow(ar.tin_d, nc * MAX_M * I_, "ar.tin_d");
        grow(ar.tout_g, nc * MAX_M * I_, "ar.tout_g");
        grow(ar.tout_u, nc * MAX_M * I_, "ar.tout_u");
        grow(ar.tout_d, nc * MAX_M * H, "ar.tout_d");
        grow(ctx.prep_g, nc, "ctx.prep_g");
        grow(ctx.prep_u, nc, "ctx.prep_u");
        grow(ctx.prep_d, nc, "ctx.prep_d");
        ctx.tout_g = ar.tout_g.data();
        ctx.tout_u = ar.tout_u.data();
        ctx.tout_d = ar.tout_d.data();
        if constexpr (compact) {
            // Each chunk's kActRows * m rows follow the previous chunk's (a call of one-token chunks keeps chunk j at
            // j * kActRows rows).
            size_t rows = 0;
            for (const Chunk& ch : ctx.chunks) rows += size_t(kActRows) * ch.m;
            grow(ar.compact_g, rows * H, "ar.compact_g");
            grow(ar.compact_u, rows * H, "ar.compact_u");
            grow(ar.compact_d, rows * I_, "ar.compact_d");
        } else {
            grow(ar.splat_g, nc * MAX_M * H, "ar.splat_g");
            grow(ar.splat_u, nc * MAX_M * H, "ar.splat_u");
            grow(ar.splat_d, nc * MAX_M * I_, "ar.splat_d");
            if constexpr (kSplatDup) {
                grow(ar.splat_dup_g, nc * MAX_M * H, "ar.splat_dup_g");
                grow(ar.splat_dup_u, nc * MAX_M * H, "ar.splat_dup_u");
                grow(ar.splat_dup_d, nc * MAX_M * I_, "ar.splat_dup_d");
            }
        }
        // One scale and sum per (128-input block, quantized row).
        const size_t sh = MAX_M * (H / kActBlock), si = MAX_M * (I_ / kActBlock);
        grow(ar.bq_g, nc * sh, "ar.bq_g");
        grow(ar.bq_u, nc * sh, "ar.bq_u");
        grow(ar.bq_d, nc * si, "ar.bq_d");
        grow(ar.bsum_g, nc * sh, "ar.bsum_g");
        grow(ar.bsum_u, nc * sh, "ar.bsum_u");
        grow(ar.bsum_d, nc * si, "ar.bsum_d");
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
        if constexpr (Traits::kSplitTiles < 8) {
            const size_t blocks = nc * (H / 128);
            grow(ar.down_tiles_done, blocks, "ar.down_tiles_done");
            std::fill_n(ar.down_tiles_done.data(), blocks, 0);
            ctx.down_tiles_done = ar.down_tiles_done.data();
        }
        ctx.claims = nullptr;
        if constexpr (Traits::kClaimMixedChunks) {
            const bool mixed = std::any_of(ctx.chunks.begin(), ctx.chunks.end(),
                                           [&](const Chunk& ch) { return ch.m != ctx.chunks.front().m; });
            if (mixed) {
                grow(ar.claims, 32, "ar.claims");
                std::fill_n(ar.claims.data(), 32, 0);
                ctx.claims = ar.claims.data();
            }
        }
    }

    // One phase of the team's sequence: run it, then wait for the whole team unless it is the last.
    template <Phase P>
    static void step(ForwardCtx& ctx, const Experts<Shape>& E, Team& team, int worker, int n, bool grouped, bool wide,
                     Capture& trace)
    {
        trace.begin(worker, static_cast<int>(P));
        phase<P>(ctx, E, worker, n, grouped, wide, trace);
        trace.work_end(worker, static_cast<int>(P));
        if constexpr (P != Phase::Accumulate) team.barrier();
        trace.end(worker, static_cast<int>(P));
    }

    // One phase for this worker; P picks the phase at compile time, so each instantiation holds one phase's code.
    template <Phase P>
    static void phase(ForwardCtx& c, const Experts<Shape>& E, int worker, int num_workers, bool grouped, bool wide,
                      Capture& trace)
    {
        [[maybe_unused]] const int nc = int(c.chunks.size());
        [[maybe_unused]] const int H = Shape::hidden(*c.layer);
        [[maybe_unused]] const int I_ = Shape::intermediate(*c.layer);

        if constexpr (P == Phase::PrepareGateUp) {
#if defined(SGLANG_CPU_EXPERTS_AVX2)
            if (wide) prepare_gate_up_blocks<Shape, true>(c, E, worker, num_workers);
            else prepare_gate_up_blocks<Shape, false>(c, E, worker, num_workers);
#else
            prepare_gate_up_scalar(c, E, worker, num_workers);
#endif
        } else if constexpr (P == Phase::GateUp) {
            // Gate + up GEMVs (assign_gemvs; claim_groups for a call of differing chunk sizes).
            const auto gemv = [&](int j, int t0, int t1) {
                const Chunk& ch = c.chunks[j / 2];
                const bool up = j % 2;
                const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / 2];
                float* tout = (up ? c.tout_u : c.tout_g) + size_t(j / 2) * MAX_M * I_;
                trace.add_work(worker, static_cast<int>(P), int64_t(t1 - t0) * ch.m);
                run_tiles(up ? E.up(ch.expert) : E.gate(ch.expert), p, tout, ch.m, t0, t1, grouped);
            };
            if (c.claims) claim_groups(c.claims[0], nc * 2, I_ / 16, gemv);
            else assign_plan_gemvs<Traits::kSplitTiles>(worker, num_workers, nc * 2, I_ / 16,
                                                        [&](int j) { return c.chunks[j / 2].m; }, gemv,
                                                        E.params.row_weighted != 0);
        } else if constexpr (P == Phase::Middle) {
#if defined(SGLANG_CPU_EXPERTS_AVX2)
            if (wide) middle_blocks<Shape, true>(c, E, worker, num_workers);
            else middle_blocks<Shape, false>(c, E, worker, num_workers);
#else
            middle_scalar(c, E, worker, num_workers);
#endif
        } else if constexpr (P == Phase::Down) {
            // Down GEMVs (split or claimed as gate/up's are), each with its output transform.
            const auto gemv = [&](int j, int t0, int t1) {
                const Chunk& ch = c.chunks[j];
                const Exl3Projection down = E.down(ch.expert);
                float* tout = c.tout_d + size_t(j) * MAX_M * H;
                trace.add_work(worker, static_cast<int>(P), int64_t(t1 - t0) * ch.m);
                run_tiles(down, c.prep_d[j], tout, ch.m, t0, t1, grouped);
                if constexpr (Traits::kSplitTiles == 8) {
                    transform_owned_blocks(down, tout, ch.m, t0, t1);
                } else {
                    // A block's tiles may come from several workers: the one that completes it transforms it. The
                    // acq_rel count makes the other workers' tile stores visible to that worker.
                    for (int b = t0 / 8; b * 8 < t1; ++b) {
                        const int done = std::min(t1, b * 8 + 8) - std::max(t0, b * 8);
                        std::atomic_ref<int> count(c.down_tiles_done[size_t(j) * (H / 128) + b]);
                        if (count.fetch_add(done, std::memory_order_acq_rel) + done == 8)
                            transform_owned_blocks(down, tout, ch.m, b * 8, b * 8 + 8);
                    }
                }
            };
            if (c.claims) claim_groups(c.claims[16], nc, H / 16, gemv);
            else assign_plan_gemvs<Traits::kSplitTiles>(worker, num_workers, nc, H / 16,
                                                        [&](int j) { return c.chunks[j].m; }, gemv,
                                                        E.params.row_weighted != 0);
        } else {
            static_assert(P == Phase::Accumulate);
            accumulate(c, H, worker, num_workers);
        }
    }

    // out[token] (+)= weight * down row, each worker over its own columns.
    static void accumulate(ForwardCtx& c, int H, int worker, int num_workers)
    {
        const int c0 = ((H / 16) * worker / num_workers) * 16;
        const int c1 = ((H / 16) * (worker + 1) / num_workers) * 16;
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
            for (int i = 0; i < c.n_unrouted; ++i)
                std::fill(c.out + size_t(c.unrouted[i]) * H + c0, c.out + size_t(c.unrouted[i]) * H + c1, 0.0f);
    }
};

// Forwards by plan since load: [0] the DSV4.1 plan, [1] the generic plan. Relaxed: counts only, read through
// exl3_cpu_plan_calls (kernel.cpp, the one file that includes this header).
std::atomic<int64_t> g_plan_calls[2];

// Runs the layer's plan (p.plan, which registration checked against the layer and this build).
void run_plan(ForwardCtx& ctx, const Exl3Quant::Params& p, ForwardArena& ar, Team& team)
{
#if defined(SGLANG_CPU_EXPERTS_AVX512)
    static_assert(Dsv41Shape::kMinIsa == Isa::Bw, "the DSV4.1 plan is compiled where SGLANG_CPU_EXPERTS_AVX512 is");
    if (p.plan == Dsv41Shape::kPlan) {
        g_plan_calls[0].fetch_add(1, std::memory_order_relaxed);
        ForwardPlan<Dsv41Shape>::run(ctx, Experts<Dsv41Shape>{ctx.layer, p}, ar, team);
        return;
    }
#endif
    g_plan_calls[1].fetch_add(1, std::memory_order_relaxed);
    ForwardPlan<GenericShape>::run(ctx, Experts<GenericShape>{ctx.layer, p}, ar, team);
}

// The plan entry: groups the routes by expert and splits them into chunks of CHUNK_M tokens (GENERIC_CHUNK_M for the
// generic plan). This grouping, not the framework's RouteTable, fixes EXL3's accumulation order (expert ascending, then
// token, then route), so it reads the call's slots as given and runs zero-weight routes. Each weight is rounded to
// FP16, the registered kernel's convention. ExpertForward has validated the call (every slot in [-1, capacity), finite
// weights, rows and k in range).
int Exl3Quant::dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa, Team& team)
{
    ForwardCtx ctx;
    ctx.layer = &l;
    ctx.x = static_cast<const Fp16*>(c.x);
    ctx.out = c.out;
    ctx.m_total = c.rows;
    ctx.zero_out = !c.accumulate();
    ctx.stage_two = c.stage_two;

    ForwardArena& ar = ForwardArena::get();
    ctx.chunks = std::move(ar.chunks);
    ctx.chunks.clear();
    ctx.prep_g = std::move(ar.prep_g);
    ctx.prep_u = std::move(ar.prep_u);
    ctx.prep_d = std::move(ar.prep_d);
    const auto give_back = [&] {
        ar.chunks = std::move(ctx.chunks);
        ar.prep_g = std::move(ctx.prep_g);
        ar.prep_u = std::move(ctx.prep_u);
        ar.prep_d = std::move(ctx.prep_d);
    };

    // The live routes as (slot, route index t * k + j), sorted: each expert's routes in token, then route, order.
    auto& routes = ar.routes;
    routes.clear();
    const int n = c.rows * c.k;
    for (int i = 0; i < n; ++i)
        if (c.slots[i] >= 0) routes.emplace_back(c.slots[i], i);
    std::sort(routes.begin(), routes.end());
    const int chunk_m = p.plan == Dsv41Shape::kPlan ? CHUNK_M : GENERIC_CHUNK_M;
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
    // A token's first write is the first of its rows in Accumulate's order (chunk, then row); the rest add to it.
    ctx.store_first = c.store_first();
    ctx.unrouted = nullptr;
    ctx.n_unrouted = 0;
    if (ctx.store_first) {
        auto& written = ar.written;
        written.assign(size_t(c.rows), 0);
        for (Chunk& ch : ctx.chunks)
            for (int r = 0; r < ch.m; ++r)
                if (!written[ch.token[r]]) {
                    written[ch.token[r]] = 1;
                    ch.first |= 1u << r;
                }
        ar.unrouted.clear();
        for (int t = 0; t < c.rows; ++t)
            if (!written[t]) ar.unrouted.push_back(t);
        ctx.unrouted = ar.unrouted.data();
        ctx.n_unrouted = int(ar.unrouted.size());
    }
    if (ctx.chunks.empty()) {
        if (ctx.stage_two) ctx.stage_two->wait();
        if (ctx.zero_out) std::memset(ctx.out, 0, size_t(ctx.m_total) * l.hidden * sizeof(float));
    } else {
        run_plan(ctx, p, ar, team);
    }
    give_back();
    return 0;
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
