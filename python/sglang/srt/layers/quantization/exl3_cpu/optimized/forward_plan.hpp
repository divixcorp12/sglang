// Included by moe_mul1.cpp inside its anonymous namespace, after the phase helpers (prepare_rows, run_tiles,
// transform_out, middle_blocks, prepare_gu_blocks, transform_owned_blocks, assign_gemvs) and ForwardCtx/ForwardArena.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is
// the ISA tier. The primary template is the generic plan. PlanTraits<Dsv41Shape, Isa::Bw> turns on the fast path that
// was measured and validated bit-exact on AVX-512BW (exl3_cpu/optimized/README.txt); every other (Shape, I) pair runs
// the generic plan.

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

template <class Shape, Isa I>
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
        const int H = Shape::hidden(ctx.info);
        const int I_ = Shape::intermediate(ctx.info);
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

    template <class Experts>
    static void run_team(ForwardCtx& ctx, const Experts& E, int count, bool grouped, bool wide)
    {
        // Freeze the configured cores once. Steady-state forwards acquire no pool mutex.
        if (!g_compute_started.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(g_cores_mutex);
            if (!g_compute_started.load(std::memory_order_relaxed)) {
                g_compute_cores = g_configured_cores;
                g_compute_started.store(true, std::memory_order_release);
            }
        }
        TORCH_CHECK(g_compute_cores.empty() || size_t(count)<=g_compute_cores.size(),
                    "CPU expert worker count exceeds configured cores");
        const bool prof=g_prof_enabled.load(std::memory_order_relaxed);
        double phase_us[6]{};
        std::atomic<int> pin_error{0};
        std::atomic<int> actual_workers{0};
        #pragma omp parallel num_threads(count) shared(ctx,pin_error,actual_workers,phase_us)
        {
            const int worker=omp_get_thread_num(),n=omp_get_num_threads();
            if(worker==0)actual_workers.store(n,std::memory_order_relaxed);
            if(!g_compute_cores.empty()) {
                const int core=g_compute_cores[worker];
                static thread_local int pinned_core=-1;
                if(pinned_core!=core || sched_getcpu()!=core) {
                    cpu_set_t set;CPU_ZERO(&set);CPU_SET(core,&set);
                    if(pthread_setaffinity_np(pthread_self(),sizeof(set),&set))
                        pin_error.store(1,std::memory_order_relaxed);
                    else pinned_core=core;
                }
            }
            if(n==count) {
                step<Phase::PrepareGateUp>(ctx,E,worker,n,grouped,wide,prof,phase_us);
                step<Phase::GateUp>(ctx,E,worker,n,grouped,wide,prof,phase_us);
                step<Phase::Middle>(ctx,E,worker,n,grouped,wide,prof,phase_us);
                step<Phase::Down>(ctx,E,worker,n,grouped,wide,prof,phase_us);
                step<Phase::Accumulate>(ctx,E,worker,n,grouped,wide,prof,phase_us);
            }
        }
        TORCH_CHECK(!pin_error.load(),"cannot pin CPU expert worker to its configured core");
        TORCH_CHECK(actual_workers.load()==count,"OpenMP returned fewer CPU expert workers than requested");
        if(prof)printf("moe_cpu phases(us): %.1f %.1f %.1f %.1f %.1f %.1f\n",phase_us[0],phase_us[1],phase_us[2],phase_us[3],phase_us[4],phase_us[5]);
    }

    // One phase of the team's sequence: run it, then wait for the whole team unless it is the last. Called inside
    // run_team's parallel region (an orphaned barrier binds to that team). phase_us is indexed by the phase's value.
    template <Phase P, class Experts>
    static void step(ForwardCtx& ctx, const Experts& E, int worker, int n, bool grouped, bool wide, bool prof,
                     double* phase_us)
    {
        const auto begin=prof?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        phase<P>(ctx,E,worker,n,grouped,wide);
        if constexpr (P != Phase::Accumulate) {
            #pragma omp barrier
        }
        if(prof && worker==0)phase_us[static_cast<int>(P)]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
    }

    // One phase for this worker; P picks the phase at compile time, so each instantiation holds one phase's code.
    template <Phase P, class Experts>
    static void phase(ForwardCtx& c, const Experts& E, int worker, int num_workers, bool grouped, bool wide)
    {
        [[maybe_unused]] const int nc = static_cast<int>(c.chunks.size());
        [[maybe_unused]] const int H = Shape::hidden(c.info);
        [[maybe_unused]] const int I_ = Shape::intermediate(c.info);

        if constexpr (P == Phase::PrepareGateUp)
        {
            if (EXL3_MOE_CPU_ACT_BLOCK==128 && act_blocked(H) && I==Isa::Bw) {
                if (wide) prepare_gu_blocks<Shape, I, true>(c,E,worker,num_workers);
                else prepare_gu_blocks<Shape, I, false>(c,E,worker,num_workers);
                return;
            }
            // Prepare gate and up inputs, distributed over (chunk, gate/up)
            const int gu = !Shape::gated(c.info) ? 1 : 2;
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
            const int gu = !Shape::gated(c.info) ? 1 : 2;
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
            const bool gated = Shape::gated(c.info);
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
                const float lim = Shape::act_limit(c.info) != 0.0f
                    ? Shape::act_limit(c.info) : std::numeric_limits<float>::infinity();
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
