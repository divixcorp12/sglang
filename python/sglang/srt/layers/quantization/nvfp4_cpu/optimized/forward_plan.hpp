// Included by moe_mul1.cpp inside its anonymous namespace, after the arithmetic (dot, swiglu, q8_representable,
// quantize_block), the worker-core helpers (freeze_compute_cores, pin_compute_worker) and ForwardCtx/ForwardArena.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is the
// dot product's tier, fixed when the library is compiled (kBuildIsa). The primary PlanTraits is the generic plan's.

// The forward's phases, in team order; a barrier separates each from the next.
enum class Phase : int
{
    PrepareInput = 0,  // fp16 input to fp32, then its Q8_0 blocks
    GateUp = 1,        // every route's gate/up rows, gated SiLU into its intermediate
    Middle = 2,        // every route's intermediate to Q8_0 blocks
    Down = 3,          // every route's down rows, routing-weighted sum into out
};

// What a (Shape, ISA) plan sets. The primary template is the generic plan.
template <class Shape, Isa I>
struct PlanTraits
{
    static constexpr int kRowUnit = 16;  // output rows per split unit: 16 fp32 outputs fill one cache line
};

// Worker `worker`'s contiguous share [first, last) of `total` items, cut on multiples of Unit.
template <int Unit>
std::pair<int64_t, int64_t> share(int64_t total, int worker, int workers)
{
    const int64_t units = (total + Unit - 1) / Unit;
    return {std::min(total, units * worker / workers * Unit), std::min(total, units * (worker + 1) / workers * Unit)};
}

template <class Shape, Isa I>
struct ForwardPlan
{
    using Traits = PlanTraits<Shape, I>;

    // Runs ctx's routes through E on `threads` workers (the caller is worker 0). Returns 0, or 2 when Q8_0 cannot
    // represent the input or an intermediate (out is then untouched). Throws when the team is short or a worker
    // cannot be pinned.
    static int run(ForwardCtx& ctx, const StridedExperts<Shape>& E, ForwardArena& ar, int threads)
    {
        bind_routes(ctx, E);
        prepare_scratch(ctx, ar, threads);
        run_team(ctx, threads);
        return ctx.invalid.load(std::memory_order_relaxed) ? 2 : 0;
    }

private:
    // Each route's projections and scaled alphas, in routing order; the team reads only these.
    static void bind_routes(ForwardCtx& c, const StridedExperts<Shape>& E)
    {
        for (int r = 0; r < c.routes; ++r) {
            const int slot = c.route[r].slot;
            c.gate[r] = E.gate(slot);
            c.up[r] = E.up(slot);
            c.down[r] = E.down(slot);
            c.gate_alpha[r] = c.gate[r].alpha * c.info.inv_input_scale13;
            c.up_alpha[r] = c.up[r].alpha * c.info.inv_input_scale13;
            c.down_alpha[r] = c.down[r].alpha * c.info.inv_input_scale2 * c.route[r].weight;
        }
    }

    // Sizes this call's scratch from the arena. The padded tails are zeroed on every call: the arena is shared by
    // every layer this thread forwards, and Q8_0 blocks and the dot product read whole 64-value blocks.
    static void prepare_scratch(ForwardCtx& c, ForwardArena& ar, int threads)
    {
        const size_t H = size_t(Shape::hidden(c.info)), N = size_t(Shape::intermediate(c.info));
        const size_t Hp = rounded(H, 64), Np = rounded(N, 64), routes = size_t(c.routes);
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(ar.xf, Hp);
        grow(ar.qx, Hp / 32);
        grow(ar.inter, routes * Np);
        grow(ar.qi, routes * Np / 32);
        std::fill(ar.xf.begin() + H, ar.xf.begin() + Hp, 0.f);
        for (size_t r = 0; r < routes; ++r)
            std::fill(ar.inter.begin() + r * Np + N, ar.inter.begin() + (r + 1) * Np, 0.f);
        c.xf = ar.xf.data();
        c.qx = ar.qx.data();
        c.inter = ar.inter.data();
        c.qi = ar.qi.data();
#if defined(NVFP4_CPU_UPSTREAM_BASELINE)
        c.row_scratch_stride = std::max(Hp, Np) / 64;
        grow(ar.row_scratch, size_t(threads) * c.row_scratch_stride);
        c.row_scratch = ar.row_scratch.data();
#else
        (void)threads;
        c.row_scratch = nullptr;
        c.row_scratch_stride = 0;
#endif
    }

    static void run_team(ForwardCtx& ctx, int count)
    {
        freeze_compute_cores();
        if (!g_compute_cores.empty() && size_t(count) > g_compute_cores.size())
            throw std::runtime_error("CPU expert worker count exceeds configured cores");
        std::atomic<int> pin_error{0};
        std::atomic<int> actual_workers{0};
        #pragma omp parallel num_threads(count) shared(ctx, pin_error, actual_workers)
        {
            const int worker = omp_get_thread_num(), n = omp_get_num_threads();
            if (worker == 0) actual_workers.store(n, std::memory_order_relaxed);
            pin_compute_worker(worker, pin_error);
            if (n == count) {
                step<Phase::PrepareInput>(ctx, worker, n);
                // `invalid` is read only after a barrier, so every worker takes the same branch.
                if (!ctx.invalid.load(std::memory_order_relaxed)) {
                    step<Phase::GateUp>(ctx, worker, n);
                    step<Phase::Middle>(ctx, worker, n);
                    if (!ctx.invalid.load(std::memory_order_relaxed)) phase<Phase::Down>(ctx, worker, n);
                }
            }
        }
        if (pin_error.load()) throw std::runtime_error("cannot pin CPU expert worker to its configured core");
        if (actual_workers.load() != count)
            throw std::runtime_error("OpenMP returned fewer CPU expert workers than requested");
    }

    // One phase, then wait for the whole team. Called inside run_team's parallel region (an orphaned barrier binds to
    // that team).
    template <Phase P>
    static void step(ForwardCtx& c, int worker, int workers)
    {
        phase<P>(c, worker, workers);
        #pragma omp barrier
    }

    // One phase for this worker; P picks the phase at compile time.
    template <Phase P>
    static void phase(ForwardCtx& c, int worker, int workers)
    {
        const int H = Shape::hidden(c.info), N = Shape::intermediate(c.info);
        const size_t Hp = rounded(size_t(H), 64), Np = rounded(size_t(N), 64);
        [[maybe_unused]] block_nvfp4* scratch =
            c.row_scratch ? c.row_scratch + size_t(worker) * c.row_scratch_stride : nullptr;

        if constexpr (P == Phase::PrepareInput) {
            const auto [b0, b1] = share<1>(int64_t(Hp / 32), worker, workers);
            for (int64_t b = b0; b < b1; ++b) {
                for (int64_t i = b * 32; i < std::min<int64_t>(b * 32 + 32, H); ++i) {
                    uint16_t v;
                    std::memcpy(&v, c.x + 2 * i, 2);
                    c.xf[i] = ggml_compute_fp16_to_fp32(v);
                }
                if (!q8_representable(c.xf + b * 32)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(c.xf + b * 32, c.qx[b]);
            }
        } else if constexpr (P == Phase::GateUp) {
            const auto [r0, r1] = share<Traits::kRowUnit>(int64_t(c.routes) * N, worker, workers);
            for (int64_t row = r0; row < r1; ++row) {
                const int r = int(row / N), i = int(row % N);
                int gate_row, up_row;
                w13_rows(c.info.w13_layout, N, i, gate_row, up_row);
                const float g = dot(c.gate[r].w, c.gate[r].sf, gate_row, H, c.qx, scratch) * c.gate_alpha[r];
                const float u = dot(c.up[r].w, c.up[r].sf, up_row, H, c.qx, scratch) * c.up_alpha[r];
                c.inter[size_t(r) * Np + size_t(i)] = swiglu(g, u, Shape::act_limit(c.info));
            }
        } else if constexpr (P == Phase::Middle) {
            // Route r's intermediate is Np floats at r * Np, so block b of the flat range is route b / (Np / 32)'s.
            const auto [b0, b1] = share<1>(int64_t(c.routes) * int64_t(Np / 32), worker, workers);
            for (int64_t b = b0; b < b1; ++b) {
                const float* v = c.inter + b * 32;
                if (!q8_representable(v)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(v, c.qi[b]);
            }
        } else {
            static_assert(P == Phase::Down);
            const auto [h0, h1] = share<Traits::kRowUnit>(H, worker, workers);
            for (int64_t h = h0; h < h1; ++h) {
                float sum = 0.f;
                for (int r = 0; r < c.routes; ++r)
                    sum += dot(c.down[r].w, c.down[r].sf, int(h), N, c.qi + size_t(r) * (Np / 32), scratch)
                           * c.down_alpha[r];
                c.out[h] = c.accumulate ? c.out[h] + sum : sum;
            }
        }
    }
};
