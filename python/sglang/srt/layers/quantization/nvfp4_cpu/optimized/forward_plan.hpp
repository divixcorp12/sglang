// Included by moe_mul1.cpp inside its anonymous namespace, after the definitions moe_mul1.h declares (dot_rows,
// swiglu, q8_representable, quantize_block, freeze_compute_cores, pin_compute_worker).
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is the
// dot product's tier, fixed when the library is compiled (kBuildIsa). The primary PlanTraits is the generic plan's;
// PlanTraits<MimoV26ProShape, Isa::Avx2> is MiMo V2.6 Pro's on an AVX2 build.
//
// A call's routes are grouped by slot into units, one per (token, slot), and units into chunks of up to kChunkRows of
// one slot, so dot_rows decodes each weight row once per chunk. Each token's output is still its own routes' sum in
// routing order, so a token's result does not depend on the other rows of the call.

// The forward's phases, in team order; a barrier separates each from the next.
enum class Phase : int
{
    PrepareInput = 0,  // every token's fp16 input to fp32, then its Q8_0 blocks
    GateUp = 1,        // every chunk's gate/up rows, gated SiLU into each unit's intermediate
    Middle = 2,        // every unit's intermediate to Q8_0 blocks
    Down = 3,          // every chunk's down rows, each token's routing-weighted sum into out
};

// The calling thread's per-forward storage, kept across calls so a steady state allocates nothing.
struct ForwardArena
{
    std::vector<float, CacheAligned<float>> xf, inter, partial;
    std::vector<block_q8_0> qx, qi;
    std::vector<Route> route;
    std::vector<int> route_count, unit_token;
    std::vector<RouteRef> refs;
    std::vector<Chunk> chunk;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

// What a (Shape, ISA) plan sets. The primary template is the generic plan.
template <class Shape, Isa I>
struct PlanTraits
{
    static constexpr int kRowUnit = 16;  // output rows per split unit: 16 fp32 outputs fill one cache line
};

// MiMo V2.6 Pro on AVX2: the shape's constants make every loop bound, sf_index group count and row stride a
// compile-time value, and the SwiGLU clamp compiles away. kRowUnit 16 splits 2048 gate/up rows into 128 units and
// 6144 down rows into 384, both even over 16 workers.
template <>
struct PlanTraits<MimoV26ProShape, Isa::Avx2>
{
    static constexpr int kRowUnit = 16;
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
    // represent an input or an intermediate (out is then untouched). Throws when the team is short or a worker
    // cannot be pinned.
    static int run(ForwardCtx& ctx, const StridedExperts<Shape>& E, ForwardArena& ar, int threads)
    {
        bind_routes(ctx, E, ar);
        prepare_scratch(ctx, ar, threads);
        run_team(ctx, threads);
        return ctx.invalid.load(std::memory_order_relaxed) ? 2 : 0;
    }

private:
    // Groups the live routes into units and chunks, and binds each chunk's projections and scaled alphas and each
    // route's unit and down alpha; the team reads only these.
    static void bind_routes(ForwardCtx& c, const StridedExperts<Shape>& E, ForwardArena& ar)
    {
        ar.refs.clear();
        for (int t = 0; t < c.rows; ++t)
            for (int j = 0; j < c.route_count[t]; ++j)
                ar.refs.push_back({c.route[size_t(t) * kMaxRoutes + j].slot, t, j});
        std::sort(ar.refs.begin(), ar.refs.end(), [](const RouteRef& a, const RouteRef& b) {
            return a.slot != b.slot ? a.slot < b.slot : a.token != b.token ? a.token < b.token : a.index < b.index;
        });
        ar.chunk.clear();
        ar.unit_token.clear();
        for (size_t i = 0; i < ar.refs.size(); ++i) {
            const RouteRef& ref = ar.refs[i];
            const bool same_slot = i && ar.refs[i - 1].slot == ref.slot;
            if (!same_slot || ar.refs[i - 1].token != ref.token) {
                if (!same_slot || ar.chunk.back().units == kChunkRows) {
                    Chunk ch;
                    ch.gate = E.gate(ref.slot);
                    ch.up = E.up(ref.slot);
                    ch.down = E.down(ref.slot);
                    ch.gate_alpha = ch.gate.alpha * c.info.inv_input_scale13;
                    ch.up_alpha = ch.up.alpha * c.info.inv_input_scale13;
                    ch.unit0 = int(ar.unit_token.size());
                    ch.units = 0;
                    ar.chunk.push_back(ch);
                }
                ++ar.chunk.back().units;
                ar.unit_token.push_back(ref.token);
            }
            Route& r = c.route[size_t(ref.token) * kMaxRoutes + ref.index];
            r.unit = int(ar.unit_token.size()) - 1;
            r.down_alpha = ar.chunk.back().down.alpha * c.info.inv_input_scale2 * r.weight;
        }
        c.chunk = ar.chunk.data();
        c.chunks = int(ar.chunk.size());
        c.unit_token = ar.unit_token.data();
        c.units = int(ar.unit_token.size());
    }

    // Sizes this call's scratch from the arena. The padded tails are zeroed on every call: the arena is shared by
    // every layer this thread forwards, and Q8_0 blocks and the dot product read whole 64-value blocks.
    static void prepare_scratch(ForwardCtx& c, ForwardArena& ar, int threads)
    {
        const size_t H = size_t(Shape::hidden(c.info)), N = size_t(Shape::intermediate(c.info));
        const size_t Hp = rounded(H, 64), Np = rounded(N, 64), rows = size_t(c.rows), units = size_t(c.units);
        auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
        grow(ar.xf, rows * Hp);
        grow(ar.qx, rows * Hp / 32);
        grow(ar.inter, units * Np);
        grow(ar.qi, units * Np / 32);
        for (size_t t = 0; t < rows; ++t)
            std::fill(ar.xf.begin() + t * Hp + H, ar.xf.begin() + (t + 1) * Hp, 0.f);
        for (size_t u = 0; u < units; ++u)
            std::fill(ar.inter.begin() + u * Np + N, ar.inter.begin() + (u + 1) * Np, 0.f);
        c.xf = ar.xf.data();
        c.qx = ar.qx.data();
        c.inter = ar.inter.data();
        c.qi = ar.qi.data();
        c.partial_stride = rounded(std::max<size_t>(units, 1), 16);  // whole cache lines per worker
        grow(ar.partial, size_t(threads) * c.partial_stride);
        c.partial = ar.partial.data();
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
                // `invalid` and `pin_error` are read only after a barrier (every pin precedes PrepareInput's), so every
                // worker takes the same branch; a failed pin computes nothing into out.
                if (!ctx.invalid.load(std::memory_order_relaxed) && !pin_error.load(std::memory_order_relaxed)) {
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

        if constexpr (P == Phase::PrepareInput) {
            // Token t's blocks are [t * Hp / 32, (t + 1) * Hp / 32) of the flat range.
            const int64_t per_row = int64_t(Hp / 32);
            const auto [b0, b1] = share<1>(int64_t(c.rows) * per_row, worker, workers);
            for (int64_t gb = b0; gb < b1; ++gb) {
                const int64_t t = gb / per_row, b = gb % per_row;
                const uint8_t* x = c.x + size_t(t) * size_t(H) * 2;
                float* xf = c.xf + size_t(t) * Hp;
                for (int64_t i = b * 32; i < std::min<int64_t>(b * 32 + 32, H); ++i) {
                    uint16_t v;
                    std::memcpy(&v, x + 2 * i, 2);
                    xf[i] = ggml_compute_fp16_to_fp32(v);
                }
                if (!q8_representable(xf + b * 32)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(xf + b * 32, c.qx[gb]);
            }
        } else if constexpr (P == Phase::GateUp) {
            const auto [r0, r1] = share<Traits::kRowUnit>(int64_t(c.chunks) * N, worker, workers);
            for (int64_t row = r0; row < r1; ++row) {
                const Chunk& ch = c.chunk[row / N];
                const int i = int(row % N);
                int gate_row, up_row;
                w13_rows(c.info.w13_layout, N, i, gate_row, up_row);
                const block_q8_0* xs[kChunkRows];
                for (int j = 0; j < ch.units; ++j) xs[j] = c.qx + size_t(c.unit_token[ch.unit0 + j]) * (Hp / 32);
                float g[kChunkRows], u[kChunkRows];
                dot_rows(ch.gate.w, ch.gate.sf, gate_row, H, xs, ch.units, g);
                dot_rows(ch.up.w, ch.up.sf, up_row, H, xs, ch.units, u);
                for (int j = 0; j < ch.units; ++j)
                    c.inter[size_t(ch.unit0 + j) * Np + size_t(i)] =
                        swiglu(g[j] * ch.gate_alpha, u[j] * ch.up_alpha, Shape::act_limit(c.info));
            }
        } else if constexpr (P == Phase::Middle) {
            // Unit u's intermediate is Np floats at u * Np, so block b of the flat range is unit b / (Np / 32)'s.
            const auto [b0, b1] = share<1>(int64_t(c.units) * int64_t(Np / 32), worker, workers);
            for (int64_t b = b0; b < b1; ++b) {
                const float* v = c.inter + b * 32;
                if (!q8_representable(v)) c.invalid.store(true, std::memory_order_relaxed);
                else quantize_block(v, c.qi[b]);
            }
        } else {
            static_assert(P == Phase::Down);
            float* partial = c.partial + size_t(worker) * c.partial_stride;
            const auto [h0, h1] = share<Traits::kRowUnit>(H, worker, workers);
            for (int64_t h = h0; h < h1; ++h) {
                for (int ci = 0; ci < c.chunks; ++ci) {
                    const Chunk& ch = c.chunk[ci];
                    const block_q8_0* xs[kChunkRows];
                    for (int j = 0; j < ch.units; ++j) xs[j] = c.qi + size_t(ch.unit0 + j) * (Np / 32);
                    dot_rows(ch.down.w, ch.down.sf, int(h), N, xs, ch.units, partial + ch.unit0);
                }
                for (int t = 0; t < c.rows; ++t) {
                    const Route* route = c.route + size_t(t) * kMaxRoutes;
                    float sum = 0.f;
                    for (int r = 0; r < c.route_count[t]; ++r) sum += partial[route[r].unit] * route[r].down_alpha;
                    float& out = c.out[size_t(t) * size_t(H) + size_t(h)];
                    out = c.accumulate ? out + sum : sum;
                }
            }
        }
    }
};
