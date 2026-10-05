// Included by kernel.cpp inside quant.hpp's namespace, after the arithmetic (math.hpp: swiglu, q8_representable,
// quantize_block; dot_rows<I, M> from math_scalar.hpp and math_avx2.hpp) and kChunkRows.
//
// One forward = ForwardPlan<Shape, I>::run. Shape (shapes.hpp) fixes what the plan may assume about the layer; I is the
// dot product's tier, the one ExpertForward detected at run time.
//
// A call's routes are grouped by slot into units, one per (token, slot), and units into chunks of up to kChunkRows of
// one slot, so dot_rows decodes each weight row once per chunk. Each token's output is still its own routes' sum in
// routing order, so a token's result does not depend on the other rows of the call.

// 64-byte-aligned storage, so each kRowUnit share of fp32 outputs owns whole cache lines.
template <class T>
struct CacheAligned
{
    using value_type = T;
    CacheAligned() = default;
    template <class U> CacheAligned(const CacheAligned<U>&) {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{64})); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t{64}); }
    template <class U> bool operator==(const CacheAligned<U>&) const { return true; }
};

// What the plan binds to route (t, i) of the call's RouteTable, stored at t * k + i like the table.
struct RouteBinding
{
    int unit;          // the (token, slot) unit computing this route's expert
    float down_alpha;  // the slot's down alpha * inv_input_scale2 * weight
};

// A route found while grouping a call's routes by slot: token `token`'s route number `index`.
struct RouteRef
{
    int slot, token, index;
};

// Up to kChunkRows units of one slot, run together so each weight row is decoded once for all of them. A unit is one
// (token, slot) pair: one intermediate row and one down result per output, however many of the token's routes name
// the slot.
struct Chunk
{
    Projection gate, up, down;
    float gate_alpha, up_alpha;  // the projection's alpha * inv_input_scale13
    int unit0, units;            // units [unit0, unit0 + units)
};

struct ForwardCtx
{
    const ExpertLayer* layer;
    Nvfp4Quant::Params params;
    const uint8_t* x;  // fp16 [rows][hidden], possibly unaligned
    float* out;        // fp32 [rows][hidden]
    int rows;
    bool accumulate;
    RouteTable routes;  // each token's live routes in routing order
    // From the calling thread's ForwardArena; the plan binds everything below.
    RouteBinding* binding;   // [rows][routes.k]
    const Chunk* chunk;      // [chunks]
    int chunks;
    const int* unit_token;   // [units], the token each unit reads
    int units;
    float* xf;                  // [rows][rounded(hidden, 64)]
    block_q8_0* qx;             // [rows][rounded(hidden, 64) / 32]
    float* inter;               // [units][rounded(intermediate, 64)]
    block_q8_0* qi;             // [units][rounded(intermediate, 64) / 32]
    float* partial;             // [workers][partial_stride]: one down output of every unit
    size_t partial_stride;
    // Q8_0 cannot represent the input or an intermediate: the forward returns 2 and leaves out untouched.
    std::atomic<bool> invalid{false};
};

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
    std::vector<RouteBinding> binding;
    std::vector<int> unit_token;
    std::vector<RouteRef> refs;
    std::vector<Chunk> chunk;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

// Output rows per split unit: 16 fp32 outputs fill one cache line. At MiMo V2.6 Pro's shape it splits 2048 gate/up
// rows into 128 units and 6144 down rows into 384, both even over 16 workers. That shape's plan gains from the Shape's
// constants alone: every loop bound, sf_index group count and row stride is a compile-time value inside the AVX2
// entries (gate_up_avx2, down_avx2, which inline the dot product), and the SwiGLU clamp compiles away.
constexpr int kRowUnit = 16;

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
    // Runs the call's routes through layer l on c.threads workers (the caller is worker 0). Returns 0, or 2 when Q8_0
    // cannot represent an input or an intermediate (out is then untouched). Throws when the team is short or a worker
    // cannot be pinned.
    static int run(const ExpertLayer& l, const Nvfp4Quant::Params& p, const ForwardCall& c, const RouteTable& r)
    {
        ForwardArena& ar = ForwardArena::get();
        ForwardCtx ctx;
        ctx.layer = &l;
        ctx.params = p;
        ctx.x = static_cast<const uint8_t*>(c.x);
        ctx.out = c.out;
        ctx.rows = c.rows;
        ctx.accumulate = c.accumulate != 0;
        ctx.routes = r;
        bind_routes(ctx, l, ar);
        prepare_scratch(ctx, ar, c.threads);
        run_team(c.threads, [&ctx](int worker, int n) {
            step<Phase::PrepareInput>(ctx, worker, n);
            // `invalid` is read only after a barrier, so every worker takes the same branch.
            if (!ctx.invalid.load(std::memory_order_relaxed)) {
                step<Phase::GateUp>(ctx, worker, n);
                step<Phase::Middle>(ctx, worker, n);
                if (!ctx.invalid.load(std::memory_order_relaxed)) phase<Phase::Down>(ctx, worker, n);
            }
        });
        return ctx.invalid.load(std::memory_order_relaxed) ? 2 : 0;
    }

private:
    // Groups the live routes into units and chunks, and binds each chunk's projections and scaled alphas and each
    // route's unit and down alpha; the team reads only these.
    static void bind_routes(ForwardCtx& c, const ExpertLayer& l, ForwardArena& ar)
    {
        const RouteTable& r = c.routes;
        if (ar.binding.size() < size_t(r.rows) * r.k) ar.binding.resize(size_t(r.rows) * r.k);
        ar.refs.clear();
        for (int t = 0; t < r.rows; ++t)
            for (int j = 0; j < r.count[t]; ++j) ar.refs.push_back({r.route(t, j).slot, t, j});
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
                    const Nvfp4Quant::Expert e = Nvfp4Quant::expert(l[ref.slot]);
                    Chunk ch;
                    ch.gate = e.gate;
                    ch.up = e.up;
                    ch.down = e.down;
                    ch.gate_alpha = e.gate.alpha * c.params.inv_input_scale13;
                    ch.up_alpha = e.up.alpha * c.params.inv_input_scale13;
                    ch.unit0 = int(ar.unit_token.size());
                    ch.units = 0;
                    ar.chunk.push_back(ch);
                }
                ++ar.chunk.back().units;
                ar.unit_token.push_back(ref.token);
            }
            RouteBinding& b = ar.binding[size_t(ref.token) * r.k + ref.index];
            b.unit = int(ar.unit_token.size()) - 1;
            b.down_alpha = ar.chunk.back().down.alpha * c.params.inv_input_scale2 * r.route(ref.token, ref.index).weight;
        }
        c.binding = ar.binding.data();
        c.chunk = ar.chunk.data();
        c.chunks = int(ar.chunk.size());
        c.unit_token = ar.unit_token.data();
        c.units = int(ar.unit_token.size());
    }

    // Sizes this call's scratch from the arena. The padded tails are zeroed on every call: the arena is shared by
    // every layer this thread forwards, and Q8_0 blocks and the dot product read whole 64-value blocks.
    static void prepare_scratch(ForwardCtx& c, ForwardArena& ar, int threads)
    {
        const size_t H = size_t(Shape::hidden(*c.layer)), N = size_t(Shape::intermediate(*c.layer));
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

    // Projection p's weight row `row` (k columns) against a chunk's m Q8_0 vectors xs, at tier I.
    static void dot_chunk(const Projection& p, int row, int k, const block_q8_0* const* xs, int m, float* out)
    {
        static_assert(kChunkRows == 4, "dot_chunk dispatches m in [1, 4]");
        const int n = int(rounded(k, 64));
        const GpuRow x(p.w, p.sf, row, k);
        switch (m) {
            case 1: dot_rows<I, 1>(n, x, xs, out); break;
            case 2: dot_rows<I, 2>(n, x, xs, out); break;
            case 3: dot_rows<I, 3>(n, x, xs, out); break;
            default: dot_rows<I, 4>(n, x, xs, out); break;
        }
    }

    // Gate/up rows [r0, r1) of the flat (chunk, output) range: each chunk's gate and up dot products, gated SiLU into
    // each unit's intermediate.
    static inline void gate_up_rows(ForwardCtx& c, int64_t r0, int64_t r1)
    {
        const int H = Shape::hidden(*c.layer), N = Shape::intermediate(*c.layer);
        const size_t Hp = rounded(size_t(H), 64), Np = rounded(size_t(N), 64);
        for (int64_t row = r0; row < r1; ++row) {
            const Chunk& ch = c.chunk[row / N];
            const int i = int(row % N);
            int gate_row, up_row;
            w13_rows(c.params.w13_layout, N, i, gate_row, up_row);
            const block_q8_0* xs[kChunkRows];
            for (int j = 0; j < ch.units; ++j) xs[j] = c.qx + size_t(c.unit_token[ch.unit0 + j]) * (Hp / 32);
            float g[kChunkRows], u[kChunkRows];
            dot_chunk(ch.gate, gate_row, H, xs, ch.units, g);
            dot_chunk(ch.up, up_row, H, xs, ch.units, u);
            for (int j = 0; j < ch.units; ++j)
                c.inter[size_t(ch.unit0 + j) * Np + size_t(i)] =
                    swiglu(g[j] * ch.gate_alpha, u[j] * ch.up_alpha, Shape::act_limit(*c.layer));
        }
    }

    // Down rows [h0, h1): every chunk's down dot products into this worker's partial, then each token's
    // routing-weighted sum, in routing order, into out.
    static inline void down_rows(ForwardCtx& c, float* partial, int64_t h0, int64_t h1)
    {
        const int H = Shape::hidden(*c.layer), N = Shape::intermediate(*c.layer);
        const size_t Np = rounded(size_t(N), 64);
        for (int64_t h = h0; h < h1; ++h) {
            for (int ci = 0; ci < c.chunks; ++ci) {
                const Chunk& ch = c.chunk[ci];
                const block_q8_0* xs[kChunkRows];
                for (int j = 0; j < ch.units; ++j) xs[j] = c.qi + size_t(ch.unit0 + j) * (Np / 32);
                dot_chunk(ch.down, int(h), N, xs, ch.units, partial + ch.unit0);
            }
            for (int t = 0; t < c.rows; ++t) {
                const RouteBinding* b = c.binding + size_t(t) * c.routes.k;
                float sum = 0.f;
                for (int r = 0; r < c.routes.count[t]; ++r) sum += partial[b[r].unit] * b[r].down_alpha;
                float& out = c.out[size_t(t) * size_t(H) + size_t(h)];
                out = c.accumulate ? out + sum : sum;
            }
        }
    }

    // The AVX2 tier's entries: one call per phase per worker, compiled for AVX2 with everything below inlined
    // (flatten), so the dot product, GpuRow, swiglu and the down sum run under one target and the Shape's constants fold
    // into them. Instantiated only by the Avx2 plans. -ffp-contract=off holds here as everywhere in the library.
    SGLANG_TARGET_AVX2 __attribute__((flatten)) static void gate_up_avx2(ForwardCtx& c, int64_t r0, int64_t r1)
    {
        gate_up_rows(c, r0, r1);
    }

    SGLANG_TARGET_AVX2 __attribute__((flatten)) static void down_avx2(ForwardCtx& c, float* partial, int64_t h0,
                                                                       int64_t h1)
    {
        down_rows(c, partial, h0, h1);
    }

    // One phase, then wait for the whole team. Called inside run_team's parallel region (team.hpp; an orphaned barrier
    // binds to that team).
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
        const int H = Shape::hidden(*c.layer), N = Shape::intermediate(*c.layer);
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
            const auto [r0, r1] = share<kRowUnit>(int64_t(c.chunks) * N, worker, workers);
            if constexpr (I == Isa::Avx2) gate_up_avx2(c, r0, r1);
            else gate_up_rows(c, r0, r1);
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
            const auto [h0, h1] = share<kRowUnit>(H, worker, workers);
            if constexpr (I == Isa::Avx2) down_avx2(c, partial, h0, h1);
            else down_rows(c, partial, h0, h1);
        }
    }
};
