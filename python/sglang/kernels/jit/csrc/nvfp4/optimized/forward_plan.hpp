// Included by kernel.cpp inside quant.hpp's namespace, after the arithmetic (math.hpp: swiglu, q8_representable,
// quantize_block; dot_rows<I, M> from math_scalar.hpp and math_avx2.hpp) and kChunkRows.
//
// One job = ForwardPlan<Shape, I>::prepare at admission, then the four phases (Phase) as tasks that any worker runs
// (tasks, run). Shape (shapes.hpp) fixes what the plan may assume about the layer; I is the dot product's tier, the
// build's (kBuildIsa, at most Avx2).
//
// A call's routes are grouped by slot into units, one per (token, slot), and units into chunks of up to kChunkRows of
// one slot, so dot_rows decodes each weight row once per chunk. Each token's output is still its own routes' sum in
// routing order, so a token's result does not depend on the other rows of the call.

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
    // From the job's state; prepare binds everything below.
    RouteBinding* binding;   // [rows][routes.k]
    const Chunk* chunk;      // [chunks]
    int chunks;
    const int* unit_token;   // [units], the token each unit reads
    int units;
    // In the job's scratch; inter's rows start cache lines, so each kRowUnit share of a row owns whole lines.
    block_q8_0* qx;             // [rows][rounded(hidden, 64) / 32]
    float* inter;               // [units][rounded(intermediate, 64)]
    block_q8_0* qi;             // [units][rounded(intermediate, 64) / 32]
    // Q8_0 cannot represent the input or an intermediate: the job reports status 2 and leaves out untouched.
    std::atomic<bool> invalid{false};
};

// The job's phases, in order; a phase's tasks run once the previous phase's have all finished.
enum class Phase : int
{
    PrepareInput = 0,  // task: a 32-value block of a token's fp16 input to fp32, then its Q8_0 block
    GateUp = 1,        // task: kRowUnit rows of the flat (chunk, output) range; gated SiLU into each unit's intermediate
    Middle = 2,        // task: a Q8_0 block of a unit's intermediate
    Down = 3,          // task: kRowUnit output rows; every chunk's down rows, each token's routing-weighted sum into out
};
constexpr int kPhaseCount = 4;
// The first phase that reads the second stage's slabs (w2, sf2); the down alpha is read by prepare, outside it.
constexpr int kGatePhase = int(Phase::Down);

// A job's state: the context and the scratch it points into, kept across jobs so a steady state allocates nothing.
struct ForwardJob
{
    ForwardCtx ctx;
    RouteStore routes;
    std::vector<RouteBinding> binding;
    std::vector<int> unit_token;
    std::vector<Chunk> chunk;
    Scratch scratch;    // qx, inter, qi (prepare_scratch)
    bool mimo = false;  // the MiMo V2.6 Pro plan runs this job (chosen at prepare)
};

// Output rows per task: 16 fp32 outputs fill one cache line. At MiMo V2.6 Pro's shape it splits 2048 gate/up
// rows into 128 tasks and 6144 down rows into 384, both even over 16 workers. That shape's plan gains from the Shape's
// constants alone: every loop bound, sf_index group count and row stride is a compile-time value inside the AVX2
// entries (gate_up_avx2, down_avx2, which inline the dot product), and the SwiGLU clamp compiles away.
constexpr int kRowUnit = 16;

template <class Shape, Isa I>
struct ForwardPlan
{
    // Binds the call into `job`: its routes grouped into chunks, its scratch sized. Throws std::invalid_argument for a
    // call with more tasks in a phase than an int counts.
    static void prepare(ForwardJob& job, const ExpertLayer& l, const ForwardCall& call)
    {
        ForwardCtx& c = job.ctx;
        c.layer = &l;
        c.params = l.params;
        c.x = static_cast<const uint8_t*>(call.x);
        c.out = call.out;
        c.rows = call.rows;
        c.accumulate = call.accumulate();
        c.routes = RouteTable::build(call.slots, call.weights, call.rows, call.k, job.routes);
        c.invalid.store(false, std::memory_order_relaxed);
        bind_routes(c, l, job);
        prepare_scratch(c, job);
        const int64_t H = Shape::hidden(l), N = Shape::intermediate(l);
        const int64_t counts[] = {int64_t(c.rows) * int64_t(rounded(size_t(H), 64) / 32),
                                  int64_t(c.chunks) * N / kRowUnit,
                                  int64_t(c.units) * int64_t(rounded(size_t(N), 64) / 32)};
        for (const int64_t n : counts)
            if (n > INT32_MAX) throw std::invalid_argument("NVFP4 CPU experts: the call has too many tasks");
    }

    // Phase `phase`'s tasks. Read when the phase becomes ready, so after `invalid` is set the phases that follow have
    // none.
    static Tasks tasks(const ForwardCtx& c, int phase)
    {
        const int H = Shape::hidden(*c.layer), N = Shape::intermediate(*c.layer);
        const size_t Hp = rounded(size_t(H), 64), Np = rounded(size_t(N), 64);
        const bool invalid = c.invalid.load(std::memory_order_relaxed);
        switch (Phase(phase)) {
            case Phase::PrepareInput: return {int(int64_t(c.rows) * int64_t(Hp / 32)), 1};
            case Phase::GateUp: return {invalid ? 0 : int((int64_t(c.chunks) * N + kRowUnit - 1) / kRowUnit), 1};
            case Phase::Middle: return {invalid ? 0 : int(int64_t(c.units) * int64_t(Np / 32)), 1};
            case Phase::Down: return {invalid ? 0 : (H + kRowUnit - 1) / kRowUnit, 1};
        }
        return {};
    }

    // Tasks [t0, t1) of phase `phase`. Which worker runs a task changes none of its arithmetic: a Down task owns whole
    // output rows and sums each token's routes in routing order inside the task.
    static void run(ForwardCtx& c, int phase, int t0, int t1)
{
        const int H = Shape::hidden(*c.layer), N = Shape::intermediate(*c.layer);
        const size_t Hp = rounded(size_t(H), 64);

        switch (Phase(phase)) {
            case Phase::PrepareInput: {
                // Token t's blocks are [t * Hp / 32, (t + 1) * Hp / 32) of the flat range. Each is staged in fp32 on the
                // task's stack, its values past H zero.
                const int64_t per_row = int64_t(Hp / 32);
                for (int64_t gb = t0; gb < t1; ++gb) {
                    const int64_t t = gb / per_row, b = gb % per_row;
                    const uint8_t* x = c.x + size_t(t) * size_t(H) * 2;
                    alignas(64) float xf[32] = {};
                    for (int64_t i = b * 32; i < std::min<int64_t>(b * 32 + 32, H); ++i) {
                        uint16_t v;
                        std::memcpy(&v, x + 2 * i, 2);
                        xf[i - b * 32] = ggml_compute_fp16_to_fp32(v);
                    }
                    if (!q8_representable(xf)) c.invalid.store(true, std::memory_order_relaxed);
                    else quantize_block(xf, c.qx[gb]);
                }
                break;
            }
            case Phase::GateUp: {
                const int64_t r0 = int64_t(t0) * kRowUnit, r1 = std::min(int64_t(c.chunks) * N, int64_t(t1) * kRowUnit);
                if constexpr (I == Isa::Avx2) gate_up_avx2(c, r0, r1);
                else gate_up_rows(c, r0, r1);
                break;
            }
            case Phase::Middle: {
                // Unit u's intermediate is Np floats at u * Np, so block b of the flat range is unit b / (Np / 32)'s.
                for (int64_t b = t0; b < t1; ++b) {
                    const float* v = c.inter + b * 32;
                    if (!q8_representable(v)) c.invalid.store(true, std::memory_order_relaxed);
                    else quantize_block(v, c.qi[b]);
                }
                break;
            }
            case Phase::Down: {
                // One down output of every unit, the task's own: on its stack, unless the call has more units than fit.
                constexpr int kStackUnits = 256;
                alignas(64) float stack[kStackUnits];
                std::vector<float> heap;
                float* partial = stack;
                if (c.units > kStackUnits) {
                    heap.resize(size_t(c.units));
                    partial = heap.data();
                }
                const int64_t h0 = int64_t(t0) * kRowUnit, h1 = std::min<int64_t>(H, int64_t(t1) * kRowUnit);
                if constexpr (I == Isa::Avx2) down_avx2(c, partial, h0, h1);
                else down_rows(c, partial, h0, h1);
                break;
            }
        }
    }

private:
    // Groups the live routes into units and chunks, and binds each chunk's projections and scaled alphas and each
    // route's unit and down alpha; the tasks read only these.
    static void bind_routes(ForwardCtx& c, const ExpertLayer& l, ForwardJob& job)
{
        const RouteTable& r = c.routes;
        if (job.binding.size() < size_t(r.rows) * r.k) job.binding.resize(size_t(r.rows) * r.k);
        std::vector<RouteRef> refs;
        refs.reserve(size_t(r.rows) * r.k);
        for (int t = 0; t < r.rows; ++t)
            for (int j = 0; j < r.count[t]; ++j) refs.push_back({r.route(t, j).slot, t, j});
        std::sort(refs.begin(), refs.end(), [](const RouteRef& a, const RouteRef& b) {
            return a.slot != b.slot ? a.slot < b.slot : a.token != b.token ? a.token < b.token : a.index < b.index;
        });
        job.chunk.clear();
        job.unit_token.clear();
        for (size_t i = 0; i < refs.size(); ++i) {
            const RouteRef& ref = refs[i];
            const bool same_slot = i && refs[i - 1].slot == ref.slot;
            if (!same_slot || refs[i - 1].token != ref.token) {
                if (!same_slot || job.chunk.back().units == kChunkRows) {
                    const Nvfp4Quant::Expert e = Nvfp4Quant::expert(l[ref.slot]);
                    Chunk ch;
                    ch.gate = e.gate;
                    ch.up = e.up;
                    ch.down = e.down;
                    ch.gate_alpha = e.gate.alpha * c.params.inv_input_scale13;
                    ch.up_alpha = e.up.alpha * c.params.inv_input_scale13;
                    ch.unit0 = int(job.unit_token.size());
                    ch.units = 0;
                    job.chunk.push_back(ch);
                }
                ++job.chunk.back().units;
                job.unit_token.push_back(ref.token);
            }
            RouteBinding& b = job.binding[size_t(ref.token) * r.k + ref.index];
            b.unit = int(job.unit_token.size()) - 1;
            b.down_alpha = job.chunk.back().down.alpha * c.params.inv_input_scale2 * r.route(ref.token, ref.index).weight;
        }
        c.binding = job.binding.data();
        c.chunk = job.chunk.data();
        c.chunks = int(job.chunk.size());
        c.unit_token = job.unit_token.data();
        c.units = int(job.unit_token.size());
    }

    // Lays out this job's scratch. The intermediates' padded tails are zeroed on every job: the scratch is reused by
    // every layer a slot runs, and Q8_0 blocks and the dot product read whole 64-value blocks.
    static void prepare_scratch(ForwardCtx& c, ForwardJob& job)
    {
        const size_t N = size_t(Shape::intermediate(*c.layer));
        const size_t Hp = rounded(size_t(Shape::hidden(*c.layer)), 64), Np = rounded(N, 64);
        const size_t rows = size_t(c.rows), units = size_t(c.units);
        ScratchLayout layout;
        const size_t qx = layout.add<block_q8_0>(rows * Hp / 32);
        const size_t inter = layout.add<float>(units * Np);
        const size_t qi = layout.add<block_q8_0>(units * Np / 32);
        job.scratch.fit(layout);
        c.qx = job.scratch.at<block_q8_0>(qx);
        c.inter = job.scratch.at<float>(inter);
        c.qi = job.scratch.at<block_q8_0>(qi);
        for (size_t u = 0; u < units; ++u) std::fill(c.inter + u * Np + N, c.inter + (u + 1) * Np, 0.f);
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

    // Down rows [h0, h1): every chunk's down dot products into `partial`, then each token's routing-weighted sum, in
    // routing order, into out.
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

    // The AVX2 tier's entries: one call per run of tasks, with everything below inlined (flatten), so the dot
    // product, GpuRow, swiglu and the down sum are one function and the Shape's constants fold into them.
    __attribute__((flatten)) static void gate_up_avx2(ForwardCtx& c, int64_t r0, int64_t r1)
    {
        gate_up_rows(c, r0, r1);
    }

    __attribute__((flatten)) static void down_avx2(ForwardCtx& c, float* partial, int64_t h0, int64_t h1)
    {
        down_rows(c, partial, h0, h1);
    }
};
