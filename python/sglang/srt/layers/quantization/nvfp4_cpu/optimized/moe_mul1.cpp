// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ForwardPlan<Shape, Isa>::run (forward_plan.hpp) on one OpenMP team per call; layers are read through
// LayerInfo and StridedExperts (experts.hpp) under a Shape (shapes.hpp). Types and declarations: moe_mul1.h.
#include "moe_mul1.h"

namespace {

// -------------------------------------------------------------------------------------------
//   Layer registry
// -------------------------------------------------------------------------------------------

LayerInfo info_of(const SglangNvfp4CpuLayer& d)
{
    return {d.capacity, d.hidden, d.intermediate, d.w13_layout, d.act_limit, d.inv_input_scale13,
            d.inv_input_scale2, d.slabs[kUpAlpha] != nullptr};
}

StridedExperts<GenericShape> strided_of(const SglangNvfp4CpuLayer& d)
{
    StridedExperts<GenericShape> e{};
    for (int i = 0; i < kSlabNames; ++i) {
        e.base[i] = static_cast<const uint8_t*>(d.slabs[i]);
        e.stride[i] = d.slot_bytes[i];
    }
    return e;
}

std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<const RegisteredLayer>> layers;
int64_t next_handle = 1;
std::shared_ptr<const RegisteredLayer> lookup(int64_t h) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = layers.find(h); return it == layers.end() ? nullptr : it->second;
}

// One forward, core configuration or free at a time: the others return 3 rather than race the forward's scratch.
std::mutex forward_mutex;

// -------------------------------------------------------------------------------------------
//   Worker cores
// -------------------------------------------------------------------------------------------

// Worker cores set by sglang_nvfp4_cpu_experts_set_cores; copied into g_compute_cores at the first forward.
std::mutex g_cores_mutex;
std::vector<int> g_configured_cores;
std::atomic<bool> g_compute_started{false};
std::vector<int> g_compute_cores;  // Immutable after release publication at first forward.

inline void freeze_compute_cores()
{
    if (g_compute_started.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_cores_mutex);
    if (!g_compute_started.load(std::memory_order_relaxed)) {
        g_compute_cores = g_configured_cores;
        g_compute_started.store(true, std::memory_order_release);
    }
}

inline void pin_compute_worker(int worker, std::atomic<int>& pin_error)
{
    if (g_compute_cores.empty()) return;
    const int core = g_compute_cores[worker];
    static thread_local int pinned_core = -1;
    if (pinned_core != core || sched_getcpu() != core) {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
            pin_error.store(1, std::memory_order_relaxed);
        else pinned_core = core;
    }
}

// -------------------------------------------------------------------------------------------
//   Arithmetic (unchanged from the pre-OpenMP kernel)
// -------------------------------------------------------------------------------------------

template <int M>
void dot_rows_of(const uint8_t* w, const uint8_t* sf, int row, int k, const block_q8_0* const* xs,
                 block_nvfp4* scratch, float* out) {
    GpuRow view(w,sf,row,k);
    const int padded=int(rounded(k,64));
#if defined(NVFP4_CPU_UPSTREAM_BASELINE)
    // Convert into worker-local scratch on every row; the source slabs may mutate.
    for (int ib=0;ib<padded/64;++ib) {
        auto& block=scratch[ib];
        for (int g=0;g<4;++g) {
            int group=ib*4+g;
            uint8_t scale=group<k/16?sf[sf_index(row,group,k/16)]:0;
            block.d[g]=scale&127;
            const auto q=view.bytes(ib)+g*8;
            const uint8_t flip=scale&128?8:0;
            for (int j=0;j<8;++j) {
                // GGML puts columns j and j+8 in a byte; GPU puts 2j and 2j+1.
                uint8_t lo=(q[j/2]>>(4*(j%2)))&15;
                uint8_t hi=(q[(j+8)/2]>>(4*((j+8)%2)))&15;
                block.qs[g*8+j]=(lo^flip)|((hi^flip)<<4);
            }
        }
    }
    for (int t=0;t<M;++t) ggml_vec_dot_nvfp4_q8_0(padded,&out[t],0,scratch,0,xs[t],0,1);
#else
    (void)scratch;
    dot_gpu_rows<M>(padded,view,xs,out);
#endif
}

void dot_rows(const uint8_t* w, const uint8_t* sf, int row, int k, const block_q8_0* const* xs, int m,
              block_nvfp4* scratch, float* out) {
    static_assert(kChunkRows == 4, "dot_rows dispatches m in [1, 4]");
    switch (m) {
        case 1: dot_rows_of<1>(w, sf, row, k, xs, scratch, out); break;
        case 2: dot_rows_of<2>(w, sf, row, k, xs, scratch, out); break;
        case 3: dot_rows_of<3>(w, sf, row, k, xs, scratch, out); break;
        default: dot_rows_of<4>(w, sf, row, k, xs, scratch, out); break;
    }
}

bool q8_representable(const float* v)
{
    for (int j = 0; j < 32; ++j)
        if (!std::isfinite(v[j]) || std::abs(v[j]) > 65504.f * 127.f) return false;
    return true;
}

// A zero delta also avoids overflowing the reciprocal for a subnormal FP32 amax in the upstream reference quantizer.
void quantize_block(const float* v, block_q8_0& out)
{
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(v[j]));
    if (!ggml_compute_fp32_to_fp16(amax / 127.f)) out = block_q8_0{};
    else quantize_row_q8_0_ref(v, &out, 32);
}

inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}

ForwardArena& ForwardArena::get()
{
    static thread_local ForwardArena arena;
    return arena;
}

#include "forward_plan.hpp"

// Both plans read the same slabs; the MiMo plan through the view checked for it. A template so a scalar build discards,
// and never instantiates, the AVX2 plan.
template <Isa I>
int run_plan(ForwardCtx& ctx, const RegisteredLayer& layer, int threads)
{
    if constexpr (I == Isa::Avx2) {
        if (MimoV26ProShape::accepts(ctx.info))
            return ForwardPlan<MimoV26ProShape, Isa::Avx2>::run(ctx, layer.strided.as<MimoV26ProShape>(),
                                                                ForwardArena::get(), threads);
    }
    return ForwardPlan<GenericShape, I>::run(ctx, layer.strided, ForwardArena::get(), threads);
}

bool valid(const SglangNvfp4CpuLayer& d) {
    if (d.abi_version != 1 || d.capacity < 1 || d.hidden < 16 || d.intermediate < 16
        || d.hidden > (1 << 20) || d.intermediate > (1 << 20)
        || d.hidden % 16 || d.intermediate % 16 || d.w13_layout < 0 || d.w13_layout > 2
        || (d.w13_layout == 2 && d.intermediate % 64) || d.activation != 0
        || !std::isfinite(d.act_limit) || d.act_limit < 0
        || !std::isfinite(d.inv_input_scale13) || d.inv_input_scale13 <= 0
        || !std::isfinite(d.inv_input_scale2) || d.inv_input_scale2 <= 0) return false;
    const SlabRowBytes minimum = SlabRowBytes::of(d.hidden, d.intermediate);
    for (int i = 0; i < kSlabNames; ++i) {
        if (i == kUpAlpha && !d.slabs[i]) continue;
        if (!d.slabs[i] || d.slot_bytes[i] < minimum.bytes[i]
            || d.slot_bytes[i] > SIZE_MAX / uint64_t(d.capacity)) return false;
    }
    return true;
}
} // namespace

extern "C" int sglang_nvfp4_cpu_experts_register_slabs(const SglangNvfp4CpuLayer* d, int64_t* h) noexcept {
    if (!d || !h || !valid(*d)) return 2;
    try {
        auto l = std::make_shared<const RegisteredLayer>(RegisteredLayer{info_of(*d), strided_of(*d)});
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (next_handle == INT64_MAX) return 1;
        const auto id = next_handle++; layers.emplace(id, std::move(l)); *h = id; return 0;
    } catch (...) { return 1; }
}

extern "C" int sglang_nvfp4_cpu_experts_free_layer(int64_t h) noexcept {
    try {
        std::unique_lock<std::mutex> forward_lock(forward_mutex, std::try_to_lock);
        if (!forward_lock.owns_lock()) return 3;
        std::lock_guard<std::mutex> lock(registry_mutex); return layers.erase(h) ? 0 : 2;
    }
    catch (...) { return 1; }
}

// Worker i on cores[i], the caller as worker 0. Cores must be distinct and allowed by this thread's affinity; refused
// (2) once the first forward has frozen them.
extern "C" int sglang_nvfp4_cpu_experts_set_cores(const int32_t* c, int32_t n) noexcept {
    try {
        std::unique_lock<std::mutex> forward_lock(forward_mutex, std::try_to_lock);
        if (!forward_lock.owns_lock()) return 3;
        if (!c || n < 1 || n > 4096) return 2;
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed)) return 2;
        for (int i = 0; i < n; ++i) {
            if (c[i] < 0 || c[i] >= CPU_SETSIZE || !CPU_ISSET(c[i], &allowed)) return 2;
            for (int j = 0; j < i; ++j) if (c[j] == c[i]) return 2;
        }
        std::lock_guard<std::mutex> lock(g_cores_mutex);
        if (g_compute_started.load(std::memory_order_relaxed)) return 2;
        g_configured_cores.assign(c, c + n);
        return 0;
    } catch (...) { return 1; }
}

extern "C" int sglang_nvfp4_cpu_experts_forward(const SglangCpuExpertsForward* call) noexcept {
    try {
        if (!call || call->abi_version != SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION) return 2;
        const SglangCpuExpertsForward& c = *call;
        const int rows = c.rows, k = c.k;
        if (!c.x || !c.out || rows < 1 || rows > kMaxRows || k < 0 || k > kMaxRoutes || c.threads < 1
            || c.threads > 4096 || (k && (!c.slots || !c.weights)) || (c.accumulate != 0 && c.accumulate != 1))
            return 2;
        std::unique_lock<std::mutex> lock(forward_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return 3;
        const auto l = lookup(c.layer); if (!l) return 2;
        const auto& E = l->strided;
        const size_t n = size_t(rows) * size_t(k);
        for (size_t j = 0; j < n; ++j) {
            const int slot = c.slots[j];
            if (slot < -1 || slot >= l->info.capacity || !std::isfinite(c.weights[j])) return 2;
            if (slot >= 0 && (!std::isfinite(E.alpha(kGateAlpha, slot))
                || !std::isfinite(E.alpha(kDownAlpha, slot))
                || (l->info.up_alpha && !std::isfinite(E.alpha(kUpAlpha, slot))))) return 2;
        }
        ForwardArena& ar = ForwardArena::get();
        auto grow = [](auto& v, size_t size) { if (v.size() < size) v.resize(size); };
        grow(ar.route, size_t(rows) * kMaxRoutes);
        grow(ar.route_count, size_t(rows));
        for (int t = 0; t < rows; ++t) {
            int live = 0;
            for (int j = 0; j < k; ++j) {
                const int slot = c.slots[size_t(t) * k + j];
                const float weight = c.weights[size_t(t) * k + j];
                if (slot == -1 || weight == 0) continue;
                ar.route[size_t(t) * kMaxRoutes + live++] = {slot, weight, -1, 0.f};
            }
            ar.route_count[t] = live;
        }
        ForwardCtx ctx;
        ctx.info = l->info;
        ctx.x = static_cast<const uint8_t*>(c.x);
        ctx.out = c.out;
        ctx.rows = rows;
        ctx.accumulate = c.accumulate != 0;
        ctx.route = ar.route.data();
        ctx.route_count = ar.route_count.data();
        return run_plan(ctx, *l, c.threads);
    } catch (...) { return 1; }
}
