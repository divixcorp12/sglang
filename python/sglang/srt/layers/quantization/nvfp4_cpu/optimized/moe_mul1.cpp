// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ForwardPlan<Shape, Isa>::run (forward_plan.hpp) on one OpenMP team per call; layers are read through
// LayerInfo and StridedExperts (experts.hpp) under a Shape (shapes.hpp).
#if !defined(__linux__) || !defined(_OPENMP)
#error The NVFP4 CPU expert kernel requires Linux and OpenMP.
#endif
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__AVX2__) && !defined(NVFP4_CPU_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace {
#include "dot_nvfp4.h"

// The dot product's tier is fixed when the library is compiled (dot_nvfp4.h's #if chain): AVX2 under -march=native
// on an AVX2 host, else the scalar loop.
enum class Isa { Scalar, Avx2 };
#if defined(__AVX2__)
constexpr Isa kBuildIsa = Isa::Avx2;
#else
constexpr Isa kBuildIsa = Isa::Scalar;
#endif

#include "experts.hpp"
#include "shapes.hpp"

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

struct RegisteredLayer
{
    LayerInfo info;
    StridedExperts<GenericShape> strided;
};
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

// Freezes the configured cores into g_compute_cores once, at the first forward. Steady-state calls acquire no mutex.
inline void freeze_compute_cores()
{
    if (g_compute_started.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(g_cores_mutex);
    if (!g_compute_started.load(std::memory_order_relaxed)) {
        g_compute_cores = g_configured_cores;
        g_compute_started.store(true, std::memory_order_release);
    }
}

// Inside a parallel region: pins OpenMP worker `worker` to its compute core (none configured: no-op), setting
// pin_error if it cannot.
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

float dot(const uint8_t* w, const uint8_t* sf, int row, int k,
          const block_q8_0* x, block_nvfp4* scratch) {
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
    float result;
    ggml_vec_dot_nvfp4_q8_0(padded,&result,0,scratch,0,x,0,1);
    return result;
#else
    (void)scratch;
    return dot_gpu(padded,view,x);
#endif
}

// Whether Q8_0 represents the 32 values of a block: finite, with a delta within FP16 range.
bool q8_representable(const float* v)
{
    for (int j = 0; j < 32; ++j)
        if (!std::isfinite(v[j]) || std::abs(v[j]) > 65504.f * 127.f) return false;
    return true;
}

// One Q8_0 block. A zero FP16 delta contributes zero; this also avoids overflowing the reciprocal for a subnormal
// FP32 amax in the upstream reference quantizer.
void quantize_block(const float* v, block_q8_0& out)
{
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(v[j]));
    if (!ggml_compute_fp32_to_fp16(amax / 127.f)) out = block_q8_0{};
    else quantize_row_q8_0_ref(v, &out, 32);
}

// Gate/up output of the gated SiLU: optional pre-SiLU clamp (gate from above, up both ways), stable SiLU including
// large negative gates, times up.
inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}

// -------------------------------------------------------------------------------------------
//   Forward context and scratch
// -------------------------------------------------------------------------------------------

constexpr int kMaxRoutes = 8;  // the C ABI's k limit

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

struct Route
{
    int slot;
    float weight;
};

struct ForwardCtx
{
    LayerInfo info;
    const uint8_t* x;  // fp16 [hidden], possibly unaligned
    float* out;        // fp32 [hidden]
    bool accumulate;
    Route route[kMaxRoutes];  // live routes (no -1 slot, no zero weight), in routing order
    int routes = 0;
    // Bound by the plan from the layer's experts, per route.
    Projection gate[kMaxRoutes], up[kMaxRoutes], down[kMaxRoutes];
    float gate_alpha[kMaxRoutes], up_alpha[kMaxRoutes], down_alpha[kMaxRoutes];
    // Scratch from the calling thread's ForwardArena.
    float* xf;                  // [rounded(hidden, 64)]
    block_q8_0* qx;             // [rounded(hidden, 64) / 32]
    float* inter;               // [routes][rounded(intermediate, 64)]
    block_q8_0* qi;             // [routes][rounded(intermediate, 64) / 32]
    block_nvfp4* row_scratch;   // [workers][row_scratch_stride], upstream-baseline builds only
    size_t row_scratch_stride;
    // Q8_0 cannot represent the input or an intermediate: the forward returns 2 and leaves out untouched.
    std::atomic<bool> invalid{false};
};

struct ForwardArena
{
    std::vector<float, CacheAligned<float>> xf, inter;
    std::vector<block_q8_0> qx, qi;
    std::vector<block_nvfp4> row_scratch;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

#include "forward_plan.hpp"

// Runs the call's plan.
int run_plan(ForwardCtx& ctx, const RegisteredLayer& layer, int threads)
{
    return ForwardPlan<GenericShape, kBuildIsa>::run(ctx, layer.strided, ForwardArena::get(), threads);
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

extern "C" int sglang_nvfp4_cpu_experts_forward(int64_t h, const void* x,
    const int32_t* slots, const float* weights, int32_t k, float* out, int32_t threads, int32_t accumulate) noexcept {
    try {
        if (!x || !out || k < 0 || k > kMaxRoutes || threads < 1 || threads > 4096
            || (k && (!slots || !weights)) || (accumulate != 0 && accumulate != 1)) return 2;
        std::unique_lock<std::mutex> lock(forward_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return 3;
        const auto l = lookup(h); if (!l) return 2;
        const auto& E = l->strided;
        for (int j = 0; j < k; ++j) {
            if (slots[j] < -1 || slots[j] >= l->info.capacity || !std::isfinite(weights[j])) return 2;
            if (slots[j] >= 0 && (!std::isfinite(E.alpha(kGateAlpha, slots[j]))
                || !std::isfinite(E.alpha(kDownAlpha, slots[j]))
                || (l->info.up_alpha && !std::isfinite(E.alpha(kUpAlpha, slots[j]))))) return 2;
        }
        ForwardCtx ctx;
        ctx.info = l->info;
        ctx.x = static_cast<const uint8_t*>(x);
        ctx.out = out;
        ctx.accumulate = accumulate != 0;
        for (int j = 0; j < k; ++j) {
            if (slots[j] == -1 || weights[j] == 0) continue;
            ctx.route[ctx.routes++] = {slots[j], weights[j]};
        }
        return run_plan(ctx, *l, threads);
    } catch (...) { return 1; }
}
