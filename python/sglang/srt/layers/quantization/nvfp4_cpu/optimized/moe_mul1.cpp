// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__AVX2__) && !defined(NVFP4_CPU_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace {
#include "dot_nvfp4.h"
#include "experts.hpp"
#include "shapes.hpp"

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

struct Layer {
    LayerInfo info;
    StridedExperts<GenericShape> strided;
    // Activations only. Sized once; mutable slab contents are read each job.
    std::vector<float> x, intermediate, result;
    std::vector<block_q8_0> qx, qi;
    std::vector<std::vector<block_nvfp4>> row_scratch;
    explicit Layer(const SglangNvfp4CpuLayer& d) : info(info_of(d)), strided(strided_of(d)), x(d.hidden),
        intermediate(d.intermediate), result(d.hidden),
        qx(rounded(d.hidden,64)/32), qi(rounded(d.intermediate,64)/32) {
        x.resize(rounded(d.hidden,64)); intermediate.resize(rounded(d.intermediate,64));
    }
};
std::mutex registry_mutex;
std::unordered_map<int64_t, std::shared_ptr<Layer>> layers;
int64_t next_handle = 1;
std::shared_ptr<Layer> lookup(int64_t h) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = layers.find(h); return it == layers.end() ? nullptr : it->second;
}

// Persistent helpers. One engine owns the process-wide pool; overlapping
// forwards are rejected, rather than racing activation scratch or task state.
class Workers {
    std::mutex mutex;
    std::condition_variable wake, done;
    std::vector<std::thread> helpers;
    std::vector<int32_t> cores;
    uint64_t generation = 0;
    int remaining = 0, count = 0, active = 0;
    bool stop = false, pin_failed = false;
    void (*task)(void*, int, int) = nullptr;
    void* context = nullptr;
    bool pin(int rank) {
        if (cores.empty()) return true;
#if defined(__linux__)
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cores[rank], &set);
        return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
        (void)rank;
        return false;
#endif
    }
    void helper(int rank) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!pin(rank)) pin_failed = true;
        --remaining; done.notify_one();
        uint64_t seen = 0;
        while (true) {
            wake.wait(lock, [&] { return stop || generation != seen; });
            if (stop) return;
            seen = generation;
            auto fn = task; auto ctx = context; const int n = active;
            lock.unlock();
            if (rank < n) fn(ctx, rank, n);
            lock.lock(); --remaining; if (!remaining) done.notify_one();
        }
    }
public:
    std::mutex forward_mutex;
    ~Workers() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        wake.notify_all(); for (auto& t : helpers) t.join();
    }
    int configure(const int32_t* c, int n) {
        std::lock_guard<std::mutex> lock(mutex);
        if (count || !c || n < 1 || n > 4096) return 2;
#if defined(__linux__)
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed)) return 2;
        for (int i = 0; i < n; ++i) {
            if (c[i] < 0 || c[i] >= CPU_SETSIZE || !CPU_ISSET(c[i], &allowed)) return 2;
            for (int j = 0; j < i; ++j) if (c[j] == c[i]) return 2;
        }
        cores.assign(c, c + n); return 0;
#else
        return 2; // Standalone unpinned tests work on other hosts.
#endif
    }
    void initialize(int n) {
        std::unique_lock<std::mutex> lock(mutex);
        if (count) {
            if (n > count || pin_failed) throw std::runtime_error("pool size or affinity");
            return;
        }
        if (!cores.empty() && size_t(n) > cores.size()) throw std::runtime_error("insufficient cores");
        if (!pin(0)) throw std::runtime_error("caller affinity");
        count = n;
        // Increment only for helpers actually launched: thread creation failure
        // leaves the pool marked unusable, with joinable helpers retained.
        try {
            for (int i = 1; i < n; ++i) {
                ++remaining;
                try { helpers.emplace_back([this, i] { helper(i); }); }
                catch (...) { --remaining; throw; }
            }
        } catch (...) { pin_failed = true; throw; }
        done.wait(lock, [&] { return remaining == 0; });
        if (pin_failed) throw std::runtime_error("worker affinity");
    }
    void run(void (*fn)(void*, int, int), void* ctx, int n) {
        { std::lock_guard<std::mutex> lock(mutex);
          task = fn; context = ctx; active = n; remaining = int(helpers.size()); ++generation; }
        wake.notify_all(); fn(ctx, 0, n);
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return remaining == 0; });
    }
} workers;

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
bool quantize(const std::vector<float>& input, std::vector<block_q8_0>& output) {
    for (float v:input) if (!std::isfinite(v) || std::abs(v)>65504.f*127.f) return false;
    for (size_t b=0;b<output.size();++b) {
        float amax=0;
        for (int j=0;j<32;++j) amax=std::max(amax,std::abs(input[b*32+j]));
        // A zero FP16 delta contributes zero. Avoid overflowing the reciprocal
        // for subnormal FP32 amax in the upstream reference quantizer.
        if (!ggml_compute_fp32_to_fp16(amax/127.f)) output[b]=block_q8_0{};
        else quantize_row_q8_0_ref(input.data()+b*32,output.data()+b,32);
    }
    return true;
}
// Gate/up output of the gated SiLU: optional pre-SiLU clamp (gate from above, up both ways), stable SiLU including
// large negative gates, times up.
inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}
struct Work { Layer* layer; int slot; float route; int stage; };
void calculate(void* p, int rank, int n) {
    auto& c = *static_cast<Work*>(p); auto& l = *c.layer; const LayerInfo& d = l.info;
    if (c.stage == 0) {
        const Projection gate = l.strided.gate(c.slot), up = l.strided.up(c.slot);
        const float gate_alpha = gate.alpha * d.inv_input_scale13;
        const float up_alpha = up.alpha * d.inv_input_scale13;
        for (int i = rank; i < d.intermediate; i += n) {
            int gate_row, up_row;
            w13_rows(d.w13_layout, d.intermediate, i, gate_row, up_row);
            const float g = dot(gate.w, gate.sf, gate_row, d.hidden, l.qx.data(), l.row_scratch[rank].data()) * gate_alpha;
            const float u = dot(up.w, up.sf, up_row, d.hidden, l.qx.data(), l.row_scratch[rank].data()) * up_alpha;
            l.intermediate[i] = swiglu(g, u, d.act_limit);
        }
    } else {
        const Projection down = l.strided.down(c.slot);
        const float alpha = down.alpha * d.inv_input_scale2 * c.route;
        for (int i = rank; i < d.hidden; i += n)
            l.result[i] += dot(down.w, down.sf, i, d.intermediate, l.qi.data(), l.row_scratch[rank].data()) * alpha;
    }
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
        auto l = std::make_shared<Layer>(*d);
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (next_handle == INT64_MAX) return 1;
        const auto id = next_handle++; layers.emplace(id, std::move(l)); *h = id; return 0;
    } catch (...) { return 1; }
}
extern "C" int sglang_nvfp4_cpu_experts_free_layer(int64_t h) noexcept {
    try {
        std::unique_lock<std::mutex> forward_lock(workers.forward_mutex, std::try_to_lock);
        if (!forward_lock.owns_lock()) return 3;
        std::lock_guard<std::mutex> lock(registry_mutex); return layers.erase(h) ? 0 : 2;
    }
    catch (...) { return 1; }
}
extern "C" int sglang_nvfp4_cpu_experts_set_cores(const int32_t* c, int32_t n) noexcept {
    try {
        std::unique_lock<std::mutex> lock(workers.forward_mutex, std::try_to_lock);
        return lock.owns_lock() ? workers.configure(c, n) : 3;
    } catch (...) { return 1; }
}
extern "C" int sglang_nvfp4_cpu_experts_forward(int64_t h, const void* x,
    const int32_t* slots, const float* weights, int32_t k, float* out, int32_t threads, int32_t accumulate) noexcept {
    try {
        if (!x || !out || k < 0 || k > 8 || threads < 1 || threads > 4096
            || (k && (!slots || !weights)) || (accumulate != 0 && accumulate != 1)) return 2;
        std::unique_lock<std::mutex> lock(workers.forward_mutex, std::try_to_lock);
        if (!lock.owns_lock()) return 3;
        auto l = lookup(h); if (!l) return 2;
        const auto& E = l->strided;
        for (int j = 0; j < k; ++j) {
            if (slots[j] < -1 || slots[j] >= l->info.capacity || !std::isfinite(weights[j])) return 2;
            if (slots[j] >= 0 && (!std::isfinite(E.alpha(kGateAlpha, slots[j]))
                || !std::isfinite(E.alpha(kDownAlpha, slots[j]))
                || (l->info.up_alpha && !std::isfinite(E.alpha(kUpAlpha, slots[j]))))) return 2;
        }
        workers.initialize(threads);
        for (int i = 0; i < l->info.hidden; ++i) {
            uint16_t v; std::memcpy(&v, static_cast<const uint8_t*>(x) + 2 * i, 2); l->x[i] = ggml_compute_fp16_to_fp32(v);
        }
        if (l->row_scratch.size()<size_t(threads)) {
            l->row_scratch.resize(threads);
            for (auto& row:l->row_scratch) row.resize(rounded(std::max(l->info.hidden,l->info.intermediate),64)/64);
        }
        if (!quantize(l->x,l->qx)) return 2;
        std::fill(l->result.begin(), l->result.end(), 0.f);
        for (int j = 0; j < k; ++j) {
            if (slots[j] == -1 || weights[j] == 0) continue;
            Work c{l.get(), slots[j], weights[j], 0};
            workers.run(calculate, &c, threads);
            if (!quantize(l->intermediate,l->qi)) return 2;
            c.stage = 1; workers.run(calculate, &c, threads);
        }
        for (int i = 0; i < l->info.hidden; ++i) {
            if (accumulate) out[i] += l->result[i]; else out[i] = l->result[i];
        }
        return 0;
    } catch (...) { return 1; }
}
