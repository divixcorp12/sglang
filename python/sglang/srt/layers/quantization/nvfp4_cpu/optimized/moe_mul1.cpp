// CPU W4A16 expert calculation over the GPU's existing packed NVFP4 slabs.
// No CUDA, PyTorch, GGML, OpenMP, expanded-weight cache or weight repacking.
#include "cpu_experts_cabi.h"
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
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {
constexpr float fp4[16] = {0,.5f,1,1.5f,2,3,4,6,-0.f,-.5f,-1,-1.5f,-2,-3,-4,-6};
float half(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = e == 0 ? std::ldexp(float(m), -24)
        : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(float(1024 + m), e - 25);
    return h & 32768 ? -v : v;
}
float e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = (b & 127) == 127 ? NAN
        : e == 0 ? std::ldexp(float(m), -9) : std::ldexp(float(8 + m), e - 10);
    return b & 128 ? -v : v;
}
const auto scale_values = [] {
    std::array<float, 256> values{};
    for (int i = 0; i < 256; ++i) values[i] = e4m3(static_cast<uint8_t>(i));
    return values;
}();
size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
// Inverse address of utils.swizzle_blockscale's reshape/permute.
size_t sf_index(int row, int group, int groups) {
    const size_t tiles_k = rounded(groups, 4) / 4;
    return (((size_t(row / 128) * tiles_k + group / 4) * 32 + row % 32) * 4
            + (row % 128) / 32) * 4 + group % 4;
}
struct Layer {
    SglangNvfp4CpuLayer d;
    // Activations only. Sized once; mutable slab contents are read each job.
    std::vector<float> x, intermediate, result;
    explicit Layer(const SglangNvfp4CpuLayer& desc) : d(desc), x(d.hidden),
        intermediate(d.intermediate), result(d.hidden) {}
    const uint8_t* slab(int i, int slot) const {
        return static_cast<const uint8_t*>(d.slabs[i]) + size_t(slot) * d.slot_bytes[i];
    }
    float alpha(int i, int slot) const {
        float v; std::memcpy(&v, slab(i, slot), sizeof(v)); return v;
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

float dot(const uint8_t* w, const uint8_t* sf, int row, int k, const float* x) {
    w += size_t(row) * (k / 2);
    float sum = 0;
    for (int g = 0; g < k / 16; ++g) {
        const float scale = scale_values[sf[sf_index(row, g, k / 16)]];
        const uint8_t* q = w + g * 8;
        float block = 0;
#if defined(__AVX2__)
        const __m256 lut = _mm256_setr_ps(0,.5f,1,1.5f,2,3,4,6);
        // Expand sixteen nibbles in registers; no full-row unpack buffer.
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q));
        const __m128i mask = _mm_set1_epi8(15);
        const __m128i lo = _mm_and_si128(bytes, mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        const __m128i nibbles = _mm_unpacklo_epi8(lo, hi);
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < 2; ++j) {
            const __m128i part = j ? _mm_srli_si128(nibbles, 8) : nibbles;
            const __m256i ids = _mm256_cvtepu8_epi32(part);
            __m256 values = _mm256_permutevar8x32_ps(lut, _mm256_and_si256(ids, _mm256_set1_epi32(7)));
            values = _mm256_xor_ps(values, _mm256_castsi256_ps(_mm256_slli_epi32(
                _mm256_and_si256(ids, _mm256_set1_epi32(8)), 28)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(values, _mm256_loadu_ps(x + g * 16 + j * 8)));
        }
        alignas(32) float lanes[8]; _mm256_store_ps(lanes, acc);
        for (float v : lanes) block += v;
#else
        for (int j = 0; j < 8; ++j) {
            block += fp4[q[j] & 15] * x[g * 16 + 2 * j];
            block += fp4[q[j] >> 4] * x[g * 16 + 2 * j + 1];
        }
#endif
        sum += block * scale;
    }
    return sum;
}
struct Work { Layer* layer; int slot; float route; int stage; };
void calculate(void* p, int rank, int n) {
    auto& c = *static_cast<Work*>(p); auto& l = *c.layer; auto& d = l.d;
    if (c.stage == 0) {
        const auto w = l.slab(0, c.slot), sf = l.slab(2, c.slot);
        const float gate_alpha = l.alpha(4, c.slot) * d.inv_input_scale13;
        const float up_alpha = l.alpha(d.slabs[6] ? 6 : 4, c.slot) * d.inv_input_scale13;
        for (int i = rank; i < d.intermediate; i += n) {
            int gate = i, up = i + d.intermediate;
            if (d.w13_layout == 1) std::swap(gate, up);
            if (d.w13_layout == 2) { up = (i / 64) * 128 + i % 64; gate = up + 64; }
            float g = dot(w, sf, gate, d.hidden, l.x.data()) * gate_alpha;
            float u = dot(w, sf, up, d.hidden, l.x.data()) * up_alpha;
            if (d.act_limit > 0) { g = std::min(g, d.act_limit); u = std::clamp(u, -d.act_limit, d.act_limit); }
            // Stable SiLU, including large negative gates.
            const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
            l.intermediate[i] = silu * u;
        }
    } else {
        const auto w = l.slab(1, c.slot), sf = l.slab(3, c.slot);
        const float alpha = l.alpha(5, c.slot) * d.inv_input_scale2 * c.route;
        for (int i = rank; i < d.hidden; i += n)
            l.result[i] += dot(w, sf, i, d.intermediate, l.intermediate.data()) * alpha;
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
    const uint64_t minimum[7] = {uint64_t(d.intermediate) * d.hidden,
        uint64_t(d.hidden) * d.intermediate / 2,
        rounded(2 * d.intermediate, 128) * rounded(d.hidden / 16, 4),
        rounded(d.hidden, 128) * rounded(d.intermediate / 16, 4), 4, 4, 4};
    for (int i = 0; i < 7; ++i) {
        if (i == 6 && !d.slabs[i]) continue;
        if (!d.slabs[i] || d.slot_bytes[i] < minimum[i]
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
        for (int j = 0; j < k; ++j) {
            if (slots[j] < -1 || slots[j] >= l->d.capacity || !std::isfinite(weights[j])) return 2;
            if (slots[j] >= 0 && (!std::isfinite(l->alpha(4, slots[j])) || !std::isfinite(l->alpha(5, slots[j]))
                || (l->d.slabs[6] && !std::isfinite(l->alpha(6, slots[j]))))) return 2;
        }
        workers.initialize(threads);
        for (int i = 0; i < l->d.hidden; ++i) {
            uint16_t v; std::memcpy(&v, static_cast<const uint8_t*>(x) + 2 * i, 2); l->x[i] = half(v);
        }
        std::fill(l->result.begin(), l->result.end(), 0.f);
        for (int j = 0; j < k; ++j) {
            if (slots[j] == -1 || weights[j] == 0) continue;
            Work c{l.get(), slots[j], weights[j], 0};
            workers.run(calculate, &c, threads); c.stage = 1; workers.run(calculate, &c, threads);
        }
        for (int i = 0; i < l->d.hidden; ++i) {
            if (accumulate) out[i] += l->result[i]; else out[i] = l->result[i];
        }
        return 0;
    } catch (...) { return 1; }
}
