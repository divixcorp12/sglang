// Transitional, deleted by plan 2026-10-04-cpu-expert-kernel-interface Task 9: the six C functions of the old CPU
// expert C ABI, over a quant library's CpuExpertKernel (kernel.hpp), with the ABI's layer handles, engines, statuses
// (0 ok, 1 internal error, 2 invalid arguments, 3 a free_layer while a forward runs) and last_error().
#pragma once
#include "../cpu_experts_abi.h"
#include "kernel.hpp"
#include <sched.h>
#include <cstdio>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state.
namespace {

// Why this thread's last register, free, forward or keep-warm call returned 1 ("" otherwise); each clears it on entry.
// Per library, like the rest of the framework; the C ABI does not export it, a quant's own wrappers read it.
inline std::string& last_error()
{
    static thread_local std::string error;
    return error;
}

// Records `what` as last_error(), prints it to stderr, and returns 1; never throws (it runs in the C functions' catch
// blocks).
inline int fail(const char* what) noexcept
{
    try {
        last_error() = what;
        std::fprintf(stderr, "cpu_experts: %s\n", what);
    } catch (...) {
    }
    return 1;
}

// This library's engines. An engine is an immutable core list: worker i of every forward and keep-warm naming it runs
// on cores[i], the caller as worker 0. Each CPU expert engine thread (one per NUMA group) names its own, so two teams
// run at once on disjoint cores. Handle h is table[h - 1]; a destroyed entry is reset and its index never reused, so a
// stale handle is refused (2). Handle 0 is no engine: its workers run unpinned.
struct Engines
{
    using Cores = std::shared_ptr<const std::vector<int>>;

    // Cores must be distinct and in [0, CPU_SETSIZE), else 2. Not checked against the caller's affinity: the engine
    // thread creates its engine from a thread whose inherited mask may exclude the expert cores, and the workers pin
    // themselves outside it. A core that cannot be pinned fails the first call's pin (1).
    static int create(const int32_t* cores, int32_t n, int64_t* engine) noexcept
    {
        try {
            if (!cores || !engine || n < 1 || n > CPU_SETSIZE) return 2;
            for (int i = 0; i < n; ++i) {
                if (cores[i] < 0 || cores[i] >= CPU_SETSIZE) return 2;
                for (int j = 0; j < i; ++j)
                    if (cores[j] == cores[i]) return 2;
            }
            Cores list = std::make_shared<const std::vector<int>>(cores, cores + n);
            std::lock_guard<std::mutex> lock(mutex);
            table.push_back(std::move(list));
            *engine = int64_t(table.size());
            return 0;
        } catch (...) {
            return 1;
        }
    }

    // A call already running on `engine` keeps its cores (it holds the list). Returns 0, or 2 for a handle never
    // created or already destroyed.
    static int destroy(int64_t engine) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (engine < 1 || engine > int64_t(table.size()) || !table[size_t(engine - 1)]) return 2;
            table[size_t(engine - 1)].reset();
            return 0;
        } catch (...) {
            return 1;
        }
    }

    // The cores of `engine`, null for engine 0. *found is false for a handle never created or destroyed.
    static Cores find(int64_t engine, bool* found)
    {
        *found = true;
        if (engine == 0) return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        if (engine < 1 || engine > int64_t(table.size()) || !table[size_t(engine - 1)]) {
            *found = false;
            return nullptr;
        }
        return table[size_t(engine - 1)];
    }

private:
    static inline std::mutex mutex;
    static inline std::vector<Cores> table;
};

// The C ABI's layer handles: index h is layers[h]; a freed entry is reset and its index never reused. Forwards hold
// layer_mutex shared; free_layer takes it exclusively and returns 3 while a forward runs.
struct CabiLayers
{
    static inline std::vector<std::shared_ptr<const CpuExpertLayer>> layers;
    static inline std::mutex registry_mutex;
    static inline std::shared_mutex layer_mutex;

    static int64_t add(std::shared_ptr<const CpuExpertLayer> layer)
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        layers.push_back(std::move(layer));
        return int64_t(layers.size() - 1);
    }

    static std::shared_ptr<const CpuExpertLayer> lookup(int64_t handle)
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (handle < 0 || handle >= int64_t(layers.size())) return nullptr;
        return layers[size_t(handle)];
    }
};

// Runs `f`, mapping what it throws to a C status: std::invalid_argument 2, anything else 1 (printed by fail()).
template <class F>
int cabi_status(F&& f) noexcept
{
    last_error().clear();
    try {
        f();
        return 0;
    } catch (const std::invalid_argument& e) {
        try { last_error() = e.what(); } catch (...) {}
        return 2;
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("unknown exception");
    }
}

inline LayerSlabs cabi_slabs(const SglangCpuExpertsLayer& d)
{
    LayerSlabs s;
    s.capacity = d.capacity;
    s.hidden = d.hidden;
    s.intermediate = d.intermediate;
    s.activation = d.activation;
    s.act_limit = d.act_limit;
    s.slab_count = d.slab_count;
    for (int i = 0; i < SGLANG_CPU_EXPERTS_MAX_SLABS; ++i) {
        s.slabs[i] = d.slabs[i];
        s.slot_bytes[i] = d.slot_bytes[i];
    }
    return s;
}

template <class Quant>
int cabi_register(const CpuExpertKernel& kernel, const SglangCpuExpertsLayer* d, int64_t* handle) noexcept
{
    return cabi_status([&] {
        if (!d || !handle || d->abi_version != SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION)
            throw std::invalid_argument("bad layer descriptor");
        const std::span<const std::byte> params =
            d->params ? std::span<const std::byte>(static_cast<const std::byte*>(d->params), sizeof(typename Quant::Params))
                      : std::span<const std::byte>();
        *handle = CabiLayers::add(kernel.make_layer(cabi_slabs(*d), params));
    });
}

inline int cabi_free(int64_t handle) noexcept
{
    last_error().clear();
    std::unique_lock<std::shared_mutex> layer_lock(CabiLayers::layer_mutex, std::try_to_lock);
    if (!layer_lock.owns_lock()) return 3;
    std::lock_guard<std::mutex> lock(CabiLayers::registry_mutex);
    if (handle < 0 || handle >= int64_t(CabiLayers::layers.size()) || !CabiLayers::layers[size_t(handle)]) return 2;
    CabiLayers::layers[size_t(handle)].reset();
    return 0;
}

inline int cabi_forward(const CpuExpertKernel& kernel, const SglangCpuExpertsForward* call) noexcept
{
    return cabi_status([&] {
        if (!call || call->abi_version != SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION
            || (call->accumulate != 0 && call->accumulate != 1))
            throw std::invalid_argument("bad forward call");
        bool found = false;
        const Engines::Cores cores = Engines::find(call->engine, &found);
        if (!found) throw std::invalid_argument("unknown engine");
        const std::shared_lock<std::shared_mutex> lock(CabiLayers::layer_mutex);
        const std::shared_ptr<const CpuExpertLayer> layer = CabiLayers::lookup(call->layer);
        if (!layer) throw std::invalid_argument("unknown layer handle");
        ForwardCall c;
        c.rows = call->rows;
        c.k = call->k;
        c.threads = call->threads;
        c.x = call->x;
        c.slots = call->slots;
        c.weights = call->weights;
        c.out = call->out;
        c.accumulate = call->accumulate == 1;
        if (cores) c.cores = *cores;
        kernel.forward(*layer, c);
    });
}

inline int cabi_keep_warm(const CpuExpertKernel& kernel, int64_t engine, int32_t threads, const uint32_t* word,
                          uint32_t seen, int64_t deadline_ns) noexcept
{
    return cabi_status([&] {
        bool found = false;
        const Engines::Cores cores = Engines::find(engine, &found);
        if (!found) throw std::invalid_argument("unknown engine");
        kernel.keep_warm(cores ? std::span<const int>(*cores) : std::span<const int>(), threads, word, seen, deadline_ns);
    });
}

}  // namespace
}  // namespace sglang::cpu_experts

#define SGLANG_CPU_EXPERTS_DEFINE_CABI(prefix, Quant, accessor)                                                       \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_register_layer(              \
        const SglangCpuExpertsLayer* d, int64_t* handle) noexcept                                                   \
    { return ::sglang::cpu_experts::cabi_register<Quant>(accessor(), d, handle); }                                  \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_free_layer(                  \
        int64_t handle) noexcept                                                                                    \
    { return ::sglang::cpu_experts::cabi_free(handle); }                                                            \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_forward(                     \
        const SglangCpuExpertsForward* call) noexcept                                                               \
    { return ::sglang::cpu_experts::cabi_forward(accessor(), call); }                                               \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_keep_warm(                   \
        int64_t engine, int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept         \
    { return ::sglang::cpu_experts::cabi_keep_warm(accessor(), engine, threads, word, seen, deadline_ns); }         \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_engine_create(               \
        const int32_t* cores, int32_t n, int64_t* engine) noexcept                                                  \
    { return ::sglang::cpu_experts::Engines::create(cores, n, engine); }                                            \
    extern "C" __attribute__((visibility("default"))) int sglang_##prefix##_cpu_experts_engine_free(                 \
        int64_t engine) noexcept                                                                                    \
    { return ::sglang::cpu_experts::Engines::destroy(engine); }
