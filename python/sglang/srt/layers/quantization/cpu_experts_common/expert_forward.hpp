// The registry, validation and dispatch every CPU expert quant shares, generic over the quant (the Quant contract:
// kName, kSlabs, kOptionalSlabs, kMaxRoutes, kMaxRows, kTopIsa, kIsaCapEnv, kIsaReportEnv, Params, Layer, Row,
// min_slot_bytes, validate, make_layer, check_slot, dispatch, decode). Each quant's library holds its own registry and forward lock.
#pragma once
#include "../../../../kernels/jit/csrc/moe/expert_stream/host/cpu_experts_abi.h"
#include "buffer_row.hpp"
#include "isa.hpp"
#include "keep_warm.hpp"
#include "routes.hpp"
#include "team.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

template <class Quant>
struct ExpertForward
{
    static_assert(Quant::kSlabs >= 1 && Quant::kSlabs <= SGLANG_CPU_EXPERTS_MAX_SLABS);
    static_assert(Quant::kMaxRows >= 1 && Quant::kMaxRoutes >= 1);

    using Layer = typename Quant::Layer;
    using Params = typename Quant::Params;

    // Handles are indices; a freed entry is reset and its index never reused, so a stale handle is refused (2).
    static inline std::vector<std::shared_ptr<const Layer>> layers;
    static inline std::mutex registry_mutex;
    // One forward or free at a time: the others return 3 rather than race the forward's scratch.
    static inline std::mutex forward_mutex;

    // Computed at the first call, so a test may set the cap variable before it. Then, when Quant::kIsaReportEnv (may
    // be null) is "1", prints "<kName> isa <tier>" to stderr, once.
    static Isa isa()
    {
        static const Isa isa = [] {
            const Isa detected = detect_isa(Quant::kTopIsa, Quant::kIsaCapEnv);
            report_isa(Quant::kName, detected, Quant::kIsaReportEnv);
            return detected;
        }();
        return isa;
    }

    static std::shared_ptr<const Layer> lookup(int64_t handle)
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (handle < 0 || handle >= int64_t(layers.size())) return nullptr;
        return layers[size_t(handle)];
    }

    static int register_layer(const SglangCpuExpertsLayer* d, int64_t* handle) noexcept
    {
        try {
            if (!d || !handle || d->abi_version != SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION || d->capacity < 1
                || d->slab_count != Quant::kSlabs)
                return 2;
            const Params* params = static_cast<const Params*>(d->params);
            if (Quant::validate(*d, params) != 0) return 2;
            const Params no_params{};
            const std::array<uint64_t, Quant::kSlabs> minimum = Quant::min_slot_bytes(*d, params ? *params : no_params);
            for (int i = 0; i < Quant::kSlabs; ++i) {
                if (!d->slabs[i]) {
                    if (Quant::kOptionalSlabs >> i & 1u) continue;
                    return 2;
                }
                if (d->slot_bytes[i] < minimum[i] || d->slot_bytes[i] > SIZE_MAX / uint64_t(d->capacity)) return 2;
            }
            auto layer = std::make_shared<const Layer>(Quant::make_layer(*d, params));
            std::lock_guard<std::mutex> lock(registry_mutex);
            layers.push_back(std::move(layer));
            *handle = int64_t(layers.size() - 1);
            return 0;
        } catch (...) {
            return 1;
        }
    }

    static int free_layer(int64_t handle) noexcept
    {
        try {
            std::unique_lock<std::mutex> forward_lock(forward_mutex, std::try_to_lock);
            if (!forward_lock.owns_lock()) return 3;
            std::lock_guard<std::mutex> lock(registry_mutex);
            if (handle < 0 || handle >= int64_t(layers.size()) || !layers[size_t(handle)]) return 2;
            layers[size_t(handle)].reset();
            return 0;
        } catch (...) {
            return 1;
        }
    }

    static int forward(const SglangCpuExpertsForward* call) noexcept
    {
        try {
            if (!call || call->abi_version != SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION) return 2;
            const SglangCpuExpertsForward& c = *call;
            if (!c.x || !c.out || c.rows < 1 || c.rows > Quant::kMaxRows || c.k < 0 || c.k > Quant::kMaxRoutes
                || c.threads < 1 || c.threads > 4096 || (c.k && (!c.slots || !c.weights))
                || (c.accumulate != 0 && c.accumulate != 1))
                return 2;
            std::unique_lock<std::mutex> lock(forward_mutex, std::try_to_lock);
            if (!lock.owns_lock()) return 3;
            const std::shared_ptr<const Layer> layer = lookup(c.layer);
            if (!layer) return 2;
            const int capacity = layer->rows.capacity;
            const size_t n = size_t(c.rows) * size_t(c.k);
            for (size_t j = 0; j < n; ++j) {
                const int32_t slot = c.slots[j];
                if (slot < -1 || slot >= capacity || !std::isfinite(c.weights[j])) return 2;
                if (slot >= 0 && Quant::check_slot(*layer, slot) != 0) return 2;
            }
            const RouteTable routes = RouteTable::build(c.slots, c.weights, c.rows, c.k);
            return Quant::dispatch(*layer, c, routes, isa());
        } catch (...) {
            return 1;
        }
    }

    // keep_warm (keep_warm.hpp) at this quant's tier, compiling only the loops up to kTopIsa.
    static int keep_warm(int32_t threads, const uint32_t* word, uint32_t seen, int64_t deadline_ns) noexcept
    {
        try {
            return ::sglang::cpu_experts::keep_warm<Quant::kTopIsa>(isa(), threads, word, seen, deadline_ns);
        } catch (...) {
            return 1;
        }
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
