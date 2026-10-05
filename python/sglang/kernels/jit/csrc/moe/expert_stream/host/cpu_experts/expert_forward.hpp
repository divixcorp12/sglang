// The validation and dispatch every CPU expert quant shares, generic over the quant. ExpertForward<Quant> is the
// quant's CpuExpertKernel (kernel.hpp): each library holds one behind its accessor; it holds no layer table and takes
// no lock. The Quant contract:
//   kName, kSlabs, kOptionalSlabs (a bit per slab that may be absent), kMaxRoutes, kMaxRows,
//   kTopIsa, kIsaCapEnv, kIsaReportEnv   the quant's facts and its ISA tier's caps
//   Params                               its per-layer parameters, trivially copyable, stored in the layer
//   row_bytes(layer, params)             the fewest bytes one slot's row of each slab holds
//   validate(layer, params)              nullptr when the quant runs the layer, else why not; may normalize params
//   usable(layer, params, slot)          whether a routed slot's contents can be run (check() only)
//   dispatch(layer, params, call, isa)   the forward: 0, 2 when the input refuses, else a failure status
#pragma once
#include "isa.hpp"
#include "keep_warm.hpp"
#include "kernel.hpp"
#include "team.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

template <class Quant>
class ExpertForward final : public CpuExpertKernel
{
public:
    static_assert(Quant::kSlabs >= 1 && Quant::kSlabs <= kMaxSlabs);
    static_assert(Quant::kMaxRows >= 1 && Quant::kMaxRoutes >= 1);
    using Params = typename Quant::Params;
    static_assert(std::is_trivially_copyable_v<Params> && sizeof(Params) <= kMaxParamBytes,
                  "params are stored in the layer as bytes");

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

    const char* name() const noexcept override { return Quant::kName; }

    ExpertLayer make_layer(const ExpertLayer& shape, std::span<const std::byte> params) const override
    {
        if (params.size() != sizeof(Params))
            refuse("params hold " + std::to_string(params.size()) + " bytes, the quant's hold "
                   + std::to_string(sizeof(Params)));
        Params p;
        std::memcpy(&p, params.data(), sizeof(Params));
        if (shape.capacity < 1 || shape.slab_count != Quant::kSlabs)
            refuse("a layer needs capacity >= 1 and " + std::to_string(Quant::kSlabs) + " slabs");
        if (const char* why = Quant::validate(shape, p)) refuse(why);
        const std::array<uint64_t, Quant::kSlabs> minimum = Quant::row_bytes(shape, p);
        for (int i = 0; i < Quant::kSlabs; ++i) {
            if (!shape.slabs[i]) {
                if (Quant::kOptionalSlabs >> i & 1u) continue;
                refuse("slab " + std::to_string(i) + " is required");
            }
            if (shape.slot_bytes[i] < minimum[i] || shape.slot_bytes[i] > SIZE_MAX / uint64_t(shape.capacity))
                refuse("slab " + std::to_string(i) + "'s slots hold " + std::to_string(shape.slot_bytes[i])
                       + " bytes, at least " + std::to_string(minimum[i]));
        }
        ExpertLayer layer = shape;
        layer.kernel = this;
        layer.params = {};
        std::memcpy(layer.params.data(), &p, sizeof(Params));
        return layer;
    }

    int32_t max_routes() const noexcept override { return Quant::kMaxRoutes; }
    int32_t max_rows() const noexcept override { return Quant::kMaxRows; }

    void check(const ExpertLayer& layer, const ForwardCall& c) const override
    {
        if (layer.kernel != this)
            refuse(std::string("a layer of kernel ") + (layer.kernel ? layer.kernel->name() : "(none)")
                   + " (another library's or object's)");
        if (!c.x || !c.out || c.rows < 1 || c.rows > Quant::kMaxRows || c.k < 0 || c.k > Quant::kMaxRoutes
            || c.threads < 1 || c.threads > 4096 || (c.k && (!c.slots || !c.weights)))
            refuse("rows, k, threads or a buffer out of range");
        if (!c.cores.empty() && size_t(c.threads) > c.cores.size())
            refuse(std::to_string(c.threads) + " workers on " + std::to_string(c.cores.size()) + " cores");
        check_cores(c.cores);
        const Params p = layer.params_as<Params>();
        const size_t n = size_t(c.rows) * size_t(c.k);
        for (size_t j = 0; j < n; ++j) {
            const int32_t slot = c.slots[j];
            if (slot < -1 || slot >= layer.capacity || !std::isfinite(c.weights[j]))
                refuse("slot " + std::to_string(slot) + " outside the layer's " + std::to_string(layer.capacity)
                       + " or a non-finite weight");
            if (slot >= 0 && !Quant::usable(layer, p, slot)) refuse("slot " + std::to_string(slot) + " is unusable");
        }
    }

    void forward(const ExpertLayer& layer, const ForwardCall& c) const override
    {
        const CallCores on_cores(c.cores);
        const int status = Quant::dispatch(layer, layer.params_as<Params>(), c, isa());
        if (status != 0) [[unlikely]]
            failed(status);
    }

    // keep_warm (keep_warm.hpp) at this quant's tier, compiling only the loops up to kTopIsa. The cores are the
    // caller's, checked where they were configured (the engine checks them when it is built).
    void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
                   int64_t deadline_ns) const override
    {
        ::sglang::cpu_experts::keep_warm<Quant::kTopIsa>(isa(), cores, threads, word, seen, deadline_ns);
    }

private:
    // Out of line and cold, so forward's failure path costs it one predicted branch.
    [[noreturn, gnu::noinline, gnu::cold]] static void failed(int status)
    {
        if (status == 2) refuse("the forward refused its input (status 2)");
        throw std::runtime_error(std::string(Quant::kName) + " CPU experts: forward failed (status "
                                 + std::to_string(status) + ")");
    }

    [[noreturn]] static void refuse(const std::string& why)
    {
        throw std::invalid_argument(std::string(Quant::kName) + " CPU experts: " + why);
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
