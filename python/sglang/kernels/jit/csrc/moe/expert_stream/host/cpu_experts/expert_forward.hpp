// The validation and dispatch every CPU expert quant shares, generic over the quant (the Quant contract: kName,
// kSlabs, kOptionalSlabs, kMaxRoutes, kMaxRows, kTopIsa, kIsaCapEnv, kIsaReportEnv, Params, Layer, Row,
// min_slot_bytes, validate, make_layer, check_slot, dispatch, decode). ExpertForward<Quant> is the quant's
// CpuExpertKernel (kernel.hpp): each library holds one behind its accessor; it holds no layer table and takes no lock.
// A Quant's dispatch may ignore the RouteTable and read the request directly: EXL3 does, to keep its frozen
// accumulation order, so it runs the zero-weight routes that RouteTable drops.
#pragma once
#include "buffer_row.hpp"
#include "isa.hpp"
#include "keep_warm.hpp"
#include "kernel.hpp"
#include "routes.hpp"
#include "team.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
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
    static_assert(std::is_trivially_copyable_v<Params>, "params cross the interface as bytes");

    // A layer this kernel made: the quant's own, tagged with the kernel.
    class Layer final : public CpuExpertLayer
    {
    public:
        Layer(const CpuExpertKernel& k, typename Quant::Layer q) : CpuExpertLayer(k), quant(std::move(q)) {}
        const typename Quant::Layer quant;
    };

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

    std::unique_ptr<CpuExpertLayer> make_layer(const LayerSlabs& d, std::span<const std::byte> params) const override
    {
        if (!params.empty() && params.size() != sizeof(Params))
            refuse("params hold " + std::to_string(params.size()) + " bytes, the quant's hold "
                   + std::to_string(sizeof(Params)));
        Params p{};
        if (!params.empty()) std::memcpy(&p, params.data(), sizeof(Params));
        const Params* given = params.empty() ? nullptr : &p;
        if (d.capacity < 1 || d.slab_count != Quant::kSlabs)
            refuse("a layer needs capacity >= 1 and " + std::to_string(Quant::kSlabs) + " slabs");
        if (Quant::validate(d, given) != 0) refuse("the layer's shape, activation or parameters");
        const std::array<uint64_t, Quant::kSlabs> minimum = Quant::min_slot_bytes(d, p);
        for (int i = 0; i < Quant::kSlabs; ++i) {
            if (!d.slabs[i]) {
                if (Quant::kOptionalSlabs >> i & 1u) continue;
                refuse("slab " + std::to_string(i) + " is required");
            }
            if (d.slot_bytes[i] < minimum[i] || d.slot_bytes[i] > SIZE_MAX / uint64_t(d.capacity))
                refuse("slab " + std::to_string(i) + "'s slots hold " + std::to_string(d.slot_bytes[i])
                       + " bytes, at least " + std::to_string(minimum[i]));
        }
        return std::make_unique<Layer>(*this, Quant::make_layer(d, given));
    }

    // Wraps a quant layer built without LayerSlabs (EXL3's per-expert table layers), as make_layer would; the caller
    // has validated it.
    std::unique_ptr<CpuExpertLayer> wrap(typename Quant::Layer quant) const
    {
        return std::make_unique<Layer>(*this, std::move(quant));
    }

    void forward(const CpuExpertLayer& layer, const ForwardCall& c) const override
    {
        if (&layer.kernel() != this)
            refuse(std::string("a layer of kernel ") + layer.kernel().name() + " (another library's or object's)");
        if (!c.x || !c.out || c.rows < 1 || c.rows > Quant::kMaxRows || c.k < 0 || c.k > Quant::kMaxRoutes
            || c.threads < 1 || c.threads > 4096 || (c.k && (!c.slots || !c.weights)))
            refuse("rows, k, threads or a buffer out of range");
        if (!c.cores.empty() && size_t(c.threads) > c.cores.size())
            refuse(std::to_string(c.threads) + " workers on " + std::to_string(c.cores.size()) + " cores");
        check_cores(c.cores);
        const typename Quant::Layer& l = static_cast<const Layer&>(layer).quant;
        const int capacity = l.rows.capacity;
        const size_t n = size_t(c.rows) * size_t(c.k);
        for (size_t j = 0; j < n; ++j) {
            const int32_t slot = c.slots[j];
            if (slot < -1 || slot >= capacity || !std::isfinite(c.weights[j]))
                refuse("slot " + std::to_string(slot) + " outside the layer's " + std::to_string(capacity)
                       + " or a non-finite weight");
            if (slot >= 0 && Quant::check_slot(l, slot) != 0) refuse("slot " + std::to_string(slot) + " is unusable");
        }
        const RouteTable routes = RouteTable::build(c.slots, c.weights, c.rows, c.k);
        const CallCores on_cores(c.cores);
        const int status = Quant::dispatch(l, c, routes, isa());
        if (status == 2) refuse("the forward refused its input (status 2)");
        if (status != 0)
            throw std::runtime_error(std::string(Quant::kName) + " CPU experts: forward failed (status "
                                     + std::to_string(status) + ")");
    }

    // keep_warm (keep_warm.hpp) at this quant's tier, compiling only the loops up to kTopIsa.
    void keep_warm(std::span<const int> cores, int32_t threads, const uint32_t* word, uint32_t seen,
                   int64_t deadline_ns) const override
    {
        check_cores(cores);
        ::sglang::cpu_experts::keep_warm<Quant::kTopIsa>(isa(), cores, threads, word, seen, deadline_ns);
    }

private:
    [[noreturn]] static void refuse(const std::string& why)
    {
        throw std::invalid_argument(std::string(Quant::kName) + " CPU experts: " + why);
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
