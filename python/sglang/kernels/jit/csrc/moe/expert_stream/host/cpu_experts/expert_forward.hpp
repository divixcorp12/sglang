// The validation every CPU expert quant shares, generic over the quant. ExpertForward<Quant> implements the layer and
// call operations of a format's kernel type (binding.hpp): the kernel's translation unit defines the kernel's static
// functions by calling these (its job interface is its own). It holds no layer table and takes no lock. The Quant
// contract:
//   kName, kSlabs, kOptionalSlabs (a bit per slab that may be absent), kMaxRoutes, kMaxRows,
//   kTopIsa                              the quant's facts and its highest ISA tier
//   Params                               its per-layer parameters, stored in the layer
//   row_bytes(layer, params)             the fewest bytes one slot's row of each slab holds
//   validate(layer, params)              nullptr when the quant runs the layer, else why not; may normalize params
//   usable(layer, params, slot)          whether a routed slot's contents can be run (check() only)
#pragma once
#include "isa.hpp"
#include "keep_warm.hpp"
#include "kernel.hpp"
#include "team.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace sglang::cpu_experts {
// Internal linkage: each kernel translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

template <class Quant>
struct ExpertForward
{
    static_assert(Quant::kSlabs >= 1 && Quant::kSlabs <= kMaxSlabs);
    static_assert(Quant::kMaxRows >= 1 && Quant::kMaxRoutes >= 1);

    using Params = typename Quant::Params;
    using Layer = ExpertLayer<Params>;

    // The tier forwards run at: the build's (isa.hpp's kBuildIsa), or the quant's top tier when that is lower.
    static constexpr Isa isa() { return kBuildIsa < Quant::kTopIsa ? kBuildIsa : Quant::kTopIsa; }

    static const char* name() noexcept { return Quant::kName; }

    // `shape` with its slabs and params, accepted: the layer to forward over (the quant may normalize its params).
    // Throws std::invalid_argument for a shape, slab or parameter the quant cannot run.
    static Layer make_layer(const Layer& shape)
    {
        Layer layer = shape;
        if (shape.capacity < 1 || shape.slab_count != Quant::kSlabs)
            refuse("a layer needs capacity >= 1 and " + std::to_string(Quant::kSlabs) + " slabs");
        if (const char* why = Quant::validate(shape, layer.params)) refuse(why);
        const std::array<uint64_t, Quant::kSlabs> minimum = Quant::row_bytes(shape, layer.params);
        for (int i = 0; i < Quant::kSlabs; ++i) {
            if (!shape.slabs[i]) {
                if (Quant::kOptionalSlabs >> i & 1u) continue;
                refuse("slab " + std::to_string(i) + " is required");
            }
            if (shape.slot_bytes[i] < minimum[i] || shape.slot_bytes[i] > SIZE_MAX / uint64_t(shape.capacity))
                refuse("slab " + std::to_string(i) + "'s slots hold " + std::to_string(shape.slot_bytes[i])
                       + " bytes, at least " + std::to_string(minimum[i]));
        }
        return layer;
    }

    static int32_t max_routes() noexcept { return Quant::kMaxRoutes; }
    static int32_t max_rows() noexcept { return Quant::kMaxRows; }

    // Throws std::invalid_argument for a call forward may not take: rows, k or a buffer out of range, a slot outside
    // the layer's capacity or unusable, a non-finite weight (the Team checks its workers and cores when made).
    // A job's prepare checks none of it; a caller whose calls are not built to fit (a test, a harness) calls check first.
    static void check(const Layer& layer, const ForwardCall& c)
    {
        if (!c.x || !c.out || c.rows < 1 || c.rows > Quant::kMaxRows || c.k < 0 || c.k > Quant::kMaxRoutes
            || (c.k && (!c.slots || !c.weights)))
            refuse("rows, k or a buffer out of range");
        const size_t n = size_t(c.rows) * size_t(c.k);
        for (size_t j = 0; j < n; ++j) {
            const int32_t slot = c.slots[j];
            if (slot < -1 || slot >= layer.capacity || !is_finite_bits(c.weights[j]))
                refuse("slot " + std::to_string(slot) + " outside the layer's " + std::to_string(layer.capacity)
                       + " or a non-finite weight");
            if (slot >= 0 && !Quant::usable(layer, layer.params, slot)) refuse("slot " + std::to_string(slot) + " is unusable");
        }
    }

    // Register-only work at the forward's vector width (keep_warm.hpp) until *word != seen or CLOCK_MONOTONIC reaches
    // deadline_ns: a Team's idle loop, so a core keeps the license the forward runs at. Returns a value for the caller
    // to sink.
    static int32_t warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
    {
        return keep_warm_loop<isa()>(word, seen, deadline_ns);
    }

private:
    [[noreturn]] static void refuse(const std::string& why)
    {
        throw std::invalid_argument(std::string(Quant::kName) + " CPU experts: " + why);
    }
};

}  // namespace
}  // namespace sglang::cpu_experts
