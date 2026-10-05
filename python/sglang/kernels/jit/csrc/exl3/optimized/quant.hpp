// Exl3Quant: the EXL3 CPU expert quant for ExpertForward (host/cpu_experts/expert_forward.hpp), with the layer facts
// and the expert accessors a forward plan reads. forward_plan.hpp defines Exl3Quant::dispatch.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include "cpu_experts_cabi.h"
#include "../../moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include <c10/util/Half.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>

namespace sglang::exl3_cpu {
// Internal linkage, like the framework: Exl3Quant's Layer embeds MoeBufferRows, which has it.
namespace {
using namespace ::sglang::cpu_experts;

// What a forward needs to know about a layer besides its matrices.
struct LayerInfo
{
    int num_experts;   // the routable slots: a slab layer's capacity, a table layer's expert count
    int hidden;        // k of gate/up, n of down
    int intermediate;  // n of gate/up, k of down
    bool gated;
    int activation;    // 0 silu, 1 gelu, 2 relu2 (gateless), 3 swiglu_oai
    float act_limit;
};

// The pinned tier's slabs, in exl3_expert_format.EXL3_STREAMED_NAMES order: one row per expert slot.
enum SlabName { kW13Trellis, kW13Suh, kW13Svh, kW2Trellis, kW2Suh, kW2Svh, kSlabNames };

// The fewest bytes one slot's row of each slab holds. w13 rows hold gate (part 0) then up (part 1); w2 rows hold
// down. A trellis is [k/16][n/16][16 * bits] uint16, i.e. k * n * bits / 8 bytes; sign vectors are fp16.
// Registration refuses a smaller stride; at exactly these strides every address equals the packed layout's.
struct SlabRowBytes
{
    uint64_t bytes[kSlabNames];

    static constexpr SlabRowBytes of(int hidden, int intermediate, int bits)
    {
        const uint64_t trellis = uint64_t(hidden) * uint64_t(intermediate) * uint64_t(bits) / 8;
        return {{2 * trellis, 2 * 2 * uint64_t(hidden), 2 * 2 * uint64_t(intermediate),
                 trellis, 2 * uint64_t(intermediate), 2 * uint64_t(hidden)}};
    }
};

// One projection's matrix over a slot's trellis and sign vectors (k inputs, n outputs).
inline MoeCpuMatrix exl3_matrix(const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int k, int n, int bits,
                                int swz)
{
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis);
    m.suh = reinterpret_cast<const at::Half*>(suh);
    m.svh = reinterpret_cast<const at::Half*>(svh);
    m.bias = nullptr;
    m.k = k;
    m.n = n;
    m.bits = bits;
    m.swz = swz;
    return m;
}

// Part 0 (gate) or 1 (up) of a slot's w13 rows, given the slot's first byte in each w13 slab.
inline MoeCpuMatrix w13_part(const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int part, int hidden,
                             int intermediate, int bits, int swz)
{
    const SlabRowBytes r = SlabRowBytes::of(hidden, intermediate, bits);
    return exl3_matrix(trellis + part * (r.bytes[kW13Trellis] / 2), suh + part * (r.bytes[kW13Suh] / 2),
                       svh + part * (r.bytes[kW13Svh] / 2), hidden, intermediate, bits, swz);
}

struct Exl3Quant
{
    static constexpr const char* kName = "exl3";
    static constexpr int kSlabs = kSlabNames;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 32;     // the C ABI's k limit
    static constexpr int kMaxRows = 65536;    // the C ABI's rows limit
    // The library holds every tier and runs min(host, EXL3_MOE_CPU_MAX_ISA).
    static constexpr Isa kTopIsa = Isa::Vbmi;
    static constexpr const char* kIsaCapEnv = "EXL3_MOE_CPU_MAX_ISA";
    static constexpr const char* kIsaReportEnv = "EXL3_MOE_CPU_REPORT_ISA";
    using Params = SglangExl3CpuParams;

    // One slot's three projections, as the plans read them.
    struct Row
    {
        MoeCpuMatrix gate, up, down;
    };

    // A slab layer (register_layer): rows' bases and strides, and the slabs' bits/swz. A table layer
    // (exl3_moe_cpu_make_layer): `table`, with rows holding only the capacity (its expert count).
    struct Layer
    {
        LayerInfo info;
        MoeBufferRows<Exl3Quant> rows;
        int bits;
        int swz;
        std::unique_ptr<MoeCpuLayer> table;
    };

    static std::array<uint64_t, kSlabs> min_slot_bytes(const LayerSlabs& d, const Params& p)
    {
        const SlabRowBytes minimum = SlabRowBytes::of(d.hidden, d.intermediate, p.bits);
        std::array<uint64_t, kSlabs> bytes;
        std::copy(std::begin(minimum.bytes), std::end(minimum.bytes), bytes.begin());
        return bytes;
    }

    // The descriptor's scalars and the parameters; ExpertForward checks the slabs against min_slot_bytes. The
    // dimensions are make_matrix's limits: 128-element blocks, and k <= 8192 for the int32 accumulators.
    static int validate(const LayerSlabs& d, const Params* p)
    {
        if (!p || p->bits < 1 || p->bits > 8 || (p->swizzled != 0 && p->swizzled != 1)) return 2;
        if (d.hidden < 128 || d.intermediate < 128 || d.hidden % 128 || d.intermediate % 128 || d.hidden > 8192
            || d.intermediate > 8192)
            return 2;
        if (d.activation != 0 || !std::isfinite(d.act_limit) || d.act_limit < 0.0f) return 2;
        return 0;
    }

    static Layer make_layer(const LayerSlabs& d, const Params* p)
    {
        return {{d.capacity, d.hidden, d.intermediate, true, 0, d.act_limit},
                MoeBufferRows<Exl3Quant>::of(d),
                p->bits,
                p->swizzled && p->bits != 8 ? 1 : 0,  // make_matrix's rule: K8 is never swizzled
                nullptr};
    }

    static int check_slot(const Layer&, int) { return 0; }

    // A slab layer's slot, from its first byte in each slab (MoeBufferRow::base).
    static Row decode(const uint8_t* const* base, const Layer& l)
    {
        const int h = l.info.hidden, n = l.info.intermediate;
        return {w13_part(base[kW13Trellis], base[kW13Suh], base[kW13Svh], 0, h, n, l.bits, l.swz),
                w13_part(base[kW13Trellis], base[kW13Suh], base[kW13Svh], 1, h, n, l.bits, l.swz),
                exl3_matrix(base[kW2Trellis], base[kW2Suh], base[kW2Svh], n, h, l.bits, l.swz)};
    }

    // Defined at the end of forward_plan.hpp.
    static int dispatch(const Layer& l, const ForwardCall& c, const RouteTable& r, Isa isa);
};

// The plans read a layer's experts through an accessor: gate(e), up(e), down(e) of slot e.

// A table layer: one MoeCpuMatrix per expert and projection, wherever each tensor lives.
struct TableExperts
{
    const MoeCpuLayer* layer;
    const MoeCpuMatrix& gate(int e) const { return layer->gates[e]; }
    const MoeCpuMatrix& up(int e) const { return layer->ups[e]; }
    const MoeCpuMatrix& down(int e) const { return layer->downs[e]; }
};

// A slab layer: slot e of slab i at rows.base[i] + e * rows.stride[i], nothing stored per expert. A fixed Shape
// (Dsv41Shape) makes the dimensions compile-time constants; GenericShape reads them from the layer.
template <class Shape>
struct StridedExperts
{
    const Exl3Quant::Layer* layer;

    int H() const { if constexpr (Shape::kFixed) return Shape::kHidden; else return layer->info.hidden; }
    int I() const { if constexpr (Shape::kFixed) return Shape::kIntermediate; else return layer->info.intermediate; }
    int B() const { if constexpr (Shape::kFixed) return Shape::kBits; else return layer->bits; }

    MoeCpuMatrix gate(int e) const { return w13(e, 0); }
    MoeCpuMatrix up(int e) const { return w13(e, 1); }
    MoeCpuMatrix down(int e) const
    {
        return exl3_matrix(at(kW2Trellis, e), at(kW2Suh, e), at(kW2Svh, e), I(), H(), B(), layer->swz);
    }

    // The same slabs under another shape's assumptions (the caller has checked they hold).
    template <class Other>
    StridedExperts<Other> as() const { return {layer}; }

private:
    // MoeBufferRows::slot(e).slab(name), without forming the other slabs' bases.
    const uint8_t* at(SlabName name, int e) const
    {
        return layer->rows.base[name] + size_t(e) * layer->rows.stride[name];
    }

    MoeCpuMatrix w13(int e, int part) const
    {
        return w13_part(at(kW13Trellis, e), at(kW13Suh, e), at(kW13Svh, e), part, H(), I(), B(), layer->swz);
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
