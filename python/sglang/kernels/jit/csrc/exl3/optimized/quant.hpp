// Exl3Quant: the EXL3 CPU expert quant for ExpertForward (host/cpu_experts/expert_forward.hpp), and its typed view of
// a layer's expert slots (Experts<Shape>: each projection a Exl3Projection over the slot's ExpertRow). forward_plan.hpp
// defines Exl3Quant::dispatch.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include "../../moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include "kernel.h"
#include <c10/util/Half.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>


namespace sglang::exl3_cpu {
// Internal linkage, like the framework's templates: each library's translation unit owns its state.
namespace {
using namespace ::sglang::cpu_experts;

// The pinned tier's slabs, in exl3_expert_format.EXL3_STREAMED_NAMES order: one row per expert slot.
enum SlabName { kW13Trellis, kW13Suh, kW13Svh, kW2Trellis, kW2Suh, kW2Svh, kSlabNames };

// The fewest bytes one slot's row of each slab holds. w13 rows hold gate (part 0) then up (part 1); w2 rows hold
// down. A trellis is [k/16][n/16][16 * bits] uint16, i.e. k * n * bits / 8 bytes; sign vectors are fp16.
// Registration refuses a smaller stride; at exactly these strides every address equals the packed layout's.
constexpr std::array<uint64_t, kSlabNames> row_bytes(int hidden, int intermediate, int bits)
{
    const uint64_t trellis = uint64_t(hidden) * uint64_t(intermediate) * uint64_t(bits) / 8;
    return {2 * trellis, 2 * 2 * uint64_t(hidden), 2 * 2 * uint64_t(intermediate),
            trellis, 2 * uint64_t(intermediate), 2 * uint64_t(hidden)};
}

// One projection's matrix over a slot's trellis and sign vectors (k inputs, n outputs).
inline Exl3Projection exl3_matrix(const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int k, int n, int bits,
                                int swz)
{
    Exl3Projection m;
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

struct Exl3Quant
{
    static constexpr const char* kName = "exl3";
    static constexpr int kSlabs = kSlabNames;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 32;     // the forward's k limit
    static constexpr int kMaxRows = 65536;    // the forward's rows limit
    // The library holds every tier and runs min(host, EXL3_MOE_CPU_MAX_ISA).
    static constexpr Isa kTopIsa = Isa::Vbmi;
    static constexpr const char* kIsaCapEnv = "EXL3_MOE_CPU_MAX_ISA";
    static constexpr const char* kIsaReportEnv = "EXL3_MOE_CPU_REPORT_ISA";
    using Params = SglangExl3CpuParams;

    static std::array<uint64_t, kSlabs> row_bytes(const ExpertLayer& l, const Params& p)
    {
        return ::sglang::exl3_cpu::row_bytes(l.hidden, l.intermediate, p.bits);
    }

    // The layer's shape and the parameters (ExpertForward checks the slabs against row_bytes): make_matrix's limits,
    // 128-element blocks and k <= 8192 for the int32 accumulators, and gated SiLU. Normalizes swizzled to the layout
    // the slabs hold: 8-bit trellises are never swizzled.
    static const char* validate(const ExpertLayer& l, Params& p)
    {
        if (p.bits < 1 || p.bits > 8 || (p.swizzled != 0 && p.swizzled != 1))
            return "bits must be in [1, 8] and swizzled 0 or 1";
        if (l.hidden < 128 || l.intermediate < 128 || l.hidden % 128 || l.intermediate % 128 || l.hidden > 8192
            || l.intermediate > 8192)
            return "hidden and intermediate must be multiples of 128 in [128, 8192]";
        if (l.activation != 0 || !std::isfinite(l.act_limit) || l.act_limit < 0.0f)
            return "the activation must be gated SiLU with a finite, non-negative act_limit";
        if (p.bits == 8) p.swizzled = 0;
        return nullptr;
    }

    static bool usable(const ExpertLayer&, const Params&, int) { return true; }

    // Defined at the end of forward_plan.hpp.
    static int dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa isa);
};

// The layer's experts as the plans read them: slot e's gate, up and down, each a Exl3Projection over the slot's
// ExpertRow. A fixed Shape (Dsv41Shape) makes the dimensions compile-time constants; GenericShape reads them from the
// layer and its params.
template <class Shape>
struct Experts
{
    const ExpertLayer* layer;
    Exl3Quant::Params params;

    int H() const { if constexpr (Shape::kFixed) return Shape::kHidden; else return layer->hidden; }
    int I() const { if constexpr (Shape::kFixed) return Shape::kIntermediate; else return layer->intermediate; }
    int B() const { if constexpr (Shape::kFixed) return Shape::kBits; else return params.bits; }

    Exl3Projection gate(int e) const { return w13(e, 0); }
    Exl3Projection up(int e) const { return w13(e, 1); }
    Exl3Projection down(int e) const
    {
        const ExpertRow r = (*layer)[e];
        return exl3_matrix(r.slab[kW2Trellis], r.slab[kW2Suh], r.slab[kW2Svh], I(), H(), B(), params.swizzled);
    }

private:
    // Part 0 (gate) or 1 (up) of the slot's w13 rows.
    Exl3Projection w13(int e, int part) const
    {
        const ExpertRow r = (*layer)[e];
        const std::array<uint64_t, kSlabNames> row = row_bytes(H(), I(), B());
        return exl3_matrix(r.slab[kW13Trellis] + part * (row[kW13Trellis] / 2),
                           r.slab[kW13Suh] + part * (row[kW13Suh] / 2), r.slab[kW13Svh] + part * (row[kW13Svh] / 2),
                           H(), I(), B(), params.swizzled);
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
