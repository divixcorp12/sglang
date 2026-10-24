// Exl3Quant: the EXL3 CPU expert quant for ExpertForward (host/cpu_experts/expert_forward.hpp), and its typed view of
// a layer's expert slots (Experts<Shape>: each projection a Exl3Projection over the slot's ExpertRow). EXL3 runs through
// the job interface (entry.h), not ExpertForward::forward, so Exl3Quant has no dispatch.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "projection.hpp"
#include "../../moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include "kernel.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>


namespace sglang::exl3_cpu {
// Internal linkage, like the framework's templates: each library's translation unit owns its state.
namespace {
using namespace ::sglang::cpu_experts;

// The layer type of this quant's kernel (entry.h's Exl3CpuKernel::Layer).
using ExpertLayer = ::sglang::cpu_experts::ExpertLayer<SglangExl3CpuParams>;

// The pinned tier's slabs, in exl3_expert_format.EXL3_STREAMED_NAMES order: one row per expert slot.
enum SlabName { kW13Trellis, kW13Suh, kW13Svh, kW2Trellis, kW2Suh, kW2Svh, kSlabNames };

// The fewest bytes one slot's row of each slab holds. w13 rows hold gate (part 0) then up (part 1), each at bits_w13;
// w2 rows hold down at bits_w2. A trellis is [k/16][n/16][16 * bits] uint16, i.e. k * n * bits / 8 bytes; sign
// vectors are fp16. Registration refuses a smaller stride; at exactly these strides every address equals the packed
// layout's.
constexpr std::array<uint64_t, kSlabNames> row_bytes(int hidden, int intermediate, int bits_w13, int bits_w2)
{
    const uint64_t weights = uint64_t(hidden) * uint64_t(intermediate);
    return {2 * (weights * uint64_t(bits_w13) / 8), 2 * 2 * uint64_t(hidden), 2 * 2 * uint64_t(intermediate),
            weights * uint64_t(bits_w2) / 8, 2 * uint64_t(intermediate), 2 * uint64_t(hidden)};
}

// Why the plan p.plan names cannot run layer l (nullptr: it can). Defined in forward_plan.hpp, beside the plans.
const char* plan_rejects(const ExpertLayer& l, const SglangExl3CpuParams& p);

// One projection's matrix over a slot's trellis and sign vectors (k inputs, n outputs).
inline Exl3Projection exl3_projection(
    const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int k, int n, int bits, int swz)
{
    return {reinterpret_cast<const uint16_t*>(trellis), reinterpret_cast<const Fp16*>(suh),
            reinterpret_cast<const Fp16*>(svh), nullptr, k, n, bits, swz, k / 16};
}

struct Exl3Quant
{
    static constexpr const char* kName = "exl3";
    static constexpr int kSlabs = kSlabNames;
    static constexpr uint32_t kOptionalSlabs = 0;
    static constexpr int kMaxRoutes = 64;     // the forward's k limit: a job's lanes, up to the wire's 64 (prepare_job sizes everything from k)
    static constexpr int kMaxRows = 65536;    // the forward's rows limit
    // The library compiles one tier, the build's (isa.hpp's kBuildIsa).
    static constexpr Isa kTopIsa = Isa::Vbmi;
    using Params = SglangExl3CpuParams;

    static std::array<uint64_t, kSlabs> row_bytes(const ExpertLayer& l, const Params& p)
    {
        return ::sglang::exl3_cpu::row_bytes(l.hidden, l.intermediate, p.bits_w13, p.bits_w2);
    }

    // The layer's shape and the parameters (ExpertForward checks the slabs against row_bytes): make_matrix's limits,
    // 128-element blocks and k <= 8192 for the int32 accumulators, gated SiLU, and a plan that accepts the layer.
    static const char* validate(const ExpertLayer& l, Params& p)
    {
        if (p.bits_w13 < 1 || p.bits_w13 > 8 || p.bits_w2 < 1 || p.bits_w2 > 8)
            return "bits_w13 and bits_w2 must be in [1, 8]";
        if (p.swizzled != 0 && p.swizzled != 1) return "swizzled must be 0 or 1";
        if (l.hidden < 128 || l.intermediate < 128 || l.hidden % 128 || l.intermediate % 128 || l.hidden > 8192
            || l.intermediate > 8192)
            return "hidden and intermediate must be multiples of 128 in [128, 8192]";
        if (l.activation != 0 || !is_finite_bits(l.act_limit) || l.act_limit < 0.0f)
            return "the activation must be gated SiLU with a finite, non-negative act_limit";
        return plan_rejects(l, p);
    }

    static bool usable(const ExpertLayer&, const Params&, int) { return true; }
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
    int B13() const { if constexpr (Shape::kFixed) return Shape::kBitsW13; else return params.bits_w13; }
    int B2() const { if constexpr (Shape::kFixed) return Shape::kBitsW2; else return params.bits_w2; }

    Exl3Projection gate(int e) const { return w13(e, 0); }
    Exl3Projection up(int e) const { return w13(e, 1); }
    Exl3Projection down(int e) const
    {
        const ExpertRow r = (*layer)[e];
        return exl3_projection(r.slab[kW2Trellis], r.slab[kW2Suh], r.slab[kW2Svh], I(), H(), B2(), swz(B2()));
    }

private:
    // A fixed Shape's plan accepted the layer's layout as it is; otherwise an 8-bit trellis is never swizzled.
    int swz(int bits) const
    {
        if constexpr (Shape::kFixed) return params.swizzled;
        else return bits == 8 ? 0 : params.swizzled;
    }

    // Part 0 (gate) or 1 (up) of the slot's w13 rows.
    Exl3Projection w13(int e, int part) const
    {
        const ExpertRow r = (*layer)[e];
        const std::array<uint64_t, kSlabNames> row = row_bytes(H(), I(), B13(), B2());
        return exl3_projection(r.slab[kW13Trellis] + part * (row[kW13Trellis] / 2),
                           r.slab[kW13Suh] + part * (row[kW13Suh] / 2), r.slab[kW13Svh] + part * (row[kW13Svh] / 2),
                           H(), I(), B13(), swz(B13()));
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
