// The shapes a forward plan can be specialized for. A plan reads every layer fact through its Shape: GenericShape takes
// each from the ExpertLayer; Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time: hidden 5120, intermediate
// 2304, 3-bit gate/up and down, unswizzled, clamped at its swiglu_limit of 10. Every layer is gated SiLU
// (Exl3Quant::validate). A Chunk is one expert's share of a call (up to CHUNK_M tokens).
// Derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe. MIT License, Copyright (c) 2025 Turboderp;
// see ../LICENSE.exllamav3.
#pragma once
#include "math.hpp"
#include <vector>

namespace sglang::exl3_cpu {
namespace {

struct Chunk
{
    int expert;
    int m;
    int token[MAX_M];
    float weight[MAX_M];
    unsigned first = 0;  // bit r: row r is token[r]'s first write of the call (ForwardCtx::store_first)
    int row0 = 0;        // the chunk's first compact activation row: the previous chunks' kActRows * m rows
};

// Each Shape is one plan's facts, which the host's cpu_kernel_plans lists and registration checks (plan_rejects): its id
// and name and the lowest ISA tier it runs at.
struct GenericShape
{
    static constexpr bool kFixed = false;
    static constexpr int32_t kPlan = kSglangExl3CpuPlanGeneric;
    static constexpr const char* kName = "generic";
    static constexpr Isa kMinIsa = Isa::Scalar;
    static int hidden(const ExpertLayer& l) { return l.hidden; }
    static int intermediate(const ExpertLayer& l) { return l.intermediate; }
    static float act_limit(const ExpertLayer& l) { return l.act_limit; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int32_t kPlan = kSglangExl3CpuPlanDsv41;
    static constexpr const char* kName = "dsv41";
    // Its register kernels need AVX-512BW (a VNNI build accumulates with vpdpwssd); its chunks hold 1..CHUNK_M tokens
    // (register_tiles<M>, kRegisterBudget), which prepare_job makes for this plan.
    static constexpr Isa kMinIsa = Isa::Bw;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBitsW13 = 3, kBitsW2 = 3, kSwizzled = 0;
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const ExpertLayer&) { return kHidden; }
    static constexpr int intermediate(const ExpertLayer&) { return kIntermediate; }
    static constexpr float act_limit(const ExpertLayer&) { return kActLimit; }

    // Whether the layer has every fact above: shape, limit 10, unswizzled 3-bit gate, up and down.
    static bool matches(const ExpertLayer& l, const Exl3Quant::Params& p)
    {
        if (l.hidden != kHidden || l.intermediate != kIntermediate || l.act_limit != kActLimit) return false;
        return p.bits_w13 == kBitsW13 && p.bits_w2 == kBitsW2 && p.swizzled == kSwizzled;
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
