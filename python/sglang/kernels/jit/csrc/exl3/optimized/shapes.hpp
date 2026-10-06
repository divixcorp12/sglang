// The shapes a forward plan can be specialized for. A plan reads every layer fact through its Shape: GenericShape takes
// each from the ExpertLayer; Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time: hidden 5120, intermediate
// 2304, 3-bit, unswizzled, clamped at its swiglu_limit of 10. Every layer is gated SiLU (Exl3Quant::validate). A Chunk
// is one expert's share of a call (up to MAX_M token rows), which Dsv41Shape::accepts reads.
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
};

struct GenericShape
{
    static constexpr bool kFixed = false;
    static int hidden(const ExpertLayer& l) { return l.hidden; }
    static int intermediate(const ExpertLayer& l) { return l.intermediate; }
    static float act_limit(const ExpertLayer& l) { return l.act_limit; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBits = 3;
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const ExpertLayer&) { return kHidden; }
    static constexpr int intermediate(const ExpertLayer&) { return kIntermediate; }
    static constexpr float act_limit(const ExpertLayer&) { return kActLimit; }

    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // every fact above (shape, limit 10, unswizzled 3-bit), and every chunk holds one token (a prefill chunk of two
    // tokens takes the generic plan).
    static bool accepts(const ExpertLayer& l, const Exl3Quant::Params& p, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (l.hidden != kHidden || l.intermediate != kIntermediate || l.act_limit != kActLimit) return false;
        if (p.bits != kBits || p.swizzled) return false;
        for (const auto& ch : chunks)
            if (ch.m != 1) return false;
        return true;
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
