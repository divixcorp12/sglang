// The shapes a forward plan can be specialized for. A plan reads every layer fact through its Shape: GenericShape takes
// each from the layer (its LayerSlabs and gated); Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time: hidden 5120,
// intermediate 2304, 3-bit, unswizzled, gated SiLU clamped at its swiglu_limit of 10. A Chunk is one expert's share
// of a call (up to MAX_M token rows), which Dsv41Shape::accepts reads.
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
    static int hidden(const Exl3Quant::Layer& l) { return l.slabs.hidden; }
    static int intermediate(const Exl3Quant::Layer& l) { return l.slabs.intermediate; }
    static bool gated(const Exl3Quant::Layer& l) { return l.gated; }
    static int activation(const Exl3Quant::Layer& l) { return l.slabs.activation; }
    static float act_limit(const Exl3Quant::Layer& l) { return l.slabs.act_limit; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBits = 3;
    static constexpr bool kGated = true;
    static constexpr int kActivation = 0;  // SiLU
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const Exl3Quant::Layer&) { return kHidden; }
    static constexpr int intermediate(const Exl3Quant::Layer&) { return kIntermediate; }
    static constexpr bool gated(const Exl3Quant::Layer&) { return kGated; }
    static constexpr int activation(const Exl3Quant::Layer&) { return kActivation; }
    static constexpr float act_limit(const Exl3Quant::Layer&) { return kActLimit; }

    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // every fact above (shape, gated SiLU, limit 10), every routed expert is unswizzled 3-bit, and every chunk holds
    // one token (a prefill chunk of two tokens takes the generic plan).
    template <class Experts>
    static bool accepts(const Exl3Quant::Layer& l, const Experts& E, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        const LayerSlabs& s = l.slabs;
        if (s.hidden != kHidden || s.intermediate != kIntermediate || l.gated != kGated) return false;
        if (s.activation != kActivation || s.act_limit != kActLimit) return false;
        for (const auto& ch : chunks) {
            const MoeCpuMatrix& g = E.gate(ch.expert);
            const MoeCpuMatrix& u = E.up(ch.expert);
            const MoeCpuMatrix& d = E.down(ch.expert);
            if (ch.m != 1 || g.bits != kBits || u.bits != kBits || d.bits != kBits || g.swz || u.swz || d.swz)
                return false;
        }
        return true;
    }
};

}  // namespace
}  // namespace sglang::exl3_cpu
