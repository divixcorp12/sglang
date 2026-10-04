// Included by moe_mul1.cpp inside sglang::exl3_cpu's anonymous namespace, after struct Chunk and quant.hpp.
//
// The shapes a forward plan can be specialized for. A plan reads every layer fact through its Shape: GenericShape takes
// each from the layer's LayerInfo; Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time: hidden 5120,
// intermediate 2304, 3-bit, unswizzled, gated SiLU clamped at its swiglu_limit of 10.

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
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
    static bool gated(const LayerInfo& info) { return info.gated; }
    static int activation(const LayerInfo& info) { return info.activation; }
    static float act_limit(const LayerInfo& info) { return info.act_limit; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBits = 3;
    static constexpr bool kGated = true;
    static constexpr int kActivation = 0;  // SiLU
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const LayerInfo&) { return kHidden; }
    static constexpr int intermediate(const LayerInfo&) { return kIntermediate; }
    static constexpr bool gated(const LayerInfo&) { return kGated; }
    static constexpr int activation(const LayerInfo&) { return kActivation; }
    static constexpr float act_limit(const LayerInfo&) { return kActLimit; }

    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // every fact above (shape, gated SiLU, limit 10), every routed expert is unswizzled 3-bit, and every chunk holds
    // one token (a prefill chunk of two tokens takes the generic plan).
    template <class Experts>
    static bool accepts(const LayerInfo& info, const Experts& E, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (info.hidden != kHidden || info.intermediate != kIntermediate || info.gated != kGated) return false;
        if (info.activation != kActivation || info.act_limit != kActLimit) return false;
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
