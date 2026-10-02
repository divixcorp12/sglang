// Included by moe_mul1.cpp inside its anonymous namespace, after struct Chunk and experts.hpp.
//
// The shapes a forward plan can be specialized for. GenericShape takes every dimension from the layer. Dsv41Shape is
// DeepSeek V4.1's routed expert: hidden 5120, intermediate 2304, 3-bit, unswizzled, gated.

struct GenericShape
{
    static constexpr bool kFixed = false;
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
};

struct Dsv41Shape
{
    static constexpr bool kFixed = true;
    static constexpr int kHidden = 5120, kIntermediate = 2304, kBits = 3;
    static constexpr int hidden(const LayerInfo&) { return kHidden; }
    static constexpr int intermediate(const LayerInfo&) { return kIntermediate; }

    // Whether this call may take the DSV4.1 plan: the build quantizes activations residual/block-128, the layer has
    // the shape and is gated, every routed expert is unswizzled 3-bit, and every chunk holds one token (a prefill
    // chunk of two tokens takes the generic plan).
    template <class Experts>
    static bool accepts(const LayerInfo& info, const Experts& E, const std::vector<Chunk>& chunks)
    {
        if (ACT_ROWS != 2 || EXL3_MOE_CPU_ACT_BLOCK != 128) return false;
        if (info.hidden != kHidden || info.intermediate != kIntermediate || !info.gated) return false;
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
