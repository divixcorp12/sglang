// Included by kernel.cpp inside quant.hpp's namespace, after quant.hpp.
//
// The shapes a forward plan can be specialized for. A plan reads every layer fact a shape may fix through its Shape:
// GenericShape takes each from the layer's LayerSlabs; MimoV26ProShape fixes MiMo V2.6 Pro's routed expert at compile
// time: hidden 6144, intermediate 2048, gated SiLU with no clamp. The w13 row order and the alphas' input scales stay
// runtime facts of the descriptor under every shape.

struct GenericShape
{
    static int hidden(const Nvfp4Quant::Layer& l) { return l.slabs.hidden; }
    static int intermediate(const Nvfp4Quant::Layer& l) { return l.slabs.intermediate; }
    static float act_limit(const Nvfp4Quant::Layer& l) { return l.slabs.act_limit; }
};

// MiMo V2.6 Pro (mimo_v2: hidden_size 6144, moe_intermediate_size 2048, hidden_act silu, no SwiGLU limit; 384 routed
// experts, 8 per token).
struct MimoV26ProShape
{
    static constexpr int kHidden = 6144, kIntermediate = 2048;
    static constexpr float kActLimit = 0.0f;  // no clamp
    static constexpr int hidden(const Nvfp4Quant::Layer&) { return kHidden; }
    static constexpr int intermediate(const Nvfp4Quant::Layer&) { return kIntermediate; }
    static constexpr float act_limit(const Nvfp4Quant::Layer&) { return kActLimit; }

    // Whether this layer may take the MiMo V2.6 Pro plan: it has every fact above.
    static bool accepts(const Nvfp4Quant::Layer& l)
    {
        return l.slabs.hidden == kHidden && l.slabs.intermediate == kIntermediate && l.slabs.act_limit == kActLimit;
    }
};
// No 64-column tail, so GpuRow never copies one, and whole Q8_0 blocks: what the plan's constants rely on.
static_assert(MimoV26ProShape::kHidden % 64 == 0 && MimoV26ProShape::kIntermediate % 64 == 0);
static_assert(SlabRowBytes::of(MimoV26ProShape::kHidden, MimoV26ProShape::kIntermediate).bytes[kSf13] == 4096ull * 384);
