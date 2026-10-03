// Included by moe_mul1.cpp inside its anonymous namespace, after experts.hpp.
//
// The shapes a forward plan can be specialized for. A plan reads every layer fact a shape may fix through its Shape:
// GenericShape takes each from the layer's LayerInfo; Dsv41Shape fixes DeepSeek V4.1's routed expert at compile time:
// hidden 5120, intermediate 2304, gated SiLU clamped at its swiglu_limit of 10. The w13 row order and the alphas'
// input scales stay runtime facts of the descriptor under every shape.

struct GenericShape
{
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
    static float act_limit(const LayerInfo& info) { return info.act_limit; }
};

struct Dsv41Shape
{
    static constexpr int kHidden = 5120, kIntermediate = 2304;
    static constexpr float kActLimit = 10.0f;
    static constexpr int hidden(const LayerInfo&) { return kHidden; }
    static constexpr int intermediate(const LayerInfo&) { return kIntermediate; }
    static constexpr float act_limit(const LayerInfo&) { return kActLimit; }

    // Whether this layer may take the DSV4.1 plan: it has every fact above.
    static bool accepts(const LayerInfo& info)
    {
        return info.hidden == kHidden && info.intermediate == kIntermediate && info.act_limit == kActLimit;
    }
};
// No 64-column tail, so GpuRow never copies one, and whole Q8_0 blocks: what the plan's constants rely on.
static_assert(Dsv41Shape::kHidden % 64 == 0 && Dsv41Shape::kIntermediate % 64 == 0);
static_assert(SlabRowBytes::of(Dsv41Shape::kHidden, Dsv41Shape::kIntermediate).bytes[kSf13] == 4608ull * 320);
