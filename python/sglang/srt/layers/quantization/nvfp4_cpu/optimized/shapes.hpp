// Included by moe_mul1.cpp inside its anonymous namespace, after experts.hpp.
//
// The shapes a forward plan can be specialized for. A plan reads every layer fact a shape may fix through its Shape:
// GenericShape takes each from the layer's LayerInfo. The w13 row order and the alphas' input scales stay runtime
// facts of the descriptor under every shape.

struct GenericShape
{
    static int hidden(const LayerInfo& info) { return info.hidden; }
    static int intermediate(const LayerInfo& info) { return info.intermediate; }
    static float act_limit(const LayerInfo& info) { return info.act_limit; }
};
