// Included by moe_mul1.cpp inside its anonymous namespace, after MoeCpuMatrix/MoeCpuLayer are declared.
// The forward reads a layer through these: LayerInfo for the facts, an Experts accessor for each expert's
// gate/up/down matrix. Accessors are cheap to copy and allocate nothing.

// What a forward needs to know about a layer besides its matrices.
struct LayerInfo
{
    int num_experts;
    int hidden;        // k of gate/up, n of down
    int intermediate;  // n of gate/up, k of down
    bool gated;
    int activation;    // 0 silu, 1 gelu, 2 relu2 (gateless), 3 swiglu_oai
    float act_limit;
};

// make_layer's registration: one MoeCpuMatrix per expert and projection, wherever each tensor lives.
struct TableExperts
{
    const MoeCpuLayer* layer;
    const MoeCpuMatrix& gate(int e) const { return layer->gates[e]; }
    const MoeCpuMatrix& up(int e) const { return layer->ups[e]; }
    const MoeCpuMatrix& down(int e) const { return layer->downs[e]; }
};
