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

// The pinned tier's slabs, in exl3_expert_format.EXL3_STREAMED_NAMES order: one row per expert slot.
enum SlabName { kW13Trellis, kW13Suh, kW13Svh, kW2Trellis, kW2Suh, kW2Svh, kSlabNames };

// Bytes of one slot's row of each slab. w13 rows hold gate (part 0) then up (part 1); w2 rows hold down.
// A trellis is [k/16][n/16][16 * bits] uint16, i.e. k * n * bits / 8 bytes; sign vectors are fp16.
struct SlabRowBytes
{
    size_t bytes[kSlabNames];

    static constexpr SlabRowBytes of(int hidden, int intermediate, int bits)
    {
        const size_t trellis = size_t(hidden) * size_t(intermediate) * size_t(bits) / 8;
        return {{2 * trellis, 2 * 2 * size_t(hidden), 2 * 2 * size_t(intermediate),
                 trellis, 2 * size_t(intermediate), 2 * size_t(hidden)}};
    }
};

// The slab registration: expert e of every slab at base + e * row bytes, nothing stored per expert. A fixed Shape
// (Dsv41Shape) makes the dimensions, and so every row size, compile-time constants; GenericShape reads them from the
// registered values. The kernel keeps no reference to the slabs: the registrant keeps them alive.
template <class Shape>
struct StridedExperts
{
    const uint8_t* base[kSlabNames];
    int hidden, intermediate, bits;  // as registered; a fixed Shape's constants take their place
    int swz;

    int H() const { if constexpr (Shape::kFixed) return Shape::kHidden; else return hidden; }
    int I() const { if constexpr (Shape::kFixed) return Shape::kIntermediate; else return intermediate; }
    int B() const { if constexpr (Shape::kFixed) return Shape::kBits; else return bits; }

    MoeCpuMatrix gate(int e) const { return w13_part(e, 0); }
    MoeCpuMatrix up(int e) const { return w13_part(e, 1); }
    MoeCpuMatrix down(int e) const
    {
        const SlabRowBytes r = SlabRowBytes::of(H(), I(), B());
        return matrix(at(kW2Trellis, e, r), at(kW2Suh, e, r), at(kW2Svh, e, r), I(), H());
    }

    // The same slabs under another shape's assumptions (the caller has checked they hold).
    template <class Other>
    StridedExperts<Other> as() const
    {
        StridedExperts<Other> o;
        std::copy(std::begin(base), std::end(base), std::begin(o.base));
        o.hidden = hidden; o.intermediate = intermediate; o.bits = bits; o.swz = swz;
        return o;
    }

private:
    const uint8_t* at(SlabName name, int e, const SlabRowBytes& r) const
    {
        return base[name] + size_t(e) * r.bytes[name];
    }

    MoeCpuMatrix w13_part(int e, int part) const
    {
        const SlabRowBytes r = SlabRowBytes::of(H(), I(), B());
        return matrix(at(kW13Trellis, e, r) + part * (r.bytes[kW13Trellis] / 2),
                      at(kW13Suh, e, r) + part * (r.bytes[kW13Suh] / 2),
                      at(kW13Svh, e, r) + part * (r.bytes[kW13Svh] / 2), H(), I());
    }

    MoeCpuMatrix matrix(const uint8_t* trellis, const uint8_t* suh, const uint8_t* svh, int k, int n) const
    {
        MoeCpuMatrix m;
        m.trellis = reinterpret_cast<const uint16_t*>(trellis);
        m.suh = reinterpret_cast<const at::Half*>(suh);
        m.svh = reinterpret_cast<const at::Half*>(svh);
        m.bias = nullptr;
        m.k = k;
        m.n = n;
        m.bits = B();
        m.swz = swz;
        return m;
    }
};
