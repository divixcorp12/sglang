// Included by moe_mul1.cpp inside its anonymous namespace, after cpu_experts_cabi.h and layout.h (rounded).
// The forward reads a layer through these: LayerInfo for the facts, StridedExperts for each slot's projections.
// Accessors are cheap to copy and allocate nothing.

// What a forward needs to know about a layer besides its slabs: SglangNvfp4CpuLayer's scalar fields.
struct LayerInfo
{
    int capacity;
    int hidden;               // columns of gate/up, rows of down
    int intermediate;         // rows of gate and of up, columns of down
    int w13_layout;           // 0 [gate, up], 1 [up, gate], 2 alternating 64-row [up, gate] chunks
    float act_limit;          // 0: no clamp
    float inv_input_scale13;  // cancels an activation scale folded into the GPU gate/up alphas
    float inv_input_scale2;   // the same for down
    bool up_alpha;            // slab kUpAlpha is registered; else up shares the gate alpha
};

// The descriptor's slabs, in cpu_experts_cabi.h order.
enum SlabName { kW13, kW2, kSf13, kSf2, kGateAlpha, kDownAlpha, kUpAlpha, kSlabNames };
static_assert(kW13 == 0 && kSf13 == 2 && kGateAlpha == 4 && kUpAlpha == 6 && kSlabNames == 7,
              "SlabName indexes SglangNvfp4CpuLayer::slabs");

// The fewest bytes one slot's row of each slab holds: packed E2M1 weights (two per byte), E4M3 scales in the GPU's
// 128x4 swizzle (rows padded to 128, scale groups to 4), one fp32 alpha. Registration refuses a smaller stride.
struct SlabRowBytes
{
    uint64_t bytes[kSlabNames];

    static constexpr SlabRowBytes of(int hidden, int intermediate)
    {
        const uint64_t h = uint64_t(hidden), n = uint64_t(intermediate);
        return {{n * h, h * n / 2, rounded(2 * n, 128) * rounded(h / 16, 4), rounded(h, 128) * rounded(n / 16, 4),
                 4, 4, 4}};
    }
};
// The sanitizer harness's 80 x 80 layer (test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp) registers exactly these.
static_assert(SlabRowBytes::of(80, 80).bytes[kW13] == 80 * 80 && SlabRowBytes::of(80, 80).bytes[kW2] == 80 * 80 / 2);
static_assert(SlabRowBytes::of(80, 80).bytes[kSf13] == 256 * 8 && SlabRowBytes::of(80, 80).bytes[kSf2] == 128 * 8);

// One projection of one slot: packed E2M1 rows, their swizzled E4M3 scales, and the slot's fp32 GPU GEMM alpha. Gate
// and up share the w13 rows (w13_rows maps an output to its row); down's rows are its own.
struct Projection
{
    const uint8_t* w;
    const uint8_t* sf;
    float alpha;
};

// The w13 rows holding gate and up output i of n, per LayerInfo::w13_layout.
inline void w13_rows(int layout, int n, int i, int& gate, int& up)
{
    gate = i;
    up = i + n;
    if (layout == 1) std::swap(gate, up);
    if (layout == 2) { up = (i / 64) * 128 + i % 64; gate = up + 64; }
}

// The descriptor registration: slot s of every slab at base + s * stride, nothing stored per slot. Strides are the
// registrant's (at least SlabRowBytes) under any Shape. Shape names the plan this view is checked for: run_plan makes
// a StridedExperts<MimoV26ProShape> only after MimoV26ProShape::accepts, and ForwardPlan<Shape, I> takes only its own Shape's
// view. The kernel keeps no reference to the slabs: the registrant keeps them alive.
template <class Shape>
struct StridedExperts
{
    const uint8_t* base[kSlabNames];  // base[kUpAlpha] may be null
    uint64_t stride[kSlabNames];

    Projection gate(int slot) const { return {at(kW13, slot), at(kSf13, slot), alpha(kGateAlpha, slot)}; }
    Projection up(int slot) const
    {
        return {at(kW13, slot), at(kSf13, slot), alpha(base[kUpAlpha] ? kUpAlpha : kGateAlpha, slot)};
    }
    Projection down(int slot) const { return {at(kW2, slot), at(kSf2, slot), alpha(kDownAlpha, slot)}; }

    // A slot's alpha, read on every call: slab rows change when a slot is reused.
    float alpha(SlabName name, int slot) const
    {
        float v;
        std::memcpy(&v, at(name, slot), sizeof(v));
        return v;
    }

    // The same slabs under another shape's assumptions (the caller has checked they hold).
    template <class Other>
    StridedExperts<Other> as() const
    {
        StridedExperts<Other> o;
        std::copy(std::begin(base), std::end(base), std::begin(o.base));
        std::copy(std::begin(stride), std::end(stride), std::begin(o.stride));
        return o;
    }

private:
    const uint8_t* at(SlabName name, int slot) const { return base[name] + size_t(slot) * stride[name]; }
};
