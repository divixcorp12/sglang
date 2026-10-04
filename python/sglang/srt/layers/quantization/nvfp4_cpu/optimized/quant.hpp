// Nvfp4Quant: the NVFP4 CPU expert quant for ExpertForward (cpu_experts_common/expert_forward.hpp), with the layer facts
// and slot projections a forward reads. moe_mul1.cpp defines the arithmetic, includes forward_plan.hpp and defines
// Nvfp4Quant::dispatch after it.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error The NVFP4 CPU expert kernel requires Linux and OpenMP.
#endif
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include "../../cpu_experts_common/expert_forward.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <new>
#include <utility>
#include <vector>
#if defined(__AVX2__) && !defined(NVFP4_CPU_FORCE_SCALAR)
#include <immintrin.h>
#endif

namespace sglang::nvfp4_cpu {
// Internal linkage, like the framework: Nvfp4Quant's Layer embeds MoeBufferRows, which has it.
namespace {
using namespace ::sglang::cpu_experts;

constexpr size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
// Inverse address of utils.swizzle_blockscale's reshape/permute.
inline size_t sf_index(int row, int group, int groups) {
    const size_t tiles_k = rounded(groups, 4) / 4;
    return (((size_t(row / 128) * tiles_k + group / 4) * 32 + row % 32) * 4
            + (row % 128) / 32) * 4 + group % 4;
}

#include "dot_nvfp4.h"

// The dot product's tier is fixed when the library is compiled (dot_nvfp4.h's #if chain): AVX2 under -march=native
// on an AVX2 host, else the scalar loop.
#if defined(__AVX2__)
constexpr Isa kBuildIsa = Isa::Avx2;
#else
constexpr Isa kBuildIsa = Isa::Scalar;
#endif

// What a forward needs to know about a layer besides its slabs: the descriptor's and SglangNvfp4CpuParams' scalars.
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
              "SlabName indexes SglangCpuExpertsLayer::slabs as cpu_experts_cabi.h orders them");

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

// A slot's fp32 alpha, read on every call: slab rows change when a slot is reused.
inline float alpha_at(const uint8_t* p)
{
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

struct Nvfp4Quant
{
    static constexpr const char* kName = "nvfp4";
    static constexpr int kSlabs = kSlabNames;
    static constexpr uint32_t kOptionalSlabs = 1u << kUpAlpha;
    static constexpr int kMaxRoutes = 8;      // the C ABI's k limit
    static constexpr int kMaxRows = 1 << 16;  // the C ABI's rows limit; arbitrary, it keeps every scratch index in range
    static constexpr Isa kTopIsa = kBuildIsa;
    static constexpr const char* kIsaCapEnv = nullptr;
    using Params = SglangNvfp4CpuParams;

    // One slot's three projections. The slabs are read through the slot's bases, nothing stored per slot.
    struct Row
    {
        Projection gate, up, down;
    };

    struct Layer
    {
        LayerInfo info;
        MoeBufferRows<Nvfp4Quant> rows;
    };

    static std::array<uint64_t, kSlabs> min_slot_bytes(const SglangCpuExpertsLayer& d, const Params&)
    {
        const SlabRowBytes minimum = SlabRowBytes::of(d.hidden, d.intermediate);
        std::array<uint64_t, kSlabs> bytes;
        std::copy(std::begin(minimum.bytes), std::end(minimum.bytes), bytes.begin());
        return bytes;
    }

    // The descriptor's scalars and the parameters; ExpertForward checks the slabs against min_slot_bytes.
    static int validate(const SglangCpuExpertsLayer& d, const Params* p)
    {
        if (!p || d.hidden < 16 || d.intermediate < 16 || d.hidden > (1 << 20) || d.intermediate > (1 << 20)
            || d.hidden % 16 || d.intermediate % 16 || p->w13_layout < 0 || p->w13_layout > 2
            || (p->w13_layout == 2 && d.intermediate % 64) || d.activation != 0 || !std::isfinite(d.act_limit)
            || d.act_limit < 0 || !std::isfinite(p->inv_input_scale13) || p->inv_input_scale13 <= 0
            || !std::isfinite(p->inv_input_scale2) || p->inv_input_scale2 <= 0)
            return 2;
        return 0;
    }

    static Layer make_layer(const SglangCpuExpertsLayer& d, const Params* p)
    {
        return {{d.capacity, d.hidden, d.intermediate, p->w13_layout, d.act_limit, p->inv_input_scale13,
                 p->inv_input_scale2, d.slabs[kUpAlpha] != nullptr},
                MoeBufferRows<Nvfp4Quant>::of(d)};
    }

    // A routed slot's alphas must be finite.
    static int check_slot(const Layer& l, int slot)
    {
        const MoeBufferRow<Nvfp4Quant> s = l.rows.slot(slot);
        return std::isfinite(alpha_at(s.slab(kGateAlpha))) && std::isfinite(alpha_at(s.slab(kDownAlpha)))
                       && (!l.info.up_alpha || std::isfinite(alpha_at(s.slab(kUpAlpha))))
                   ? 0
                   : 2;
    }

    static Row decode(const uint8_t* const* base, const Layer&)
    {
        const float gate_alpha = alpha_at(base[kGateAlpha]);
        return {{base[kW13], base[kSf13], gate_alpha},
                {base[kW13], base[kSf13], base[kUpAlpha] ? alpha_at(base[kUpAlpha]) : gate_alpha},
                {base[kW2], base[kSf2], alpha_at(base[kDownAlpha])}};
    }

    // Defined in moe_mul1.cpp after forward_plan.hpp.
    static int dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r, Isa isa);
};

}  // namespace
}  // namespace sglang::nvfp4_cpu
