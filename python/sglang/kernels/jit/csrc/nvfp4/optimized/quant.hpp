// Nvfp4Quant: the NVFP4 CPU expert quant for ExpertForward (host/cpu_experts/expert_forward.hpp), and its typed view of
// an expert slot (Expert: three Projections over the slot's ExpertRow). The arithmetic is math.hpp's and its tiers'
// (math_scalar.hpp, math_avx2.hpp); kernel.cpp includes forward_plan.hpp and defines Nvfp4Quant::dispatch after it.
#pragma once
#if !defined(__linux__) || !defined(_OPENMP)
#error The NVFP4 CPU expert kernel requires Linux and OpenMP.
#endif
#include "../upstream/kernels.h"
#include "../../moe/expert_stream/host/cpu_experts/expert_forward.hpp"
#include "../../moe/expert_stream/host/cpu_experts/routes.hpp"
#include "kernel.h"
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


namespace sglang::nvfp4_cpu {
// Internal linkage, like the framework's templates: each library's translation unit owns its state.
namespace {
using namespace ::sglang::cpu_experts;

constexpr size_t rounded(size_t x, size_t n) { return (x + n - 1) / n * n; }
// Inverse address of utils.swizzle_blockscale's reshape/permute.
inline size_t sf_index(int row, int group, int groups) {
    const size_t tiles_k = rounded(groups, 4) / 4;
    return (((size_t(row / 128) * tiles_k + group / 4) * 32 + row % 32) * 4
            + (row % 128) / 32) * 4 + group % 4;
}

// The layer's slabs, in the slab order of SglangNvfp4CpuParams' comment.
enum SlabName { kW13, kW2, kSf13, kSf2, kGateAlpha, kDownAlpha, kUpAlpha, kSlabNames };
static_assert(kW13 == 0 && kSf13 == 2 && kGateAlpha == 4 && kUpAlpha == 6 && kSlabNames == 7,
              "SlabName indexes ExpertLayer::slabs and ExpertRow::slab in the quant's slab order");

// The fewest bytes one slot's row of each slab holds: packed E2M1 weights (two per byte), E4M3 scales in the GPU's
// 128x4 swizzle (rows padded to 128, scale groups to 4), one fp32 alpha. Registration refuses a smaller stride.
constexpr std::array<uint64_t, kSlabNames> row_bytes(int hidden, int intermediate)
{
    const uint64_t h = uint64_t(hidden), n = uint64_t(intermediate);
    return {n * h, h * n / 2, rounded(2 * n, 128) * rounded(h / 16, 4), rounded(h, 128) * rounded(n / 16, 4), 4, 4, 4};
}
// The sanitizer harness's 80 x 80 layer (test/registered/unit/kernels/nvfp4_cpu_sanitizer.cpp) registers exactly these.
static_assert(row_bytes(80, 80)[kW13] == 80 * 80 && row_bytes(80, 80)[kW2] == 80 * 80 / 2);
static_assert(row_bytes(80, 80)[kSf13] == 256 * 8 && row_bytes(80, 80)[kSf2] == 128 * 8);

// One projection of one slot: packed E2M1 rows, their swizzled E4M3 scales, and the slot's fp32 GPU GEMM alpha. Gate
// and up share the w13 rows (w13_rows maps an output to its row); down's rows are its own.
struct Projection
{
    const uint8_t* w;
    const uint8_t* sf;
    float alpha;
};

// The w13 rows holding gate and up output i of n, per SglangNvfp4CpuParams::w13_layout.
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
    static constexpr uint32_t kOptionalSlabs = 1u << kUpAlpha;  // absent: up shares the gate alpha
    static constexpr int kMaxRoutes = 8;      // the forward's k limit
    static constexpr int kMaxRows = 1 << 16;  // the forward's rows limit; arbitrary, it keeps every scratch index in range
    // The library holds both tiers and runs min(host, NVFP4_CPU_MAX_ISA): AVX2 on an AVX2/FMA host, else scalar.
    static constexpr Isa kTopIsa = Isa::Avx2;
    static constexpr const char* kIsaCapEnv = "NVFP4_CPU_MAX_ISA";
    static constexpr const char* kIsaReportEnv = "NVFP4_CPU_REPORT_ISA";
    using Params = SglangNvfp4CpuParams;

    // One expert slot read as NVFP4: its three projections, over its ExpertRow.
    struct Expert
    {
        Projection gate, up, down;
    };

    static Expert expert(const ExpertRow& r)
    {
        const float gate_alpha = alpha_at(r.slab[kGateAlpha]);
        return {{r.slab[kW13], r.slab[kSf13], gate_alpha},
                {r.slab[kW13], r.slab[kSf13], r.slab[kUpAlpha] ? alpha_at(r.slab[kUpAlpha]) : gate_alpha},
                {r.slab[kW2], r.slab[kSf2], alpha_at(r.slab[kDownAlpha])}};
    }

    static std::array<uint64_t, kSlabs> row_bytes(const ExpertLayer& l, const Params&)
    {
        return ::sglang::nvfp4_cpu::row_bytes(l.hidden, l.intermediate);
    }

    // The layer's shape and the parameters (ExpertForward checks the slabs against row_bytes): hidden (columns of
    // gate/up, rows of down) and intermediate (rows of gate and of up, columns of down) multiples of 16, gated SiLU,
    // act_limit 0 (no clamp) or a finite clamp.
    static const char* validate(const ExpertLayer& l, Params& p)
    {
        if (l.hidden < 16 || l.intermediate < 16 || l.hidden > (1 << 20) || l.intermediate > (1 << 20)
            || l.hidden % 16 || l.intermediate % 16)
            return "hidden and intermediate must be multiples of 16 in [16, 2^20]";
        if (p.w13_layout < 0 || p.w13_layout > 2 || (p.w13_layout == 2 && l.intermediate % 64))
            return "w13_layout must be 0, 1 or 2 (2 needs intermediate % 64 == 0)";
        if (l.activation != 0 || !std::isfinite(l.act_limit) || l.act_limit < 0)
            return "the activation must be gated SiLU with a finite, non-negative act_limit";
        if (!std::isfinite(p.inv_input_scale13) || p.inv_input_scale13 <= 0 || !std::isfinite(p.inv_input_scale2)
            || p.inv_input_scale2 <= 0)
            return "the inverse input scales must be finite and positive";
        return nullptr;
    }

    // A routed slot's alphas must be finite.
    static bool usable(const ExpertLayer& l, const Params&, int slot)
    {
        const Expert e = expert(l[slot]);
        return std::isfinite(e.gate.alpha) && std::isfinite(e.up.alpha) && std::isfinite(e.down.alpha);
    }

    // Defined in kernel.cpp after forward_plan.hpp.
    static int dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa isa, Team& team);
};

}  // namespace
}  // namespace sglang::nvfp4_cpu
