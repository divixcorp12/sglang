// NVFP4's ISA-independent arithmetic: a GPU weight row's view (GpuRow), Q8_0 activation quantization and the gated
// SiLU, plus dot_rows<I, M>, whose tiers are math_scalar.hpp and math_avx2.hpp.
// GpuRow is derived from GGML x86 quants.c, llama.cpp 889edf43ddae0cfe9a4564a882764dc879759870.
// MIT, ggml authors; see ../upstream/LICENSE.llama.cpp.
// Changes: GPU row/scale access, adjacent-nibble interleave.
#pragma once
#include "quant.hpp"

namespace sglang::nvfp4_cpu {
namespace {

// One GPU weight row of k columns: packed E2M1 bytes and its swizzled E4M3 scales. A partial last 64-column block is
// copied into `tail`, zero padded, so a dot product reads whole blocks without reading beyond the row.
struct GpuRow {
    const uint8_t* w;
    const uint8_t* sf;
    int row, k;
    uint8_t tail[32]{};
    GpuRow(const uint8_t* weights, const uint8_t* scales, int r, int columns)
        : w(weights + size_t(r)*(columns/2)), sf(scales), row(r), k(columns) {
        if (k%64) std::memcpy(tail, w+(k/64)*32, (k%64)/2);
    }
    const uint8_t* bytes(int ib) const { return ib*64+64 <= k ? w+ib*32 : tail; }
    float scale(int ib, int group) const {
        int g=ib*4+group;
        if (g>=k/16) return 0;
        uint8_t b=sf[sf_index(row,g,k/16)];
        // GGML's LUT is doubled. Preserve the signed GPU E4M3 convention.
        float v=ggml_ue4m3_to_fp32(b & 127);
        return b & 128 ? -v : v;
    }
};

// Tier I's dot product; math_scalar.hpp and math_avx2.hpp specialize it.
template <Isa I>
struct DotRows;

// x (n columns, a multiple of 64) against M Q8_0 vectors ys[0..M) into out[0..M): each weight block is decoded once for
// all M. Each out[t] is computed by exactly the operations a one-vector call of the same tier makes, so it is bitwise
// that call's result.
template <Isa I, int M>
inline void dot_rows(int n, const GpuRow& x, const block_q8_0* const* ys, float* out)
{
    DotRows<I>::template rows<M>(n, x, ys, out);
}

inline bool q8_representable(const float* v)
{
    for (int j = 0; j < 32; ++j)
        if (!std::isfinite(v[j]) || std::abs(v[j]) > 65504.f * 127.f) return false;
    return true;
}

// A zero delta also avoids overflowing the reciprocal for a subnormal FP32 amax in the upstream reference quantizer.
inline void quantize_block(const float* v, block_q8_0& out)
{
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(v[j]));
    if (!ggml_compute_fp32_to_fp16(amax / 127.f)) out = block_q8_0{};
    else quantize_row_q8_0_ref(v, &out, 32);
}

inline float swiglu(float g, float u, float limit)
{
    if (limit > 0) { g = std::min(g, limit); u = std::clamp(u, -limit, limit); }
    const float silu = g >= 0 ? g / (1 + std::exp(-g)) : g * std::exp(g) / (1 + std::exp(g));
    return silu * u;
}

}  // namespace
}  // namespace sglang::nvfp4_cpu
