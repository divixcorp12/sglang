// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ExpertForward<Nvfp4Quant> (cpu_experts_common/expert_forward.hpp) dispatching to
// ForwardPlan<Shape, Isa>::run (forward_plan.hpp); Nvfp4Quant (quant.hpp) reads a layer's slots under a Shape
// (shapes.hpp).
#include "quant.hpp"
#include "../../cpu_experts_common/cabi.hpp"

namespace sglang::nvfp4_cpu {
namespace {

#include "shapes.hpp"

// Tokens one weight-row decode serves: dot_rows's accumulators, one AVX2 register each, sit beside the decoded row.
constexpr int kChunkRows = 4;

// -------------------------------------------------------------------------------------------
//   Arithmetic (unchanged from the pre-OpenMP kernel)
// -------------------------------------------------------------------------------------------

template <int M>
void dot_rows_of(const uint8_t* w, const uint8_t* sf, int row, int k, const block_q8_0* const* xs, float* out) {
    dot_gpu_rows<M>(int(rounded(k, 64)), GpuRow(w, sf, row, k), xs, out);
}

void dot_rows(const uint8_t* w, const uint8_t* sf, int row, int k, const block_q8_0* const* xs, int m, float* out) {
    static_assert(kChunkRows == 4, "dot_rows dispatches m in [1, 4]");
    switch (m) {
        case 1: dot_rows_of<1>(w, sf, row, k, xs, out); break;
        case 2: dot_rows_of<2>(w, sf, row, k, xs, out); break;
        case 3: dot_rows_of<3>(w, sf, row, k, xs, out); break;
        default: dot_rows_of<4>(w, sf, row, k, xs, out); break;
    }
}

bool q8_representable(const float* v)
{
    for (int j = 0; j < 32; ++j)
        if (!std::isfinite(v[j]) || std::abs(v[j]) > 65504.f * 127.f) return false;
    return true;
}

// A zero delta also avoids overflowing the reciprocal for a subnormal FP32 amax in the upstream reference quantizer.
void quantize_block(const float* v, block_q8_0& out)
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

#include "forward_plan.hpp"

// The MiMo plan when the layer is that model's routed expert on an AVX2 build, else the generic plan for this build's
// tier. A template so a scalar build discards, and never instantiates, the AVX2 plan.
template <Isa I>
int run_plan(const Nvfp4Quant::Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r)
{
    if constexpr (I == Isa::Avx2) {
        if (MimoV26ProShape::accepts(l.info)) return ForwardPlan<MimoV26ProShape, Isa::Avx2>::run(l, c, r);
    }
    return ForwardPlan<GenericShape, I>::run(l, c, r);
}

// The detected tier is not read: kTopIsa is the build's tier, so it never differs from kBuildIsa on a host that can
// run this build.
int Nvfp4Quant::dispatch(const Layer& l, const SglangCpuExpertsForward& c, const RouteTable& r, Isa)
{
    return run_plan<kBuildIsa>(l, c, r);
}

}  // namespace
}  // namespace sglang::nvfp4_cpu

SGLANG_CPU_EXPERTS_DEFINE_CABI(nvfp4, ::sglang::nvfp4_cpu::Nvfp4Quant)
