// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ExpertForward<Nvfp4Quant> (expert_stream/host/cpu_experts/expert_forward.hpp) dispatching to
// ForwardPlan<Shape, Isa>::run (forward_plan.hpp) at the tier ExpertForward detected; Nvfp4Quant (quant.hpp) reads a
// layer's slots under a Shape (shapes.hpp), and each tier's dot product is in math_scalar.hpp or math_avx2.hpp.
#include "kernel.h"
#include "quant.hpp"
#include "math_scalar.hpp"
#include "math_avx2.hpp"

namespace sglang::nvfp4_cpu {
namespace {

#include "shapes.hpp"

// Tokens one weight-row decode serves: dot_rows's accumulators, one AVX2 register each, sit beside the decoded row.
constexpr int kChunkRows = 4;

#include "forward_plan.hpp"

// The detected tier (min(host, kTopIsa, NVFP4_CPU_MAX_ISA)) picks the plan: the MiMo plan when the layer is that
// model's routed expert at the AVX2 tier, else the generic plan at the tier.
int Nvfp4Quant::dispatch(const Layer& l, const ForwardCall& c, const RouteTable& r, Isa isa)
{
    if (isa >= Isa::Avx2 && MimoV26ProShape::accepts(l.info))
        return ForwardPlan<MimoV26ProShape, Isa::Avx2>::run(l, c, r);
    return isa >= Isa::Avx2 ? ForwardPlan<GenericShape, Isa::Avx2>::run(l, c, r)
                            : ForwardPlan<GenericShape, Isa::Scalar>::run(l, c, r);
}

}  // namespace

const ::sglang::cpu_experts::CpuExpertKernel& nvfp4_cpu_kernel()
{
    static const ::sglang::cpu_experts::ExpertForward<Nvfp4Quant> kernel{};
    return kernel;
}
}  // namespace sglang::nvfp4_cpu

