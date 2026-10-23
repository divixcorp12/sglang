// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is ExpertForward<Nvfp4Quant> (expert_stream/host/cpu_experts/expert_forward.hpp) dispatching to
// ForwardPlan<Shape, Isa>::run (forward_plan.hpp) at the build's tier; Nvfp4Quant (quant.hpp) reads a
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

// The build's tier (min(kBuildIsa, kTopIsa)) picks the plan: the MiMo plan when the layer is that model's routed
// expert at the AVX2 tier, else the generic plan at the tier.
template <Isa I>
int dispatch_at(const ExpertLayer& l, const Nvfp4Quant::Params& p, const ForwardCall& c, Team& team)
{
    const RouteTable r = RouteTable::build(c.slots, c.weights, c.rows, c.k);
    if constexpr (I >= Isa::Avx2) {
        if (MimoV26ProShape::accepts(l)) return ForwardPlan<MimoV26ProShape, I>::run(l, p, c, r, team);
    }
    return ForwardPlan<GenericShape, I>::run(l, p, c, r, team);
}

int Nvfp4Quant::dispatch(const ExpertLayer& l, const Params& p, const ForwardCall& c, Isa, Team& team)
{
    return dispatch_at<ExpertForward<Nvfp4Quant>::isa()>(l, p, c, team);
}

}  // namespace

const ::sglang::cpu_experts::CpuExpertKernel& nvfp4_cpu_kernel()
{
    static const ::sglang::cpu_experts::ExpertForward<Nvfp4Quant> kernel{};
    return kernel;
}
}  // namespace sglang::nvfp4_cpu

