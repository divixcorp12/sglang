// CPU experts derived from pinned GGML NVFP4 x Q8 kernels.
// GPU weight slabs stay unchanged; see ../upstream/README.md for provenance.
// A forward is a job of four phases of tasks, ForwardPlan<Shape, Isa> (forward_plan.hpp) at the build's tier, behind the
// job interface of Nvfp4CpuKernel (entry.h); ExpertForward<Nvfp4Quant> (expert_stream/host/cpu_experts/expert_forward.hpp)
// holds the layer and call checks. Nvfp4Quant (quant.hpp) reads a
// layer's slots under a Shape (shapes.hpp), and each tier's dot product is in math_scalar.hpp or math_avx2.hpp.
#include "entry.h"
#include "../../moe/expert_stream/host/cpu_experts/binding.hpp"
#include "quant.hpp"
#include "math_scalar.hpp"
#include "math_avx2.hpp"

#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

namespace sglang::nvfp4_cpu {
namespace {

#include "shapes.hpp"

// Tokens one weight-row decode serves: dot_rows's accumulators, one AVX2 register each, sit beside the decoded row.
constexpr int kChunkRows = 4;

#include "forward_plan.hpp"

// The build's tier (min(kBuildIsa, kTopIsa)) picks the plan: the MiMo plan when the layer is that model's routed
// expert at the AVX2 tier, else the generic plan at the tier.
constexpr Isa kIsa = ::sglang::cpu_experts::ExpertForward<Nvfp4Quant>::isa();

void prepare_job(ForwardJob& job, const ExpertLayer& l, const ForwardCall& c)
{
    if constexpr (kIsa >= Isa::Avx2) {
        if (MimoV26ProShape::accepts(l)) {
            job.mimo = true;
            ForwardPlan<MimoV26ProShape, kIsa>::prepare(job, l, c);
            return;
        }
    }
    job.mimo = false;
    ForwardPlan<GenericShape, kIsa>::prepare(job, l, c);
}

Tasks job_tasks(const ForwardJob& job, int phase)
{
    if constexpr (kIsa >= Isa::Avx2) {
        if (job.mimo) return ForwardPlan<MimoV26ProShape, kIsa>::tasks(job.ctx, phase);
    }
    return ForwardPlan<GenericShape, kIsa>::tasks(job.ctx, phase);
}

void job_run(ForwardJob& job, int phase, int t0, int t1)
{
    if constexpr (kIsa >= Isa::Avx2) {
        if (job.mimo) return ForwardPlan<MimoV26ProShape, kIsa>::run(job.ctx, phase, t0, t1);
    }
    ForwardPlan<GenericShape, kIsa>::run(job.ctx, phase, t0, t1);
}

using Forward = ::sglang::cpu_experts::ExpertForward<Nvfp4Quant>;
}  // namespace

struct Nvfp4CpuKernel::Job::Impl
{
    ForwardJob job;
};

static_assert(::sglang::cpu_experts::CpuExpertJobKernel<Nvfp4CpuKernel>);
static_assert(Nvfp4CpuKernel::kPhases == kPhaseCount);
static_assert(Nvfp4CpuKernel::kGatePhase == kGatePhase);

Nvfp4CpuKernel::Job::Job() : impl(std::make_unique<Impl>()) {}
Nvfp4CpuKernel::Job::~Job() = default;

void Nvfp4CpuKernel::prepare(Job& job, const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call)
{
    prepare_job(job.impl->job, layer, call);
}

::sglang::cpu_experts::Tasks Nvfp4CpuKernel::tasks(const Job& job, int phase) { return job_tasks(job.impl->job, phase); }

void Nvfp4CpuKernel::run(Job& job, int phase, int t0, int t1, int) { job_run(job.impl->job, phase, t0, t1); }

int Nvfp4CpuKernel::status(const Job& job) { return job.impl->job.ctx.invalid.load(std::memory_order_relaxed) ? 2 : 0; }

const char* Nvfp4CpuKernel::name() noexcept { return Forward::name(); }
Nvfp4CpuKernel::Layer Nvfp4CpuKernel::make_layer(const Layer& shape) { return Forward::make_layer(shape); }
int32_t Nvfp4CpuKernel::max_routes() noexcept { return Forward::max_routes(); }
int32_t Nvfp4CpuKernel::max_rows() noexcept { return Forward::max_rows(); }
void Nvfp4CpuKernel::check(const Layer& layer, const ::sglang::cpu_experts::ForwardCall& call) { Forward::check(layer, call); }

int32_t Nvfp4CpuKernel::warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    return Forward::warm(word, seen, deadline_ns);
}

Nvfp4CpuKernel::Params Nvfp4CpuKernel::params(std::span<const int64_t> values)
{
    if (values.size() != 3)
        throw std::invalid_argument("NVFP4 CPU params are [w13_layout, inv_input_scale13 bits, inv_input_scale2 bits], got "
                                    + std::to_string(values.size()) + " values");
    if (values[0] < INT32_MIN || values[0] > INT32_MAX)
        throw std::invalid_argument("NVFP4 CPU params: w13_layout " + std::to_string(values[0]) + " does not fit int32");
    for (const int64_t bits : {values[1], values[2]})
        if (bits < 0 || bits > int64_t(UINT32_MAX))
            throw std::invalid_argument("NVFP4 CPU params: " + std::to_string(bits) + " is not a binary32 bit pattern");
    Params params;
    params.w13_layout = static_cast<int32_t>(values[0]);
    const uint32_t scale_bits[2] = {static_cast<uint32_t>(values[1]), static_cast<uint32_t>(values[2])};
    static_assert(sizeof(params) == 12, "SglangNvfp4CpuParams is int32 plus two binary32");
    std::memcpy(&params.inv_input_scale13, &scale_bits[0], sizeof(float));
    std::memcpy(&params.inv_input_scale2, &scale_bits[1], sizeof(float));
    return params;
}
}  // namespace sglang::nvfp4_cpu

