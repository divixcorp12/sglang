// DSV4.1 CPU expert kernels, derived from exllamav3 02aef45cd681b960a00afcd0749a4ab99e6c1bfe.
// MIT License, Copyright (c) 2025 Turboderp; see ../LICENSE.exllamav3.
// Optimized residual/block-128 path: original packed weights, register decode,
// compact activations, parallel preparation/middle stages, fused down transforms,
// and cache-line output partitioning. See README.txt for measured provenance.
// The arithmetic is math.hpp and each tier's math_scalar.hpp, math_avx2.hpp and math_avx512.hpp; the forward is
// forward_plan.hpp (with shapes.hpp). This file holds the C ABI and the torch wrappers.
#if !defined(__linux__) || !defined(_OPENMP)
#error This CPU expert implementation requires Linux and OpenMP.
#endif
#include "moe_mul1.h"
#include "quant.hpp"
#include "../../cpu_experts_common/cabi.hpp"
#include <c10/util/Half.h>
#include <ATen/ATen.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

// Last: the definition order the kernels were validated in (bit-exact per tier); another order changes what GCC
// inlines and clones.
#include "forward_plan.hpp"

// Kept for upstream's bindings. Phase timing is compile-time here (ForwardPlan's Profile, forward_plan.hpp).
void exl3_moe_cpu_set_prof(bool) {}

SGLANG_CPU_EXPERTS_DEFINE_CABI(exl3, ::sglang::exl3_cpu::Exl3Quant)

namespace {
using ::sglang::exl3_cpu::Exl3Quant;
using Exl3Forward = ::sglang::cpu_experts::ExpertForward<Exl3Quant>;

// A nonzero ExpertForward status as a torch error.
void check_status(int status, const char* what)
{
    TORCH_CHECK(status != 2, what, ": invalid arguments (status 2: an unknown or freed handle, a slot outside the "
                "layer, a non-finite weight, or rows/k/threads out of range)");
    TORCH_CHECK(status != 3, what, ": another forward or free is running (status 3)");
    TORCH_CHECK(status == 0, what, ": kernel error (status ", status, "): ", ::sglang::cpu_experts::last_error());
}
}  // namespace

// -------------------------------------------------------------------------------------------
//   Upstream link compatibility
// -------------------------------------------------------------------------------------------

// This file replaces upstream's cpu/moe_mul1.cpp inside the exllamav3 extension, whose
// bindings.cpp and cpu/moe_handoff.cu still reference these symbols. sglang never calls them:
// the staging copy belongs to upstream's handoff worker, and the pool they exercised was
// replaced by the OpenMP forward. They exist only so the extension links, and fail if called.

void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_stage_experts is not supported by the optimized CPU expert kernel");
}

int64_t exl3_moe_cpu_pool_stress(int, int, int, int)
{
    TORCH_CHECK(false, "exl3_moe_cpu_pool_stress is not supported by the optimized CPU expert kernel");
    return 0;
}

bool exl3_moe_cpu_has_avx2() { return Exl3Forward::isa() != ::sglang::cpu_experts::Isa::Scalar; }
bool exl3_moe_cpu_has_avx512_bw() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Bw; }
bool exl3_moe_cpu_has_avx512_vnni() { return Exl3Forward::isa() >= ::sglang::cpu_experts::Isa::Vnni; }
bool exl3_moe_cpu_has_avx512_vbmi() { return Exl3Forward::isa() == ::sglang::cpu_experts::Isa::Vbmi; }

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && trellis.is_contiguous(), "trellis must be contiguous CPU");
    TORCH_CHECK(trellis.dim() == 3, "trellis must be [k/16, n/16, 16K]");
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
    m.suh = reinterpret_cast<const at::Half*>(suh.data_ptr());
    m.svh = reinterpret_cast<const at::Half*>(svh.data_ptr());
    m.bias = bias ? reinterpret_cast<const at::Half*>(bias->data_ptr()) : nullptr;
    m.k = static_cast<int>(trellis.size(0)) * 16;
    m.n = static_cast<int>(trellis.size(1)) * 16;
    m.bits = static_cast<int>(trellis.size(2)) / 16;
    // K8 tensors are exempt from swizzling (routed to the dword kernel, which would gain
    // nothing) -- the child loader applies the same bits != 8 rule when repacking, so the two
    // sides agree per tensor
    m.swz = swizzled && m.bits != 8 ? 1 : 0;
    TORCH_CHECK(m.bits >= 1 && m.bits <= 8, "CPU MoE requires K in [1, 8]");
    TORCH_CHECK(m.k % 128 == 0 && m.n % 128 == 0, "dims must be divisible by 128");
    TORCH_CHECK(m.k <= 8192, "k too large for i32 accumulation");
    return m;
}

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled
)
{
    auto table = std::unique_ptr<MoeCpuLayer>(new MoeCpuLayer);
    const bool swz = swizzled != 0;
    const size_t E = up_trellis.size();
    const bool gated = !gate_trellis.empty();
    TORCH_CHECK(down_trellis.size() == E && (!gated || gate_trellis.size() == E), "expert count mismatch");
    TORCH_CHECK(gated ? (activation == 0 || activation == 1 || activation == 3) : activation == 2, "gated experts take silu/gelu/swiglu_oai, gateless take relu2");
    TORCH_CHECK(gate_bias.empty() || gate_bias.size() == E, "gate bias count mismatch");
    TORCH_CHECK(up_bias.empty() || up_bias.size() == E, "up bias count mismatch");
    TORCH_CHECK(down_bias.empty() || down_bias.size() == E, "down bias count mismatch");
    table->num_experts = static_cast<int>(E);
    table->activation = static_cast<int>(activation);
    table->act_limit = static_cast<float>(act_limit);
    for (size_t e = 0; e < E; ++e) {
        if (gated) {
            table->gates.push_back(make_matrix(gate_trellis[e], gate_suh[e], gate_svh[e],
                                               gate_bias.empty() ? nullptr : &gate_bias[e], swz));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]})
                table->refs.push_back(t);
            if (!gate_bias.empty()) table->refs.push_back(gate_bias[e]);
        }
        table->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                         up_bias.empty() ? nullptr : &up_bias[e], swz));
        table->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                           down_bias.empty() ? nullptr : &down_bias[e], swz));
        for (auto& t : {up_trellis[e], up_suh[e], up_svh[e], down_trellis[e], down_suh[e], down_svh[e]})
            table->refs.push_back(t);
        if (!up_bias.empty()) table->refs.push_back(up_bias[e]);
        if (!down_bias.empty()) table->refs.push_back(down_bias[e]);
    }
    table->hidden_size = table->ups[0].k;
    table->interm_size = table->ups[0].n;
    TORCH_CHECK(table->downs[0].k == table->interm_size && table->downs[0].n == table->hidden_size,
                "expert shape mismatch");

    // A table layer: ExpertForward refuses a slot at or past its expert count, as it does a slab layer's capacity.
    const ::sglang::exl3_cpu::LayerInfo info{table->num_experts, table->hidden_size, table->interm_size,
                                             !table->gates.empty(), table->activation, table->act_limit};
    Exl3Quant::Layer layer{info, {}, 0, 0, std::move(table)};
    layer.rows.capacity = info.num_experts;
    auto entry = std::make_shared<const Exl3Quant::Layer>(std::move(layer));
    std::lock_guard<std::mutex> lock(Exl3Forward::registry_mutex);
    Exl3Forward::layers.push_back(std::move(entry));
    return static_cast<int64_t>(Exl3Forward::layers.size() - 1);
}

void exl3_moe_cpu_free_layer(int64_t handle)
{
    const int status = Exl3Forward::free_layer(handle);
    if (status == 2) return;  // an unknown or already freed handle: a no-op, as upstream's free is
    check_status(status, "exl3_moe_cpu_free_layer");
}

// The C ABI's forward with accumulate 0: the FP16 weights widen to FP32 exactly and the kernel narrows them back.
void exl3_moe_cpu_forward_raw(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    if (rows == 0) return;  // upstream's forward of no tokens is a no-op; the C ABI refuses rows < 1
    static thread_local std::vector<float> weights;
    const size_t n = static_cast<size_t>(rows) * topk;
    weights.resize(n);
    for (size_t i = 0; i < n; ++i) weights[i] = static_cast<float>(wts[i]);
    SglangCpuExpertsForward call{};
    call.abi_version = SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION;
    call.rows = rows;
    call.layer = handle;
    call.x = x;
    call.slots = sel;
    call.weights = weights.data();
    call.out = out;
    call.k = topk;
    call.threads = std::max(threads, 1);  // upstream's forward ran fewer than one thread as one
    call.accumulate = 0;
    check_status(Exl3Forward::forward(&call), "exl3_moe_cpu_forward");
}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(x.device().is_cpu() && selected.device().is_cpu() && weights.device().is_cpu() && out.device().is_cpu(), "CPU MoE tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kFloat, "dtype mismatch");

    const int m_total = static_cast<int>(x.size(0));
    const int top_k = static_cast<int>(selected.size(-1));

    // Raw path takes int32 selection
    std::vector<int32_t> sel32(static_cast<size_t>(m_total) * top_k);
    if (selected.scalar_type() == at::kLong)
    {
        const int64_t* s = selected.data_ptr<int64_t>();
        for (size_t i = 0; i < sel32.size(); ++i) sel32[i] = static_cast<int32_t>(s[i]);
    }
    else
    {
        TORCH_CHECK(selected.scalar_type() == at::kInt, "selected must be int32 or int64");
        std::memcpy(sel32.data(), selected.data_ptr<int32_t>(), sel32.size() * 4);
    }

    exl3_moe_cpu_forward_raw
    (
        handle,
        reinterpret_cast<const at::Half*>(x.data_ptr()),
        sel32.data(),
        reinterpret_cast<const at::Half*>(weights.data_ptr()),
        out.data_ptr<float>(),
        m_total, top_k,
        static_cast<int>(num_threads)
    );
}
