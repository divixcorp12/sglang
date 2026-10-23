// The router seam of the RAM prefetch scorer: the per-expert score the next streamed layer's gate gives a token, from
// its dot product w_e . x and its bias b_e. The host scorer (host/gate_scorer.h) and the GPU score kernel
// (spec_score.cuh) both call R::score, so the two sides agree on one definition of a model's router.
#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <string_view>

// Usable from nvcc (host and device passes) and from plain host C++.
#if defined(__CUDACC__)
#define SGL_ROUTER_HD __host__ __device__
#else
#define SGL_ROUTER_HD
#endif

namespace sglang::expert_stream {

/// \brief A model's expert router score.
///
/// `score(dot, bias)` is the fp32 selection score; the caller maps a NaN to -inf. `GateWeight` and `Input` name the
/// storage types the host scorer reads. `kSupportsLearnedHead` says whether the learned L+2 head (spec_head.cuh) was
/// trained on this router's routing; the head refuses to compile for a router without it.
template <class R>
concept ExpertRouter = requires(float dot, float bias) {
  typename R::GateWeight;
  typename R::Input;
  { R::score(dot, bias) } -> std::same_as<float>;
  { R::kSupportsLearnedHead } -> std::convertible_to<bool>;
  { R::kName } -> std::convertible_to<std::string_view>;
};

/// \brief DSV4's router: `sqrt(softplus(dot)) + bias`, torch's softplus with threshold 20.
///
/// The two sides differ on purpose and each keeps its own arithmetic: the device computes `log1pf(expf(z))` and
/// rounds the sqrt and the add explicitly (`__fsqrt_rn`, `__fadd_rn`) so no fast-math flag or FMA contraction can
/// change them; the host computes `std::log1p(std::exp(z))`, `std::sqrt` and `+`, as it did before the seam.
struct SqrtSoftplusBias {
  using GateWeight = uint16_t;  // bf16 bits on the host; the device also takes fp32 gates (router_fp32)
  using Input = uint16_t;       // fp16 bits on the host; the device reads bf16
  static constexpr bool kSupportsLearnedHead = true;
  static constexpr std::string_view kName = "sqrt_softplus";

  static SGL_ROUTER_HD float score(float dot, float bias) {
#ifdef __CUDA_ARCH__
    const float softplus = dot > 20.0f ? dot : log1pf(expf(dot));
    return __fadd_rn(__fsqrt_rn(softplus), bias);
#else
    const float softplus = dot > 20.0f ? dot : std::log1p(std::exp(dot));
    return std::sqrt(softplus) + bias;
#endif
  }
};

/// \brief MiMo V2's router: `sigmoid(dot) + bias`, the e_score_correction_bias added for selection.
///
/// Plain `1 / (1 + exp(-dot))` on both sides; the learned head was not trained on this routing.
struct SigmoidBias {
  using GateWeight = uint16_t;
  using Input = uint16_t;
  static constexpr bool kSupportsLearnedHead = false;
  static constexpr std::string_view kName = "sigmoid";

  static SGL_ROUTER_HD float score(float dot, float bias) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(1.0f / (1.0f + expf(-dot)), bias);
#else
    return 1.0f / (1.0f + std::exp(-dot)) + bias;
#endif
  }
};

static_assert(ExpertRouter<SqrtSoftplusBias>);
static_assert(ExpertRouter<SigmoidBias>);
static_assert(SqrtSoftplusBias::kSupportsLearnedHead && !SigmoidBias::kSupportsLearnedHead);

/// \brief The routers by their `kName`, the spelling of the engine's -DSGLANG_EXPERT_ROUTER=<name> (engine_host.cpp,
/// engine_device.cuh pick `routers::SGLANG_EXPERT_ROUTER`).
namespace routers {
using sqrt_softplus = SqrtSoftplusBias;
using sigmoid = SigmoidBias;
}  // namespace routers

}  // namespace sglang::expert_stream
