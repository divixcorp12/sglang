#include "../expert_stream/host/cpu_experts/expert_forward.hpp"
#include "bestla_int3.h"
#include "entry.h"
#include "half.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

namespace sglang::gptq_int3 {
namespace ce = ::sglang::cpu_experts;
namespace {
using Layer = GptqInt3CpuKernel::Layer;
struct Quant {
  using Params = ::sglang::gptq_int3::Params;
  static constexpr const char* kName = "gptq_int3";
  static constexpr int kSlabs = 6, kOptionalSlabs = 0, kMaxRoutes = 8, kMaxRows = 4096;
  static constexpr auto kTopIsa = ce::Isa::Bw;
  static const char* validate(const Layer& l, Params& p) {
    if (p.group_size != 128 || l.hidden < 128 || l.intermediate < 128 || l.hidden % 128 || l.intermediate % 128 ||
        l.hidden > 65536 || l.intermediate > 65536)
      return "requires group128 and dimensions divisible by128 in [128,65536]";
    if (l.activation != 0 || !finite(l.act_limit) || l.act_limit < 0)
      return "requires SwiGLU (activation0) and a finite nonnegative clamp";
    return nullptr;
  }
  static std::array<uint64_t, 6> row_bytes(const Layer& l, const Params&) {
    const uint64_t h = l.hidden, i = l.intermediate;
    return {2 * i * h / 4, 2 * i * h / 8, 2 * i * h / 128 * 2, h * i / 4, h * i / 8, h * i / 128 * 2};
  }
  // Do not touch down slabs here: a staged job's gate may still be closed.
  static bool usable(const Layer&, const Params&, int) {
    return true;
  }
};
using Common = ce::ExpertForward<Quant>;
uint16_t half_at(const uint8_t* p) {
  uint16_t h;
  std::memcpy(&h, p, 2);
  return h;
}

// Tile-local decode, never an FP16 copy of the full layer. Rounding each weight before the dot is intentional;
// moving scale multiplication outside the dot would change the exported FP16 QDQ weights.
float dot(const uint8_t* lo, const uint8_t* hi, const uint8_t* scales, int row, int width, const float* x) {
  lo += size_t(row) * width / 4;
  hi += size_t(row) * width / 8;
  scales += size_t(row) * (width / 128) * 2;
  float total = 0;
  for (int g = 0; g < width / 128; ++g) {
    const float scale = from_half(half_at(scales + g * 2));
    if (!finite(scale) || scale == 0) return std::bit_cast<float>(0x7fc00000u);
    alignas(64) int8_t codes[128];
#ifdef SGLANG_CPU_EXPERTS_AVX512
    bestla::avx512(lo + g * 32, hi + g * 16, codes);
    bestla::avx512(lo + g * 32 + 16, hi + g * 16 + 8, codes + 64);
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
    for (int j = 0; j < 128; j += 32)
      bestla::avx2(lo + g * 32 + j / 4, hi + g * 16 + j / 8, codes + j);
#else
    bestla::scalar(lo + g * 32, hi + g * 16, codes, 128);
#endif
    alignas(64) float w[128];
#ifdef SGLANG_CPU_EXPERTS_AVX512
    for (int j = 0; j < 128; j += 16) {
      auto v = _mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(codes + j)))),
          _mm512_set1_ps(scale));
      _mm512_store_ps(w + j, _mm512_cvtph_ps(_mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
    }
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
    for (int j = 0; j < 128; j += 8) {
      auto v = _mm256_mul_ps(
          _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + j)))),
          _mm256_set1_ps(scale));
      _mm256_store_ps(w + j, _mm256_cvtph_ps(_mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
    }
#else
    for (int j = 0; j < 128; ++j)
      w[j] = round_half(float(codes[j]) * scale);
#endif
    for (int j = 0; j < 128; ++j) {
      if (!finite(w[j])) return std::bit_cast<float>(0x7fc00000u);
    }
#ifdef SGLANG_CPU_EXPERTS_AVX512
    auto sum = _mm512_setzero_ps();
    for (int j = 0; j < 128; j += 16)
      sum = _mm512_fmadd_ps(_mm512_load_ps(w + j), _mm512_loadu_ps(x + g * 128 + j), sum);
    total += _mm512_reduce_add_ps(sum);
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
    auto sum = _mm256_setzero_ps();
    for (int j = 0; j < 128; j += 8)
      sum = _mm256_fmadd_ps(_mm256_load_ps(w + j), _mm256_loadu_ps(x + g * 128 + j), sum);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    for (float v : lanes)
      total += v;
#else
    for (int j = 0; j < 128; ++j)
      total += w[j] * x[g * 128 + j];
#endif
  }
  return total;
}
}  // namespace
const char* GptqInt3CpuKernel::name() noexcept {
  return Common::name();
}
Params GptqInt3CpuKernel::params(std::span<const int64_t> p) {
  if (p.size() > 1 || (!p.empty() && p[0] != 128)) throw std::invalid_argument("gptq_int3 requires group128");
  return {};
}
GptqInt3CpuKernel::Layer GptqInt3CpuKernel::make_layer(const Layer& l) {
  return Common::make_layer(l);
}
int32_t GptqInt3CpuKernel::max_routes() noexcept {
  return Common::max_routes();
}
int32_t GptqInt3CpuKernel::max_rows() noexcept {
  return Common::max_rows();
}
void GptqInt3CpuKernel::check(const Layer& l, const ce::ForwardCall& c) {
  Common::check(l, c);
  if (c.out_mode != ce::OutMode::kOverwrite && c.out_mode != ce::OutMode::kOverwriteByStore &&
      c.out_mode != ce::OutMode::kAccumulate)
    throw std::invalid_argument("invalid GPTQ INT3 output mode");
}
int32_t GptqInt3CpuKernel::warm(const uint32_t* w, uint32_t s, int64_t d) {
  return Common::warm(w, s, d);
}

struct GptqInt3CpuKernel::Job::Impl {
  Layer layer;
  ce::ForwardCall call;
  struct Route {
    int token, slot;
    float weight;
  };
  std::vector<Route> routes;
  std::vector<size_t> offsets;
  std::vector<float> scratch;
  std::span<float> x, middle, down, result;
  std::atomic<int> error{0};
};
GptqInt3CpuKernel::Job::Job() : impl(std::make_unique<Impl>()) {}
GptqInt3CpuKernel::Job::~Job() = default;
void GptqInt3CpuKernel::prepare(Job& j, const Layer& l, const ce::ForwardCall& c) {
  check(l, c);
  auto& p = *j.impl;
  p.layer = l;
  p.call = c;
  p.error.store(0);
  p.routes.clear();
  p.offsets.resize(c.rows + 1);
  for (int t = 0; t < c.rows; ++t) {
    p.offsets[t] = p.routes.size();
    for (int r = 0; r < c.k; ++r) {
      const int n = t * c.k + r;
      if (c.slots[n] >= 0 && c.weights[n] != 0) p.routes.push_back({t, c.slots[n], c.weights[n]});
    }
  }
  p.offsets[c.rows] = p.routes.size();
  const uint64_t values = uint64_t(c.rows) * l.hidden * 2 + uint64_t(p.routes.size()) * (l.hidden + l.intermediate);
  if (values > 512ull * 1024 * 1024 / sizeof(float)) throw std::invalid_argument("gptq_int3 job scratch exceeds512MiB");
  // One allocation prevents unrelated high-water marks in four vectors from adding up across reused jobs.
  // Grow exactly, rather than vector's geometric growth crossing the scratch budget.
  if (values > p.scratch.capacity())
    std::vector<float>(values).swap(p.scratch);
  else
    p.scratch.resize(values);
  const size_t nx = size_t(c.rows) * l.hidden, nm = p.routes.size() * l.intermediate, nd = p.routes.size() * l.hidden;
  auto* base = p.scratch.data();
  p.x = {base, nx};
  p.middle = {base + nx, nm};
  p.down = {base + nx + nm, nd};
  p.result = {base + nx + nm + nd, nx};
  const auto* x = static_cast<const uint8_t*>(c.x);
  for (size_t i = 0; i < p.x.size(); ++i) {
    p.x[i] = from_half(half_at(x + i * 2));
    if (!finite(p.x[i])) p.error.store(2);
  }
}
ce::Tasks GptqInt3CpuKernel::tasks(const Job& j, int phase) {
  const auto& p = *j.impl;
  if (p.error.load()) return {};
  if (phase == 3) return {1, 1};
  return {
      int((phase < 2 ? p.routes.size() : size_t(p.call.rows)) *
          size_t(phase == 0 ? p.layer.intermediate : p.layer.hidden) / 16),
      1};
}
void GptqInt3CpuKernel::run(Job& j, int phase, int t0, int t1, int) {
  auto& p = *j.impl;
  const auto& l = p.layer;
  const auto& c = p.call;
  if (phase == 3) {
    // One transactional output task. Both reading an accumulation destination and publishing it must happen
    // inside kOutPhase: the window releases same-output successors when this phase ends, not at job completion.
    if (c.accumulate())
      for (size_t pos = 0; pos < p.result.size(); ++pos) {
        p.result[pos] += c.out[pos];
        if (!finite(p.result[pos])) p.error.store(2);
      }
    if (!p.error.load()) std::copy(p.result.begin(), p.result.end(), c.out);
    return;
  }
  const int width = phase == 0 ? l.intermediate : l.hidden, tiles = width / 16;
  for (int task = t0; task < t1; ++task) {
    const int idx = task / tiles, start = (task % tiles) * 16;
    if (phase < 2) {
      const auto r = p.routes[idx];
      const auto row = l[r.slot];
      for (int n = start; n < start + 16; ++n) {
        float v;
        if (phase == 0) {
          const auto* x = p.x.data() + size_t(r.token) * l.hidden;
          float gate = round_half(dot(row.slab[0], row.slab[1], row.slab[2], n, l.hidden, x));
          float up = round_half(dot(row.slab[0], row.slab[1], row.slab[2], n + l.intermediate, l.hidden, x));
          if (!finite(gate) || !finite(up)) p.error.store(2);
          if (l.act_limit > 0) {
            gate = round_half(std::min(gate, l.act_limit));
            up = round_half(std::clamp(up, -l.act_limit, l.act_limit));
          }
          v = round_half(round_half(gate / (1.f + std::exp(-gate))) * up);
          p.middle[size_t(idx) * l.intermediate + n] = v;
        } else {
          v = round_half(
              dot(row.slab[3],
                  row.slab[4],
                  row.slab[5],
                  n,
                  l.intermediate,
                  p.middle.data() + size_t(idx) * l.intermediate));
          p.down[size_t(idx) * l.hidden + n] = v;
        }
        if (!finite(v)) p.error.store(2);
      }
    } else
      for (int n = start; n < start + 16; ++n) {
        const size_t pos = size_t(idx) * l.hidden + n;
        if (phase == 2) {
          float sum = 0;
          for (size_t r = p.offsets[idx]; r < p.offsets[idx + 1]; ++r)
            sum += p.down[r * l.hidden + n] * p.routes[r].weight;
          p.result[pos] = sum;
          if (!finite(p.result[pos])) p.error.store(2);
        }
      }
  }
}
int GptqInt3CpuKernel::status(const Job& j) {
  return j.impl->error.load();
}
}  // namespace sglang::gptq_int3
