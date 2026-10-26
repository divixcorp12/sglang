// Test-only C ABI around the real typed kernel and scheduler. No Python
// callback runs in a forward.
#include "moe/expert_stream/host/cpu_experts/single_job.hpp"
#define SGLANG_EXPERT_FORMAT gptq_int3
#define SGLANG_EXPERT_ROUTER sigmoid
#include "moe/expert_stream/engine_select.h"
#include "moe/gptq_int3/bestla_int3.h"
#include "moe/gptq_int3/entry.h"
#include "moe/gptq_int3/half.h"
#include <chrono>
#include <cstdio>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>
namespace ce = sglang::cpu_experts;
using K = sglang::gptq_int3::GptqInt3CpuKernel;
static_assert(ce::CpuExpertKernel<K>);
static_assert(std::same_as<K, sglang::expert_stream::EngineFormat::CpuKernel>);
extern "C" {
void half_convert(const float *x, uint16_t *y, int n) {
  for (int i = 0; i < n; ++i)
    y[i] = sglang::gptq_int3::to_half(x[i]);
}
void half_roundtrip(uint16_t *y) {
  for (int i = 0; i < 65536; ++i)
    y[i] = sglang::gptq_int3::to_half(sglang::gptq_int3::from_half(i));
}
void unpack(const uint8_t *lo, const uint8_t *hi, int8_t *out, int count) {
  for (int i = 0; i < count; i += 128) {
#ifdef SGLANG_CPU_EXPERTS_AVX512
    sglang::gptq_int3::bestla::avx512(lo + i / 4, hi + i / 8, out + i);
    sglang::gptq_int3::bestla::avx512(lo + i / 4 + 16, hi + i / 8 + 8,
                                      out + i + 64);
#elif defined(SGLANG_CPU_EXPERTS_AVX2)
    for (int j = 0; j < 128; j += 32)
      sglang::gptq_int3::bestla::avx2(lo + (i + j) / 4, hi + (i + j) / 8,
                                      out + i + j);
#else
    sglang::gptq_int3::bestla::scalar(lo + i / 4, hi + i / 8, out + i, 128);
#endif
  }
}
// window=1: four simultaneous jobs on a closed gate, all accumulating to the
// same output key.
int forward(int h, int i, int cap, int rows, int routes, const void **slabs,
            const uint64_t *strides, const void *x, const int32_t *slots,
            const float *weights, float *out, int mode, int threads,
            float limit, int window, char *error) {
  try {
    struct Mapping {
      void *p;
      size_t bytes;
      Mapping(const void *source, size_t n) : bytes((n + 4095) / 4096 * 4096) {
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
          throw std::runtime_error("mmap failed");
        std::memcpy(p, source, n);
        mprotect(p, bytes, PROT_NONE);
      }
      ~Mapping() { munmap(p, bytes); }
    };
    std::vector<std::unique_ptr<Mapping>> protected_down;
    K::Layer shape;
    shape.capacity = cap;
    shape.hidden = h;
    shape.intermediate = i;
    shape.slab_count = 6;
    shape.act_limit = limit;
    for (int s = 0; s < 6; ++s) {
      shape.slabs[s] = slabs[s];
      shape.slot_bytes[s] = strides[s];
    }
    if (window == 2)
      for (int s = 3; s < 6; ++s) {
        protected_down.push_back(
            std::make_unique<Mapping>(slabs[s], strides[s] * cap));
        shape.slabs[s] = protected_down.back()->p;
      }
    auto layer = K::make_layer(shape);
    ce::ForwardCall c;
    c.rows = rows;
    c.k = routes;
    c.x = x;
    c.slots = slots;
    c.weights = weights;
    c.out = out;
    c.out_mode = static_cast<ce::OutMode>(mode);
    ce::Team team({}, threads, &K::warm, 0);
    if (!window) {
      ce::JobRunner<K> runner(team);
      runner.run(layer, c);
    } else {
      ce::JobWindow<K, 4> w(team);
      w.start();
      std::atomic<uint32_t> word{0};
      ce::StageGate gate;
      gate.word = &word;
      gate.open = 1;
      for (int n = 0; n < 4; ++n) {
        if (!w.admit(n + 1, 0, layer, c, 17, &gate))
          throw std::runtime_error("admission failed");
      }
      std::thread opener([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        for (const auto &m : protected_down)
          mprotect(m->p, m->bytes, PROT_READ);
        word.store(1);
      });
      while (w.any_active()) {
        auto seen = w.word();
        auto step = w.step(0);
        if (step != ce::WindowStep::kRan && w.any_active())
          w.idle_wait(seen, step == ce::WindowStep::kGated, 0);
      }
      opener.join();
      w.close();
      w.drain();
      w.join();
      for (int n = 0; n < 4; ++n)
        if (w.slot(n).status())
          throw std::runtime_error("concurrent job refused");
    }
    return 0;
  } catch (const std::exception &e) {
    std::snprintf(error, 512, "%s", e.what());
    return 2;
  }
}
}

#ifdef GPTQ_INT3_STANDALONE
int main() {
  constexpr int h = 256, i = 128;
  const uint64_t sizes[6] = {2 * i * h / 4, 2 * i * h / 8, 2 * i * h / 128 * 2,
                             h * i / 4,     h * i / 8,     h * i / 128 * 2};
  std::vector<uint8_t> data[6];
  const void *pointers[6];
  for (int s = 0; s < 6; ++s) {
    data[s].resize(sizes[s]);
    pointers[s] = data[s].data();
    if (s % 3 == 0)
      std::fill(data[s].begin(), data[s].end(), 0x55);
    if (s % 3 == 2)
      for (size_t n = 0; n < sizes[s]; n += 2) {
        auto half = sglang::gptq_int3::to_half(.001f);
        std::memcpy(data[s].data() + n, &half, 2);
      }
  }
  uint16_t x[h];
  std::fill_n(x, h, sglang::gptq_int3::to_half(.125f));
  int32_t slot = 0;
  float weight = 1;
  float out[h];
  char error[512];
  for (int attempt = 0; attempt < 40; ++attempt) {
    std::fill_n(out, h, 0.f);
    if (forward(h, i, 1, 1, 1, pointers, sizes, x, &slot, &weight, out, 2, 4, 0,
                2, error))
      return 1;
    for (float v : out)
      if (!sglang::gptq_int3::finite(v) || v == 0)
        return 2;
  }
  std::fill_n(out, h, 123.f);
  data[2][0] = data[2][1] = 0;
  if (forward(h, i, 1, 1, 1, pointers, sizes, x, &slot, &weight, out, 0, 4, 0,
              0, error) != 2)
    return 3;
  for (float v : out)
    if (v != 123.f)
      return 4;
  std::puts("ok protected staged slabs, concurrent accumulation, invalid input "
            "transaction");
}
#endif
