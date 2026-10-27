// Standalone harness for the EXL3 CPU kernel's job interface (exl3/optimized/entry.h): random generic-plan layers and calls,
// each run two ways that must agree bit for bit -- one job through a JobRunner on a one-worker team, and the same on a
// team of several workers (dynamic runs, whichever worker is free). Built by test_exl3_cpu_job_interface.py with kernel.cpp at the tier under test.
#include "../../../../python/sglang/kernels/jit/csrc/exl3/optimized/entry.h"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/single_job.hpp"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace sglang::cpu_experts;
using K = sglang::exl3_cpu::Exl3CpuKernel;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {
struct Cfg {
    int hidden, intermediate, bits_w13, bits_w2, swizzled;
    float act_limit;
};

uint16_t fp16_of(float a)
{
    uint32_t u;
    std::memcpy(&u, &a, 4);
    const uint32_t sign = (u >> 16) & 0x8000, exponent = (u >> 23) & 0xff, mantissa = (u >> 13) & 0x3ff;
    return exponent == 0 ? uint16_t(sign) : uint16_t(sign | ((exponent - 112) << 10) | mantissa);
}
}  // namespace

int main(int argc, char** argv)
{
    const int workers = argc > 1 ? std::atoi(argv[1]) : 4;
    std::mt19937 rng(20261024);
    const auto rnd = [&](int n) { return int(rng() % unsigned(n)); };
    // Hidden 256 / intermediate 384 puts a GEMV's end (12 tile pairs) inside a run of 8, and 512 / 640 gives a long phase.
    const std::vector<Cfg> cfgs = {{256, 384, 3, 3, 0, 10.f}, {256, 256, 2, 4, 1, 0.f}, {384, 256, 8, 5, 0, 10.f},
                                   {128, 128, 1, 6, 1, 3.f},  {512, 640, 3, 3, 0, 10.f}};
    constexpr int kCapacity = 6;
    long calls = 0;
    for (const Cfg& c : cfgs) {
        const uint64_t w = uint64_t(c.hidden) * uint64_t(c.intermediate);
        const uint64_t row_bytes[6] = {2 * (w * uint64_t(c.bits_w13) / 8), 4ull * c.hidden, 4ull * c.intermediate,
                                       w * uint64_t(c.bits_w2) / 8, 2ull * c.intermediate, 2ull * c.hidden};
        std::vector<std::vector<uint8_t>> slabs(6);
        K::Layer shape;
        shape.capacity = kCapacity;
        shape.hidden = c.hidden;
        shape.intermediate = c.intermediate;
        shape.activation = 0;
        shape.act_limit = c.act_limit;
        shape.slab_count = 6;
        for (int s = 0; s < 6; ++s) {
            slabs[s].resize(row_bytes[s] * kCapacity);
            for (uint8_t& b : slabs[s]) b = uint8_t(rng());
            if (s == 1 || s == 2 || s == 4 || s == 5)  // the sign vectors: fp16 +-1
                for (size_t i = 0; i + 1 < slabs[s].size(); i += 2) {
                    const uint16_t v = rng() & 1 ? 0x3c00 : 0xbc00;
                    std::memcpy(&slabs[s][i], &v, 2);
                }
            shape.slabs[s] = slabs[s].data();
            shape.slot_bytes[s] = row_bytes[s];
        }
        const int64_t values[4] = {c.bits_w13, c.bits_w2, c.swizzled, 0};
        shape.params = K::params(values);
        const K::Layer layer = K::make_layer(shape);

        Team one({}, 1, &K::warm, 0), many({}, workers, &K::warm, 0);
        JobRunner<K> single(one), dynamic(many);
        for (int iter = 0; iter < 45; ++iter) {
            const int rows = 1 + rnd(7), k = 1 + rnd(5);
            std::vector<uint16_t> x(size_t(rows) * c.hidden);
            for (uint16_t& v : x) v = fp16_of((float(rnd(2001)) - 1000.f) / 500.f);
            std::vector<int32_t> slots(size_t(rows) * k);
            std::vector<float> weights(slots.size());
            const int pool = 1 + rnd(kCapacity);
            for (size_t i = 0; i < slots.size(); ++i) {
                slots[i] = rnd(8) == 0 ? -1 : rnd(pool);
                weights[i] = float(1 + rnd(100)) / 64.f;
            }
            std::vector<float> start(size_t(rows) * c.hidden);
            for (float& o : start) o = (float(rnd(1000)) - 500.f) / 16.f;

            ForwardCall call;
            call.rows = rows;
            call.k = k;
            call.x = x.data();
            call.slots = slots.data();
            call.weights = weights.data();
            call.out_mode = iter % 3 == 0   ? OutMode::kOverwrite
                            : iter % 3 == 1 ? OutMode::kAccumulate
                                            : OutMode::kOverwriteByStore;
            std::atomic<uint32_t> word{1};
            StageGate gate;
            gate.word = &word;
            gate.open = 1;
            if (iter % 4 == 3) call.stage_two = &gate;
            call.out = start.data();
            K::check(layer, call);

            std::vector<float> a = start, b = start;
            call.out = a.data();
            single.run(layer, call);
            call.out = b.data();
            dynamic.run(layer, call);
            for (const float v : a) CHECK(std::isfinite(v));
            CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
            if (call.stage_two) CHECK(gate.calls == 2);
            ++calls;
        }
    }
    std::printf("ok %ld calls bit-identical across one worker and %d workers\n", calls, workers);
    return 0;
}
