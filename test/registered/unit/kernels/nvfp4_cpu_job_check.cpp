// Standalone harness for the NVFP4 CPU kernel's job interface (nvfp4/optimized/entry.h): random layers and calls, each
// run three ways that must agree bit for bit -- one job through a JobRunner on a one-worker team, the same on a team of
// several workers (dynamic runs, whichever worker is free), and Kernel::forward on a team (a static split of each phase
// between barriers). A call whose input Q8_0 cannot represent must be refused by all three with `out` untouched. Built by
// test_nvfp4_cpu_job_interface.py with kernel.cpp at the tier under test.
#include "../../../../python/sglang/kernels/jit/csrc/nvfp4/optimized/kernel.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace sglang::cpu_experts;
using K = sglang::nvfp4_cpu::Nvfp4CpuKernel;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {
struct Cfg {
    int hidden, intermediate, layout;
    float act_limit;
    bool up_alpha;
    int iterations;
};

uint16_t fp16_of(float a)
{
    uint32_t u;
    std::memcpy(&u, &a, 4);
    const uint32_t sign = (u >> 16) & 0x8000, exponent = (u >> 23) & 0xff, mantissa = (u >> 13) & 0x3ff;
    return exponent == 0 ? uint16_t(sign) : uint16_t(sign | ((exponent - 112) << 10) | mantissa);
}

uint64_t round_up(uint64_t x, uint64_t n) { return (x + n - 1) / n * n; }

template <class F>
bool refused(F&& f)
{
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
}  // namespace

int main(int argc, char** argv)
{
    const int workers = argc > 1 ? std::atoi(argv[1]) : 4;
    std::mt19937 rng(20261024);
    const auto rnd = [&](int n) { return int(rng() % unsigned(n)); };
    // 80 x 80 has a 64-column tail and fewer tasks than workers in its small phases; 256 x 192 puts several chunks in
    // GateUp; 6144 x 2048 is the MiMo V2.6 Pro plan, whose layer facts are compile-time constants.
    const std::vector<Cfg> cfgs = {{80, 80, 0, 0.f, false, 40},   {256, 192, 2, 10.f, true, 40},
                                   {144, 128, 1, 0.f, false, 40}, {512, 640, 0, 0.f, true, 25},
                                   {6144, 2048, 0, 0.f, false, 4}};
    constexpr int kCapacity = 6;
    long calls = 0, refusals = 0;
    for (const Cfg& c : cfgs) {
        const uint64_t h = c.hidden, n = c.intermediate;
        const uint64_t stride[7] = {n * h, h * n / 2, round_up(2 * n, 128) * round_up(h / 16, 4),
                                    round_up(h, 128) * round_up(n / 16, 4), 4, 4, 4};
        std::vector<std::vector<uint8_t>> slabs(7);
        K::Layer shape;
        shape.capacity = kCapacity;
        shape.hidden = c.hidden;
        shape.intermediate = c.intermediate;
        shape.activation = 0;
        shape.act_limit = c.act_limit;
        shape.slab_count = 7;
        for (int s = 0; s < 7; ++s) {
            if (s == 6 && !c.up_alpha) continue;
            slabs[s].resize(stride[s] * kCapacity);
            for (size_t i = 0; i < slabs[s].size(); ++i) {
                if (s < 2) slabs[s][i] = uint8_t(rng());
                else if (s < 4) slabs[s][i] = uint8_t((rnd(2) << 7) | ((5 + rnd(4)) << 3) | rnd(8));  // finite E4M3
            }
            if (s >= 4)
                for (int slot = 0; slot < kCapacity; ++slot) {
                    const float a = 0.01f * float(1 + slot + s);
                    std::memcpy(&slabs[s][size_t(slot) * stride[s]], &a, 4);
                }
            shape.slabs[s] = slabs[s].data();
            shape.slot_bytes[s] = stride[s];
        }
        shape.params.w13_layout = c.layout;
        shape.params.inv_input_scale13 = 0.5f;
        shape.params.inv_input_scale2 = 0.75f;
        const K::Layer layer = K::make_layer(shape);

        Team one({}, 1, &K::warm, 0), many({}, workers, &K::warm, 0), barrier_team({}, workers, &K::warm, 0);
        JobRunner<K> single(one), dynamic(many);
        for (int iter = 0; iter < c.iterations; ++iter) {
            const int rows = 1 + rnd(9), k = 1 + rnd(8);
            std::vector<uint16_t> x(size_t(rows) * c.hidden);
            for (uint16_t& v : x) v = fp16_of((float(rnd(2001)) - 1000.f) / 500.f);
            const bool bad_input = iter % 11 == 10;
            if (bad_input) x[size_t(rnd(rows)) * c.hidden + size_t(rnd(c.hidden))] = 0x7c00;  // +inf
            std::vector<int32_t> slots(size_t(rows) * k);
            std::vector<float> weights(slots.size());
            const int pool = 1 + rnd(kCapacity);
            for (size_t i = 0; i < slots.size(); ++i) {
                slots[i] = rnd(8) == 0 ? -1 : rnd(pool);
                weights[i] = rnd(10) == 0 ? 0.f : float(1 + rnd(100)) / 64.f;
            }
            std::vector<float> start(size_t(rows) * c.hidden);
            for (float& o : start) o = (float(rnd(1000)) - 500.f) / 16.f;

            ForwardCall call;
            call.rows = rows;
            call.k = k;
            call.x = x.data();
            call.slots = slots.data();
            call.weights = weights.data();
            call.out_mode = iter % 2 == 0 ? OutMode::kOverwrite : OutMode::kAccumulate;
            std::atomic<uint32_t> word{1};
            StageGate gate;
            gate.word = &word;
            gate.open = 1;
            if (iter % 4 == 3) call.stage_two = &gate;
            call.out = start.data();
            K::check(layer, call);

            std::vector<float> a = start, b = start, d = start;
            if (bad_input) {
                call.out = a.data();
                CHECK(refused([&] { single.run(layer, call); }));
                call.out = b.data();
                CHECK(refused([&] { dynamic.run(layer, call); }));
                call.out = d.data();
                CHECK(refused([&] { K::forward(layer, call, barrier_team); }));
                CHECK(std::memcmp(a.data(), start.data(), a.size() * sizeof(float)) == 0);
                CHECK(std::memcmp(b.data(), start.data(), a.size() * sizeof(float)) == 0);
                CHECK(std::memcmp(d.data(), start.data(), a.size() * sizeof(float)) == 0);
                if (call.stage_two) CHECK(gate.calls == 3);
                ++refusals;
                continue;
            }
            call.out = a.data();
            single.run(layer, call);
            call.out = b.data();
            dynamic.run(layer, call);
            call.out = d.data();
            K::forward(layer, call, barrier_team);
            for (const float v : a) CHECK(std::isfinite(v));
            CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
            CHECK(std::memcmp(a.data(), d.data(), a.size() * sizeof(float)) == 0);
            if (call.stage_two) CHECK(gate.calls == 3);
            ++calls;
        }
    }
    std::printf("ok %ld calls bit-identical across one worker, %d workers and the static split; %ld refused inputs\n",
                calls, workers, refusals);
    return 0;
}
