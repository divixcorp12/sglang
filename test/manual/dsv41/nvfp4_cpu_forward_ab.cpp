// Bit-exact A/B harness for the NVFP4 CPU expert kernel; run_nvfp4_cpu_forward_checks.sh builds and runs it.
// It calls only the kernel interface (nvfp4_cpu_kernel(), kernel.h). Every one-row output and status (status_of) is
// written to OUT in a fixed order: two
// revisions agree when their files are byte-identical, each built with its own revision of this harness.
//   nvfp4_cpu_forward_ab OUT CORE [CORE...]     worker i runs on CORE i; the caller is worker 0
#include "kernel.h"
#include "quant.hpp"
#include "../upstream/kernels.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Config {
    const char* name;
    int hidden, intermediate, layout;
    float limit, inv13, inv2;
    bool up_alpha;
};
// Ordered so each small layer reuses scratch a 5120/6144-wide layer dirtied: the 80/144-wide layers' 64-column tails
// must still read zeros. mimo_v26_pro* are MiMo V2.6 Pro's routed expert (6144/2048, no clamp: the specialized plan);
// mimo_v26_pro_limit10 is its shape with a clamp, and h5120_n2304_* another large shape (both the generic plan).
const Config kConfigs[] = {
    {"mimo_v26_pro", 6144, 2048, 0, 0.f, 1.f, 1.f, false},
    {"h80_n80_l0", 80, 80, 0, 0.f, 1.f, 1.f, false},
    {"mimo_v26_pro_l2_up_scaled", 6144, 2048, 2, 0.f, .5f, .25f, true},
    {"h80_n80_l1_lim", 80, 80, 1, 2.5f, .5f, .25f, true},
    {"mimo_v26_pro_limit10", 6144, 2048, 0, 10.f, 1.f, 1.f, false},
    {"h5120_n2304_lim10", 5120, 2304, 0, 10.f, 1.f, 1.f, false},
    {"h256_n192_l2", 256, 192, 2, 0.f, 1.f, 1.f, true},
    {"h5120_n2304_l2_up_scaled", 5120, 2304, 2, 0.f, .5f, .25f, true},
    {"h144_n128_l0_lim", 144, 128, 0, 10.f, .75f, 1.f, false},
};
struct Routing {
    const char* name;
    std::vector<int32_t> slots;
    std::vector<float> weights;
};
const Routing kRoutings[] = {
    {"k0", {}, {}},
    {"k1", {3}, {.7f}},
    {"k3", {0, 5, 2}, {.5f, .3f, .2f}},
    {"k5_skip_dup_zero", {1, -1, 4, 6, 1}, {.4f, .9f, 0.f, .3f, .2f}},
    {"k8_negzero", {0, 1, 2, 3, 4, 5, 6, 7}, {.125f, -.25f, .5f, -0.f, .0625f, 1.f, .3f, .2f}},
};
constexpr int kCapacity = 8;
constexpr uint64_t kPad = 64;           // every weight/scale stride is its minimum plus this: strides are honored
constexpr uint64_t kAlphaStride = 8;    // fp32 alpha plus 4 bytes of padding

uint64_t round_up(uint64_t x, uint64_t n) { return (x + n - 1) / n * n; }

// A finite signed E4M3 scale in [2^-2, 2^1].
uint8_t scale(std::mt19937& rng) {
    const unsigned e = 5 + rng() % 4, m = rng() % 8, s = rng() & 1;
    return uint8_t(s << 7 | e << 3 | m);
}

struct Slabs {
    std::vector<uint8_t> bytes[7];
    uint64_t stride[7]{};
};

Slabs make_slabs(const Config& c, std::mt19937& rng) {
    const uint64_t h = c.hidden, n = c.intermediate;
    const uint64_t minimum[4] = {n * h, h * n / 2, round_up(2 * n, 128) * round_up(h / 16, 4),
                                 round_up(h, 128) * round_up(n / 16, 4)};
    Slabs s;
    for (int i = 0; i < 4; ++i) {
        s.stride[i] = minimum[i] + kPad;
        s.bytes[i].resize(s.stride[i] * kCapacity);
        for (auto& b : s.bytes[i]) b = i < 2 ? uint8_t(rng()) : scale(rng);
    }
    const float base[3] = {.01f, .015f, .02f};  // gate, down, up
    for (int i = 4; i < 7; ++i) {
        if (i == 6 && !c.up_alpha) continue;
        s.stride[i] = kAlphaStride;
        s.bytes[i].assign(kAlphaStride * kCapacity, 0);
        for (int slot = 0; slot < kCapacity; ++slot) {
            const float a = base[i - 4] * float(1 + slot);
            std::memcpy(s.bytes[i].data() + slot * kAlphaStride, &a, sizeof(a));
        }
    }
    return s;
}

std::vector<int> g_cores;  // the cores main was given, carried by every forward
using Layer = ::sglang::cpu_experts::CpuExpertLayer;

// The C ABI's status for what a kernel call threw: 0 none, 2 std::invalid_argument, 1 any other exception.
template <class F>
int status_of(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::invalid_argument&) {
        return 2;
    } catch (const std::exception&) {
        return 1;
    }
}

int forward(const Layer& layer, const void* x, const int32_t* slots, const float* weights, int32_t k, float* out,
            int32_t threads, int32_t accumulate, int32_t rows = 1) {
    ::sglang::cpu_experts::ForwardCall call;
    call.rows = rows; call.x = x; call.slots = slots; call.weights = weights;
    call.out = out; call.k = k; call.threads = threads; call.accumulate = accumulate != 0;
    call.cores = g_cores;
    return status_of([&] {
        const auto& kernel = ::sglang::nvfp4_cpu::nvfp4_cpu_kernel();
        kernel.check(layer, call);
        kernel.forward(layer, call);
    });
}

// Rows token rows in one call against one call per row, bitwise; returns the number of rows that differ or fail.
// Token t takes routing kRoutings[(t + shift) % 5] padded with -1 slots to k 8, or k8_negzero for all when shift < 0
// (every slot shared by every token: chunks of 4, 4 and 1 for 9 rows).
int check_batch(const Layer& handle, int hidden, int threads, int rows, int shift, std::mt19937& rng) {
    constexpr int k = 8;
    std::uniform_real_distribution<float> unit(-1.f, 1.f);
    std::vector<uint16_t> x(size_t(rows) * hidden);
    for (auto& v : x) v = ggml_compute_fp32_to_fp16(unit(rng));
    std::vector<int32_t> slots(size_t(rows) * k, -1);
    std::vector<float> weights(size_t(rows) * k, 1.f);
    for (int t = 0; t < rows; ++t) {
        const Routing& r = kRoutings[shift < 0 ? 4 : (t + shift) % 5];
        std::copy(r.slots.begin(), r.slots.end(), slots.begin() + t * k);
        std::copy(r.weights.begin(), r.weights.end(), weights.begin() + t * k);
    }
    int bad = 0;
    for (int accumulate : {0, 1}) {
        std::vector<float> batched(size_t(rows) * hidden), single;
        for (size_t i = 0; i < batched.size(); ++i) batched[i] = accumulate ? .001f * float(i % 97) - .04f : 777.f;
        single = batched;
        bad += forward(handle, x.data(), slots.data(), weights.data(), k, batched.data(), threads, accumulate, rows) != 0;
        for (int t = 0; t < rows; ++t) {
            const size_t o = size_t(t) * hidden;
            bad += forward(handle, x.data() + o, slots.data() + t * k, weights.data() + t * k, k, single.data() + o,
                           threads, accumulate) != 0;
            bad += std::memcmp(batched.data() + o, single.data() + o, hidden * sizeof(float)) != 0;
        }
    }
    return bad;
}

void put(FILE* f, const std::string& name, int status, const std::vector<float>& out) {
    const uint32_t length = uint32_t(name.size()), count = uint32_t(out.size());
    std::fwrite(&length, 4, 1, f);
    std::fwrite(name.data(), 1, length, f);
    std::fwrite(&status, 4, 1, f);
    std::fwrite(&count, 4, 1, f);
    std::fwrite(out.data(), 4, count, f);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s OUT CORE [CORE...]\n", argv[0]); return 2; }
    std::vector<int32_t> cores;
    for (int i = 2; i < argc; ++i) cores.push_back(std::atoi(argv[i]));
    g_cores.assign(cores.begin(), cores.end());
    if (status_of([&] { ::sglang::cpu_experts::check_cores(g_cores); }) != 0) {
        std::fprintf(stderr, "cores refused\n"); return 1;
    }
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return 1;
    // Descending: the pre-OpenMP pool fixes its size at the first forward and refuses a larger later one.
    std::vector<int> teams = {int(cores.size())};
    for (int t : {3, 1}) if (t < teams.back()) teams.push_back(t);
    std::mt19937 rng(20261003);
    std::mt19937 batch_rng(20261004);  // separate, so the dumped cases draw what they always drew
    int cases = 0, unexpected = 0, batch_mismatches = 0;
    for (const Config& c : kConfigs) {
        Slabs s = make_slabs(c, rng);
        SglangNvfp4CpuParams params{};
        params.w13_layout = c.layout; params.inv_input_scale13 = c.inv13; params.inv_input_scale2 = c.inv2;
        ::sglang::cpu_experts::LayerSlabs d;
        d.capacity = kCapacity; d.hidden = c.hidden;
        d.intermediate = c.intermediate; d.activation = 0; d.act_limit = c.limit; d.slab_count = 7;
        for (int i = 0; i < 7; ++i) {
            d.slabs[i] = s.bytes[i].empty() ? nullptr : s.bytes[i].data();
            d.slot_bytes[i] = s.stride[i];
        }
        std::unique_ptr<Layer> layer;
        if (status_of([&] {
                layer = ::sglang::nvfp4_cpu::nvfp4_cpu_kernel().make_layer(
                    d, std::as_bytes(std::span<const SglangNvfp4CpuParams>(&params, 1)));
            }) != 0) {
            std::fprintf(stderr, "%s: registration refused\n", c.name); return 1;
        }
        const Layer& handle = *layer;
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        std::vector<uint16_t> x(c.hidden);
        for (auto& v : x) v = ggml_compute_fp32_to_fp16(unit(rng));
        for (int threads : teams)
            for (const Routing& r : kRoutings)
                for (int accumulate : {0, 1}) {
                    std::vector<float> out(c.hidden);
                    for (int i = 0; i < c.hidden; ++i) out[i] = accumulate ? .001f * float(i) - .04f : 777.f;
                    const int status = forward(
                        handle, x.data(), r.slots.empty() ? nullptr : r.slots.data(),
                        r.weights.empty() ? nullptr : r.weights.data(), int32_t(r.slots.size()), out.data(),
                        threads, accumulate);
                    unexpected += status != 0;
                    put(f, std::string(c.name) + "/t" + std::to_string(threads) + "/" + r.name + "/acc" +
                               std::to_string(accumulate), status, out);
                    ++cases;
                }
        // Refusals: an infinite input element, and a slot past the capacity. Both must leave out untouched.
        std::vector<uint16_t> bad = x;
        bad[7] = 0x7c00;
        const int32_t slot = 2, past = kCapacity;
        const float weight = .5f;
        std::vector<float> out(c.hidden, 777.f);
        int status = forward(handle, bad.data(), &slot, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/nonfinite_x", status, out);
        out.assign(c.hidden, 777.f);
        status = forward(handle, x.data(), &past, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/slot_past_capacity", status, out);
        cases += 2;
        // Batches are checked against this run's own one-row calls and never written to OUT, which stays comparable
        // with a revision that has no batched forward.
        for (int threads : teams)
            for (int shift : {0, 2, -1}) batch_mismatches += check_batch(handle, c.hidden, threads, 9, shift, batch_rng);
    }
    std::fclose(f);
    std::printf("%d cases, %d unexpected statuses, %d batched rows differing from one-row calls\n", cases, unexpected,
                batch_mismatches);
    return unexpected || batch_mismatches ? 1 : 0;
}
