// Bit-exact A/B harness for the NVFP4 CPU expert kernel; run_nvfp4_cpu_forward_checks.sh builds and runs it.
// It calls only the C ABI (cpu_experts_cabi.h), so the same source runs against any kernel revision. Every output and
// status is written to OUT in a fixed order: two revisions agree when their files are byte-identical.
//   nvfp4_cpu_forward_ab OUT CORE [CORE...]     worker i runs on CORE i; the caller is worker 0
#include "cpu_experts_cabi.h"
#include "../upstream/kernels.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    if (sglang_nvfp4_cpu_experts_set_cores(cores.data(), int32_t(cores.size())) != 0) {
        std::fprintf(stderr, "set_cores refused\n"); return 1;
    }
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) return 1;
    // Descending: the pre-OpenMP pool fixes its size at the first forward and refuses a larger later one.
    std::vector<int> teams = {int(cores.size())};
    for (int t : {3, 1}) if (t < teams.back()) teams.push_back(t);
    std::mt19937 rng(20261003);
    int cases = 0, unexpected = 0;
    for (const Config& c : kConfigs) {
        Slabs s = make_slabs(c, rng);
        SglangNvfp4CpuLayer d{};
        d.abi_version = 1; d.capacity = kCapacity; d.hidden = c.hidden; d.intermediate = c.intermediate;
        d.w13_layout = c.layout; d.activation = 0; d.act_limit = c.limit;
        d.inv_input_scale13 = c.inv13; d.inv_input_scale2 = c.inv2;
        for (int i = 0; i < 7; ++i) {
            d.slabs[i] = s.bytes[i].empty() ? nullptr : s.bytes[i].data();
            d.slot_bytes[i] = s.stride[i];
        }
        int64_t handle = -1;
        if (sglang_nvfp4_cpu_experts_register_slabs(&d, &handle) != 0) {
            std::fprintf(stderr, "%s: registration refused\n", c.name); return 1;
        }
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        std::vector<uint16_t> x(c.hidden);
        for (auto& v : x) v = ggml_compute_fp32_to_fp16(unit(rng));
        for (int threads : teams)
            for (const Routing& r : kRoutings)
                for (int accumulate : {0, 1}) {
                    std::vector<float> out(c.hidden);
                    for (int i = 0; i < c.hidden; ++i) out[i] = accumulate ? .001f * float(i) - .04f : 777.f;
                    const int status = sglang_nvfp4_cpu_experts_forward(
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
        int status = sglang_nvfp4_cpu_experts_forward(handle, bad.data(), &slot, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/nonfinite_x", status, out);
        out.assign(c.hidden, 777.f);
        status = sglang_nvfp4_cpu_experts_forward(handle, x.data(), &past, &weight, 1, out.data(), teams[0], 0);
        unexpected += status != 2;
        put(f, std::string(c.name) + "/slot_past_capacity", status, out);
        cases += 2;
        if (sglang_nvfp4_cpu_experts_free_layer(handle) != 0) { std::fprintf(stderr, "free refused\n"); return 1; }
    }
    std::fclose(f);
    std::printf("%d cases, %d unexpected statuses\n", cases, unexpected);
    return unexpected ? 1 : 0;
}
