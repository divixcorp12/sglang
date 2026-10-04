// Standalone harness for host/cpu_experts: a toy quant through ExpertForward. Built by test_cpu_experts_common.py.
#include "cpu_experts_common_toy.hpp"
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

SGLANG_CPU_EXPERTS_DEFINE_CABI(toy, toy::ToyQuant)

namespace {
using sglang::cpu_experts::Isa;

constexpr int kCapacity = 3;
constexpr int kHidden = 16;

int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Fixture {
    std::vector<float> slab = std::vector<float>(size_t(kCapacity) * kHidden);
    toy::ToyParams params{2.0f};

    Fixture()
    {
        for (int s = 0; s < kCapacity; ++s)
            for (int h = 0; h < kHidden; ++h) slab[size_t(s) * kHidden + h] = float(s + h);
    }

    SglangCpuExpertsLayer layer() const
    {
        SglangCpuExpertsLayer d{};
        d.abi_version = SGLANG_CPU_EXPERTS_LAYER_ABI_VERSION;
        d.capacity = kCapacity;
        d.hidden = kHidden;
        d.intermediate = kHidden;
        d.activation = 0;
        d.act_limit = 0;
        d.slab_count = 1;
        d.slabs[0] = slab.data();
        d.slot_bytes[0] = uint64_t(kHidden) * 4;
        d.params = &params;
        return d;
    }
};

struct Call {
    std::vector<uint16_t> x;
    std::vector<int32_t> slots;
    std::vector<float> weights;
    std::vector<float> out;
    SglangCpuExpertsForward c{};

    Call(int64_t handle, int rows, int k, std::vector<int32_t> s, std::vector<float> w, int threads = 2)
        : x(size_t(rows) * kHidden), slots(std::move(s)), weights(std::move(w)), out(size_t(rows) * kHidden, 7.0f)
    {
        c.abi_version = SGLANG_CPU_EXPERTS_FORWARD_ABI_VERSION;
        c.rows = rows;
        c.layer = handle;
        c.x = x.data();
        c.slots = slots.data();
        c.weights = weights.data();
        c.out = out.data();
        c.k = k;
        c.threads = threads;
        c.accumulate = 0;
    }

    int run() { return sglang_toy_cpu_experts_forward(&c); }
    bool untouched() const
    {
        for (float v : out)
            if (v != 7.0f) return false;
        return true;
    }
};

int64_t register_toy(const Fixture& f)
{
    const SglangCpuExpertsLayer d = f.layer();
    int64_t handle = -1;
    CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 0);
    CHECK(handle >= 0);
    return handle;
}

std::vector<int> allowed_cores()
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CHECK(sched_getaffinity(0, sizeof(set), &set) == 0);
    std::vector<int> cores;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &set)) cores.push_back(c);
    return cores;
}

void ok(const char* name) { std::printf("ok %s\n", name); std::fflush(stdout); }

}  // namespace

int main()
{
    // isa() is computed once, at the first forward: the cap and the report switch must be in the environment before
    // it. The Python test reads the report ("toy isa scalar") from stderr.
    CHECK(setenv("TOY_CPU_MAX_ISA", "scalar", 1) == 0);
    CHECK(setenv("TOY_CPU_REPORT_ISA", "1", 1) == 0);
    Fixture f;

    // Configure the worker cores before the first forward freezes them; checked again after it.
    const std::vector<int> allowed = allowed_cores();
    const int32_t team = allowed.size() >= 2 ? 2 : 1;
    const std::vector<int32_t> cores(allowed.begin(), allowed.begin() + team);
    CHECK(sglang_toy_cpu_experts_set_cores(nullptr, 1) == 2);
    if (team == 2) {
        const int32_t twice[2] = {cores[0], cores[0]};
        CHECK(sglang_toy_cpu_experts_set_cores(twice, 2) == 2);
    }
    {
        // The engine configures from a thread whose mask may exclude the expert cores (the server runs under taskset):
        // set_cores accepts a core outside the caller's affinity; only the workers' own pins must succeed.
        cpu_set_t saved, one;
        CHECK(pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        CPU_ZERO(&one);
        CPU_SET(allowed[0], &one);
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0);
        const int32_t outside = allowed.size() >= 2 ? allowed[1] : allowed[0] + 1;
        CHECK(outside < CPU_SETSIZE && !CPU_ISSET(outside, &one));
        CHECK(sglang_toy_cpu_experts_set_cores(&outside, 1) == 0);
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        ok("set_cores_accepts_a_core_outside_the_callers_affinity");
    }
    // Before the first forward a later configuration replaces the earlier one.
    CHECK(sglang_toy_cpu_experts_set_cores(cores.data(), team) == 0);

    {
        const int64_t h = register_toy(f);
        // Token 0: slots 0 and 2; token 1: slot 1 and a skipped -1.
        Call call(h, 2, 2, {0, 2, 1, -1}, {0.5f, 0.25f, 2.0f, 1.0f}, team);
        for (int accumulate = 0; accumulate < 2; ++accumulate) {
            call.c.accumulate = accumulate;
            CHECK(call.run() == 0);
            for (int hh = 0; hh < kHidden; ++hh) {
                const float base = accumulate ? 1.0f : 0.0f;
                const float t0 = 0.5f * 2.0f * float(0 + hh) + 0.25f * 2.0f * float(2 + hh);
                const float t1 = 2.0f * 2.0f * float(1 + hh);
                CHECK(call.out[hh] == base + t0);
                CHECK(call.out[kHidden + hh] == base + t1);
            }
            std::fill(call.out.begin(), call.out.end(), 1.0f);
        }
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        ok("register_then_forward_overwrites_and_accumulates");
    }

    {
        SglangCpuExpertsLayer d = f.layer();
        d.abi_version += 1;
        int64_t handle = -1;
        CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 2);
        CHECK(handle == -1);
        const int64_t h = register_toy(f);
        Call call(h, 1, 1, {0}, {1.0f});
        call.c.abi_version += 1;
        CHECK(call.run() == 2);
        CHECK(call.untouched());
        CHECK(sglang_toy_cpu_experts_forward(nullptr) == 2);
        CHECK(sglang_toy_cpu_experts_register_layer(nullptr, &handle) == 2);
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        ok("abi_versions_are_checked");
    }

    {
        SglangCpuExpertsLayer d = f.layer();
        d.slot_bytes[0] = uint64_t(kHidden) * 4 - 4;
        int64_t handle = -1;
        CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 2);
        d = f.layer();
        d.slabs[0] = nullptr;
        CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 2);
        d = f.layer();
        d.slab_count = 2;
        CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 2);
        d = f.layer();
        d.activation = 1;
        CHECK(sglang_toy_cpu_experts_register_layer(&d, &handle) == 2);
        CHECK(handle == -1);
        ok("slot_bytes_below_the_minimum_are_refused");
    }

    {
        Call unknown(99, 1, 1, {0}, {1.0f});
        CHECK(unknown.run() == 2);
        CHECK(unknown.untouched());
        const int64_t h = register_toy(f);
        Call call(h, 1, 1, {0}, {1.0f});
        CHECK(call.run() == 0);
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        std::fill(call.out.begin(), call.out.end(), 7.0f);
        CHECK(call.run() == 2);
        CHECK(call.untouched());
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 2);
        CHECK(sglang_toy_cpu_experts_free_layer(-1) == 2);
        // Handles are never reused: a new registration does not revive the freed one.
        const int64_t next = register_toy(f);
        CHECK(next != h);
        CHECK(call.run() == 2);
        CHECK(sglang_toy_cpu_experts_free_layer(next) == 0);
        ok("unknown_handle_is_refused");
    }

    {
        const int64_t h = register_toy(f);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const int max_routes = toy::ToyQuant::kMaxRoutes, max_rows = toy::ToyQuant::kMaxRows;
        Call bad_slot(h, 1, 1, {kCapacity}, {1.0f});
        Call negative_slot(h, 1, 1, {-2}, {1.0f});
        Call nan_weight(h, 1, 1, {0}, {nan});
        Call wide(h, 1, max_routes + 1, std::vector<int32_t>(max_routes + 1, 0),
                  std::vector<float>(max_routes + 1, 1.0f));
        Call tall(h, max_rows + 1, 1, std::vector<int32_t>(max_rows + 1, 0), std::vector<float>(max_rows + 1, 1.0f));
        Call empty(h, 1, 1, {0}, {1.0f});
        empty.c.rows = 0;
        Call no_threads(h, 1, 1, {0}, {1.0f});
        no_threads.c.threads = 0;
        Call bad_accumulate(h, 1, 1, {0}, {1.0f});
        bad_accumulate.c.accumulate = 2;
        Call null_slots(h, 1, 1, {0}, {1.0f});
        null_slots.c.slots = nullptr;
        for (Call* c : {&bad_slot, &negative_slot, &nan_weight, &wide, &tall, &empty, &no_threads, &bad_accumulate,
                        &null_slots}) {
            CHECK(c->run() == 2);
            CHECK(c->untouched());
        }
        // k = 0 has no routes to read: the output is overwritten with zeros.
        Call no_routes(h, 1, 0, {}, {});
        no_routes.c.slots = nullptr;
        no_routes.c.weights = nullptr;
        CHECK(no_routes.run() == 0);
        for (float v : no_routes.out) CHECK(v == 0.0f);
        // -1 slots and zero weights are dropped, the routing order kept.
        Call sparse(h, 1, 4, {1, -1, 2, 0}, {0.0f, 1.0f, 0.5f, 0.25f});
        CHECK(sparse.run() == 0);
        const auto& kept = toy::ToyQuant::last_routes;
        CHECK(kept.size() == 2 && kept[0].slot == 2 && kept[0].weight == 0.5f && kept[1].slot == 0
              && kept[1].weight == 0.25f);
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        ok("routes_are_validated");
    }

    {
        const int64_t h = register_toy(f);
        toy::ToyQuant::inside.store(false);
        toy::ToyQuant::hold.store(true);
        int first = -1;
        Call held(h, 1, 1, {0}, {1.0f}, 1);
        std::thread runner([&] { first = held.run(); });
        while (!toy::ToyQuant::inside.load()) std::this_thread::yield();
        Call second(h, 1, 1, {0}, {1.0f}, 1);
        CHECK(second.run() == 3);
        CHECK(second.untouched());
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 3);
        toy::ToyQuant::hold.store(false);
        runner.join();
        CHECK(first == 0);
        CHECK(second.run() == 0);
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        ok("concurrent_forward_returns_3");
    }

    {
        CHECK(toy::ToyQuant::last_isa == Isa::Scalar);
        CHECK(sglang::cpu_experts::ExpertForward<toy::ToyQuant>::isa() == Isa::Scalar);
        const Isa hw = sglang::cpu_experts::detect_isa(Isa::Vbmi, nullptr);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Avx2, nullptr) <= Isa::Avx2);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Scalar, nullptr) == Isa::Scalar);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_MAX_ISA") == Isa::Scalar);
        CHECK(setenv("TOY_CPU_BOGUS_ISA", "avx9000", 1) == 0);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_BOGUS_ISA") == hw);
        // A cap above the hardware never raises the tier.
        CHECK(setenv("TOY_CPU_HIGH_ISA", "VBMI", 1) == 0);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Vbmi, "TOY_CPU_HIGH_ISA") == hw);
        CHECK(sglang::cpu_experts::detect_isa(Isa::Avx2, "TOY_CPU_HIGH_ISA") <= Isa::Avx2);
        std::printf("detected %d\n", int(hw));
        ok("isa_cap_env_lowers_the_tier");
    }

    {
        // Worker 0 is the caller, now pinned: its own core is still allowed, so 2 can only mean frozen.
        const std::vector<int> still_allowed = allowed_cores();
        const int32_t own = still_allowed.at(0);
        CHECK(sglang_toy_cpu_experts_set_cores(&own, 1) == 2);
        CHECK(sglang::cpu_experts::Cores::frozen().size() == size_t(team));
        // A team larger than the frozen cores is a kernel error, not a silent oversubscription.
        const int64_t h = register_toy(f);
        Call call(h, 1, 1, {0}, {1.0f}, team + 1);
        CHECK(call.run() == 1);
        ok("set_cores_after_the_first_forward_returns_2");
        // Status 1 keeps its reason on this thread, for a quant's own wrappers; the next call clears it.
        CHECK(sglang::cpu_experts::last_error().find("exceeds configured cores") != std::string::npos);
        Call fits(h, 1, 1, {0}, {1.0f}, team);
        CHECK(fits.run() == 0);
        CHECK(sglang::cpu_experts::last_error().empty());
        CHECK(sglang_toy_cpu_experts_free_layer(h) == 0);
        ok("last_error_names_why_a_call_failed");
    }

    {
        uint32_t word = 5;
        CHECK(sglang_toy_cpu_experts_keep_warm(0, &word, 5, now_ns() + 1000000000) == 2);
        CHECK(sglang_toy_cpu_experts_keep_warm(1, nullptr, 5, now_ns() + 1000000000) == 2);
        CHECK(sglang_toy_cpu_experts_keep_warm(team + 1, &word, 5, now_ns() + 1000000000) == 2);
        // An expired deadline returns at the first clock poll.
        CHECK(sglang_toy_cpu_experts_keep_warm(team, &word, 5, now_ns() - 1) == 0);
        const int64_t start = now_ns();
        std::thread mover([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            __atomic_store_n(&word, 6u, __ATOMIC_RELEASE);
        });
        const int r = sglang_toy_cpu_experts_keep_warm(team, &word, 5, start + 60LL * 1000000000);
        mover.join();
        CHECK(r == 0);
        CHECK(now_ns() - start < 1000000000);
        // Every tier's loop, as far as this CPU runs them, through the free function.
        for (Isa tier : {Isa::Scalar, Isa::Avx2, Isa::Bw, Isa::Vnni, Isa::Vbmi}) {
            if (tier > sglang::cpu_experts::detect_isa(Isa::Vbmi, nullptr)) break;
            CHECK(sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, team, &word, 6, now_ns() + 2000000) == 0);
            CHECK(sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, team, &word, 5, now_ns() + 1000000000) == 0);
        }
        ok("keep_warm_returns_when_the_word_moves");
    }

    std::printf("all ok\n");
    return 0;
}
