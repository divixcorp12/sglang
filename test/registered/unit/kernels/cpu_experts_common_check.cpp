// Standalone harness for host/cpu_experts: a toy quant through ExpertForward, as a CpuExpertKernel. Built by
// test_cpu_experts_common.py.
#include "cpu_experts_common_toy.hpp"
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
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

namespace toy {
const sglang::cpu_experts::CpuExpertKernel& toy_kernel()
{
    static const sglang::cpu_experts::ExpertForward<ToyQuant> kernel{};
    return kernel;
}
}  // namespace toy

namespace {
using sglang::cpu_experts::CpuExpertKernel;
using sglang::cpu_experts::ExpertLayer;
using sglang::cpu_experts::ForwardCall;
using sglang::cpu_experts::Isa;

constexpr int kCapacity = 3;
constexpr int kHidden = 16;

int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// What a call threw: 0 nothing, 2 std::invalid_argument, 1 any other std::exception.
template <class F>
int status_of(F&& f)
{
    try {
        f();
        return 0;
    } catch (const std::invalid_argument&) {
        return 2;
    } catch (const std::exception&) {
        return 1;
    }
}

struct Fixture {
    std::vector<float> slab = std::vector<float>(size_t(kCapacity) * kHidden);
    toy::ToyParams params{2.0f};

    Fixture()
    {
        for (int s = 0; s < kCapacity; ++s)
            for (int h = 0; h < kHidden; ++h) slab[size_t(s) * kHidden + h] = float(s + h);
    }

    ExpertLayer layer() const
    {
        ExpertLayer d;
        d.capacity = kCapacity;
        d.hidden = kHidden;
        d.intermediate = kHidden;
        d.slab_count = 1;
        d.slabs[0] = slab.data();
        d.slot_bytes[0] = uint64_t(kHidden) * 4;
        return d;
    }

    std::span<const std::byte> params_bytes() const { return std::as_bytes(std::span<const toy::ToyParams>(&params, 1)); }
};

ExpertLayer make_toy(const Fixture& f) { return toy::toy_kernel().make_layer(f.layer(), f.params_bytes()); }

// One forward's buffers; out starts at 7 so a refused call is seen to leave it untouched.
struct Call {
    const ExpertLayer* layer;
    std::vector<uint16_t> x;
    std::vector<int32_t> slots;
    std::vector<float> weights;
    std::vector<float> out;
    std::vector<int> cores;
    ForwardCall c;

    Call(const ExpertLayer& l, int rows, int k, std::vector<int32_t> s, std::vector<float> w, int threads = 2,
         std::vector<int> on = {})
        : layer(&l), x(size_t(rows) * kHidden), slots(std::move(s)), weights(std::move(w)),
          out(size_t(rows) * kHidden, 7.0f), cores(std::move(on))
    {
        c.rows = rows;
        c.k = k;
        c.threads = threads;
        c.x = x.data();
        c.slots = slots.data();
        c.weights = weights.data();
        c.out = out.data();
        c.cores = cores;
    }
    Call(const Call&) = delete;
    Call& operator=(const Call&) = delete;

    int run(const CpuExpertKernel& kernel = toy::toy_kernel())
    {
        return status_of([&] {
            kernel.check(*layer, c);
            kernel.forward(*layer, c);
        });
    }
    bool untouched() const
    {
        for (float v : out)
            if (v != 7.0f) return false;
        return true;
    }
};

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
    const CpuExpertKernel& kernel = toy::toy_kernel();
    CHECK(std::string(kernel.name()) == "toy");
    const std::vector<int> allowed = allowed_cores();
    const int team = allowed.size() >= 2 ? 2 : 1;
    const std::vector<int> cores(allowed.begin(), allowed.begin() + team);

    {
        const auto layer = make_toy(f);
        CHECK(layer.kernel == &kernel);
        // Token 0: slots 0 and 2; token 1: slot 1 and a skipped -1.
        Call call(layer, 2, 2, {0, 2, 1, -1}, {0.5f, 0.25f, 2.0f, 1.0f}, team, cores);
        for (int accumulate = 0; accumulate < 2; ++accumulate) {
            call.c.accumulate = accumulate != 0;
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
        ok("make_layer_then_forward_overwrites_and_accumulates");
    }

    {
        const ExpertLayer good = f.layer();
        auto refused = [&](const ExpertLayer& d, std::span<const std::byte> p) {
            return status_of([&] { kernel.make_layer(d, p); }) == 2;
        };
        ExpertLayer d = good;
        d.slot_bytes[0] = uint64_t(kHidden) * 4 - 4;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.slabs[0] = nullptr;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.slab_count = 2;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.activation = 1;
        CHECK(refused(d, f.params_bytes()));
        d = good;
        d.capacity = 0;
        CHECK(refused(d, f.params_bytes()));
        CHECK(refused(good, {}));  // the toy needs its params
        const std::byte wide[8]{};
        CHECK(refused(good, wide));  // not sizeof(ToyParams)
        ok("params_and_slabs_are_validated");
    }

    {
        // A second kernel object of the same quant (another library's, in production) refuses this one's layer.
        const sglang::cpu_experts::ExpertForward<toy::ToyQuant> other{};
        const auto layer = make_toy(f);
        Call call(layer, 1, 1, {0}, {1.0f});
        CHECK(call.run(other) == 2);
        CHECK(call.untouched());
        CHECK(call.run() == 0);
        ok("a_layer_of_another_kernel_is_refused");
    }

    {
        const auto layer = make_toy(f);
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const int max_routes = toy::ToyQuant::kMaxRoutes, max_rows = toy::ToyQuant::kMaxRows;
        Call bad_slot(layer, 1, 1, {kCapacity}, {1.0f});
        Call negative_slot(layer, 1, 1, {-2}, {1.0f});
        Call nan_weight(layer, 1, 1, {0}, {nan});
        Call wide(layer, 1, max_routes + 1, std::vector<int32_t>(max_routes + 1, 0),
                  std::vector<float>(max_routes + 1, 1.0f));
        Call tall(layer, max_rows + 1, 1, std::vector<int32_t>(max_rows + 1, 0),
                  std::vector<float>(max_rows + 1, 1.0f));
        Call empty(layer, 1, 1, {0}, {1.0f});
        empty.c.rows = 0;
        Call no_threads(layer, 1, 1, {0}, {1.0f});
        no_threads.c.threads = 0;
        Call null_slots(layer, 1, 1, {0}, {1.0f});
        null_slots.c.slots = nullptr;
        Call null_out(layer, 1, 1, {0}, {1.0f});
        null_out.c.out = nullptr;
        for (Call* c : {&bad_slot, &negative_slot, &nan_weight, &wide, &tall, &empty, &no_threads, &null_slots}) {
            CHECK(c->run() == 2);
            CHECK(c->untouched());
        }
        CHECK(null_out.run() == 2);
        // k = 0 has no routes to read: the output is overwritten with zeros.
        Call no_routes(layer, 1, 0, {}, {});
        no_routes.c.slots = nullptr;
        no_routes.c.weights = nullptr;
        CHECK(no_routes.run() == 0);
        for (float v : no_routes.out) CHECK(v == 0.0f);
        // -1 slots and zero weights are dropped, the routing order kept.
        Call sparse(layer, 1, 4, {1, -1, 2, 0}, {0.0f, 1.0f, 0.5f, 0.25f});
        CHECK(sparse.run() == 0);
        const auto& kept = toy::ToyQuant::last_routes;
        CHECK(kept.size() == 2 && kept[0].slot == 2 && kept[0].weight == 0.5f && kept[1].slot == 0
              && kept[1].weight == 0.25f);
        ok("routes_are_validated");
    }

    {
        // No lock: a forward parked inside its dispatch does not stop another on the same layer.
        const auto layer = make_toy(f);
        toy::ToyQuant::inside.store(false);
        toy::ToyQuant::hold.store(true);
        int first = -1;
        Call held(layer, 1, 1, {0}, {1.0f}, 1, cores);
        std::thread runner([&] {
            toy::ToyQuant::park_here = true;
            first = held.run();
        });
        while (!toy::ToyQuant::inside.load()) std::this_thread::yield();
        Call second(layer, 1, 1, {0}, {1.0f}, 1);
        CHECK(second.run() == 0);
        CHECK(!second.untouched());
        toy::ToyQuant::hold.store(false);
        runner.join();
        CHECK(first == 0);
        ok("forwards_run_at_once");
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

    if (allowed.size() >= 2) {
        // The engine thread may run under a mask that excludes the expert cores (the server runs under taskset): a
        // call accepts a core outside the caller's affinity; only the workers' own pins must succeed.
        const auto layer = make_toy(f);
        cpu_set_t saved, one;
        CHECK(pthread_getaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        CPU_ZERO(&one);
        CPU_SET(allowed[0], &one);
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(one), &one) == 0);
        Call outside(layer, 1, 1, {0}, {1.0f}, 1, {allowed[1]});
        CHECK(outside.run() == 0);
        CHECK(toy::ToyQuant::last_cpus == std::vector<int>{allowed[1]});
        CHECK(pthread_setaffinity_np(pthread_self(), sizeof(saved), &saved) == 0);
        ok("a_core_outside_the_callers_affinity_is_pinned");
    }

    {
        // More workers than cores, a repeated core and a core outside [0, CPU_SETSIZE) are refused, out untouched;
        // without cores the team is unpinned and has no core limit.
        const auto layer = make_toy(f);
        Call too_many(layer, 1, 1, {0}, {1.0f}, team + 1, cores);
        Call repeated(layer, 1, 1, {0}, {1.0f}, 1, {cores[0], cores[0]});
        Call past(layer, 1, 1, {0}, {1.0f}, 1, {CPU_SETSIZE});
        Call negative(layer, 1, 1, {0}, {1.0f}, 1, {-1});
        for (Call* c : {&too_many, &repeated, &past, &negative}) {
            CHECK(c->run() == 2);
            CHECK(c->untouched());
        }
        Call unpinned(layer, 1, 1, {0}, {1.0f}, team + 1);
        CHECK(unpinned.run() == 0);
        ok("cores_are_validated_and_bound_the_team");
    }

    {
        // Each call's team runs on its own cores, also while another call's team runs at once from another thread
        // (the two-team half needs four allowed CPUs).
        const auto layer = make_toy(f);
        Call one(layer, 1, 1, {0}, {1.0f}, team, cores);
        CHECK(one.run() == 0);
        CHECK(toy::ToyQuant::last_cpus == cores);
        if (allowed.size() >= 4) {
            const std::vector<int> a = {allowed[0], allowed[1]}, b = {allowed[2], allowed[3]};
            std::vector<int> seen_a, seen_b;
            std::atomic<int> bad{0};
            auto run = [&](const std::vector<int>& on, std::vector<int>* seen) {
                for (int i = 0; i < 200; ++i) {
                    Call call(layer, 1, 1, {0}, {1.0f}, 2, on);
                    if (call.run() != 0) bad.fetch_add(1);
                    seen->insert(seen->end(), toy::ToyQuant::last_cpus.begin(), toy::ToyQuant::last_cpus.end());
                }
            };
            std::thread ta(run, std::cref(a), &seen_a), tb(run, std::cref(b), &seen_b);
            ta.join();
            tb.join();
            CHECK(bad.load() == 0);
            CHECK(seen_a.size() == 400 && seen_b.size() == 400);
            for (int cpu : seen_a) CHECK(cpu == a[0] || cpu == a[1]);
            for (int cpu : seen_b) CHECK(cpu == b[0] || cpu == b[1]);
        }
        ok("each_calls_team_runs_on_its_own_cores");
    }

    {
        // A core in range but past this machine's CPUs passes validation and fails the worker's pin: a
        // std::runtime_error, out untouched; the next call runs.
        const auto layer = make_toy(f);
        Call fails(layer, 1, 1, {0}, {1.0f}, 1, {CPU_SETSIZE - 1});
        CHECK(fails.run() == 1);
        CHECK(fails.untouched());
        Call fits(layer, 1, 1, {0}, {1.0f}, team, cores);
        CHECK(fits.run() == 0);
        ok("a_failed_pin_throws_runtime_error_and_leaves_out_untouched");
    }

    {
        uint32_t word = 5;
        auto warm = [&](std::span<const int> on, int32_t threads, const uint32_t* w, int64_t deadline) {
            return status_of([&] { kernel.keep_warm(on, threads, w, 5, deadline); });
        };
        CHECK(warm(cores, 0, &word, now_ns() + 1000000000) == 2);
        CHECK(warm(cores, 1, nullptr, now_ns() + 1000000000) == 2);
        CHECK(warm(cores, team + 1, &word, now_ns() + 1000000000) == 2);
        CHECK(warm({}, team + 1, &word, now_ns() - 1) == 0);  // no cores: no core limit
        // An expired deadline returns at the first clock poll.
        CHECK(warm(cores, team, &word, now_ns() - 1) == 0);
        const int64_t start = now_ns();
        std::thread mover([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            __atomic_store_n(&word, 6u, __ATOMIC_RELEASE);
        });
        const int r = warm(cores, team, &word, start + 60LL * 1000000000);
        mover.join();
        CHECK(r == 0);
        CHECK(now_ns() - start < 1000000000);
        // Every tier's loop, as far as this CPU runs them, through the free function.
        for (Isa tier : {Isa::Scalar, Isa::Avx2, Isa::Bw, Isa::Vnni, Isa::Vbmi}) {
            if (tier > sglang::cpu_experts::detect_isa(Isa::Vbmi, nullptr)) break;
            sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, {}, team, &word, 6, now_ns() + 2000000);
            sglang::cpu_experts::keep_warm<Isa::Vbmi>(tier, {}, team, &word, 5, now_ns() + 1000000000);
        }
        ok("keep_warm_returns_when_the_word_moves");
    }

    std::printf("all ok\n");
    return 0;
}
