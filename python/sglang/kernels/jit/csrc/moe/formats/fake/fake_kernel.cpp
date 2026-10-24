// The fake CPU expert kernel (fake_kernel.h) of the test format `fake`.
#include "fake_kernel.h"

#include "../../expert_stream/host/cpu_experts/team.hpp"

#include <immintrin.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

namespace sglang::formats::fake {
namespace {
using ::sglang::cpu_experts::ForwardCall;
using ::sglang::cpu_experts::team_now_ns;

std::mutex g_mutex;
std::vector<FakeCpuKernel::Call> g_calls;
std::atomic<int64_t> g_ns{0}, g_fail{0}, g_held_core{-1};
std::atomic<bool> g_zero{false};
std::atomic<int32_t> g_max_rows{1 << 16};

// The job interface's controls and records. g_test is set between jobs (job_reset) and read by the workers after the
// scheduler's release of a phase, so it needs no atomics.
FakeCpuKernel::JobTest g_test;
std::atomic<int32_t> g_ordinal{0};
std::atomic<int64_t> g_violations{0};
constexpr uint32_t kMaxEvents = 1u << 16;
std::vector<FakeCpuKernel::JobEvent> g_events(kMaxEvents);
std::atomic<uint32_t> g_event_next{0};

// The bytes of the live slots of call row t in slabs [first, last) of a six-slab layer, summed; 0 otherwise.
double slab_sum(const FakeCpuKernel::Layer& layer, const ForwardCall& c, int32_t t, int first, int last)
{
    if (layer.slab_count != 6) return 0;
    double sum = 0;
    for (int32_t i = 0; i < c.k; ++i) {
        const int32_t slot = c.slots[static_cast<int64_t>(t) * c.k + i];
        if (slot < 0) continue;
        const ::sglang::cpu_experts::ExpertRow row = layer[slot];
        for (int name = first; name < last; ++name)
            for (uint64_t b = 0; b < layer.slot_bytes[name]; ++b)
                sum += row.slab[name][b];
    }
    return sum;
}
}  // namespace

const char* FakeCpuKernel::name() noexcept { return "fake"; }
FakeCpuKernel::Params FakeCpuKernel::params(std::span<const int64_t>) { return {}; }
FakeCpuKernel::Layer FakeCpuKernel::make_layer(const Layer& shape) { return shape; }
int32_t FakeCpuKernel::max_routes() noexcept { return kFakeMaxLanes; }
int32_t FakeCpuKernel::max_rows() noexcept { return g_max_rows.load(std::memory_order_relaxed); }
void FakeCpuKernel::set_max_rows(int32_t rows) { g_max_rows.store(rows, std::memory_order_relaxed); }
void FakeCpuKernel::check(const Layer&, const ForwardCall&) {}

void FakeCpuKernel::reset(int64_t ns_per_expert, int64_t fail, bool zero)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    g_calls.clear();
    g_ns.store(ns_per_expert, std::memory_order_relaxed);
    g_fail.store(fail, std::memory_order_relaxed);
    g_zero.store(zero, std::memory_order_relaxed);
    g_held_core.store(-1, std::memory_order_release);
    g_max_rows.store(1 << 16, std::memory_order_relaxed);
}

std::vector<FakeCpuKernel::Call> FakeCpuKernel::calls()
{
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_calls;
}

void FakeCpuKernel::hold(int64_t core, bool on) { g_held_core.store(on ? core : -1, std::memory_order_release); }

void FakeCpuKernel::forward(const Layer& layer, const ForwardCall& c, ::sglang::cpu_experts::Team& team)
{
    team.run([](int, int) {});  // one job through the team's dispatch and end barrier, as a real forward
    std::vector<double> w13(static_cast<size_t>(std::max(c.rows, 1))), w2(w13.size());
    for (size_t t = 0; t < w13.size(); ++t)
        w13[t] = slab_sum(layer, c, static_cast<int32_t>(t), 0, 3);
    if (c.stage_two) c.stage_two->wait();
    for (size_t t = 0; t < w2.size(); ++t)
        w2[t] = slab_sum(layer, c, static_cast<int32_t>(t), 3, 6);
    const int32_t core = team.cores().empty() ? -1 : team.cores().front();
    const int64_t until = team_now_ns() + c.k * g_ns.load(std::memory_order_relaxed);
    while (team_now_ns() < until)
        _mm_pause();
    while (core >= 0 && g_held_core.load(std::memory_order_acquire) == core)
        _mm_pause();
    if (const int64_t f = g_fail.load(std::memory_order_relaxed); f != 0)
        throw std::runtime_error("fake CPU expert forward failed (" + std::to_string(f) + ")");
    const int32_t hidden = std::max(layer.hidden, 1);
    const bool zero = g_zero.load(std::memory_order_relaxed);
    for (int32_t t = 0; t < std::max(c.rows, 1); ++t) {
        const int32_t* slots = c.slots + static_cast<int64_t>(t) * c.k;
        const float* weights = c.weights + static_cast<int64_t>(t) * c.k;
        float* out = c.out + static_cast<int64_t>(t) * hidden;
        double sum = 0;
        for (int32_t i = 0; i < c.k; ++i)
            sum += static_cast<double>(weights[i]) * (slots[i] + 1);
        for (int32_t j = 0; j < hidden; ++j) {
            const double base = c.accumulate() ? static_cast<double>(out[j]) : (zero ? 0.0 : static_cast<double>(j));
            out[j] = static_cast<float>(zero ? base : base + sum);
        }
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    int32_t affinity = -1;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0 && CPU_COUNT(&mask) == 1)
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) affinity = cpu;
    std::lock_guard<std::mutex> lock(g_mutex);
    for (int32_t t = 0; t < std::max(c.rows, 1); ++t) {
        Call call{core, affinity, team.workers(), c.accumulate() ? 1 : 0, std::min<int32_t>(c.k, kFakeMaxLanes), {}, {},
                  layer.capacity, c.stage_two != nullptr ? 1 : 0, w13[t], w2[t]};
        for (int32_t i = 0; i < call.k; ++i) {
            call.slots[i] = c.slots[static_cast<int64_t>(t) * c.k + i];
            call.weights[i] = c.weights[static_cast<int64_t>(t) * c.k + i];
        }
        g_calls.push_back(call);
    }
}

void FakeCpuKernel::job_reset(const JobTest& test)
{
    g_test = test;
    g_ordinal.store(0, std::memory_order_relaxed);
    g_violations.store(0, std::memory_order_relaxed);
    g_event_next.store(0, std::memory_order_relaxed);
}

std::vector<FakeCpuKernel::JobEvent> FakeCpuKernel::job_events()
{
    const uint32_t n = std::min(g_event_next.load(std::memory_order_acquire), kMaxEvents);
    return {g_events.begin(), g_events.begin() + n};
}

int64_t FakeCpuKernel::job_violations() { return g_violations.load(std::memory_order_relaxed); }

void FakeCpuKernel::prepare(Job& job, const Layer& layer, const ForwardCall& c)
{
    if (const int64_t f = g_fail.load(std::memory_order_relaxed); f != 0)
        throw std::invalid_argument("fake CPU expert prepare failed (" + std::to_string(f) + ")");
    if (c.k > 0 && c.slots[0] == kRejectSlot) throw std::invalid_argument("fake CPU expert prepare rejected the call");
    job.id = g_ordinal.fetch_add(1, std::memory_order_relaxed);
    job.rows = std::max(c.rows, 1);
    job.status = c.k > 0 && c.slots[0] == kRefuseSlot ? 2 : 0;
    job.layer = &layer;
    job.call = &c;
    for (int p = 0; p < kPhases; ++p) {
        const int32_t n = g_test.count != nullptr ? g_test.count(job.id, p, job.rows) : -1;
        job.count[p] = n < 0 ? job.rows : n;
        job.executed[p].store(0, std::memory_order_relaxed);
    }
    job.w13.assign(size_t(job.rows), 0.0);
    job.w2.assign(size_t(job.rows), 0.0);
}

::sglang::cpu_experts::Tasks FakeCpuKernel::tasks(const Job& job, int phase)
{
    return {job.count[size_t(phase)], std::max(g_test.align[size_t(phase)], 1)};
}

int FakeCpuKernel::status(const Job& job) { return job.status; }

void FakeCpuKernel::run(Job& job, int phase, int t0, int t1, int worker)
{
    const uint64_t tsc0 = __rdtsc();
    // The previous phase must have run all its tasks, and the results this one reads must be there.
    if (phase > 0 && job.executed[size_t(phase - 1)].load(std::memory_order_acquire) != job.count[size_t(phase - 1)])
        g_violations.fetch_add(1, std::memory_order_relaxed);
    const int32_t count = job.count[size_t(phase)];
    const FakeCpuKernel::Layer& layer = *job.layer;
    const ForwardCall& c = *job.call;
    const int32_t hidden = std::max(layer.hidden, 1);
    const bool zero = g_zero.load(std::memory_order_relaxed);
    for (int t = t0; t < t1; ++t) {
        if (g_test.delay_ns != nullptr)
            for (const int64_t until = team_now_ns() + g_test.delay_ns(job.id, phase, t); team_now_ns() < until;)
                _mm_pause();
        for (int32_t r = t; r < job.rows; r += count) {
            if (phase == 0) {
                job.w13[size_t(r)] = slab_sum(layer, c, r, 0, 3) + 1;
            } else if (phase == 1) {
                if (job.count[0] > 0 && job.w13[size_t(r)] == 0) g_violations.fetch_add(1, std::memory_order_relaxed);
                job.w2[size_t(r)] = slab_sum(layer, c, r, 3, 6) + 1;
            } else {
                if (job.count[1] > 0 && job.w2[size_t(r)] == 0) g_violations.fetch_add(1, std::memory_order_relaxed);
                const int32_t* slots = c.slots + static_cast<int64_t>(r) * c.k;
                const float* weights = c.weights + static_cast<int64_t>(r) * c.k;
                float* out = c.out + static_cast<int64_t>(r) * hidden;
                double sum = 0;
                for (int32_t i = 0; i < c.k; ++i) sum += static_cast<double>(weights[i]) * (slots[i] + 1);
                for (int32_t j = 0; j < hidden; ++j) {
                    const double base = c.accumulate() ? static_cast<double>(out[j]) : (zero ? 0.0 : double(j));
                    out[j] = static_cast<float>(zero ? base : base + sum);
                }
            }
        }
    }
    job.executed[size_t(phase)].fetch_add(t1 - t0, std::memory_order_release);
    const uint32_t slot = g_event_next.fetch_add(1, std::memory_order_relaxed);
    if (slot < kMaxEvents) g_events[slot] = {job.id, phase, t0, t1, worker, tsc0, __rdtsc()};
}

int32_t FakeCpuKernel::warm(const uint32_t* word, uint32_t seen, int64_t deadline_ns)
{
    while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == seen && team_now_ns() < deadline_ns)
        _mm_pause();
    return 0;
}

}  // namespace sglang::formats::fake
