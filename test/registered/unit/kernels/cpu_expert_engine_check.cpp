// Standalone harness for the CPU expert engine on its job window (host/cpu_experts.h): BasicCpuExpertEngine driven
// natively, with no tier, over the fake kernel (-DENGINE_KERNEL_FAKE: scheduler-level scenarios) or the EXL3 kernel
// (-DENGINE_KERNEL_EXL3, kernel.cpp linked at the tier under test: bit-exact scenarios). Built by
// test_cpu_expert_engine_window.py. Run with a scenario name; it prints `ok <scenario>` or fails; the fatal scenarios
// (watchdog, refused input, throwing prepare) end the process through fail_stop and the test reads their stderr.
//
//   mixed       several records in flight (a hit job, then two miss jobs adding into part 1, one of them staged), at 1 and 6
//               tokens a row: the outputs equal the same jobs run one at a time, bit for bit, and done(seq) never shows
//               an output unfinished
//   tokens32    one record of 32 tokens a row, the same comparison
//   gate        a shut stage-two gate holds only its job; done() does not pass it though a later job has finished
//   full        a full window stops admission (4 admitted, the rest wait in the ring) and the window counts the spell
//   stop        stop() admits nothing more and the admitted jobs finish
//   watchdog / refused / rejected   fail_stop naming the row (fake only)
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts.h"
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/single_job.hpp"
#if defined(ENGINE_KERNEL_EXL3)
#include "../../../../python/sglang/kernels/jit/csrc/exl3/optimized/kernel.h"
#else
#include "../../../../python/sglang/kernels/jit/csrc/moe/formats/fake/fake_kernel.h"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace es = sglang::expert_stream;
namespace ce = sglang::cpu_experts;
#if defined(ENGINE_KERNEL_EXL3)
using K = sglang::exl3_cpu::Exl3CpuKernel;
#else
using K = sglang::formats::fake::FakeCpuKernel;
#endif
using Build = es::BuildWithKernel<es::InstrBuild, K>;
using Engine = es::BasicCpuExpertEngine<Build>;
constexpr int kLanes = es::wire::Wire::kLanes;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

namespace {
constexpr int kHidden = 256;
constexpr int kCapacity = 6;
constexpr int kRows = 8;

uint16_t fp16_of(float a)
{
    uint32_t u;
    std::memcpy(&u, &a, 4);
    const uint32_t sign = (u >> 16) & 0x8000, exponent = (u >> 23) & 0xff, mantissa = (u >> 13) & 0x3ff;
    return exponent == 0 ? uint16_t(sign) : uint16_t(sign | ((exponent - 112) << 10) | mantissa);
}

int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

template <class F>
bool wait_until(F&& f, double seconds = 60.0)
{
    const int64_t deadline = now_ns() + int64_t(seconds * 1e9);
    while (!f()) {
        if (now_ns() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

// The layers every row runs over (rows alternate between two), the slabs behind them, and nothing else.
struct Layers {
    std::vector<std::vector<uint8_t>> slabs;
    es::CpuExpertLayers<K::Layer> table{kRows};

    explicit Layers(std::mt19937& rng)
    {
#if defined(ENGINE_KERNEL_EXL3)
        struct Cfg {
            int intermediate, bits_w13, bits_w2, swizzled;
            float act_limit;
        };
        const Cfg cfgs[2] = {{384, 3, 3, 0, 10.f}, {256, 2, 4, 1, 0.f}};
        for (int variant = 0; variant < 2; ++variant) {
            const Cfg& c = cfgs[variant];
            const uint64_t w = uint64_t(kHidden) * uint64_t(c.intermediate);
            const uint64_t row_bytes[6] = {2 * (w * uint64_t(c.bits_w13) / 8), 4ull * kHidden, 4ull * c.intermediate,
                                           w * uint64_t(c.bits_w2) / 8, 2ull * c.intermediate, 2ull * kHidden};
            K::Layer shape;
            shape.capacity = kCapacity;
            shape.hidden = kHidden;
            shape.intermediate = c.intermediate;
            shape.activation = 0;
            shape.act_limit = c.act_limit;
            shape.slab_count = 6;
            for (int s = 0; s < 6; ++s) {
                slabs.emplace_back(row_bytes[s] * kCapacity);
                std::vector<uint8_t>& slab = slabs.back();
                for (uint8_t& b : slab) b = uint8_t(rng());
                if (s == 1 || s == 2 || s == 4 || s == 5)  // the sign vectors: fp16 +-1
                    for (size_t i = 0; i + 1 < slab.size(); i += 2) {
                        const uint16_t v = rng() & 1 ? 0x3c00 : 0xbc00;
                        std::memcpy(&slab[i], &v, 2);
                    }
                shape.slabs[s] = slab.data();
                shape.slot_bytes[s] = row_bytes[s];
            }
            const int64_t values[4] = {c.bits_w13, c.bits_w2, c.swizzled, 0};
            shape.params = K::params(values);
            layers[variant] = K::make_layer(shape);
        }
#else
        (void)rng;
        for (int variant = 0; variant < 2; ++variant) {
            K::Layer shape;
            shape.capacity = kCapacity;
            shape.hidden = kHidden;
            shape.intermediate = 64;
            layers[variant] = K::make_layer(shape);
        }
#endif
        for (int row = 0; row < kRows; ++row)
            CHECK(table.set(row, layers[row % 2]));
    }
    K::Layer layers[2];
};

// The rows' staged inputs (and token tables), and the part buffers the engine writes.
struct Rows {
    int tokens;
    int64_t x_token_bytes, x_stride, out_part_stride, out_stride;
    std::vector<uint8_t> x;
    std::vector<float> out;  // [row][part][token][hidden]

    Rows(std::mt19937& rng, int tokens_)
        : tokens(tokens_), x_token_bytes(2 * kHidden),
          x_stride(tokens_ * 2 * kHidden + es::CpuTokenTable::kHeaderBytes + 4 * kLanes * (1 + tokens_)),
          out_part_stride(int64_t(tokens_) * kHidden * 4), out_stride(2 * out_part_stride),
          x(size_t(kRows) * size_t(x_stride)), out(size_t(kRows) * 2 * size_t(tokens_) * kHidden)
    {
        for (int row = 0; row < kRows; ++row) {
            uint8_t* base = &x[size_t(row) * size_t(x_stride)];
            for (int t = 0; t < tokens; ++t)
                for (int h = 0; h < kHidden; ++h) {
                    const uint16_t v = fp16_of((float(int(rng() % 2001)) - 1000.f) / 500.f);
                    std::memcpy(base + t * x_token_bytes + 2 * h, &v, 2);
                }
            if (tokens > 1) {  // the table: tokens, live tokens, a mask a lane, a weight a (token, lane)
                uint8_t* table = base + tokens * x_token_bytes;
                const uint32_t live = uint32_t(tokens);
                std::memcpy(table, &live, 4);
                std::memcpy(table + es::CpuTokenTable::kLiveTokens, &live, 4);
                for (int lane = 0; lane < kLanes; ++lane) {
                    uint32_t mask = uint32_t(rng());
                    if (tokens < 32) mask &= (1u << tokens) - 1;
                    std::memcpy(table + es::CpuTokenTable::kHeaderBytes + 4 * lane, &mask, 4);
                    for (int t = 0; t < tokens; ++t) {
                        const float w = float(1 + int(rng() % 100)) / 64.f;
                        std::memcpy(table + es::CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane), &w, 4);
                    }
                }
            }
        }
        for (float& v : out) v = (float(int(rng() % 1000)) - 500.f) / 16.f;
    }
    float* part(int row, int p) { return &out[(size_t(row) * 2 + size_t(p)) * size_t(tokens) * kHidden]; }
    const uint8_t* input(int row) const { return &x[size_t(row) * size_t(x_stride)]; }
};

uint32_t load_u32(const uint8_t* p)
{
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// The call the engine builds for `job` over `rows` (cpu_experts.h admit_target), for the one-at-a-time reference:
// the expanded token slots and weights for a per-token job.
struct RefCall {
    ce::ForwardCall call;
    std::vector<int32_t> slots;
    std::vector<float> weights;
};
RefCall expand(const es::CpuJob& job, Rows& rows, bool store_out, float* out)
{
    RefCall r;
    r.call.rows = 1;
    r.call.k = job.k;
    r.call.x = rows.input(int(job.row));
    r.call.out = out;
    r.call.out_mode = job.accumulate ? ce::OutMode::kAccumulate
                      : store_out    ? ce::OutMode::kOverwriteByStore
                                     : ce::OutMode::kOverwrite;
    r.slots.assign(job.slots, job.slots + job.k);
    r.weights.assign(job.weights, job.weights + job.k);
    if (job.per_token && rows.tokens > 1) {
        const uint8_t* table = rows.input(int(job.row)) + rows.tokens * rows.x_token_bytes;
        const uint32_t tokens = load_u32(table);
        r.call.rows = int(tokens);
        r.slots.assign(size_t(tokens) * size_t(job.k), -1);
        r.weights.assign(r.slots.size(), 0.f);
        for (uint32_t t = 0; t < tokens; ++t)
            for (int i = 0; i < job.k; ++i) {
                const int lane = job.lanes[i];
                const bool routed = (load_u32(table + es::CpuTokenTable::kHeaderBytes + 4 * lane) >> t & 1u) != 0;
                const uint32_t bits = load_u32(table + es::CpuTokenTable::kHeaderBytes + 4 * kLanes + 4 * (t * kLanes + lane));
                r.slots[t * size_t(job.k) + size_t(i)] = routed ? job.slots[i] : -1;
                std::memcpy(&r.weights[t * size_t(job.k) + size_t(i)], &bits, 4);
                if (!routed) r.weights[t * size_t(job.k) + size_t(i)] = 0.f;
            }
    }
    r.call.slots = r.slots.data();
    r.call.weights = r.weights.data();
    return r;
}

struct Fixture {
    std::mt19937 rng{20261024};
    Layers layers{rng};
    Rows rows;
    explicit Fixture(int tokens) : rows(rng, tokens) {}

    Engine::Config config(int threads, bool store_out) {
        Engine::Config c;
        c.layers = &layers.table;
        c.x_base = rows.x.data();
        c.x_stride = rows.x_stride;
        c.x_token_bytes = rows.x_token_bytes;
        c.tokens = rows.tokens;
        c.out_base = reinterpret_cast<uint8_t*>(rows.out.data());
        c.out_stride = rows.out_stride;
        c.out_part_stride = rows.out_part_stride;
        c.hidden = kHidden;
        c.threads = threads;
        c.check_calls = true;
        c.store_out = store_out;
        c.two_stage = true;
        return c;
    }
};

// A job of `lanes` random lanes over slots in [0, pool).
es::CpuJob make_job(Fixture& f, int row, int part, bool accumulate, bool staged, int lanes, uint32_t seq)
{
    es::CpuJob job;
    job.row = row;
    job.part = part;
    job.accumulate = accumulate;
    job.per_token = f.rows.tokens > 1;
    job.staged = staged;
    job.seq = seq;
    job.k = lanes;
    const int pool = 1 + int(f.rng() % kCapacity);
    for (int i = 0; i < lanes; ++i) {
        job.slots[i] = f.rng() % 8 == 0 ? -1 : int32_t(f.rng() % unsigned(pool));
        job.weights[i] = float(1 + int(f.rng() % 100)) / 64.f;
        job.lanes[i] = (i * 3 + row) % kLanes;
    }
    return job;
}

// Runs `jobs` (in sequence order) through the engine with several in flight and through the one-at-a-time reference;
// the engine's outputs must equal the reference's, bit for bit, and every done(seq) must show the outputs whose jobs
// are all at or below it final.
void compare(Fixture& f, const std::vector<es::CpuJob>& jobs, bool store_out, int threads)
{
    const std::vector<float> start = f.rows.out;
    // The reference: each job alone, in order, on the same initial buffers.
    std::vector<float> ref = start;
    {
        Fixture& g = f;
        for (const es::CpuJob& job : jobs) {
            float* out = ref.data() + (size_t(job.row) * 2 + size_t(job.part)) * size_t(g.rows.tokens) * kHidden;
            RefCall r = expand(job, g.rows, store_out, out);
            ce::forward_alone<K>(*g.layers.table.get(job.row), r.call, std::span<const int>(), threads);
        }
    }
    // The last job of each output.
    std::vector<uint32_t> last(size_t(kRows) * 2, 0);
    for (const es::CpuJob& job : jobs) last[size_t(job.row) * 2 + size_t(job.part)] = job.seq;

    Engine engine(f.config(threads, store_out), "engine-check: ", "engine-check");
    engine.start();
    const uint32_t first = engine.claim(int(jobs.back().seq - jobs.front().seq + 1));
    CHECK(first == jobs.front().seq);
    std::atomic<bool> stop_opening{false};
    std::thread opener([&] {  // the staged jobs' gates open a little after their jobs go in
        std::mt19937 local(7);
        for (const es::CpuJob& job : jobs) {
            if (!job.staged) continue;
            std::this_thread::sleep_for(std::chrono::microseconds(50 + local() % 600));
            engine.open_stage_two(job.seq);
        }
        stop_opening = true;
    });
    for (const es::CpuJob& job : jobs) CHECK(engine.submit(job));
    std::atomic<long> checks{0};
    const uint32_t final_seq = jobs.back().seq;
    CHECK(wait_until([&] {
        // done(s) at one instant: every output whose jobs are all <= s is final then.
        for (const es::CpuJob& probe : jobs) {
            if (!engine.done(probe.seq)) continue;
            for (int out = 0; out < kRows * 2; ++out) {
                if (last[size_t(out)] == 0 || last[size_t(out)] > probe.seq) continue;
                const size_t n = size_t(f.rows.tokens) * kHidden;
                CHECK(std::memcmp(&f.rows.out[size_t(out) * n], &ref[size_t(out) * n], n * sizeof(float)) == 0);
                ++checks;
            }
        }
        return engine.done(final_seq);
    }));
    opener.join();
    CHECK(f.rows.out.size() == ref.size());
    CHECK(std::memcmp(f.rows.out.data(), ref.data(), ref.size() * sizeof(float)) == 0);
    CHECK(engine.jobs() == int64_t(jobs.size()));
    CHECK(engine.window_counters().admissions == jobs.size());
    engine.stop();
}

// Records of (hit, miss, staged miss) jobs on distinct rows, with some rows twice.
std::vector<es::CpuJob> record_jobs(Fixture& f, int records)
{
    std::vector<es::CpuJob> jobs;
    uint32_t seq = 1;
    for (int rec = 0; rec < records; ++rec) {
        const int row = rec % kRows;
        const int lanes = 3 + int(f.rng() % 3);
        jobs.push_back(make_job(f, row, 0, false, false, lanes, seq++));
        jobs.push_back(make_job(f, row, 1, false, rec % 2 == 0, 1 + int(f.rng() % 2), seq++));
        jobs.push_back(make_job(f, row, 1, true, rec % 2 == 1, 1 + int(f.rng() % 2), seq++));
    }
    return jobs;
}

void scenario_mixed()
{
    for (const int tokens : {1, 6}) {
        for (const bool store_out : {false, true}) {
            Fixture f(tokens);
            const std::vector<es::CpuJob> jobs = record_jobs(f, 9);
            for (const int threads : {1, 4}) compare(f, jobs, store_out, threads);
        }
    }
}

void scenario_tokens32()
{
    Fixture f(32);
    const std::vector<es::CpuJob> jobs = record_jobs(f, 2);
    compare(f, jobs, false, 4);
}

#if !defined(ENGINE_KERNEL_EXL3)
using Fake = sglang::formats::fake::FakeCpuKernel;

// A fake-kernel engine over `rows` rows of one token, outputs preset to -1.
struct Fake1 {
    Fixture f{1};
    std::unique_ptr<Engine> engine;
    Fake1(int threads, int64_t fatal_wait_ns = 0)
    {
        std::fill(f.rows.out.begin(), f.rows.out.end(), -1.0f);
        Fake::reset(0, 0, false);
        engine = std::make_unique<Engine>(f.config(threads, false), "engine-check: ", "engine-check");
        engine->start();
        if (fatal_wait_ns > 0) {
            alignas(64) static std::array<uint8_t, es::draft::kChannelBytes> channel{};
            static std::vector<int32_t> slots(size_t(es::draft::kMaxRows) * es::draft::kMaxK);
            static std::vector<float> weights(size_t(es::draft::kMaxRows) * es::draft::kMaxK);
            auto source = std::make_unique<Engine::DraftSource>();
            source->channel = channel.data();
            source->x = f.rows.x.data();
            source->slots = slots.data();
            source->weights = weights.data();
            source->out = reinterpret_cast<float*>(f.rows.out.data());
            source->hidden = kHidden;
            source->layers = {f.layers.layers[0]};
            engine->attach_draft(std::move(source), fatal_wait_ns);
        }
    }
    es::CpuJob job(int row, int part, bool staged, int32_t slot0 = 1)
    {
        es::CpuJob j;
        j.row = row;
        j.part = part;
        j.staged = staged;
        j.k = 1;
        j.slots[0] = slot0;
        j.weights[0] = 1.0f;
        j.seq = engine->claim(1);
        return j;
    }
    float out0(int row, int part) { return f.rows.part(row, part)[0]; }
};

void scenario_gate()
{
    Fake1 e(4);
    const es::CpuJob staged = e.job(0, 0, true);
    const es::CpuJob plain = e.job(1, 0, false);
    CHECK(e.engine->submit(staged) && e.engine->submit(plain));
    // The plain job finishes (its row is written) while the staged one holds at its gate.
    CHECK(wait_until([&] { return e.out0(1, 0) != -1.0f; }, 10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!e.engine->done(staged.seq));
    CHECK(!e.engine->done(plain.seq));  // done never passes the unfinished one before it
    e.engine->open_stage_two(staged.seq);
    CHECK(wait_until([&] { return e.engine->done(plain.seq); }, 10));
    CHECK(e.engine->done(staged.seq));
    CHECK(e.out0(0, 0) != -1.0f);
    CHECK(e.engine->staged_jobs() == 1 && e.engine->stage_two_waits() == 1);
    e.engine->stop();
}

// Four staged jobs with shut gates fill the window; two more jobs wait in the ring behind them.
std::vector<es::CpuJob> fill_window(Fake1& e)
{
    std::vector<es::CpuJob> jobs;
    for (int i = 0; i < 6; ++i) jobs.push_back(e.job(i, 0, i < 4));
    for (const es::CpuJob& j : jobs) CHECK(e.engine->submit(j));
    CHECK(wait_until([&] { return e.engine->window_counters().admissions == 4; }, 10));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    return jobs;
}

void scenario_full()
{
    Fake1 e(4);
    CHECK(e.engine->window_counters().admissions == 0);
    const std::vector<es::CpuJob> jobs = fill_window(e);
    const ce::WindowCounters full = e.engine->window_counters();
    CHECK(full.admissions == 4);       // the window is W = 4: the others wait in the ring
    CHECK(full.admissions_full == 1);  // one spell, not one a spin of the engine's loop
    CHECK(full.both_pending == 0);
    CHECK(e.out0(4, 0) == -1.0f && e.out0(5, 0) == -1.0f);
    for (int i = 0; i < 4; ++i) e.engine->open_stage_two(jobs[size_t(i)].seq);
    CHECK(wait_until([&] { return e.engine->done(jobs.back().seq); }, 30));
    CHECK(e.engine->window_counters().admissions == 6);
    for (const es::CpuJob& j : jobs) CHECK(e.out0(int(j.row), 0) != -1.0f);
    e.engine->stop();
}

void scenario_stop()
{
    Fake1 e(4);
    const std::vector<es::CpuJob> jobs = fill_window(e);
    e.engine->stop();  // admits nothing more; the four in the window finish, their shut gates opened by the stop
    CHECK(e.engine->jobs() == 4);
    CHECK(e.engine->done(jobs[3].seq) && !e.engine->done(jobs[4].seq));
    for (int i = 0; i < 4; ++i) CHECK(e.out0(i, 0) != -1.0f);
    for (int i = 4; i < 6; ++i) CHECK(e.out0(i, 0) == -1.0f);
}

// A staged job whose gate never opens is shut open by stop().
void scenario_stop_opens_gate()
{
    Fake1 e(2);
    const es::CpuJob staged = e.job(0, 0, true);
    CHECK(e.engine->submit(staged));
    CHECK(wait_until([&] { return e.engine->window_counters().admissions == 1; }, 10));
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    CHECK(!e.engine->done(staged.seq));
    e.engine->stop();
    CHECK(e.engine->done(staged.seq));
}

void scenario_watchdog()
{
    Fake1 e(2, 200'000'000);
    Fake::JobTest test;
    test.delay_ns = [](int32_t, int32_t, int32_t) -> int64_t { return 5'000'000'000; };
    Fake::job_reset(test);
    const es::CpuJob job = e.job(3, 0, false);
    CHECK(e.engine->submit(job));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::fprintf(stderr, "not aborted\n");
    std::exit(3);
}

void scenario_refused()
{
    Fake1 e(2);
    const es::CpuJob job = e.job(5, 0, false, Fake::kRefuseSlot);
    CHECK(e.engine->submit(job));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::fprintf(stderr, "not aborted\n");
    std::exit(3);
}

void scenario_rejected()
{
    Fake1 e(2);
    const es::CpuJob job = e.job(6, 0, false, Fake::kRejectSlot);
    CHECK(e.engine->submit(job));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::fprintf(stderr, "not aborted\n");
    std::exit(3);
}
#endif
}  // namespace

int main(int argc, char** argv)
{
    CHECK(argc == 2);
    const std::string name = argv[1];
    if (name == "mixed") scenario_mixed();
    else if (name == "tokens32") scenario_tokens32();
#if !defined(ENGINE_KERNEL_EXL3)
    else if (name == "gate") scenario_gate();
    else if (name == "full") scenario_full();
    else if (name == "stop") scenario_stop();
    else if (name == "stop_opens_gate") scenario_stop_opens_gate();
    else if (name == "watchdog") scenario_watchdog();
    else if (name == "refused") scenario_refused();
    else if (name == "rejected") scenario_rejected();
#endif
    else {
        std::fprintf(stderr, "no scenario %s\n", name.c_str());
        return 2;
    }
    std::printf("ok %s\n", name.c_str());
    return 0;
}
