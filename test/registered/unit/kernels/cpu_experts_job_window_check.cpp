// Standalone harness for host/cpu_experts/job_window.hpp: the JobWindow and its scheduler over the fake kernel's job
// interface (formats/fake), on a real Team of unpinned workers. Built by test_cpu_experts_job_window.py with
// fake_kernel.cpp.
#include "../../../../python/sglang/kernels/jit/csrc/moe/expert_stream/host/cpu_experts/job_window.hpp"
#include "../../../../python/sglang/kernels/jit/csrc/moe/formats/fake/fake_kernel.h"
#include <x86intrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

namespace {
using namespace sglang::cpu_experts;
using K = sglang::formats::fake::FakeCpuKernel;
static_assert(CpuExpertJobKernel<K>);

constexpr int kTarget = 0;
constexpr int kDraft = 1;
constexpr int kHidden = 8;
constexpr int kCapacity = 4;
constexpr int kMaxJobs = 64;
constexpr int kPhases = K::kPhases;

void ok(const char* name) { std::printf("ok %s\n", name); }

// What the hooks saw, by sequence (target jobs run seq 1.. and drafts above 40).
struct Log {
    std::array<std::atomic<int>, kMaxJobs> completed{};
    std::array<std::atomic<int>, kMaxJobs> status{};
    std::array<std::atomic<uint32_t>, kMaxJobs> done_at{};  // the done word when the hook ran
    std::array<std::atomic<int>, kMaxJobs> gate_calls{}, gate_spun{};
    std::array<std::atomic<int64_t>, kMaxJobs> gate_waited{};
    std::array<std::atomic<uint64_t>, kMaxJobs> worker_tsc{};
    std::array<std::atomic<int>, kMaxJobs> order{};  // the seq that completed i-th
    std::atomic<int> completions{0};
    std::atomic<uint32_t>* done = nullptr;
};

struct Hooks {
    Log* log;
    template <class S>
    void on_complete(S& s) noexcept
    {
        const uint32_t q = s.seq();
        Log& l = *log;
        if (q >= kMaxJobs) {  // the stress case: only the count is kept
            l.completions.fetch_add(1);
            return;
        }
        l.status[q] = s.status();
        l.done_at[q] = l.done->load(std::memory_order_acquire);
        l.gate_calls[q] = s.gate_calls;
        l.gate_spun[q] = s.gate_spun ? 1 : 0;
        l.gate_waited[q] = s.gate_waited_ns;
        l.worker_tsc[q] = s.worker_tsc();
        l.order[size_t(l.completions.fetch_add(1))] = int(q);
        l.completed[q] = 1;
    }
};

// The layer every job runs over: six slabs of four 4-byte slots, so the fake's phases have bytes to sum.
struct LayerFx {
    std::array<std::vector<uint8_t>, 6> slabs;
    K::Layer layer;
    LayerFx()
    {
        layer.capacity = kCapacity;
        layer.hidden = kHidden;
        layer.slab_count = 6;
        for (size_t i = 0; i < 6; ++i) {
            slabs[i].resize(size_t(kCapacity) * 4);
            for (size_t b = 0; b < slabs[i].size(); ++b) slabs[i][b] = uint8_t(i * 7 + b);
            layer.slabs[i] = slabs[i].data();
            layer.slot_bytes[i] = 4;
        }
    }
};
LayerFx g_fx;

// One call's buffers; `out` starts at 7 (or at what the test sets).
struct TJob {
    int rows, k;
    std::vector<int32_t> slots;
    std::vector<float> weights, out, init;
    ForwardCall call;

    TJob(int rows_, int k_, int base, OutMode mode = OutMode::kOverwrite)
        : rows(rows_), k(k_), slots(size_t(rows_) * k_), weights(size_t(rows_) * k_),
          out(size_t(rows_) * kHidden, 7.0f), init(out)
    {
        for (int t = 0; t < rows; ++t)
            for (int i = 0; i < k; ++i) {
                slots[size_t(t * k + i)] = (base + t + i) % kCapacity;
                weights[size_t(t * k + i)] = 0.5f * float(i + 1) + float(t);
            }
        call.rows = rows;
        call.k = k;
        call.slots = slots.data();
        call.weights = weights.data();
        call.out = out.data();
        call.out_mode = mode;
    }
    TJob(const TJob&) = delete;
    TJob& operator=(const TJob&) = delete;

    // What the fake writes, from `from` (the buffer's contents before the job).
    std::vector<float> expected(const std::vector<float>& from) const
    {
        std::vector<float> e = from;
        for (int t = 0; t < rows; ++t) {
            double sum = 0;
            for (int i = 0; i < k; ++i) sum += double(weights[size_t(t * k + i)]) * (slots[size_t(t * k + i)] + 1);
            for (int j = 0; j < kHidden; ++j) {
                float& o = e[size_t(t * kHidden + j)];
                const double base = call.accumulate() ? double(o) : double(j);
                o = float(base + sum);
            }
        }
        return e;
    }
};

// A team, a window over it with started workers and a done word, and the log its hooks write.
struct Env {
    std::atomic<uint32_t> done{0};
    Log log;
    Team team;
    JobWindow<K, 4, Hooks> win;
    bool finished = false;

    explicit Env(int threads = 6)
        : team(std::span<const int>{}, threads, &K::warm, 20000), win(team, Hooks{&log})
    {
        log.done = &done;
        win.publish_done_to(&done, kTarget);
        win.start();
    }
    ~Env() { finish(); }

    void finish()
    {
        if (finished) return;
        finished = true;
        win.close();
        win.drain();
        win.join();
    }

    bool admit(uint32_t seq, TJob& j, uint64_t key, const StageGate* gate = nullptr, int kind = kTarget)
    {
        return win.admit(seq, kind, g_fx.layer, j.call, key, gate);
    }
    template <class P>
    void wait(P&& pred)
    {
        const int64_t start = team_now_ns();
        while (!pred()) {
            win.step(0);
            CHECK(team_now_ns() - start < 30LL * 1000000000);
        }
    }
    void admit_wait(uint32_t seq, TJob& j, uint64_t key, const StageGate* gate = nullptr, int kind = kTarget)
    {
        wait([&] { return admit(seq, j, key, gate, kind); });
    }
    void wait_done(uint32_t seq)
    {
        wait([&] { return done.load(std::memory_order_acquire) >= seq; });
    }
};

void reset(const K::JobTest& test = {})
{
    K::reset(0, 0, false);
    K::job_reset(test);
}

struct Span {
    uint64_t first = UINT64_MAX, last = 0;
    int runs = 0;
};
using Spans = std::array<std::array<Span, kPhases>, kMaxJobs>;

Spans spans_of(const std::vector<K::JobEvent>& events)
{
    Spans s{};
    for (const K::JobEvent& e : events) {
        Span& p = s[size_t(e.job)][size_t(e.phase)];
        p.first = std::min(p.first, e.tsc0);
        p.last = std::max(p.last, e.tsc1);
        ++p.runs;
    }
    return s;
}

// Every task [0, count) of (job, phase) ran in exactly one recorded run, and every run but the last starts and ends on
// the phase's alignment.
void check_covered(const std::vector<K::JobEvent>& events, int job, int phase, int count, int align)
{
    std::vector<std::pair<int, int>> runs;
    for (const K::JobEvent& e : events)
        if (e.job == job && e.phase == phase) runs.emplace_back(e.t0, e.t1);
    std::sort(runs.begin(), runs.end());
    int next = 0;
    for (auto [t0, t1] : runs) {
        CHECK(t0 == next && t1 > t0);
        CHECK(t0 % align == 0 && ((t1 - t0) % align == 0 || t1 == count));
        next = t1;
    }
    CHECK(next == count);
}

// The buffer `actual` holds what `j` writes over `from`.
void check_out(const TJob& j, const std::vector<float>& from, const std::vector<float>& actual)
{
    const std::vector<float> e = j.expected(from);
    CHECK(e.size() == actual.size());
    for (size_t i = 0; i < e.size(); ++i) CHECK(e[i] == actual[i]);
}
void check_out(const TJob& j, const std::vector<float>& from) { check_out(j, from, j.out); }

// ---- controls for the tests' jobs (plain functions: K::JobTest holds function pointers)

int32_t count_varied(int32_t job, int32_t phase, int32_t)
{
    return phase == 0 ? 13 + job % 5 : phase == 1 ? 9 : -1;
}
int64_t delay_1us(int32_t, int32_t phase, int32_t) { return phase < 2 ? 1000 : 0; }

int64_t delay_slow_first_out(int32_t job, int32_t phase, int32_t) { return job == 0 && phase == 2 ? 3000000 : 0; }

int32_t count_job1_no_out(int32_t job, int32_t phase, int32_t) { return job == 1 && phase == 2 ? 0 : -1; }

int64_t delay_slow_first_job(int32_t job, int32_t, int32_t) { return job == 0 ? 2000000 : 0; }

int32_t count_none(int32_t, int32_t, int32_t) { return 0; }
int32_t count_job1_no_gate_tasks(int32_t job, int32_t phase, int32_t) { return job == 1 && phase == 1 ? 0 : -1; }

// 0-19 us from a hash of the task, so consecutive jobs finish out of order.
int64_t delay_random(int32_t job, int32_t phase, int32_t task)
{
    uint32_t h = uint32_t(job) * 2654435761u ^ uint32_t(phase) * 40503u ^ uint32_t(task) * 97u;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    return int64_t(h % 20) * 1000;
}

int64_t delay_1ms(int32_t, int32_t, int32_t) { return 1000000; }

// ---- the checks

void every_task_runs_once_and_phases_run_in_order()
{
    reset({count_varied, delay_1us, {1, 2, 3}});
    constexpr int kJobs = 24;
    std::vector<std::unique_ptr<TJob>> jobs;
    {
        Env e;
        for (int i = 0; i < kJobs; ++i) {
            jobs.push_back(std::make_unique<TJob>(3, 2, i));
            e.admit_wait(uint32_t(i + 1), *jobs.back(), uint64_t(i));
        }
        e.wait_done(kJobs);
        CHECK(e.win.counters().admissions == kJobs);
        for (int i = 1; i <= kJobs; ++i) CHECK(e.log.completed[size_t(i)] == 1);
    }
    CHECK(K::job_violations() == 0);
    const auto events = K::job_events();
    const Spans spans = spans_of(events);
    for (int i = 0; i < kJobs; ++i) {
        check_covered(events, i, 0, 13 + i % 5, 1);
        check_covered(events, i, 1, 9, 2);
        check_covered(events, i, 2, 3, 3);
        // Phase p+1's first task starts after phase p's last task ended.
        for (int p = 0; p + 1 < kPhases; ++p)
            CHECK(spans[size_t(i)][size_t(p)].last <= spans[size_t(i)][size_t(p + 1)].first);
        check_out(*jobs[size_t(i)], jobs[size_t(i)]->init);
    }
    ok("every_task_runs_once_and_phases_run_in_order");
}

void same_output_out_phases_run_in_sequence_order_while_other_outputs_interleave()
{
    reset({nullptr, delay_slow_first_out, {1, 1, 1}});
    TJob a0(2, 2, 0), a1(2, 2, 1), a2(2, 2, 2), b(2, 2, 3);
    a1.call.out = a2.call.out = a0.call.out;
    std::vector<float> shared_init = a0.init;
    {
        Env e;
        e.admit_wait(1, a0, 100);
        e.admit_wait(2, a1, 100);
        e.admit_wait(3, a2, 100);
        e.admit_wait(4, b, 200);
        e.wait_done(4);
    }
    CHECK(K::job_violations() == 0);
    const Spans s = spans_of(K::job_events());
    CHECK(s[1][2].first >= s[0][2].last);
    CHECK(s[2][2].first >= s[1][2].last);
    // A different output does not wait: job 3 finishes its out phase while job 0 is still in its slow one, and job 1's
    // early phases overlap job 0's out phase.
    CHECK(s[3][2].last < s[0][2].last);
    CHECK(s[1][0].first < s[0][2].last);
    // The last writer wins.
    check_out(a2, shared_init, a0.out);
    check_out(b, b.init);
    ok("same_output_out_phases_run_in_sequence_order_while_other_outputs_interleave");
}

void a_job_with_no_out_tasks_keeps_the_chain_ordered()
{
    reset({count_job1_no_out, delay_slow_first_out, {1, 1, 1}});
    TJob a(2, 2, 0), b(2, 2, 1), c(2, 2, 2);
    b.call.out = c.call.out = a.call.out;
    std::vector<float> shared_init = a.init;
    {
        Env e;
        e.admit_wait(1, a, 7);
        e.admit_wait(2, b, 7);
        e.admit_wait(3, c, 7);
        e.wait_done(3);
    }
    const Spans s = spans_of(K::job_events());
    CHECK(s[1][2].runs == 0);
    CHECK(s[2][2].first >= s[0][2].last);
    check_out(c, shared_init, a.out);
    ok("a_job_with_no_out_tasks_keeps_the_chain_ordered");
}

void done_never_passes_an_unfinished_sequence()
{
    reset({nullptr, delay_slow_first_job, {1, 1, 1}});
    TJob j0(2, 2, 0), j1(2, 2, 1), j2(2, 2, 2);
    Env e;
    e.admit_wait(1, j0, 1);
    e.admit_wait(2, j1, 2);
    e.admit_wait(3, j2, 3);
    uint32_t last = 0;
    e.wait([&] {
        const uint32_t d = e.done.load(std::memory_order_acquire);
        CHECK(d >= last);
        last = d;
        for (uint32_t q = 1; q <= d; ++q) CHECK(e.log.completed[q] == 1);
        return d >= 3;
    });
    // The later jobs finished first, and the word waited for job 0.
    CHECK(e.log.order[2] == 1);
    CHECK(e.log.done_at[2] == 0 && e.log.done_at[3] == 0);
    CHECK(e.log.done_at[1] == 0);
    ok("done_never_passes_an_unfinished_sequence");
}

void a_staged_gate_holds_only_its_job()
{
    reset();
    std::atomic<uint32_t> word{0};
    std::atomic<bool> stop{false};
    StageGate gate;
    gate.word = &word;
    gate.open = 1;
    gate.stop = &stop;
    gate.timed = true;
    TJob staged(2, 2, 0), p2(2, 2, 1), p3(2, 2, 2);
    {
        Env e;
        e.admit_wait(1, staged, 1, &gate);
        e.admit_wait(2, p2, 2);
        e.admit_wait(3, p3, 3);
        e.wait([&] { return e.log.completed[2] == 1 && e.log.completed[3] == 1; });
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        (void)e.win.word();  // acquire: the events of the phases that already finished are visible
        CHECK(e.log.completed[1] == 0);
        CHECK(e.done.load() == 0);
        for (const K::JobEvent& ev : K::job_events())
            if (ev.job == 0) CHECK(ev.phase == 0);
        const uint64_t t_open = __rdtsc();
        word.store(1, std::memory_order_release);
        e.wait_done(3);
        for (const K::JobEvent& ev : K::job_events())
            if (ev.job == 0 && ev.phase >= 1) CHECK(ev.tsc0 >= t_open);
        CHECK(e.log.gate_calls[1] == 1 && e.log.gate_spun[1] == 1 && e.log.gate_waited[1] > 0);
        CHECK(e.log.gate_calls[2] == 0);
        check_out(staged, staged.init);

        // A gate already open on arrival: one call, not spun, no wait.
        TJob open_gate(2, 2, 3);
        e.admit_wait(4, open_gate, 4, &gate);
        e.wait_done(4);
        CHECK(e.log.gate_calls[4] == 1 && e.log.gate_spun[4] == 0 && e.log.gate_waited[4] == 0);

        // A shut gate opens on its stop flag.
        std::atomic<uint32_t> never{0};
        std::atomic<bool> stop5{false};
        StageGate stoppable = gate;
        stoppable.word = &never;
        stoppable.stop = &stop5;
        TJob stopped(2, 2, 0);
        e.admit_wait(5, stopped, 5, &stoppable);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        CHECK(e.log.completed[5] == 0);
        stop5.store(true, std::memory_order_release);
        e.wait_done(5);
        CHECK(e.log.gate_calls[5] == 1 && e.log.gate_spun[5] == 1);
    }
    CHECK(K::job_violations() == 0);
    ok("a_staged_gate_holds_only_its_job");
}

void a_full_window_stops_admission()
{
    reset();
    std::atomic<uint32_t> word{0};
    StageGate gate;
    gate.word = &word;
    gate.open = 1;
    std::vector<std::unique_ptr<TJob>> jobs;
    {
        Env e;
        for (int i = 0; i < 4; ++i) {
            jobs.push_back(std::make_unique<TJob>(2, 2, i));
            CHECK(e.admit(uint32_t(i + 1), *jobs.back(), uint64_t(i), &gate));
        }
        TJob fifth(2, 2, 0);
        CHECK(!e.admit(5, fifth, 9, &gate));
        CHECK(e.win.counters().admissions == 4 && e.win.counters().admissions_full == 1);
        word.store(1, std::memory_order_release);
        e.wait_done(4);
        e.admit_wait(5, fifth, 9);
        e.wait_done(5);
        // Drafts publish nothing: their slots free on completion, so more than W of them pass through.
        std::vector<std::unique_ptr<TJob>> drafts;
        for (int i = 0; i < 6; ++i) {
            drafts.push_back(std::make_unique<TJob>(2, 2, i));
            e.admit_wait(uint32_t(40 + i), *drafts.back(), kNoOutKey, nullptr, kDraft);
        }
        e.wait([&] { return e.log.completed[45] == 1 && !e.win.any_active(); });
        CHECK(e.done.load() == 5);
        e.win.note_both_pending();
        CHECK(e.win.counters().both_pending == 1);
    }
    ok("a_full_window_stops_admission");
}

void stop_drains()
{
    reset({nullptr, delay_1ms, {1, 1, 1}});
    std::vector<std::unique_ptr<TJob>> jobs;
    Env e;
    for (int i = 0; i < 4; ++i) {
        jobs.push_back(std::make_unique<TJob>(2, 2, i));
        CHECK(e.admit(uint32_t(i + 1), *jobs.back(), uint64_t(i)));
    }
    e.win.close();
    TJob late(2, 2, 0);
    CHECK(!e.admit(5, late, 9));
    e.win.drain();
    CHECK(!e.win.any_active());
    for (int i = 1; i <= 4; ++i) CHECK(e.log.completed[size_t(i)] == 1);
    CHECK(e.done.load() == 4);
    e.win.join();  // the workers left their loops
    for (auto& j : jobs) check_out(*j, j->init);
    ok("stop_drains");
}

void status_2_is_surfaced_at_completion()
{
    reset();
    TJob j1(2, 2, 0), j2(2, 2, 1), j3(2, 2, 2);
    j2.slots[0] = K::kRefuseSlot;
    Env e;
    e.admit_wait(1, j1, 1);
    e.admit_wait(2, j2, 2);
    e.admit_wait(3, j3, 3);
    e.wait_done(3);
    CHECK(e.log.status[1] == 0 && e.log.status[2] == 2 && e.log.status[3] == 0);
    ok("status_2_is_surfaced_at_completion");
}

void a_throwing_prepare_surfaces_to_the_admitter()
{
    reset();
    std::atomic<uint32_t> word{0};
    StageGate gate;
    gate.word = &word;
    gate.open = 1;
    TJob bad(2, 2, 0);
    bad.slots[0] = K::kRejectSlot;
    std::vector<std::unique_ptr<TJob>> jobs;
    Env e;
    for (int i = 0; i < 5; ++i) {
        bool threw = false;
        try {
            e.admit(1, bad, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
    K::reset(0, 1, false);  // every prepare fails
    {
        TJob fine(2, 2, 0);
        bool threw = false;
        try {
            e.admit(1, fine, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK(threw);
    }
    K::reset(0, 0, false);
    CHECK(e.win.counters().admissions == 0 && e.win.counters().admissions_full == 0);
    // No slot was taken: all four admit with none completing, and the next sequence is still 1.
    for (int i = 0; i < 4; ++i) {
        jobs.push_back(std::make_unique<TJob>(2, 2, i));
        CHECK(e.admit(uint32_t(i + 1), *jobs.back(), uint64_t(i), &gate));
    }
    word.store(1, std::memory_order_release);
    e.wait_done(4);
    ok("a_throwing_prepare_surfaces_to_the_admitter");
}

void phases_with_no_tasks_complete_at_once_or_wait_for_their_gate()
{
    reset({count_none, nullptr, {1, 1, 1}});
    {
        Env e;
        TJob j(2, 2, 0);
        CHECK(e.admit(1, j, 1));
        // Every phase is empty: the job completed inside admit, on this thread.
        CHECK(e.log.completed[1] == 1 && e.done.load() == 1 && !e.win.any_active());
    }
    reset({count_job1_no_gate_tasks, nullptr, {1, 1, 1}});
    std::atomic<uint32_t> word{0};
    StageGate gate;
    gate.word = &word;
    gate.open = 1;
    TJob a(2, 2, 0), b(2, 2, 1);
    Env e;
    e.admit_wait(1, a, 1);
    e.admit_wait(2, b, 2, &gate);
    e.wait_done(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // An empty gate phase still holds the job until its gate opens.
    CHECK(e.log.completed[2] == 0);
    word.store(1, std::memory_order_release);
    e.wait_done(2);
    CHECK(e.log.gate_calls[2] == 1);
    const Spans s = spans_of(K::job_events());
    CHECK(s[1][1].runs == 0 && s[1][0].runs > 0 && s[1][2].runs > 0);
    check_out(b, b.init);
    ok("phases_with_no_tasks_complete_at_once_or_wait_for_their_gate");
}

// Thousands of jobs through a four-slot window on eight threads with random task durations: the done word must reach
// the last sequence (a lost advance stalls it and fills the window). Env::wait fails the run after 30 s, never hangs.
void the_done_word_reaches_the_last_sequence_under_out_of_order_completions()
{
    reset({nullptr, delay_random, {1, 1, 1}});
    constexpr int kJobs = 4000;
    std::vector<std::unique_ptr<TJob>> jobs;
    for (int i = 0; i < kJobs; ++i) jobs.push_back(std::make_unique<TJob>(2, 2, i));
    {
        Env e(8);
        for (int i = 0; i < kJobs; ++i) e.admit_wait(uint32_t(i + 1), *jobs[size_t(i)], uint64_t(i));
        e.wait_done(kJobs);
        CHECK(e.done.load() == kJobs);
        CHECK(e.win.counters().admissions == kJobs);
        CHECK(e.log.completions.load() == kJobs);
    }
    CHECK(K::job_violations() == 0);
    for (auto& j : jobs) check_out(*j, j->init);
    ok("the_done_word_reaches_the_last_sequence_under_out_of_order_completions");
}

}  // namespace

int main()
{
    every_task_runs_once_and_phases_run_in_order();
    same_output_out_phases_run_in_sequence_order_while_other_outputs_interleave();
    a_job_with_no_out_tasks_keeps_the_chain_ordered();
    done_never_passes_an_unfinished_sequence();
    a_staged_gate_holds_only_its_job();
    a_full_window_stops_admission();
    stop_drains();
    status_2_is_surfaced_at_completion();
    a_throwing_prepare_surfaces_to_the_admitter();
    phases_with_no_tasks_complete_at_once_or_wait_for_their_gate();
    the_done_word_reaches_the_last_sequence_under_out_of_order_completions();
    std::printf("all ok\n");
    return 0;
}
