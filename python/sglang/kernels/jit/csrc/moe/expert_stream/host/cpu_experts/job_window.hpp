// The window of admitted CPU expert jobs and the scheduler every Team worker runs over it.
//
// A job is a kernel's phases of tasks (binding.hpp's CpuExpertJobKernel). The window holds W jobs in slots; a worker
// takes a run of tasks from the oldest slot whose current phase is ready and whose dependencies hold, so jobs overlap
// and no phase waits on a team barrier. Worker 0, the owner of the Team, is the one admitter and calls step() between
// its admissions; workers 1..n-1 run worker_loop() through Team::serve (start()).
//
// Dependencies, checked when a worker scans a ready phase:
//   gate        a staged job's kGatePhase waits until its gate word reads `open` or its stop flag is set;
//   same output the kOutPhase of a job waits until the out phase of the latest earlier job with the same out_key is
//               done.
//
// A slot lives Free -> Active -> (Complete ->) Free. A job of the published kind stays Complete until the done word has
// passed its sequence, so a later job finishing first never lets the word skip an unfinished one.
//
// Stale views: a worker may scan a slot that finished and was re-admitted since it read it. Everything a scan reads is
// an atomic, and a claim (one CAS on the phase's cursor) carries the job's generation, so a stale view claims nothing.
#pragma once
#include "binding.hpp"
#include "clock.hpp"
#include "kernel.hpp"
#include "team.hpp"
#include <x86intrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>

namespace sglang::cpu_experts {

/// \brief The kernel's out phase: Kernel::kOutPhase when it declares one, else its last phase.
template <class Kernel>
constexpr int out_phase_of()
{
    if constexpr (requires { Kernel::kOutPhase; }) return Kernel::kOutPhase;
    else return Kernel::kPhases - 1;
}

/// \brief An out_key no job shares: its out phase waits for no earlier job.
inline constexpr uint64_t kNoOutKey = UINT64_MAX;

/// \brief The default completion policy: nothing.
struct NoWindowHooks {};

/// \brief What step() did.
enum class WindowStep {
    kRan,    // ran a run of tasks, or settled a phase
    kIdle,   // nothing ready
    kGated,  // nothing ready, and a staged job waits on its gate word, which moves without moving the window word
};

/// \brief The window's always-on counters.
struct WindowCounters {
    uint64_t admissions = 0;       // jobs admitted
    uint64_t admissions_full = 0;  // admit() calls that found no free slot
    uint64_t both_pending = 0;     // admissions that found the target ring and the draft channel both pending (engine)
};

/// \brief A window of W job slots over `Kernel`.
///
/// Hooks may offer `on_complete(Slot&)` (run once per job by the worker that finishes it, before the done word can pass
/// the job: stats, fit, trace) and `on_run(Slot&, phase, t0, t1, worker, tsc_cycles)` (after each run of tasks); both
/// are called through the template, never virtually, and must not throw.
template <CpuExpertJobKernel Kernel, int W = 4, class Hooks = NoWindowHooks>
class JobWindow {
public:
    using Layer = typename Kernel::Layer;
    static constexpr int kPhases = Kernel::kPhases;
    static constexpr int kGatePhase = Kernel::kGatePhase;
    static constexpr int kOutPhase = out_phase_of<Kernel>();
    static constexpr int kSlots = W;
    static_assert(W >= 1 && W <= 64);
    static_assert(kOutPhase >= 0 && kOutPhase < kPhases);

    enum SlotState : uint8_t { kFree, kActive, kComplete };

    /// \brief One slot: a job in flight. Fields marked "hook" are for on_complete and on_run to read.
    struct Slot {
        // Written by the admitter before the slot turns Active; read by the workers that hold a claim, and by hooks.
        typename Kernel::Job job{};
        const Layer* layer = nullptr;
        ForwardCall call{};  // stage_two is cleared: the scheduler owns the gate
        uint64_t out_key = kNoOutKey;
        int64_t admit_ns = 0;
        uint32_t gen = 0;

        // The staged job's gate stats, as StageGate records them (hook). Written once each: calls and spun when the
        // gate phase is entered, waited_ns by the worker that sees the gate open.
        bool gate_timed = false;  // read the clock (called_ns, waited_ns)
        int32_t gate_calls = 0;
        bool gate_spun = false;
        int64_t gate_called_ns = 0;
        int64_t gate_waited_ns = 0;

        uint32_t seq() const { return seq_.load(std::memory_order_relaxed); }
        int kind() const { return kind_.load(std::memory_order_relaxed); }
        // The summed TSC cycles of every run of this job, on every worker (hook; final once the job is complete).
        uint64_t worker_tsc() const { return worker_tsc_.load(std::memory_order_relaxed); }
        int status() const { return Kernel::status(job); }

    private:
        friend class JobWindow;
        // Below: everything a scan reads, all atomic. A cursor is (generation << 32 | next task), alone on its line.
        struct alignas(64) Cursor {
            std::atomic<uint64_t> v{0};
        };
        struct alignas(64) Done {
            std::atomic<int> v{0};
        };

        std::atomic<uint8_t> state_{kFree};
        std::atomic<uint32_t> seq_{0};
        std::atomic<int> kind_{0};
        std::atomic<uint64_t> order_{0};  // admission order: the age the scheduler sorts by
        // (generation << 32 | phase + 1) of the phase a worker may scan; low half 0: none.
        alignas(64) std::atomic<uint64_t> ready_{0};
        std::array<std::atomic<int>, kPhases> count_{};
        std::array<std::atomic<int>, kPhases> align_{};
        std::atomic<uint32_t> out_complete_{0};  // generation of the last job whose out phase finished in this slot
        std::atomic<int> pred_{-1};              // slot of the previous job on the same output, -1: none
        std::atomic<uint32_t> pred_gen_{0};
        std::atomic<uint64_t> worker_tsc_{0};
        std::atomic<const std::atomic<uint32_t>*> gate_word_{nullptr};
        std::atomic<uint32_t> gate_open_{0};
        std::atomic<const std::atomic<bool>*> gate_stop_{nullptr};
        std::atomic<uint32_t> gate_state_{0};  // 0 not reached, 1 shut or unseen, 2 being opened, 3 open
        std::array<Cursor, kPhases> cursor_{};
        std::array<Done, kPhases> done_{};
    };

    /// \brief A window over `team`'s workers. The caller of the Team is worker 0, the only admitter.
    explicit JobWindow(Team& team, Hooks hooks = {}) : team_(team), hooks_(std::move(hooks)) {}
    JobWindow(const JobWindow&) = delete;
    JobWindow& operator=(const JobWindow&) = delete;

    /// \brief Publishes sequence numbers of jobs of kind `kind` to *word: it moves to s once every job of that kind
    /// up to seq s has finished (sequences run consecutively from *word + 1). Before start().
    void publish_done_to(std::atomic<uint32_t>* word, int kind)
    {
        done_word_ = word;
        done_kind_ = kind;
    }

    /// \brief Starts workers 1..n-1 on the scheduler loop. Worker 0 keeps the caller.
    void start() { team_.serve(loop_); }

    /// \brief Admits a job: prepares it into a free slot and makes its first phase ready. Worker 0 only. Returns false,
    /// admitting nothing, when the window is closed or full. `gate` (may be null) is a staged job's gate: its word,
    /// open value and stop flag must outlive the job. `out_key` names the output (row, part) the job writes, kNoOutKey
    /// for none: an earlier job still in the window with the same key is this one's predecessor.
    /// \throws whatever Kernel::prepare throws (std::invalid_argument); the window is then unchanged.
    bool admit(uint32_t seq, int kind, const Layer& layer, const ForwardCall& call, uint64_t out_key,
               const StageGate* gate)
    {
        if (closed_.load(std::memory_order_relaxed)) return false;
        Slot* const s = free_slot();
        if (s == nullptr) {
            full_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        s->call = call;
        s->call.stage_two = nullptr;
        s->layer = &layer;
        Kernel::prepare(s->job, layer, s->call);  // the slot is still Free: a throw leaves nothing behind

        do ++order_; while (uint32_t(order_) == 0);  // generation 0 is the idle out_complete_
        const uint32_t gen = uint32_t(order_);
        s->gen = gen;
        s->out_key = out_key;
        s->admit_ns = team_now_ns();
        s->seq_.store(seq, std::memory_order_relaxed);
        s->kind_.store(kind, std::memory_order_relaxed);
        s->order_.store(order_, std::memory_order_relaxed);
        s->worker_tsc_.store(0, std::memory_order_relaxed);

        int pred = -1;
        if (out_key != kNoOutKey) {
            uint64_t newest = 0;
            for (int i = 0; i < W; ++i) {
                const Slot& o = slots_[i];
                if (&o == s || o.state_.load(std::memory_order_acquire) == kFree || o.out_key != out_key) continue;
                const uint64_t age = o.order_.load(std::memory_order_relaxed);
                if (age > newest) newest = age, pred = i;
            }
        }
        s->pred_.store(pred, std::memory_order_relaxed);
        s->pred_gen_.store(pred < 0 ? 0 : slots_[pred].gen, std::memory_order_relaxed);

        s->gate_calls = 0;
        s->gate_spun = false;
        s->gate_called_ns = s->gate_waited_ns = 0;
        s->gate_timed = gate != nullptr && gate->timed;
        s->gate_state_.store(0, std::memory_order_relaxed);
        s->gate_open_.store(gate != nullptr ? gate->open : 0, std::memory_order_relaxed);
        s->gate_stop_.store(gate != nullptr ? gate->stop : nullptr, std::memory_order_relaxed);
        s->gate_word_.store(gate != nullptr ? gate->word : nullptr, std::memory_order_relaxed);

        for (int p = 0; p < kPhases; ++p) {
            s->cursor_[p].v.store(pack(gen, 0), std::memory_order_relaxed);
            s->done_[p].v.store(0, std::memory_order_relaxed);
            s->count_[p].store(0, std::memory_order_relaxed);
            s->align_[p].store(1, std::memory_order_relaxed);
        }
        s->ready_.store(pack(gen, 0), std::memory_order_relaxed);
        // Release: a worker that sees Active sees every field above.
        s->state_.store(kActive, std::memory_order_release);
        admissions_.fetch_add(1, std::memory_order_relaxed);
        advance(*s, gen, -1);
        return true;
    }

    /// \brief One scheduler step for `worker`: runs one claimed run of tasks of the oldest ready slot, or settles a
    /// zero-task phase whose dependencies now hold. Worker 0 calls it between admissions.
    WindowStep step(int worker) noexcept
    {
        int idx[W];
        uint64_t age[W];
        int n = 0;
        for (int i = 0; i < W; ++i) {
            if (slots_[i].state_.load(std::memory_order_acquire) != kActive) continue;
            const uint64_t a = slots_[i].order_.load(std::memory_order_relaxed);
            int j = n++;
            for (; j > 0 && age[j - 1] > a; --j) age[j] = age[j - 1], idx[j] = idx[j - 1];
            age[j] = a, idx[j] = i;
        }
        bool gated = false;
        for (int k = 0; k < n; ++k) {
            Slot& s = slots_[idx[k]];
            // Acquire: pairs with the release that published this phase, so the tasks' inputs are visible.
            const uint64_t r = s.ready_.load(std::memory_order_acquire);
            const int ph = int(uint32_t(r)) - 1;
            if (ph < 0) continue;
            const uint32_t gen = uint32_t(r >> 32);
            if (!deps_ok(s, ph)) {
                if (ph == kGatePhase && s.gate_word_.load(std::memory_order_relaxed) != nullptr
                    && s.gate_state_.load(std::memory_order_relaxed) != 3)
                    gated = true;
                continue;
            }
            // Read before the cursor: a recycled slot shows the new generation there, which fails the claim.
            const int count = s.count_[ph].load(std::memory_order_relaxed);
            if (count == 0) {
                uint64_t expect = r;
                if (s.ready_.compare_exchange_strong(expect, pack(gen, 0), std::memory_order_acq_rel,
                                                     std::memory_order_relaxed)) {
                    advance(s, gen, ph);
                    return WindowStep::kRan;
                }
                continue;
            }
            int t0, t1;
            if (!claim(s, gen, ph, count, t0, t1)) continue;
            const uint64_t c0 = __rdtsc();
            Kernel::run(s.job, ph, t0, t1, worker);
            const uint64_t cycles = __rdtsc() - c0;
            s.worker_tsc_.fetch_add(cycles, std::memory_order_relaxed);
            if constexpr (requires { hooks_.on_run(s, ph, t0, t1, worker, cycles); })
                hooks_.on_run(s, ph, t0, t1, worker, cycles);
            // Release publishes the run's outputs; the acquire half makes the worker that completes the phase see
            // every earlier run's, and its later release of the next phase hands them on.
            const int prev = s.done_[ph].v.fetch_add(t1 - t0, std::memory_order_acq_rel);
            if (prev + (t1 - t0) == count) advance(s, gen, ph);
            return WindowStep::kRan;
        }
        return gated ? WindowStep::kGated : WindowStep::kIdle;
    }

    /// \brief The window word: moves on every admission, phase transition, job completion and close. Read it before a
    /// step() and hand it to idle_wait.
    uint32_t word() const { return word_.load(std::memory_order_acquire); }

    /// \brief Idles until the window word differs from `seen`: the Team's wait (the kernel's register work until
    /// warm_until_ns, PAUSE after). With poll_gates it also returns after kGatePollNs, because a gate word moves
    /// without moving the window word.
    void idle_wait(uint32_t seen, bool poll_gates, int64_t warm_until_ns) const
    {
        const int64_t until = poll_gates ? team_now_ns() + kGatePollNs : INT64_MAX;
        team_.wait(reinterpret_cast<const uint32_t*>(&word_), seen, warm_until_ns, until);
    }

    /// \brief No more admissions; admitted jobs drain. Worker 0 only. The workers leave worker_loop once no slot is
    /// Active.
    void close()
    {
        closed_.store(true, std::memory_order_release);
        bump();
    }

    /// \brief Worker 0 after close(): steps until every admitted job has completed.
    void drain() noexcept
    {
        while (any_active()) {
            const uint32_t seen = word();
            const WindowStep r = step(0);
            if (r != WindowStep::kRan) idle_wait(seen, r == WindowStep::kGated, 0);
        }
    }

    /// \brief After close() and drain(): waits for workers 1..n-1 to leave their loops.
    void join() const { team_.join_serve(); }

    bool any_active() const
    {
        for (const Slot& s : slots_)
            if (s.state_.load(std::memory_order_acquire) == kActive) return true;
        return false;
    }

    /// \brief The engine counts admissions that found both sources pending here.
    void note_both_pending() { both_pending_.fetch_add(1, std::memory_order_relaxed); }

    WindowCounters counters() const
    {
        return {admissions_.load(std::memory_order_relaxed), full_.load(std::memory_order_relaxed),
                both_pending_.load(std::memory_order_relaxed)};
    }

    const Slot& slot(int i) const { return slots_[i]; }
    int workers() const { return team_.workers(); }

    /// \brief The loop of workers 1..n-1.
    void worker_loop(int worker) noexcept
    {
        int64_t warm_until = 0;
        bool was_running = false;
        for (;;) {
            const uint32_t seen = word();  // before the scan: a move during it ends the wait at once
            const WindowStep r = step(worker);
            if (r == WindowStep::kRan) {
                was_running = true;
                continue;
            }
            if (closed_.load(std::memory_order_acquire) && !any_active()) return;
            if (was_running) {
                warm_until = team_now_ns() + team_.keep_warm_ns();
                was_running = false;
            }
            idle_wait(seen, r == WindowStep::kGated, warm_until);
        }
    }

private:
    static constexpr int64_t kGatePollNs = 2000;
    static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free);

    static constexpr uint64_t pack(uint32_t gen, uint32_t low) { return uint64_t(gen) << 32 | low; }

    struct Loop {
        JobWindow* window;
        void operator()(int worker, int) const noexcept { window->worker_loop(worker); }
    };

    void bump() { word_.fetch_add(1, std::memory_order_release); }

    Slot* free_slot()
    {
        for (Slot& s : slots_)
            if (s.state_.load(std::memory_order_acquire) == kFree) return &s;
        return nullptr;
    }

    // Takes a run of tasks of phase `ph`: max(align, remaining / 2n) rounded up to align, at most what remains. A CAS
    // and not a fetch_add: a stale view of a recycled slot must be able to fail without moving the new job's cursor.
    // Relaxed: the phase's inputs came with the acquire of ready_, and the cursor orders nothing else.
    bool claim(Slot& s, uint32_t gen, int ph, int count, int& t0, int& t1)
    {
        const int align = std::max(s.align_[ph].load(std::memory_order_relaxed), 1);
        std::atomic<uint64_t>& cursor = s.cursor_[ph].v;
        uint64_t v = cursor.load(std::memory_order_relaxed);
        for (;;) {
            if (uint32_t(v >> 32) != gen) return false;
            const int next = int(uint32_t(v));
            const int remaining = count - next;
            if (remaining <= 0) return false;
            int run = std::max(align, remaining / (2 * team_.workers()));
            run = std::min((run + align - 1) / align * align, remaining);
            if (cursor.compare_exchange_weak(v, v + uint64_t(run), std::memory_order_relaxed)) {
                t0 = next, t1 = next + run;
                return true;
            }
        }
    }

    bool deps_ok(Slot& s, int ph) noexcept
    {
        if (ph == kGatePhase && !gate_open(s)) return false;
        if (ph == kOutPhase) {
            const int p = s.pred_.load(std::memory_order_relaxed);
            // Acquire: the predecessor's out writes are visible before this job writes the same rows. The signed
            // difference holds across the 32-bit generation wrap, and a recycled predecessor slot holds a later one.
            if (p >= 0 && int32_t(slots_[p].out_complete_.load(std::memory_order_acquire)
                                  - s.pred_gen_.load(std::memory_order_relaxed)) < 0)
                return false;
        }
        return true;
    }

    // The scheduler's StageGate::wait: true once the gate word reads `open` or the stop flag is set. The worker that
    // first sees it records the wait; the others read "shut" until it has, so the stats are complete before any task of
    // the gate phase runs.
    bool gate_open(Slot& s) noexcept
    {
        const std::atomic<uint32_t>* word = s.gate_word_.load(std::memory_order_relaxed);
        if (word == nullptr) return true;
        uint32_t state = s.gate_state_.load(std::memory_order_acquire);
        if (state == 3) return true;
        if (state != 1) return false;
        const std::atomic<bool>* stop = s.gate_stop_.load(std::memory_order_relaxed);
        if (word->load(std::memory_order_acquire) != s.gate_open_.load(std::memory_order_relaxed)
            && !(stop != nullptr && stop->load(std::memory_order_relaxed)))
            return false;
        if (!s.gate_state_.compare_exchange_strong(state, 2, std::memory_order_acq_rel, std::memory_order_relaxed))
            return false;
        s.gate_waited_ns = s.gate_timed && s.gate_spun ? team_now_ns() - s.gate_called_ns : 0;
        s.gate_state_.store(3, std::memory_order_release);
        return true;
    }

    // The gate phase is entered: StageGate::wait's call, one per staged job.
    void arrive(Slot& s) noexcept
    {
        s.gate_calls = 1;
        if (s.gate_timed) s.gate_called_ns = team_now_ns();
        s.gate_spun = s.gate_word_.load(std::memory_order_relaxed)->load(std::memory_order_acquire)
                      != s.gate_open_.load(std::memory_order_relaxed);
        s.gate_state_.store(1, std::memory_order_release);  // publishes the stats above with the phase
    }

    // Called by the one thread that finished phase `ph` of generation `gen` (-1: at admission): enters the next phase,
    // or completes the job. A phase with no tasks and no pending dependency is skipped here; one whose dependency
    // is pending is published like any other and settled by whichever worker's scan finds it holding.
    void advance(Slot& s, uint32_t gen, int ph) noexcept
    {
        for (;;) {
            // Release: the successor's out phase acquires it, and so sees this job's out writes.
            if (ph == kOutPhase) s.out_complete_.store(gen, std::memory_order_release);
            if (++ph == kPhases) {
                complete(s, gen);
                return;
            }
            const Tasks t = Kernel::tasks(s.job, ph);
            s.count_[ph].store(t.count, std::memory_order_relaxed);
            s.align_[ph].store(std::max(t.align, 1), std::memory_order_relaxed);
            if (ph == kGatePhase && s.gate_word_.load(std::memory_order_relaxed) != nullptr) arrive(s);
            if (t.count == 0 && deps_ok(s, ph)) continue;
            // Release: publishes the phase's count and the previous phases' outputs to the workers that scan it.
            s.ready_.store(pack(gen, uint32_t(ph) + 1), std::memory_order_release);
            bump();
            return;
        }
    }

    void complete(Slot& s, uint32_t gen) noexcept
    {
        s.ready_.store(pack(gen, 0), std::memory_order_relaxed);
        if constexpr (requires { hooks_.on_complete(s); }) hooks_.on_complete(s);
        if (done_word_ != nullptr && s.kind() == done_kind_) {
            s.state_.store(kComplete, std::memory_order_release);
            publish_done();
        } else {
            s.state_.store(kFree, std::memory_order_release);
        }
        bump();
    }

    // Moves the done word over the Complete slots in sequence order. The CAS publishes each value once, by the one
    // thread that moved it, and that thread frees the slot; a job whose predecessor is unfinished is not found.
    void publish_done() noexcept
    {
        for (;;) {
            uint32_t p = done_word_->load(std::memory_order_acquire);
            Slot* hit = nullptr;
            for (Slot& s : slots_)
                if (s.state_.load(std::memory_order_acquire) == kComplete && s.kind() == done_kind_
                    && s.seq() == p + 1) {
                    hit = &s;
                    break;
                }
            if (hit == nullptr) return;
            // Release: the engine that reads the word sees the job's outputs and its on_complete bookkeeping.
            if (done_word_->compare_exchange_strong(p, p + 1, std::memory_order_acq_rel, std::memory_order_acquire))
                hit->state_.store(kFree, std::memory_order_release);
        }
    }

    Team& team_;
    Hooks hooks_;
    Loop loop_{this};
    std::array<Slot, W> slots_{};
    std::atomic<uint32_t>* done_word_ = nullptr;
    int done_kind_ = 0;
    uint64_t order_ = 0;  // the admitter's
    std::atomic<bool> closed_{false};
    alignas(64) std::atomic<uint32_t> word_{0};
    std::atomic<uint64_t> admissions_{0}, full_{0}, both_pending_{0};
};

}  // namespace sglang::cpu_experts
