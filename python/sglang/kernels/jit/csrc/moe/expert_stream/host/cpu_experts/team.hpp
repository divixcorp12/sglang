// The team every forward runs: one OpenMP team per call, the calling thread as worker 0, worker i pinned to core i of
// the call's cores (none: unpinned).
#pragma once
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

// The cores of the call this thread is running (empty: unpinned), which run_team pins its workers to. ExpertForward
// sets it around a forward's dispatch (CallCores), so a quant's plan calls run_team without passing the cores down.
inline std::span<const int>& call_cores()
{
    static thread_local std::span<const int> cores;
    return cores;
}

struct CallCores
{
    explicit CallCores(std::span<const int> cores) : saved(call_cores()) { call_cores() = cores; }
    ~CallCores() { call_cores() = saved; }
    CallCores(const CallCores&) = delete;
    CallCores& operator=(const CallCores&) = delete;
    std::span<const int> saved;
};

// Throws std::invalid_argument unless every core is in [0, CPU_SETSIZE) and none repeats. Not checked against the
// caller's affinity: the engine thread may run under a mask that excludes the expert cores, and the workers pin
// themselves outside it; a core that cannot be pinned fails the call's pin (std::runtime_error).
inline void check_cores(std::span<const int> cores)
{
    for (size_t i = 0; i < cores.size(); ++i) {
        if (cores[i] < 0 || cores[i] >= CPU_SETSIZE)
            throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " is outside [0, CPU_SETSIZE)");
        for (size_t j = 0; j < i; ++j)
            if (cores[j] == cores[i])
                throw std::invalid_argument("CPU expert core " + std::to_string(cores[i]) + " repeats");
    }
}

// Inside a team: pins worker `worker` to cores[worker] (empty cores: no-op), setting `error` when it cannot; the caller
// checks that worker < cores.size(). Each thread remembers its last core, so a team on the same cores repins nothing.
inline void pin(int worker, std::span<const int> cores, std::atomic<int>& error)
{
    if (cores.empty()) return;
    const int core = cores[size_t(worker)];
    static thread_local int pinned_core = -1;
    if (pinned_core != core || sched_getcpu() != core) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set))
            error.store(1, std::memory_order_relaxed);
        else
            pinned_core = core;
    }
}

// body(worker, workers) on every worker of a team of `threads` on call_cores(). Every pin precedes a barrier and the
// body runs only on a full, fully pinned team, so all workers take the same branch; body must not throw (it runs
// inside an OpenMP region).
struct NoTeamObserver {
    void worker_enter(int) {} void worker_ready(int) {}
};

template <class Body, class Observer = NoTeamObserver>
void run_team(int threads, Body&& body, Observer&& observer = NoTeamObserver{})
{
    if (threads < 1) throw std::runtime_error("CPU expert team needs at least one worker");
    const std::span<const int> cores = call_cores();  // the caller's: a worker's call_cores() is its own thread's
    if (!cores.empty() && size_t(threads) > cores.size())
        throw std::runtime_error("CPU expert worker count exceeds the call's cores");
    std::atomic<int> pin_error{0};
    std::atomic<int> actual_workers{0};
    #pragma omp parallel num_threads(threads) shared(body, cores, pin_error, actual_workers, observer)
    {
        const int worker = omp_get_thread_num(), n = omp_get_num_threads();
        observer.worker_enter(worker);
        if (worker == 0) actual_workers.store(n, std::memory_order_relaxed);
        pin(worker, cores, pin_error);
        #pragma omp barrier
        observer.worker_ready(worker);
        if (n == threads && !pin_error.load(std::memory_order_relaxed)) body(worker, n);
    }
    if (pin_error.load()) throw std::runtime_error("cannot pin CPU expert worker to its core");
    if (actual_workers.load() != threads)
        throw std::runtime_error("OpenMP returned fewer CPU expert workers than requested");
}

}  // namespace
}  // namespace sglang::cpu_experts
