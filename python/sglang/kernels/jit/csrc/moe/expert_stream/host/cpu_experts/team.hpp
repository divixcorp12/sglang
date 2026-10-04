// The CPU expert engines and the team every forward runs: one OpenMP team per call, the calling thread as worker 0,
// worker i pinned to core i of the call's engine (engine 0: unpinned).
#pragma once
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sglang::cpu_experts {
// Internal linkage: each quant library's translation unit owns its state. Inline statics with external linkage
// are STB_GNU_UNIQUE, which the dynamic linker merges across every library in the process, even RTLD_LOCAL ones.
namespace {

// Why this thread's last register, free, forward or keep-warm call returned 1 ("" otherwise); each clears it on entry.
// Per library, like the rest of the framework; the C ABI does not export it, a quant's own wrappers read it.
inline std::string& last_error()
{
    static thread_local std::string error;
    return error;
}

// Records `what` as last_error(), prints it to stderr, and returns 1; never throws (it runs in the C ABI's catch blocks).
inline int fail(const char* what) noexcept
{
    try {
        last_error() = what;
        std::fprintf(stderr, "cpu_experts: %s\n", what);
    } catch (...) {
    }
    return 1;
}

// This library's engines. An engine is an immutable core list: worker i of every forward and keep-warm naming it runs
// on cores[i], the caller as worker 0. Each CPU expert engine thread (one per NUMA group) names its own, so two teams
// run at once on disjoint cores. Handle h is table[h - 1]; a destroyed entry is reset and its index never reused, so a
// stale handle is refused (2). Handle 0 is no engine: its workers run unpinned.
struct Engines
{
    using Cores = std::shared_ptr<const std::vector<int>>;

    // Cores must be distinct and in [0, CPU_SETSIZE), else 2. Not checked against the caller's affinity: the engine
    // thread creates its engine from a thread whose inherited mask may exclude the expert cores, and the workers pin
    // themselves outside it. A core that cannot be pinned fails the first call's pin (1).
    static int create(const int32_t* cores, int32_t n, int64_t* engine) noexcept
    {
        try {
            if (!cores || !engine || n < 1 || n > CPU_SETSIZE) return 2;
            for (int i = 0; i < n; ++i) {
                if (cores[i] < 0 || cores[i] >= CPU_SETSIZE) return 2;
                for (int j = 0; j < i; ++j)
                    if (cores[j] == cores[i]) return 2;
            }
            Cores list = std::make_shared<const std::vector<int>>(cores, cores + n);
            std::lock_guard<std::mutex> lock(mutex);
            table.push_back(std::move(list));
            *engine = int64_t(table.size());
            return 0;
        } catch (...) {
            return 1;
        }
    }

    // A call already running on `engine` keeps its cores (it holds the list). Returns 0, or 2 for a handle never
    // created or already destroyed.
    static int destroy(int64_t engine) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (engine < 1 || engine > int64_t(table.size()) || !table[size_t(engine - 1)]) return 2;
            table[size_t(engine - 1)].reset();
            return 0;
        } catch (...) {
            return 1;
        }
    }

    // The cores of `engine`, null for engine 0. *found is false for a handle never created or destroyed.
    static Cores find(int64_t engine, bool* found)
    {
        *found = true;
        if (engine == 0) return nullptr;
        std::lock_guard<std::mutex> lock(mutex);
        if (engine < 1 || engine > int64_t(table.size()) || !table[size_t(engine - 1)]) {
            *found = false;
            return nullptr;
        }
        return table[size_t(engine - 1)];
    }

private:
    static inline std::mutex mutex;
    static inline std::vector<Cores> table;
};

// The cores of the call this thread is running (null: engine 0), which run_team pins its workers to. ExpertForward
// sets it around a forward's dispatch (CallCores), so a quant's plan calls run_team without passing the engine down.
inline const std::vector<int>*& call_cores()
{
    static thread_local const std::vector<int>* cores = nullptr;
    return cores;
}

struct CallCores
{
    explicit CallCores(const std::vector<int>* cores) : saved(call_cores()) { call_cores() = cores; }
    ~CallCores() { call_cores() = saved; }
    CallCores(const CallCores&) = delete;
    CallCores& operator=(const CallCores&) = delete;
    const std::vector<int>* saved;
};

// Inside a team: pins worker `worker` to cores[worker] (null cores: no-op), setting `error` when it cannot; the caller
// checks that worker < cores->size(). Each thread remembers its last core, so a team on the same engine repins nothing.
inline void pin(int worker, const std::vector<int>* cores, std::atomic<int>& error)
{
    if (!cores) return;
    const int core = (*cores)[size_t(worker)];
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
template <class Body>
void run_team(int threads, Body&& body)
{
    if (threads < 1) throw std::runtime_error("CPU expert team needs at least one worker");
    const std::vector<int>* cores = call_cores();  // the caller's: a worker's call_cores() is its own thread's
    if (cores && size_t(threads) > cores->size())
        throw std::runtime_error("CPU expert worker count exceeds the engine's cores");
    std::atomic<int> pin_error{0};
    std::atomic<int> actual_workers{0};
    #pragma omp parallel num_threads(threads) shared(body, cores, pin_error, actual_workers)
    {
        const int worker = omp_get_thread_num(), n = omp_get_num_threads();
        if (worker == 0) actual_workers.store(n, std::memory_order_relaxed);
        pin(worker, cores, pin_error);
        #pragma omp barrier
        if (n == threads && !pin_error.load(std::memory_order_relaxed)) body(worker, n);
    }
    if (pin_error.load()) throw std::runtime_error("cannot pin CPU expert worker to its engine's core");
    if (actual_workers.load() != threads)
        throw std::runtime_error("OpenMP returned fewer CPU expert workers than requested");
}

}  // namespace
}  // namespace sglang::cpu_experts
